"""
Author: Daniel Norlan[212888], Elias Silva[261670]
        
Config edit: Josef Rosland-Borlaug[258139]   
    - Create a bridge between this system and config module (unified).
    - Find out where the ouputs are sent ,"catch" them and present them in GUI
        - I think everything is saved as features .npy array features. Not sure what to do with that?              

              
"""
#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

# mlp.py
#
# Summary
# We classify engine condition (good / broken / heavyload) from WAV files using:
#   - Hand-crafted audio features:
#       - log-mel spectrogram + deltas (Δ) + delta-deltas (ΔΔ)
#       - MFCC (mel-frequency cepstral coefficients) + Δ + ΔΔ
#       - pitch/tonal features: chroma + tonnetz
#       - a few spectral stats: centroid, bandwidth, roll-off, contrast, flatness, ZCR (zero-crossing rate), RMS (root-mean-square energy)
#   - One scikit-learn pipeline: StandardScaler -> PCA(256) -> MLPClassifier
#   - Train-time augments (tiny time-stretch/pitch/gain) and test-time averaging (TTA)

#   - Segment-level predictions pooled to one file-level decision
#
# Example runs:
#   python src/mlp.py --ensemble 7 --tta 2 --hop 0.5
#   python src/mlp.py --ensemble 3 --tta 1 --hop 1.0
#

import argparse, os, warnings, hashlib, random, math
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import joblib
import librosa
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight

#Connection to config module 
import sys #I might need to create a fallback getter, because I can't seem to connect with my local config module...? BUG SOLVED: forgot the "()" at the end
from configs import Config #Connect it to the "global root system created in config".
cfg = Config() 


#

# librosa tends to warn a lot on some operations; hide the noise in normal runs
warnings.filterwarnings("ignore", category=UserWarning)

# CONFIG 

# Audio is resampled to 22.05 kHz. Common choice for audio ML, keeps arrays small and fast.
SR = 22050

# STFT settings for two “views” of the spectrum:
# - fine: shorter window, more frames, better timing
# - coarse: longer window, fewer frames, a bit more frequency context
N_FFT_FINE, HOP_FINE   = 1024, 256
N_FFT_COAR, HOP_COAR   = 2048, 512

# Mel filterbank sizes for the two branches
MELS_FINE, MELS_COAR   = 128, 64

# Segmentation:
#   seg: window length in seconds
#   hop: step between windows in seconds
# Small hop = more overlap = more segments = steadier voting (but slower).
DEFAULT_SEG   = 2.0
DEFAULT_HOP_S = 0.5

# Ensemble and TTA:
#   ensemble: how many independently seeded MLPs to train and average
#   tta: at test time, how many lightly jittered copies per file to average
DEFAULT_ENSEMBLE = 7
DEFAULT_TTA      = 2

# Slightly stronger augmentation for “good” and “heavyload” helps reduce false “broken”.
DEFAULT_STRONG_GOOD_AUG = True

# File-level pooling:
#   We drop the outer 5% of segments (edges are often silence or clicks),
#   then, for each class, average only the top 15% most confident log-probabilities.
CENTER_FRACTION = 0.95
TOPQ_FRACTION   = 0.15

# PCA size: the raw feature vector is large; 256 dims is a good trade-off
PCA_DIM = 256

# MLP settings (picked earlier and left fixed here)
MLP_HIDDEN = (512, 256)
MLP_ALPHA  = 5e-5
MLP_LR     = 5e-4
MLP_BS     = 64
MLP_MAXIT  = 320

# Seeds for reproducibility
SEED0 = 1337
RNG = np.random.default_rng(123)
random.seed(123)

# I/O helpers
def list_audio_by_class(root: Path) -> List[Tuple[Path, str]]:
    """
    Scan a folder like:
        data/train_cut/engine1_good/*.wav
        data/train_cut/engine2_broken/*.wav
        data/train_cut/engine3_heavyload/*.wav
    Returns [(path, label), ...]. Fails early if the folder is empty so users notice.
    """
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    pairs: List[Tuple[Path,str]] = []               # Make an empty list to store (file_path, label) pairs.
    for cls_dir in sorted([p for p in root.iterdir() if p.is_dir()]): # Go through all subfolders inside root, sorted alphabetically.
        for wav in sorted(cls_dir.rglob("*.wav")):  #Find .wav files under class folder, then add to list
            pairs.append((wav, cls_dir.name))
    if not pairs:                                   # If we didn’t find any WAVs, fail 
        raise RuntimeError(f"No .wav files found under {root}")
    return pairs                                    # Give back the full list of (file, label) items


def group_id_from_path(p: Path, label: str) -> str: # Make a short string that uniquely identifies the original file, so we can group all its segments later.
    """
    Return <foldername>/<filename_without_extension>. p.parent.name is the label folder 
    (e.g., engine1_good).

    p.stem is the file name without .wav.

    Combining them yields an ID thats unique within the dataset and human-readable, e.g., 
    engine1_good/00123.
    """
    return f"{p.parent.name}/{p.stem}"              


def load_wav(path: Path, sr: int = SR) -> np.ndarray:
    """
    Read a WAV file, convert it to mono, resample to our sample rate, 
    and scale it to fit between −1 and 1.
    """
    y, _ = librosa.load(str(path), sr=sr, mono=True)    # Read the audio as a NumPy array y, resampled to sr, collapsed to mono.
    m = float(np.max(np.abs(y)) + 1e-12)                # Find peak magnitude; add a tiny epsilon to avoid divide-by-zero.
    return (y / m).astype(np.float32)                   # Scale the whole waveform so the peak is 1.0 (or −1.0), and make sure the data type is 32-bit float.

def segment_wave(y: np.ndarray, sr: int, seg_dur_s: float, hop_dur_s: float):
    """
    Cut the waveform y into many overlapping chunks. If the file is too short for one chunk, 
    pad it by mirroring the signal.
    """
    seg_len = int(round(seg_dur_s * sr))
    hop_len = int(round(hop_dur_s * sr))    # Convert the segment length and hop (both given in seconds) into sample counts.
    if y.size < seg_len:                    # If the audio isn’t long enough for even one segment, extend it by reflecting the waveform to reach seg_len.
        y = np.pad(y, (0, seg_len - y.size), mode="reflect")
    segs = [y[i:i+seg_len] for i in range(0, max(y.size - seg_len + 1, 1), hop_len)]
    # Make a list of slices of length seg_len, starting at i = 0, hop_len, 2*hop_len, … until we run out of audio.
    if not segs:                            
        segs = [y[:seg_len]]                # # If, for any reason, we didn’t get any segments, force a single segment from the start.
    return segs                             # Return the list of 1D NumPy arrays, one per segment.

# ---------------- Features (input to MLP model) ----------------
"""
Build the actual inputs to the neural net. We take spectrogram-like features over time, then squash each 
segment to a fixed-size vector by taking the mean and standard deviation across time.
"""
def _agg_time(M: np.ndarray) -> np.ndarray:
    """This helper takes a 2D matrix where rows are features and columns are time frames. 
    It returns one long vector: all the means followed by all the standard deviations."""
    return np.concatenate([np.mean(M, axis=1), np.std(M, axis=1, ddof=1)], axis=0)

def _spec_stack(y: np.ndarray, sr: int, n_fft: int, hop: int, n_mels: int) -> np.ndarray:
    """
    "Spectral stack":
      - mel-spectrogram (log-scaled) + deltas + delta-deltas
      - spectral centroid, bandwidth, rolloff, contrast, flatness
      - zero-crossing rate (ZCR), RMS (frame energy)
    We compute all features frame-wise, convert mel to dB, then summarize each by mean/std over time.
    """

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))        # Compute the magnitude spectrogram via STFT.
    mel = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=n_mels) # Build mel features from the STFT magnitude and convert to dB; we keep the recipe consistent and time-pool via mean/std.
    logmel = librosa.power_to_db(mel + 1e-10)                       # Add epsilon before log to avoid -inf; 1e-10 corresponds to a ~-100 dB floor (10*log10(1e-10)).
    d1 = librosa.feature.delta(logmel, order=1)                     # Compute Δ (first derivative over time) and ΔΔ (second derivative) of the log-mel features.
    d2 = librosa.feature.delta(logmel, order=2)                     # ΔΔ (second derivative) of the log-mel features.
    cent = librosa.feature.spectral_centroid(S=S, sr=sr)            # Centroid: “center of mass” of the spectrum (higher = brighter).
    bw   = librosa.feature.spectral_bandwidth(S=S, sr=sr)           # Bandwidth: spread of the spectrum around the centroid.
    roll = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85) #  Rolloff (85%): frequency below which 85% of energy lies.
    contr= librosa.feature.spectral_contrast(S=S, sr=sr)            # Contrast: difference between peaks and valleys per frequency band (captures “noisiness vs tonality”).
    flat = librosa.feature.spectral_flatness(S=S)                   # Flatness: how noise-like the spectrum is (flat noise ≈ 1, peaky/tonal ≈ 0).
    zcr  = librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=hop) # ZCR (zero-crossing rate): proxy for high-frequency/noisy content (more zero-crossings ⇒ more high-frequency energy).
    rms  = librosa.feature.rms(S=S, frame_length=n_fft, hop_length=hop, center=True, pad_mode="reflect") # RMS: frame energy (simple loudness proxy).

    return np.concatenate([
        _agg_time(logmel), _agg_time(d1), _agg_time(d2),
        _agg_time(cent), _agg_time(bw), _agg_time(roll),
        _agg_time(contr), _agg_time(flat), _agg_time(zcr), _agg_time(rms)
    ], axis=0)
   

def _mfcc_stack(y: np.ndarray, sr: int, n_fft: int, hop: int, n_mfcc: int = 20) -> np.ndarray:
    """
    MFCC stack: compute MFCC + Δ + ΔΔ from log-mel features, then summarize each by mean/std over time.
    Captures broad spectral shape and how it changes across frames.
    """

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    mel = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=128)
    logmel = librosa.power_to_db(mel + 1e-10)       # Make a magnitude spectrogram, turn it into a 128-band mel spectrogram, then take log.
    mfcc = librosa.feature.mfcc(S=logmel, sr=sr, n_mfcc=n_mfcc)
    d1 = librosa.feature.delta(mfcc, order=1)
    d2 = librosa.feature.delta(mfcc, order=2)       # Compute MFCCs and their time derivatives Δ and ΔΔ.
    return np.concatenate([_agg_time(mfcc), _agg_time(d1), _agg_time(d2)], axis=0)  # Mean/std-pool each of MFCC, Δ, ΔΔ; then concatenate.

def _chroma_stack(y: np.ndarray, sr: int, n_fft: int, hop: int) -> np.ndarray:
    """
    Compute chroma (12 pitch classes) and tonnetz (tonal centroids) features, then we will mean/std them.
    Chroma: compresses frequency to 12 classes (C..B), folding octaves together. 
    It picks up harmonic content regardless of the octave.
    Tonnetz: (tonal centroid features) maps pitch/chord relations into a geometric space (captures intervals like fifths/thirds). In librosa, 
    its computed from the harmonic component of the signal.
    """
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)            # Magnitude spectrogram → chroma features per frame.
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr) # Extract the harmonic part of the audio, then compute tonnetz features.
    return np.concatenate([_agg_time(chroma), _agg_time(tonnetz)], axis=0)  # Mean/std-pool chroma and tonnetz, then concatenate them.

def feature_vector(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    For one segment, build four blocks of features (coarse spectral, fine spectral, MFCC, and chroma/tonnetz), 
    then glue them together into one long vector and cast to float32.
    """
    coarse = _spec_stack(y, sr, N_FFT_COAR, HOP_COAR, MELS_COAR)
    fine   = _spec_stack(y, sr, N_FFT_FINE, HOP_FINE, MELS_FINE)    # Build the coarse and fine spectral stacks with their respective FFT/hop/mel settings.
    mfccs  = _mfcc_stack(y, sr, N_FFT_FINE, HOP_FINE, n_mfcc=20)    # Compute 20 MFCCs (plus Δ and ΔΔ) using the fine STFT timing.
    chroma = _chroma_stack(y, sr, N_FFT_FINE, HOP_FINE)             # Compute chroma and tonnetz (with fine timing), then mean/std them.
    return np.concatenate([coarse, fine, mfccs, chroma], axis=0).astype(np.float32) # Stick all four parts together into one long vector and convert to 32-bit float.

# ---------------- Train-time augmentation (small and realistic) ----------------
# These tweaks simulate small RPM/pitch and loudness differences. Reason: nudge the training audio so the model isn’t brittle to tiny timing/tuning changes.


def aug_time_stretch(y, sr):
    """Change speed a tiny bit (±5%). If anything fails, use the original. 
    Make sure the output is exactly the same length as the input by padding or trimming. Return as 32-bit float."""
    rate = float(np.clip(RNG.uniform(0.95, 1.05), 0.9, 1.1))
    try: out = librosa.effects.time_stretch(y, rate=rate)
    except Exception: out = y
    if out.size < y.size: out = np.pad(out, (0, y.size - out.size), mode="reflect")
    return out[:y.size].astype(np.float32)

def aug_pitch_shift(y, sr):
    """Move pitch up/down by at most 1 semitone (small musical step). Keep same length, pad/trim as needed, return float32."""
    steps = float(RNG.uniform(-1.0, 1.0))
    try: out = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
    except Exception: out = y
    if out.size < y.size: out = np.pad(out, (0, y.size - out.size), mode="reflect")
    return out[:y.size].astype(np.float32)

def aug_rand_gain(y):
    """Multiply the signal by a small random factor (around 1.0). Then renormalize so peaks sit at ±1 again. Return float32."""
    g = float(np.clip(RNG.normal(1.0, 0.08), 0.8, 1.25))
    out = y * g
    m = float(np.max(np.abs(out)) + 1e-12)
    return (out / m).astype(np.float32)

def make_augments(y, sr, label: str, strong_good=True):
    """
    Small, fast aug set (kept tiny on purpose).
    - In 'strong' mode for good/heavyload, we include time-stretch + pitch; in other cases time-stretch + gain.
    - We always cap the list to 3 items: [original, aug1, aug2] to keep training time reasonable.
    """
    base = [y]
    if label in ("engine1_good", "engine3_heavyload") and strong_good:
        base += [aug_time_stretch(y, sr), aug_pitch_shift(y, sr), aug_rand_gain(y)]
    else:
        base += [aug_time_stretch(y, sr), aug_rand_gain(y)]
    return base[:3]

# ---------------- Caching + parallel feature extraction ----------------
# We save computed features to disk so we don’t recompute them every run.

def _hash_key(path: Path, seg: float, hop_s: float, sr: int, strong_good: bool):
    """Build a unique ID string from the file path and the important settings. We use MD5 (fast hashing function) to get a short, fixed hash."""
    h = hashlib.md5()
    h.update(str(path).encode())
    h.update(f"|{seg}|{hop_s}|{sr}|mspec_mfcc_chroma_v1".encode())
    h.update(f"|aug{int(strong_good)}".encode())
    return h.hexdigest()

def extract_file_features(args):
    """
    Does all the work for one WAV: load → (maybe) augment → segment → make features → save .npy → return arrays.
    """
    wav_path, label, sr, seg_dur, hop_dur, cache_dir, do_aug, strong_good = args    # Unpack the arguments tuple.
    key = _hash_key(wav_path, seg_dur, hop_dur, sr, strong_good)                    # Create a unique hash key for this file and settings.
    npy = cache_dir / f"{key}.npy"                                                  # Build the cache file path based on the hash key.
    if npy.exists():                                                                # If the .npy cache file already exists, load and return its contents.
        data = np.load(npy, allow_pickle=True).item()
        return data["X"], data["y"], data["groups"]

    y = load_wav(wav_path, sr)                                          # Load the audio (mono, normalized, resampled) and cut it into overlapping segments.
    segs = segment_wave(y, sr, seg_dur, hop_dur)                       
    seg_sources = segs                                                 
    if do_aug:                                                          # If augmentation is enabled (training), create 2–3 augmented versions of the waveform,
        seg_sources = []                                                # segment each, and use those segments instead.                                      
        for a in make_augments(y, sr, label, strong_good=strong_good):  
            seg_sources.extend(segment_wave(a, sr, seg_dur, hop_dur))   

    rows, labels, groups = [], [], []                                   # Prepare empty lists to collect features (rows), labels (labels), and group IDs (groups). 
    gid = group_id_from_path(wav_path, label)                           # Build a group ID string for the original file.               
    for s in seg_sources:                                               # For every segment, compute its feature vector and store it, along with its class label and file ID.
        rows.append(feature_vector(s, sr))
        labels.append(label)
        groups.append(gid)

    X = np.vstack(rows); yv = np.array(labels); gv = np.array(groups)   # Stack all feature vectors into one 2D array X and turn the label and group lists into 1D arrays.
    np.save(npy, {"X": X, "y": yv, "groups": gv}, allow_pickle=True)    # Save everything to the cache and return it.
    return X, yv, gv

def build_dataset(pairs, sr, seg_dur, hop_dur, cache_dir: Path, n_jobs: int, do_aug: bool, strong_good: bool):
    """
    Run extract_file_features for every file, in parallel, then join everything into one big dataset.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    Xs, Ys, Gs = [], [], []
    tasks = [(p, lab, sr, seg_dur, hop_dur, cache_dir, do_aug, strong_good) for (p, lab) in pairs]
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = [ex.submit(extract_file_features, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Extracting (cached)"):
            X, y, g = fut.result()
            Xs.append(X); Ys.append(y); Gs.append(g)
    X = np.vstack(Xs); y = np.concatenate(Ys); groups = np.concatenate(Gs)
    return X, y, groups

# ---------------- Test-time augmentation (TTA) ----------------
def tta_variants(y: np.ndarray, sr: int, n: int):
    """Build a list with the original waveform plus n copies, each either slightly time-stretched or slightly pitch-shifted."""
    outs = [y]
    for _ in range(n):
        outs.append(aug_time_stretch(y, sr) if RNG.random() < 0.5 else aug_pitch_shift(y, sr))
    return outs

def features_for_file_with_tta(path: Path, sr: int, seg: float, hop_s: float, tta: int):
    """
    Load the file, make TTA jittered copies, cut each into seg-length overlapping segments, compute the feature vector for every segment, 
    and stack them into one big matrix.
    """
    y0 = load_wav(path, sr)
    Ys = tta_variants(y0, sr, n=tta) if tta > 0 else [y0]
    feats = []
    for y in Ys:
        segs = segment_wave(y, sr, seg, hop_s)
        feats.extend([feature_vector(s, sr) for s in segs])
    return np.vstack(feats)

def _feat_for_one_file(args):
    """Helper used in parallel: for one file, compute its test-time features and return them along with the known label and path."""
    p, true_lab, sr, seg, hop_s, tta = args
    Xf = features_for_file_with_tta(p, sr, seg, hop_s, tta)
    return (true_lab, Xf, str(p))

# ---------------- One clean sklearn Pipeline ----------------
def make_mlp(random_state: int = 42):
    """
    Entire model in one pipeline:
      1) StandardScaler: normalize features (zero mean, unit var)
      2) PCA(256, whiten=True): decorrelate + compress (speeds up and regularizes)
      3) MLPClassifier: ReLU + Adam with the strong defaults above
    """
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("pca", PCA(n_components=PCA_DIM, whiten=True, random_state=random_state)),
        ("clf", MLPClassifier(
            activation="relu",              # Rectified Linear Unit; fast and works well in practice.
            solver="adam",                  # Adam optimizer: popular, robust, needs little tuning.               
            learning_rate_init=MLP_LR,      # Small learning rate; we have a lot of data/steps so we can go slow and steady.
            alpha=MLP_ALPHA,                # L2 weight regularization (prevents overfitting).
            batch_size=MLP_BS,              # Mini-batch size for each training step.
            early_stopping=True,            # Early stopping monitors validation SCORE; stops after 12 epochs with no improvement.
            n_iter_no_change=12,            # Wait this many epochs before stopping if no improvement.
            max_iter=MLP_MAXIT,             # Max epochs to prevent infinite training.
            random_state=random_state,      # Seed for reproducibility.
            hidden_layer_sizes=MLP_HIDDEN,  # Two hidden layers with these many units.
            verbose=False                   # keep training logs quiet unless debugging.
        ))
    ])

# ---------------- File-level pooling (voting) ----------------
def center_topq_pool(logits: np.ndarray, center_frac=CENTER_FRACTION, topq=TOPQ_FRACTION):
    """
    We have many segment predictions for one file. This function:
    1. drops edge segments (keep the middle chunk),
    2. for each class, keeps only the most confident segment scores,
    3. averages those to get one score per class,
    4. returns that vector so we can take the winner.
    """
    n = logits.shape[0]
    keep = max(1, int(round(center_frac * n)))
    start = (n - keep) // 2
    L = logits[start:start+keep]
    qk = max(1, int(round(topq * L.shape[0])))
    out = np.zeros(L.shape[1], dtype=np.float32)
    for c in range(L.shape[1]):
        idx = np.argsort(L[:, c])[-qk:]
        out[c] = float(np.mean(L[idx, c]))
    return out

# ---------------- Train & Evaluate ----------------
def train_and_eval(train_pairs, test_pairs, sr, seg, hop_s, cache_dir, n_jobs,
                   ensemble=DEFAULT_ENSEMBLE, tta=DEFAULT_TTA, strong_good_aug=DEFAULT_STRONG_GOOD_AUG):
    """
    This is the whole run: make features, train several MLPs, run on test files (with optional TTA), pool per file, print accuracy numbers, and return stuff we need.
    """

    # ---------- TRAIN ----------
    """
    Build the full training matrix: features for all (possibly augmented) segments, their string labels, and file IDs.
    """
    X_train, y_train_str, groups = build_dataset(
        train_pairs, sr, seg, hop_s, cache_dir, n_jobs=n_jobs, do_aug=True, strong_good=strong_good_aug
    )
    # Convert string labels to integer class indices.
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_str)     

    # Compute class weights (inverse-frequency) so rare classes count more, then expand to a sample weight per segment.
    classes_idx = np.arange(len(le.classes_))
    cw = compute_class_weight("balanced", classes=classes_idx, y=y_train)
    sample_weight = cw[y_train]

    # Train several MLPs with different seeds for ensembling.
    seeds = [SEED0 + 271 * i for i in range(ensemble)]
    estimators = []
    for sd in seeds:
        est = make_mlp(random_state=sd)
        est.fit(X_train, y_train, clf__sample_weight=sample_weight)
        estimators.append(est)

    # ---------- TEST ----------
    def infer_files(files_pairs, tta):
        """
        Given a list of (path, label), run the ensemble, do per-file pooling, and return the true and predicted labels.
        """
        y_true_files, y_pred_files = [], []

        if tta <= 0:
            # Fast path: use cached segments for all test files
            X_test, y_test_str, groups_test = build_dataset(
                files_pairs, sr, seg, hop_s, cache_dir, n_jobs=n_jobs, do_aug=False, strong_good=False
            )
            probs_members = [est.predict_proba(X_test) for est in estimators]
            probs_mean = np.mean(probs_members, axis=0)
            logits = np.log(np.clip(probs_mean, 1e-9, 1.0))

            pooled_by_file, file_true = [], []
            for fid in np.unique(groups_test):
                idx = np.where(groups_test == fid)[0]
                pooled_by_file.append(center_topq_pool(logits[idx]))
                file_true.append(y_test_str[idx[0]])

            y_pred_idx = np.argmax(np.vstack(pooled_by_file), axis=1)
            y_pred_files = list(le.classes_[y_pred_idx])
            y_true_files = file_true

        else:
            # Higher-accuracy path: rebuild per-file features with small TTA
            files = [p for (p, _) in files_pairs]
            labs  = [lab for (_, lab) in files_pairs]
            tasks = [(p, lab, sr, seg, hop_s, tta) for p, lab in zip(files, labs)]
            pooled_list, y_true_files = [], []

            # For each file (in parallel): build TTA features, average ensemble probabilities over all segments of all TTA variants, convert to log-probabilities, pool per file, and store the result.
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                futures = [ex.submit(_feat_for_one_file, t) for t in tasks]
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Inferring TEST (parallel)"):
                    true_lab, Xf, _ = fut.result()
                    P = np.mean([est.predict_proba(Xf) for est in estimators], axis=0)
                    L = np.log(np.clip(P, 1e-9, 1.0))
                    pooled_list.append(center_topq_pool(L))
                    y_true_files.append(true_lab)

            y_pred_idx = np.argmax(np.vstack(pooled_list), axis=1)
            y_pred_files = list(le.classes_[y_pred_idx])

        return y_true_files, y_pred_files
    # Get lists of true and predicted labels for all test files.
    y_true, y_pred = infer_files(test_pairs, tta)

    # Print accuracy numbers and classification report.
    acc = accuracy_score(y_true, y_pred)
    bal = balanced_accuracy_score(y_true, y_pred)
    print(f"\n[Test] accuracy (file-level): {acc:.4f} | balanced acc: {bal:.4f}\n")
    print(classification_report(y_true, y_pred, digits=2))

    return estimators, le, acc, bal # Give back the trained ensemble, the label encoder, and the two headline metrics.

# ---------------- CLI (parse flags, run, save) ----------------
def main():
    """
    Define command-line flags so we can change paths and settings without touching the code.
    """
    ap = argparse.ArgumentParser()
    
    #Replacing the local configs
    # ap.add_argument("--train_dir", type=str, default="data/train_cut") I can change the defaut path
    # ap.add_argument("--test_dir",  type=str, default="data/test_cut")
    
    #New directories path:
    ap.add_argument("--train_dir", type=str, default=str(cfg.get_train_cut()))
    ap.add_argument("--test_dir",  type=str, default=str(cfg.get_test_cut()))

    
    ap.add_argument("--sr", type=int, default=SR)
    ap.add_argument("--seg", type=float, default=DEFAULT_SEG)
    ap.add_argument("--hop", type=float, default=DEFAULT_HOP_S)
    ap.add_argument("--models_dir", type=str, default="models")
    ap.add_argument("--cache_dir", type=str, default="cache/features")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--ensemble", type=int, default=DEFAULT_ENSEMBLE)
    ap.add_argument("--tta", type=int, default=DEFAULT_TTA)
    ap.add_argument("--strong_good_aug", action="store_true", default=DEFAULT_STRONG_GOOD_AUG)
    args = ap.parse_args()
    
    # Convert strings to Path objects and make sure output folders exist.
    train_dir = Path(args.train_dir)
    test_dir  = Path(args.test_dir)
    models_dir = Path(args.models_dir); models_dir.mkdir(parents=True, exist_ok=True)
    cache_dir  = Path(args.cache_dir);  cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Echo key settings so you can sanity-check before the heavy work starts.
    print(f"Train dir: {train_dir}")
    print(f"Test  dir: {test_dir}")
    print(f"Cache dir: {cache_dir} | jobs: {args.jobs}")

    # Scan the dataset folders into lists of (path, label) pairs.
    train_pairs = list_audio_by_class(train_dir)
    test_pairs  = list_audio_by_class(test_dir)

    # Run the full training + evaluation
    estimators, le, acc, bal = train_and_eval(
        train_pairs, test_pairs, args.sr, args.seg, args.hop, cache_dir, n_jobs=args.jobs,
        ensemble=args.ensemble, tta=args.tta, strong_good_aug=args.strong_good_aug
    )

    # Save the ensemble and the label encoder for reuse
    saved = []
    if args.ensemble == 1:
        p = models_dir / "mlp_simple.joblib"
        joblib.dump(estimators[0], p); saved.append(str(p))
    else:
        for i, est in enumerate(estimators, 1):
            p = models_dir / f"mlp_simple_m{i}.joblib"
            joblib.dump(est, p); saved.append(str(p))
    joblib.dump(le, models_dir / "labels_simple.joblib")

    print(f"\n[Saved] models -> {saved}")
    print(f"[Saved] labels -> {models_dir / 'labels_simple.joblib'}")
    print("\n[Done]")

if __name__ == "__main__":
    main()




# Abbreviations used
# ACC — Accuracy
# Adam — Adaptive Moment Estimation
# argmax — Index of maximum
# BAL — Balanced accuracy
# BS — Batch size
# CLI — Command-line interface
# CPU — Central Processing Unit
# dB — Decibels
# ddof — Delta degrees of freedom
# FFT — Fast Fourier Transform
# HOP — Hop length
# ID — Identifier
# I/O — Input/Output
# LR — Learning rate
# MD5 — Message-Digest Algorithm 5
# Mel — Mel scale / mel bands
# MFCC — Mel-Frequency Cepstral Coefficients
# MLP — Multilayer Perceptron
# PCA — Principal Component Analysis
# PCA_DIM — PCA dimensionality
# ReLU — Rectified Linear Unit
# RMS — Root-Mean-Square
# RNG — Random Number Generator
# SR — Sample rate
# STFT — Short-Time Fourier Transform
# TTA — Test-Time Augmentation
# ZCR — Zero-Crossing Rate
