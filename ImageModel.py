"""
ImageModel.py
Author: Hans Josef Rosland-Borlaug [258139]
startdate: 19.09.25
Last edit (date):

Reflection and build:
  - I turn short audio clips into tiny grayscale spectrogram images.
  - I train a very small model (Scaler to PCA to Logistic Regression).
  - I can predict straight from a WAV by generating the same “tiny” spectrogram on the go.
  - I also draw a simple saliency heatmap to see which pixels influenced the prediction.

Why I kept it this tiny:
  - Easier to understand and debug.
  - No heavy DL frameworks; just numpy + sklearn + matplotlib + PIL.
"""

from pathlib import Path
import time
import io

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import wave
import joblib

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from configs import Config



# Audio helpers
def _wav_to_numpy_mono(path):
    """
    Opens a WAV and gives me:
      - y: mono audio as float32 in [-1, 1]
      - sr: sample rate

    If it’s stereo, I just average the two channels. I assume 16-bit PCM
    (same simple approach as in the preprocessor), which is fine for this toy setup.
    Returns (y, sr).
    """
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        n = wf.getnframes()
        raw = wf.readframes(n)

    # Simple path: treat as int16. 
    dtype = np.int16 if sw == 2 else np.int16
    y = np.frombuffer(raw, dtype=dtype).astype(np.float32) / 32768.0
    if ch == 2:
        y = y.reshape(-1, 2).mean(axis=1)
    return y, sr


def _resample_linear(x, sr_in, sr_out):
    """
    Resamples with linear interpolation. Not fancy, but perfect for building
    consistent spectrogram images. Returns the resampled waveform.
    """
    if sr_in == sr_out:
        return x
    duration = x.size / sr_in
    t_out = np.linspace(0.0, duration, int(round(duration * sr_out)), endpoint=False)
    t_in = np.linspace(0.0, duration, x.size, endpoint=False)
    return np.interp(t_out, t_in, x).astype(np.float32)



# Tiny spectrogram builder (the “image” we feed the model)
def _tiny_style_spectrogram_image(
    y, sr_in,
    target_sr=16000, secs=3,
    nfft=1024, noverlap=512,
    out_size=(256, 192),
):
    """
    Makes a compact grayscale spectrogram:
      - resample to a fixed rate
      - trim/pad to a fixed duration
      - plot a spectrogram with matplotlib (in memory)
      - convert to PIL image, grayscale it, resize, scale to [0, 1]
    Returns a 2D numpy array shaped (H, W).
    """
    y = _resample_linear(y, sr_in, target_sr)
    want = int(target_sr * secs)
    if y.size >= want:
        y = y[:want]
    else:
        y = np.pad(y, (0, want - y.size))

    fig = plt.figure(figsize=(4, 3), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])           # no margins, full image
    ax.specgram(y, NFFT=nfft, Fs=target_sr, noverlap=noverlap, cmap="gray")
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    buf.seek(0)
    img = Image.open(buf).convert("L").resize(out_size, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr  # (H, W)



# Image helpers
def _read_png_grayscale(path, size):
    """
    Reads a PNG, converts to grayscale, resizes, and scales to [0, 1].
    Returns the image as a 2D numpy array.
    """
    img = Image.open(path).convert("L").resize(size, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def _flatten_img(img_hw):
    """
    Flattens a 2D image (H, W) into a 1D vector (H*W).
    Returns the flattened vector. Need this later
    """
    return img_hw.reshape(-1)


def _occlusion_saliency(img, pipe, cls_idx, win=16, stride=8):
    """
    Very simple “what mattered” map:
      - hide small windows with their mean brightness
      - check how much the target class probability drops
      - the bigger the drop, the “hotter” that region

    Returns a 2D heatmap (same H, W) scaled to [0, 1].
    """
    H, W = img.shape
    base = pipe.predict_proba(_flatten_img(img)[None, :])[0][cls_idx]
    heat = np.zeros_like(img, dtype=np.float32)

    for y in range(0, H - win + 1, stride):
        for x in range(0, W - win + 1, stride):
            patch = img[y:y+win, x:x+win]
            repl = float(patch.mean())
            img2 = img.copy()
            img2[y:y+win, x:x+win] = repl

            p = pipe.predict_proba(_flatten_img(img2)[None, :])[0][cls_idx]
            drop = max(0.0, base - p)
            heat[y:y+win, x:x+win] += drop

    m = heat.max()
    if m > 0:
        heat /= m
    return heat



# Dataset loader (tiles from disk to X, y)
class _TileDataset:
    """
    Looks in preprocessed/<class> and gathers PNG tiles.
    I keep it tiny and direct: build (X, y) in memory, ready for sklearn.

    Returns from .load():
      - X: float32 array shaped (num_samples, H*W)
      - y: int labels shaped (num_samples,)
    """
    def __init__(self, root, classes, size=(256, 192)):
        self.samples = []   # list of (Path, class_index)
        self.size = size    # expected as (W, H) for the reader below
        for ci, cname in enumerate(classes):
            d = root / cname
            if not d.exists():
                continue
            for p in sorted(d.glob("*.png")):
                self.samples.append((p, ci))

    def load(self):
        X, Y = [], []
        for p, y in self.samples:
            img = _read_png_grayscale(p, self.size)  # size is (W, H)
            X.append(_flatten_img(img))
            Y.append(y)

        if not X:
            w, h = self.size
            return (np.zeros((0, w * h), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64))

        X = np.stack(X).astype(np.float32)
        Y = np.array(Y, dtype=np.int64)
        return X, Y



# The model: Scaler to PCA to Logistic Regression
class ImageModel:
    """
    A small image classifier that learns on spectrogram tiles.

    Train:
      - read tiles, flatten, scale, PCA, logistic regression
      - keep everything in a sklearn Pipeline

    Predict:
      - convert one WAV into the same tiny spectrogram
      - run the pipeline
      - draw a saliency map to see where it “looked”
    """
    def __init__(self):
        self.cfg = Config()
        self.classes = list(self.cfg.get_folders())
        self.preproc_root = self.cfg.get_preproc_root()

        # I store size as (H, W) for plotting, but some helpers want (W, H)
        self.out_size = (192, 256)  # (H, W)

        # Model bits
        self.model = None
        self.trained = False
        self.pca_components = 256
        self.max_iter = 200
        self.C = 2.0

        # Spectrogram settings for the “tiny” image
        self.target_sr = 16000
        self.slice_sec = 3
        self.tiny_nfft = 1024
        self.tiny_noverlap = 512

        # Where I save trained models
        self.models_dir = self.cfg.get_project_root() / "ImageModel_Training"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self._last_acc = 0.0  # just to remember the last training score

    def _make_dataset(self):
        """
        Builds an in-memory dataset from tiles on disk.
        Note: _TileDataset expects size=(W, H), so I flip my (H, W).
        Returns (X, y).
        """
        size_wh = (self.out_size[1], self.out_size[0])
        ds = _TileDataset(self.preproc_root, self.classes, size=size_wh)
        return ds.load()

    def build(self):
        """
        Creates the sklearn Pipeline:
          StandardScaler to PCA to LogisticRegression.
        Returns nothing
        """
        self.model = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("pca", PCA(n_components=self.pca_components, whiten=True, random_state=42)),
            ("clf", LogisticRegression(
                max_iter=self.max_iter,
                C=self.C,
                multi_class="multinomial",
                solver="lbfgs"
            )),
        ])

    def train(self, log_cb=lambda s: None):
        """
        Trains on whatever is under preprocessed/<class>.
        I log to the callback so the GUI can display progress.
        Returns nothing; updates self.model and self.trained.
        """
        if self.model is None:
            self.build()

        X, y = self._make_dataset()
        if X.shape[0] == 0:
            log_cb("No PNG tiles found under preprocessed/<class>. Preprocess first.\n")
            return

        log_cb(f"Training on {X.shape[0]} tiles (dims={X.shape[1]}, PCA={self.pca_components})\n")
        t0 = time.time()
        self.model.fit(X, y)
        dt = time.time() - t0
        acc = float(self.model.score(X, y)) * 100.0
        self.trained = True
        self._last_acc = acc
        log_cb(f"Done in {dt:.1f}s. Train accuracy={acc:.2f}%\n")

        # Save the trained pipeline
        stamp = time.strftime("%Y%m%d_%H%M%S")
        safe_acc = f"{acc:.2f}".replace(".", "_")
        out = self.models_dir / f"imagemodel_PCA{self.pca_components}_{safe_acc}pct_{stamp}.joblib"
        joblib.dump({
            "pipeline": self.model,
            "classes": self.classes,
            "size_hw": self.out_size,  # (H, W)
        }, out)
        log_cb(f"Saved model to {out}\n")

    def _ensure_model(self):
        """
        Makes sure the pipeline exists before I use it.
        If it’s missing, I rebuild it. Returns nothing.
        """
        if self.model is None:
            self.build()

    def _make_input_image(self, y, sr):
        """
        Builds the same tiny spectrogram I used for training.
        Note: builder wants (W, H), so I flip my (H, W).
        Returns a 2D (H, W) numpy array.
        """
        return _tiny_style_spectrogram_image(
            y, sr_in=sr,
            target_sr=self.target_sr,
            secs=self.slice_sec,
            nfft=self.tiny_nfft,
            noverlap=self.tiny_noverlap,
            out_size=(self.out_size[1], self.out_size[0]),  # flip
        )

    def predict_wav(self, wav_path, anti_cheat=True):
        """
        Full prediction flow for one WAV:
          - load audio, build tiny spectrogram
          - flatten and run the pipeline
          - find the top class and its probability vector
          - compute a small saliency heatmap

        Returns a dict with:
          'probs', 'pred_idx', 'pred_name', 'spec', 'saliency'
        """
        self._ensure_model()
        y, sr = _wav_to_numpy_mono(wav_path)
        spec = self._make_input_image(y, sr)                  # (H, W)
        vec = _flatten_img(spec)[None, :]                     # (1, H*W)
        probs = self.model.predict_proba(vec)[0]              # (num_classes,)
        pred_idx = int(np.argmax(probs))
        pred_name = self.classes[pred_idx]
        sal = _occlusion_saliency(spec, self.model, pred_idx, win=16, stride=8)

        return {
            "probs": probs,
            "pred_idx": pred_idx,
            "pred_name": pred_name,
            "spec": spec,
            "saliency": sal,
        }
