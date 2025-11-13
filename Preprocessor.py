# Preprocessor.py
"""
Author: Hans Josef Rosland-Borlaug[258139]

Build and reflection:
 I rewrote helpers to be small and readable.
 Adding comments so I can backtrack .
 
 I made runPreprocess accept mel-ish params (nFft, nMels, hopLength, fMin, fMax), these are hardcoded. Can be unlocked
 so preprocess_all_train_cut can pass them without errors.
 stdlib + numpy + matplotlib only (no librosa), experiencing some errors on it, look into it if time.
 Keeps saving files as <preprocessed>/<CLASS>/<stem>_sXXX.png
 
 later edit: I need to create some wrappers so I can send the data to GUI, I want this to be validated in Function_validations as I create them at the bottom.
"""

#Imports
from pathlib import Path  # nice path handling on all OSes
from typing import Optional, Dict, List, Tuple  # type hints for clarity (I keep these for readability)

import wave  # read basic WAV headers + PCM data (stdlib)
import numpy as np  # number crunching
import matplotlib.pyplot as plt  # quick spectrograms and image saving

from configs import Config  # project config (paths, folders, etc.)


#Preprocess helpers
def files_exist_for_file(cfg: Config, filename: str, class_filter: Optional[str] = None):
    """
    What this does (short version):
        I count how many spectrogram files already exist for a given audio *stem* (no .wav),
        so I can decide if I should skip or replace.

    Why I need it:
        - Prevents duplicate stacking of <stem>_s000.png, <stem>_s001.png ... in the preprocessed folder.
        - Lets me implement "skip existing" vs "replace" behavior.

    Args:
        cfg:      my Config object that knows where "preprocessed" lives.
        filename: the WAV stem, like 'engine_001' (no extension).
        class_filter: if I know the class (like 'engine1_good'), I only check there.
                      If None, I search all class folders.

    Returns:
        int: count of existing files named "<filename>_s*.png".
    """
    # 1) find the root folder where files live, like: <project>/preprocessed
    preprocessed_root = cfg.get_preproc_root()
    if not preprocessed_root.exists():
        # nothing there to no files
        return 0

    # 2) if a class is specified, only inspect that folder
    if class_filter:
        class_folder = preprocessed_root / class_filter
        if not class_folder.exists():
            return 0  # class folder missing to zero files
        return sum(1 for _ in class_folder.glob(f"{filename}_s*.png"))

    # 3) otherwise, search across all immediate subfolders that are directories
    total = 0
    for folder in preprocessed_root.iterdir():
        if folder.is_dir():
            total += sum(1 for _ in folder.glob(f"{filename}_s*.png"))
    return total


def parseMelParams(nMels: int, hopLength: int, fMin: float, fMax):
    """
    Small normalizer for "mel-ish" params I take from Tk widgets.

    
      - This is kept even if not heavily used now because other parts of my GUI expect this shape.
      - It's a future-proof thing: if/when I swap to a real mel pipeline (librosa/own code),
        I already have a clean place to parse inputs.

    Behavior:
      - fMax can be blank '' to I treat that as None (Nyquist).
      - Everything else is cast to the right basic types.

    Returns:
      dict with { "nMels", "hopLength", "fMin", "fMax" }
    """
    fmax_val: Optional[float] = None
    if isinstance(fMax, (int, float)):
        fmax_val = float(fMax)
    elif isinstance(fMax, str) and fMax.strip():
        try:
            fmax_val = float(fMax.strip())
        except ValueError:
            fmax_val = None  # fall back

    return {
        "nMels": int(nMels),
        "hopLength": int(hopLength),
        "fMin": float(fMin),
        "fMax": fmax_val,  # None => Nyquist
    }


#Processor main functions
# Some parameters here are intentionally unused (like nMels/fMin/fMax)
# because I am doing a simple linear-frequency specgram (matplotlib),
# BUT I keep the parameters so the GUI / caller code doesn't break when I switch to mel later.
def runPreprocess(
    cfg: "Config",# config with project paths
    label: Optional[str], # if None I try to infer from the WAV path (folder names)
    selectedPaths: List[Path], # list of WAVs to process
    targetSr: int,# resample target (I default to 16000 in GUI)
    forceMono: bool,  #  mono
    doNormalize: bool,# peak normalize to ~[-1,1]
    sliceSeconds: int, # seconds per spectrogram tile (like 3s)
    overwritePolicy: str,  # "skip" or "replace" (info string; the real switch is skip_if_exists)
    alsoSaveAudio: bool = False,# not used yet (future: maybe store sliced WAVs) — kept for API compatibility
    skip_if_exists: bool = True,# True to skip files that already have files
    # mel-ish API (kept so preprocess_all_train_cut can pass them safely)
    nFft: int = 1024,# actually used for specgram FFT size
    nMels: int = 128, # not used now (kept for future mel)
    hopLength: int = 256, # used to compute noverlap = NFFT - hop
    fMin: float = 20.0,# not used now
    fMax: Optional[float] = None#not used now
):
    """
    What this does:
      I take a bunch of WAV files, and for each file I cut it into fixed-size time windows
      (sliceSeconds). For each window I render a grayscale spectrogram tile and save it as
      <preprocessed>/<CLASS>/<filename>_sXXX.png.

    Why I wrote it this way:
      - I want consistent tile shapes (helps a simple image model).
      - I don't depend on librosa here to keep it portable (pure stdlib + numpy + matplotlib).
      - I support "skip" vs "replace" behavior so I don't create duplicate stacks.

    Skip/overwrite behavior:
      - skip_if_exists=True  to if any files exist for that file (in its class folder), I skip it.
      - skip_if_exists=False to I delete old files for that file first, then regenerate.

    Returns:
      (written_count, info_dict)
      - written_count: how many PNG files I actually wrote
      - info_dict: summary for the GUI/log (paths, counts, policy, params, errors, etc.)
    """

    # Where I store files (make sure it exists)
    preproc_root = cfg.get_preproc_root()
    preproc_root.mkdir(parents=True, exist_ok=True)

    # Allowed class names (to keep things tidy)
    # cfg.get_folders() should match the dataset subfolders (like engine1_good, etc.)
    class_names = set(cfg.get_folders())

    # Helper: infer class name from a file path by scanning its parts
    # If I can't find any known class in the path, I fall back to the user-specified label
    # or "unsorted" so I never crash because of missing class info.
    def infer_class_from_path(p: Path):
        for part in p.parts:
            if part in class_names:
                return part
        return label or "unsorted"

    # Read WAV file into float32 in [-1, 1]
    #  I'm assuming 16-bit PCM here (simple path). If not 16-bit, I still try to read
    # as int16. This is why 24-bit / float32 WAVs show up as "unknown format: 3" sometimes
    # elsewhere. It's on my improvement list to handle more formats properly.
    def read_wav_float32(p: Path):
        """
        Opens a WAV with the stdlib 'wave' module.
        Returns:
            y (np.ndarray float32 in [-1, 1]), sample_rate (int), channels (int)

        Limitations:
            I assume 16-bit PCM. If sample width is not 2 bytes, I still force int16 which
            is obviously not correct for all files. Kept simple on purpose for now.
        """
        with wave.open(str(p), "rb") as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            n = wf.getnframes()
            raw = wf.readframes(n)

        # Currently forcing int16 path for simplicity (future work: real 24-bit/float paths)
        dtype = np.int16 if sw == 2 else np.int16
        y = np.frombuffer(raw, dtype=dtype).astype(np.float32) / 32768.0

        # If stereo and we want mono to average channels
        if ch == 2 and forceMono:
            y = y.reshape(-1, 2).mean(axis=1)
            ch = 1

        return y, sr, ch

    # A tiny, dependency-free linear resampler
    def resample_linear(y: np.ndarray, sr_src: int, sr_tgt: int):
        """
        Quick resampler just so I can normalize everything to targetSr.
        I use linear interpolation (np.interp). It's not audiophile-quality, but it's fine
        for spectrogram files.

        Returns:
            y_resampled (np.ndarray float32)
        """
        if sr_src == sr_tgt or y.size == 0:
            return y
        dur = len(y) / float(sr_src)
        t_src = np.linspace(0.0, dur, num=len(y), endpoint=False)
        n_tgt = int(round(dur * sr_tgt))
        t_tgt = np.linspace(0.0, dur, num=n_tgt, endpoint=False)
        return np.interp(t_tgt, t_src, y).astype(np.float32)

    # Simple peak normalization so spectrograms look visually consistent
    def normalize_peak(y: np.ndarray):
        """
        Scales the waveform so the loudest absolute sample is ~1.0.
        I multiply by 0.999 just to avoid exact 1.0 for safety.

        Returns:
            y_norm (np.ndarray float32)
        """
        m = float(np.max(np.abs(y))) if y.size else 0.0
        return (y / m * 0.999).astype(np.float32) if m > 0 else y.astype(np.float32)

    #("scoreboard" for later summary)
    written = 0            # how many new PNG files I wrote
    replaced = 0           # how many old files I deleted in replace-mode
    skipped_errors = 0     # count of files I skipped because something blew up
    skipped_existing = 0   # count of files I skipped because files were already there
    errors: List[str] = []  # human-readable error strings I can show in GUI/log
    processed_files: List[str] = []  # book-keeping of files that went through fine

    #Main loop over all chosen WAVs 
    for src in selectedPaths:
        try:
            # (a) existence check (GUI lists can go stale)
            if not src.exists():
                skipped_errors += 1
                errors.append(f"Missing file: {src}")
                continue

            # (b) decide output folder: either the given 'label', or inferred from path
            class_dir_name = label if label else infer_class_from_path(src)
            out_dir = preproc_root / class_dir_name
            out_dir.mkdir(parents=True, exist_ok=True)

            # (c) skip-or-replace logic BEFORE I do heavy work
            if skip_if_exists and files_exist_for_file(cfg, src.stem, class_filter=class_dir_name) > 0:
                skipped_existing += 1
                continue

            if not skip_if_exists:
                # replace mode: delete old files of this file in this class folder
                for old in out_dir.glob(f"{src.stem}_s*.png"):
                    old.unlink(missing_ok=True)
                    replaced += 1

            # (d) read audio to y (float32), sr (sample rate)
            y, sr, _ = read_wav_float32(src)

            # (e) resample if needed
            if sr != targetSr:
                y = resample_linear(y, sr, targetSr)
                sr = targetSr

            # optional normalization for consistent look
            if doNormalize:
                y = normalize_peak(y)

            # (f) convert seconds-per-tile to samples
            win = int(sr * sliceSeconds)
            if win <= 0:
                skipped_errors += 1
                errors.append(f"Bad sliceSeconds={sliceSeconds} for file {src}")
                continue

            # (g) spectrogram params (linear-frequency specgram)
            #  I keep mel-ish API args but here I only use NFFT + hopLength (noverlap)
            NFFT = max(32, int(nFft))  # make sure it's not absurdly small
            hop = max(1, int(hopLength))
            noverlap = int(max(0, min(NFFT - 1, NFFT - hop)))  # clamp for safety

            # (h) slice + render + save files
            idx = 0
            for i in range(0, len(y), win):
                seg = y[i:i + win]
                if seg.size == 0:
                    break

                # pad last chunk so all images stay the same shape
                if seg.size < win:
                    seg = np.concatenate([seg, np.zeros(win - seg.size, dtype=np.float32)])

                # draw the spectrogram in grayscale, no axes (clean tile)
                fig = plt.figure(figsize=(4, 3), dpi=100)
                ax = fig.add_axes([0, 0, 1, 1])
                ax.specgram(seg, NFFT=NFFT, Fs=sr, noverlap=noverlap, cmap="gray")
                ax.axis("off")

                # save pattern: <stem>_s000.png, <stem>_s001.png, ...
                out_path = out_dir / f"{src.stem}_s{idx:03d}.png"
                fig.savefig(str(out_path), bbox_inches="tight", pad_inches=0)
                plt.close(fig)

                written += 1
                idx += 1

            processed_files.append(str(src))

        except Exception as e:
            # (i) safety net so one weird file doesn't kill the whole batch
            skipped_errors += 1
            errors.append(f"Error on {src}: {e}")

    #  Build info dict for GUI/logging and return it together with 'written' 
    info = {
        "vision_root": str(preproc_root),
        "label": label if label else "(auto)",
        "written": written,
        "replaced": replaced,
        "skipped_existing": skipped_existing,
        "policy": overwritePolicy,
        "slice_seconds": sliceSeconds,
        "target_sr": targetSr,
        "nFft": nFft,           # used
        "nMels": nMels,         # not used (kept for future mel)
        "hopLength": hopLength, # used (via noverlap)
        "fMin": fMin,           # not used (future mel)
        "fMax": fMax,           # not used (future mel)
        "processed_files": processed_files,
        "errors": errors,
    }
    return written, info



# Batch convenience for train_cut

def preprocess_all_train_cut(
    cfg: "Config",
    targetSr: int = 16000,
    sliceSeconds: int = 3,
    forceMono: bool = True,
    doNormalize: bool = True,
    skip_if_exists: bool = True,
):
    """
    What this does:
      I walk the training dataset folders and collect *all* WAVs we should process,
      then I call `runPreprocess` once with that whole list.


    Skip rules:
      - If skip_if_exists=True:DO NOT send a file if files already exist for that file in its class folder.
      - If skip_if_exists=False: I send everything in; 'runPreprocess' will handle replacing.

    Returns:
      (written, info)
      - written (int): total number of files written
      - info (dict):   summary including my pre-scan 'skipped_existing' count
    """
    # 1) grab paths and lists from Config
    train_cut = cfg.get_train_cut()
    classes = cfg.get_folders()
    exts = cfg.get_exts()

    # 2) collect all WAVs (and optionally skip if they already have files)
    all_wavs: List[Path] = []
    skipped_existing = 0

    for c in classes:
        class_dir = train_cut / c
        for ext in exts:
            for p in sorted(class_dir.glob(f"*{ext}")):
                if skip_if_exists and files_exist_for_file(cfg, p.stem, class_filter=c) > 0:
                    skipped_existing += 1
                else:
                    all_wavs.append(p)

    # 3)call the main processor
    #  I pass mel-ish params for API stability, but only NFFT/hopLength are used here.
    written, info = runPreprocess(
        cfg=cfg,
        label=None,                 # let the function infer class from each file path
        selectedPaths=all_wavs,     # everything collected above
        targetSr=targetSr,
        forceMono=forceMono,
        doNormalize=doNormalize,
        sliceSeconds=sliceSeconds,
        nFft=1024,
        nMels=128,
        hopLength=256,
        fMin=20.0,
        fMax=None,
        overwritePolicy=("skip" if skip_if_exists else "replace"),
        alsoSaveAudio=False,        # not used yet (kept for a future "save sliced wavs" feature)
        skip_if_exists=skip_if_exists,
    )

    # 4) merge my pre-scan information into the final info for transparency
    info["skipped_existing"] = info.get("skipped_existing", 0) + skipped_existing
    return written, info



# GUI helper class + wrappers

"""
This was really messy, because I made the instances in GUI, but I created them with camelCase logic, as I am used to in C# so I got a bunch of errors, 
Hence changing the wrappers in the Preprocessor class to camelcase
"""
class Preprocessor:
    """
    I use this small class as a bridge between the core preprocessor logic and the GUI.

    Why it exists:
      - Keeps defaults in one place.
      - Lets the GUI list WAV candidates for a given class folder.
      - Tracks which files the user selected.
      - Provides tiny methods the GUI can call without knowing internals.

    
      I keep both snake_case and camelCase wrappers because I mixed styles earlier.
      The wrappers just forward the calls (so old GUI code keeps working).
    """

    def __init__(self):
        # Shared Config for paths
        self.cfg = Config()

        # Default source folder for listing candidates (I start with train_cut)
        self.source_root: Path = self.cfg.get_train_cut()

        # Tracks which WAVs are selected in the GUI (Path -> bool)
        self.selected: Dict[Path, bool] = {}

        # Default parameters shown in the GUI controls (were editable earlier, now often disabled)
        # Some keys (like nMels/fMin/fMax) are here for consistency with the API,
        # even if the simple specgram path does not use them yet.
        self.params = {
            "targetSr": 16000,
            "sliceSeconds": 3,
            "nMels": 128,     # not used now (future mel)
            "hopLength": 256, # used via noverlap
            "fMin": 20.0,     # not used now
            "fMax": None,     # not used now
        }

    def list_preprocessed_images(self, cfg: "Config", label: str, split: Optional[str] = None):
        """
        Tiny file lister I call from the GUI to show what's already produced.

       

        Returns:
          list[Path]: all .png files found under the preprocessed root, sorted.
        """
        root = cfg.get_preproc_root()
        if not root.exists():
            return []
        return sorted(root.rglob("*.png"))

    #methods that wrappers will call 
    def set_source_root(self, path: str):
        """
        Change which root folder I scan for WAVs (e.g., switch between train_cut and test_cut).
        GUI calls this when the user switches a base folder.
        """
        self.source_root = Path(path)

    def list_candidates(self, subfolder: str):
        """
        I scan one class folder (like 'engine1_good') and build rows for a GUI table.

        Each row is a dict with:
          - "path": Path to the WAV
          - "name": filename only
          - "rel":  relative path from source_root (nice for display)
          - "seconds": duration in seconds (0.0 if unreadable)
          - "readonly": True if files already exist (so I can gray it out in GUI)

        Returns:
          list[dict]: rows ready to be displayed.
        """
        
        folder = self.source_root / subfolder
        rows: List[Dict] = []
        if not folder.exists():
            return rows

        # include both .wav and .WAV (from config)
        paths: List[Path] = []
        for ext in self.cfg.get_exts():
            paths.extend(sorted(folder.glob(f"*{ext}")))

        for p in paths:
            sec = self._safe_duration(p)
            readonly = files_exist_for_file(self.cfg, p.stem, class_filter=subfolder) > 0
            rows.append({
                "path": p,
                "name": p.name,
                "rel": str(p.relative_to(self.source_root)),
                "seconds": sec,
                "readonly": readonly,
            })
        return rows

    def load_image_array(self, path: str):
        """
        Quick helper to read a PNG tile into a numpy array for preview in the GUI.
        I use matplotlib's imread to keep deps simple.

        Returns:
          np.ndarray (H,W) or (H,W,3/4) depending on the image.
        """
        import matplotlib.pyplot as _plt
        return _plt.imread(path)

    def _safe_duration(self, wav_path: Path):
        """
        Hidden helper (only used internally here).

        What it does:
          - Tries to open a WAV and compute its duration in seconds.
          - If anything fails, returns 0.0 so the GUI doesn't crash.

        Returns:
          float: seconds (0.0 if unreadable).
        """
        try:
            with wave.open(str(wav_path), "rb") as wf:
                n = wf.getnframes()
                sr = wf.getframerate()
                if sr > 0:
                    return float(n) / float(sr)
        except Exception:
            pass
        return 0.0

    #Wrappers  so my older GUI code still works 
    def getDefaults(self):
        """
        Wrapper:
          - Returns a shallow copy of current default parameters for the GUI.
        """
        return dict(self.params)

    def setSourceRoot(self, path: str):
        """
        Wrapper around set_source_root(path).
        Keeping camelCase because I mixed styles earlier and I don't want to break GUI code.
        """
        self.set_source_root(path)

    def listCandidates(self, subfolder: str):
        """
        Wrapper around list_candidates(subfolder).
        Returns rows with metadata about files in a class folder.
        """
        return self.list_candidates(subfolder)

    def markFile(self, path, selected: bool):
        """
        Marks/unmarks a WAV as selected in the GUI table.

        """
        self.selected[Path(path)] = bool(selected)

    def getSelectedPaths(self):
        """
        Returns a list of Paths for all files marked as selected in the GUI.

        Used by:
          - The "Process selected" button handler.
        """
        return [p for p, sel in self.selected.items() if sel]

    def listPreprocessedImages(self, cfg: "Config", label: str, split: Optional[str] = None):
        """
        Wrapper around list_preprocessed_images(cfg, label, split).
        Gives the GUI a simple way to show produced files.
        """
        return self.list_preprocessed_images(cfg, label, split)

    def loadImageArray(self, path: str):
        """
        Wrapper around load_image_array(path).
        Lets the GUI preview files without importing matplotlib in GUI code.
        """
        return self.load_image_array(path)
