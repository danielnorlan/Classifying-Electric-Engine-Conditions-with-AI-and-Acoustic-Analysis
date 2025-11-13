# Gui_Helpers.py
"""
Author: Hans Josef Rosland-Borlaug[258139]
startdate: 19.09.25
Last edit (date):

MAKE AS I GO, This holds everything
"""

from pathlib import Path
import tkinter as tk
import os, sys, shlex, queue, threading, subprocess
import numpy as np
from tkinter import ttk, messagebox, filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Optional
from datetime import datetime
from heartPTE import build_heartbeat_tab

from Preprocessor import (
    Preprocessor,
    runPreprocess,
    preprocess_all_train_cut,
   
)
from configs import Config


"""
Struggling to import the parseMelParams and tiles_exist_for_file from Preprocessor module, hardcoding it into GUI_Helpers and fixing it later....

"""
###########
def parseMelParams(nMels: int, hopLength: int, fMin: float, fMax):
    """
    Normalize/validate mel parameters from Tk widgets.
    fMax may arrive as '' (blank); treat that as None (Nyquist).
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

def tiles_exist_for_file(cfg: Config, filename: str, class_filter: Optional[str] = None) -> int:
    """
    Check how many preprocessed image tiles already exist for a given audio file.

    Args:
        cfg: Config object (knows where preprocessed folder lives).
        filename: the audio stem (no extension), e.g. "engine_001".
        class_filter: if set, only look in that class folder; else search all class folders.

    Returns:
        count of PNG tiles named like "<filename>_sXXX.png".
    """

    # 1) figure out the root where tiles live, like: <project>/preprocessed
    preprocessed_root = cfg.get_preproc_root()
    if not preprocessed_root.exists():
        # nothing there → no tiles
        return 0

    # 2) if a class is specified, only inspect that folder
    if class_filter:
        class_folder = preprocessed_root / class_filter
        if not class_folder.exists():
            return 0  # class folder missing → zero tiles
        return sum(1 for _ in class_folder.glob(f"{filename}_s*.png"))

    # 3) otherwise, search across all immediate subfolders that are directories
    total = 0
    for folder in preprocessed_root.iterdir():
        if folder.is_dir():
            total += sum(1 for _ in folder.glob(f"{filename}_s*.png"))
    return total
###########



#PREPROCESSOR TAB
class PreprocessorTab:
    """
    Encapsulates the entire Preprocessor UI state and handlers.

    Usage:
        tab = PreprocessorTab(notebook)
        tab.build()  # adds a "Preprocessor" tab to the notebook
    """
    def __init__(self, notebook: ttk.Notebook):
        self.notebook = notebook
        self.cfg = Config()
        self.proc = Preprocessor()

        # Tk variables
        self.sourceVar = tk.StringVar(value=str(self.cfg.get_train_cut()))
        self.sliceVar  = tk.IntVar(value=3)
        self.srVar     = tk.IntVar(value=16000)
        self.nMelsVar  = tk.IntVar(value=128)
        self.hopVar    = tk.IntVar(value=256)
        self.fMinVar   = tk.DoubleVar(value=20.0)
        self.fMaxVar   = tk.StringVar(value="Nyquist")

        self.classBox: ttk.Combobox | None = None
        self.table: ttk.Treeview | None = None
        self.readonly_iids: set[str] = set()

        # Preview widgets
        self.imgList: tk.Listbox | None = None
        self.ax = None
        self.canvas = None

    #Interface build
    def build(self) -> None:
        pp_tab = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(pp_tab, text="Preprocessor")
        pp_tab.rowconfigure(0, weight=1)
        pp_tab.columnconfigure(0, weight=1)

        inner = ttk.Notebook(pp_tab)
        inner.grid(row=0, column=0, sticky="nsew")

        self._build_select_tab(inner)
        self._build_preview_tab(inner)

        # initial population
        self.search()

    def _build_select_tab(self, inner: ttk.Notebook) -> None:
        select_tab = ttk.Frame(inner, padding=8)
        inner.add(select_tab, text="Select & Process")
        select_tab.columnconfigure(1, weight=1)
        select_tab.rowconfigure(0, weight=1)

        # Left column
        left = ttk.Frame(select_tab)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 10))
        left.grid_rowconfigure(99, weight=1)

        ttk.Label(left, text="Source root").grid(row=0, column=0, sticky="w")
        src_entry = ttk.Entry(left, width=50, textvariable=self.sourceVar)
        src_entry.grid(row=1, column=0, sticky="w", pady=(0, 6))

        ttk.Label(left, text="Class / subfolder").grid(row=2, column=0, sticky="w")
        self.classBox = ttk.Combobox(left, state="readonly", values=self.cfg.get_folders(), width=40)
        self.classBox.grid(row=3, column=0, sticky="w", pady=(0, 8))
        self.classBox.set(self.cfg.get_folders()[0])

        #left side box for skip, overwrite, etc. (Removing the other options, only keeping "ski)
        ttk.Label(left, text="If image exists").grid(row=4, column=0, sticky="w")
        overwriteBox = ttk.Combobox(left, state="readonly", values=["skip"], width=12)
        overwriteBox.grid(row=5, column=0, sticky="w")
        overwriteBox.set("skip")

        rowA = ttk.Frame(left); rowA.grid(row=6, column=0, sticky="w", pady=(6, 0))
        ttk.Label(rowA, text="Slice sec", state="readonly").pack(side="left")
        ttk.Entry(rowA, width=6, textvariable=self.sliceVar, state="disabled").pack(side="left", padx=6)
        ttk.Label(rowA, text="Target SR", state="readonly").pack(side="left", padx=(10, 0))
        ttk.Entry(rowA, width=8, textvariable=self.srVar, state="disabled").pack(side="left", padx=6)

        specBox = ttk.LabelFrame(left, text="Mel settings - Read Only", padding=6)
        specBox.grid(row=7, column=0, sticky="ew", pady=(10, 0))

        rowB = ttk.Frame(specBox); rowB.grid(row=0, column=0, sticky="w", pady=2)
        ttk.Label(rowB, text="Mel bands", state="readonly").pack(side="left")
        ttk.Entry(rowB, width=6, textvariable=self.nMelsVar, state="disabled").pack(side="left", padx=6)

        rowC = ttk.Frame(specBox); rowC.grid(row=1, column=0, sticky="w", pady=2)
        ttk.Label(rowC, text="Hop length").pack(side="left")
        ttk.Entry(rowC, width=8, textvariable=self.hopVar, state="disabled").pack(side="left", padx=6)

        rowD = ttk.Frame(specBox); rowD.grid(row=2, column=0, sticky="w", pady=2)
        ttk.Label(rowD, text="fMin (Hz)").pack(side="left")
        ttk.Entry(rowD, width=8, textvariable=self.fMinVar, state="disabled").pack(side="left", padx=6)
        ttk.Label(rowD, text="fMax (Hz, blank=Nyquist)").pack(side="left", padx=(10, 0))
        ttk.Entry(rowD, width=10, textvariable=self.fMaxVar, state="disabled").pack(side="left", padx=6)
        
        tutorial_frame = ttk.LabelFrame(left, text="How to use:", padding=6)
        tutorial_frame.grid(row=11, column=0, sticky="ew", pady=(12, 0))
        tutorial_box = tk.Text(tutorial_frame, height=12, width=45, wrap="word", bg="#f9f9f9",
                               relief="flat", font=("Segoe UI", 8))
        
        tutorial_box.pack(fill="both", expand=True)
        tutorial_text = ("Imagemodel is using all items from preprocessor."
                         "\n Items residing in the Original folders:"
                         "\n engine1_good = 105 items"
                         "\n engine2_broken = 124"
                         "\n enigne3_heavyload =128 "
                         "\n Check this by clicking 'preprocess all' and check number of items"
                         "\n Either in folder (try delete all) and see items"
                         "\n Use Function_Validation and run count_preprocessed_items")
        
        tutorial_box.insert("1.0", tutorial_text)
        tutorial_box.configure(state="disabled")

        # Buttons
        btns = ttk.Frame(left); btns.grid(row=8, column=0, sticky="w", pady=(10, 0))
        ttk.Button(btns, text="Search", command=self.search).pack(side="left")
        ttk.Button(btns, text="Select all", command=self.select_all).pack(side="left", padx=6)
        ttk.Button(btns, text="Clear", command=self.clear_all).pack(side="left", padx=6)

        # Right column (table + run)
        right = ttk.Frame(select_tab)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.table = ttk.Treeview(
            right,
            columns=("pick", "name", "rel", "sec", "status"),
            show="headings",
            selectmode="extended"
        )
        for col, text, w, anc in [
            ("pick","X",40,"center"),
            ("name","File",220,"w"),
            ("rel","Relative",360,"w"),
            ("sec","Sec",60,"e"),
            ("status","Status",90,"center")
        ]:
            self.table.heading(col, text=text)
            self.table.column(col, width=w, anchor=anc, stretch=(col in ("name","rel")))
        self.table.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(right, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set); scroll.grid(row=0, column=1, sticky="ns")

        self.table.bind("<Button-1>", self._on_table_click)

        runBar = ttk.Frame(right)
        runBar.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(runBar, text="Process selected", command=self.run_selected).pack(side="left")

        self.btn_all = ttk.Button(runBar, text="Preprocess ALL (train_cut)", command=self.run_all)
        self.btn_all.pack(side="left", padx=8)

    def _build_preview_tab(self, inner: ttk.Notebook) -> None:
        preview_tab = ttk.Frame(inner, padding=8)
        inner.add(preview_tab, text="Preview Images")
        preview_tab.columnconfigure(1, weight=1)
        preview_tab.rowconfigure(0, weight=1)

        pl = ttk.Frame(preview_tab)
        pl.grid(row=0, column=0, sticky="nsw", padx=(0, 10))
        ttk.Label(pl, text="Folder: preprocessed").grid(row=0, column=0, sticky="w")

        self.PREPROC_DIR = self.cfg.get_preproc_root()

        self.imgList = tk.Listbox(pl, width=40, height=24)
        self.imgList.grid(row=1, column=0, sticky="nsew")
        scrollImg = ttk.Scrollbar(pl, orient="vertical", command=self.imgList.yview)
        self.imgList.configure(yscrollcommand=scrollImg.set)
        scrollImg.grid(row=1, column=1, sticky="ns")

        ttk.Button(pl, text="Refresh", command=self.refresh_list).grid(row=2, column=0, sticky="w", pady=(6, 0))

        pr = ttk.Frame(preview_tab)
        pr.grid(row=0, column=1, sticky="nsew")
        pr.columnconfigure(0, weight=1)
        pr.rowconfigure(0, weight=1)

        fig = Figure(figsize=(5.2, 4.0), dpi=100)
        self.ax = fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(fig, master=pr)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.imgList.bind("<<ListboxSelect>>", self.show_selected)
        self.refresh_list()

    # Event handlers
    def search(self):
        self.proc.setSourceRoot(self.sourceVar.get())
        rows = self.proc.listCandidates(subfolder=self.classBox.get() if self.classBox else None)

        # Always clear Treeview first
        if self.table:
            for iid in self.table.get_children():
                self.table.delete(iid)

        self.readonly_iids.clear()

        if not self.table:
            return

        for row in rows:
            iid = str(row["path"])
            if iid in self.readonly_iids or iid in self.table.get_children():
                continue

            # Prefer Preprocessor.tiles_exist_for_stem, else filesystem fallback
            if tiles_exist_for_file is not None:
                exists_count = tiles_exist_for_file(self.cfg, Path(iid).stem)
            else:
                exists_count = tiles_exist_for_file(self.cfg, Path(iid).stem)

            readonly = bool(row.get("readonly")) or (exists_count > 0)

            status = "Read-only" if readonly else ""
            if readonly:
                self.readonly_iids.add(iid)

            self.table.insert(
                "", "end", iid=iid,
                values=("", row["name"], row["rel"], f"{row['seconds']:.1f}", status)
            )

    def select_all(self):
        if not self.table: return
        for iid in self.table.get_children():
            if iid in self.readonly_iids:
                continue
            self.table.set(iid, "pick", "X")
            self.proc.markFile(Path(iid), True)

    def clear_all(self):
        if not self.table: return
        for iid in self.table.get_children():
            self.table.set(iid, "pick", "")
            self.proc.markFile(Path(iid), False)

    def _on_table_click(self, e):
        if not self.table: return
        if self.table.identify("region", e.x, e.y) != "cell":
            return
        if self.table.identify_column(e.x) != "#1":
            return
        row = self.table.identify_row(e.y)
        if not row:
            return
        if row in self.readonly_iids:
            return
        curr = self.table.set(row, "pick")
        newv = "" if curr == "X" else "X"
        self.table.set(row, "pick", newv)
        self.proc.markFile(Path(row), selected=(newv == "X"))

    def run_selected(self):
        selectedPaths = self.proc.getSelectedPaths()
        if not selectedPaths:
            messagebox.showinfo("Preprocessor", "No files selected.")
            return

        # Use Preprocessor.parseMelParams if available; else local helper
        if parseMelParams is not None:
            mel = parseMelParams(
                nMels=self.nMelsVar.get(),
                hopLength=self.hopVar.get(),
                fMin=self.fMinVar.get(),
                fMax=self.fMaxVar.get(),
            )
       

        written, info = runPreprocess(
            cfg=self.cfg,
            label=None,  # auto-infer class from path
            selectedPaths=selectedPaths,
            targetSr=int(self.srVar.get()),
            forceMono=True,
            doNormalize=True,
            sliceSeconds=int(self.sliceVar.get()),
            nFft=1024,
            nMels=mel["nMels"],
            hopLength=mel["hopLength"],
            fMin=mel["fMin"],
            fMax=mel["fMax"],
            overwritePolicy="skip",
            alsoSaveAudio=False,
            skip_if_exists=True,
        )

        msg = (
            f"Processed {len(selectedPaths)} WAV file(s)\n\n"
            f"New tiles written: {info.get('written', written)}\n"
            f"Already existed (skipped): {info.get('skipped_existing', 0)}\n" #I think I am doing something wrong here, cause it is stacking for some reason, I dont understand it, Can't find anything on StackOverflow.
            f"Errors: {info.get('skipped_errors', 0)}\n\n"
            f"Saved in: {info.get('vision_root', self.cfg.get_preproc_root())}"
        )
        messagebox.showinfo("Preprocessor", msg)
        self.search()  # refresh statuses

    def run_all(self):
        if not hasattr(self, "btn_all"):  # guard
            return
        self.btn_all.configure(state="disabled")
        try:
            written, info = preprocess_all_train_cut(
                cfg=self.cfg,
                targetSr=int(self.srVar.get()),
                sliceSeconds=int(self.sliceVar.get()),
                forceMono=True,
                doNormalize=True,
            )
            msg = (
                "Processed ALL files from train_cut\n\n"
                f"New tiles written: {info.get('written', 0)}\n"
                f"Already existed (skipped): {info.get('skipped_existing', 0)}\n"
                f"Errors: {info.get('skipped_errors', 0)}\n\n"
                f"Saved in: {info['vision_root']}"
            )
            messagebox.showinfo("Preprocess ALL", msg)
            self.refresh_list()
            self.search()
        except Exception as e:
            messagebox.showerror("Preprocess ALL", f"Something went wrong:\n{e}")
        finally:
            self.btn_all.configure(state="normal")

    # Preview helpers
    def refresh_list(self):
        if not self.imgList: return
        self.imgList.delete(0, "end")
        self.PREPROC_DIR.mkdir(parents=True, exist_ok=True)
        for p in sorted(self.PREPROC_DIR.rglob("*.png")):
            self.imgList.insert("end", str(p))

    def show_selected(self, _evt=None):
        if not self.imgList or self.ax is None or self.canvas is None:
            return
        sel = self.imgList.curselection()
        if not sel:
            return
        path = self.imgList.get(sel[0])
        try:
            import matplotlib.pyplot as _plt
            img = _plt.imread(path)
        except Exception:
            return
        self.ax.clear()
        self.ax.imshow(img, origin="upper", aspect="auto")
        self.ax.axis("off")
        self.ax.set_title(Path(path).name)
        self.canvas.draw()


#IMAGEMODEL TAB
"""
Lightweight training and single file testing UI. 
Trains PCA + Logistic Regression on the tiles. Tests a single .wav and shows spectrogram side by sidee with prediction label
"""
class ImageModelTab:
   
    """
    Encapsulates the ImageModel training/prediction UI.
    """
    def __init__(self, notebook: ttk.Notebook):
        self.notebook = notebook
        self.cfg = Config()
        self.pcaVar = tk.IntVar(value=256)
        self.itVar  = tk.IntVar(value=200)
        self.cVar   = tk.DoubleVar(value=2.0)
        self.procVar = tk.StringVar(value="tiny")
        self.antiVar = tk.BooleanVar(value=True)
        self.log: tk.Text | None = None
        self.predVar = tk.StringVar(value="—")
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.canvas = None
        self.imodel = None

    def build(self) -> None:
        import numpy as np  # noqa: F401
        from ImageModel import ImageModel

        tab = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(tab, text="ImageModel")

        tab.columnconfigure(0, weight=0)
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(0, weight=1)

        left = ttk.Frame(tab)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        for r in range(20): left.rowconfigure(r, weight=0)
        left.rowconfigure(19, weight=1)

        ttk.Label(left, text="Training (PCA + Logistic Regression)").grid(row=0, column=0, sticky="w")

        rowA = ttk.Frame(left); rowA.grid(row=1, column=0, sticky="w", pady=(2,0))
        ttk.Label(rowA, text="PCA comps").pack(side="left"); ttk.Entry(rowA,state="disabled", width=7, textvariable=self.pcaVar).pack(side="left", padx=6) #Make disabled atm
        ttk.Label(rowA, text="Max iters").pack(side="left"); ttk.Entry(rowA,state="disabled", width=7, textvariable=self.itVar).pack(side="left", padx=6)
        ttk.Label(rowA, text="C").pack(side="left"); ttk.Entry(rowA,state="disabled", width=7, textvariable=self.cVar).pack(side="left", padx=6)

        self.log = tk.Text(left, width=40, height=18, wrap="word")
        self.log.grid(row=2, column=0, sticky="nsew", pady=(6,0))

        def log_cb(s: str):
            self.log.insert("end", s)
            self.log.see("end")

        self.imodel = ImageModel()

        def do_train():
            self.imodel.pca_components = int(self.pcaVar.get())
            self.imodel.max_iter = int(self.itVar.get())
            self.imodel.C = float(self.cVar.get())
            try:
                train_btn.configure(state="disabled")
                self.imodel.train(log_cb=log_cb)
            except Exception as e:
                log_cb(f"[ERROR] {e}\n")
            finally:
                train_btn.configure(state="normal")

        train_btn = ttk.Button(left, text="Train on preprocessed", command=do_train)
        train_btn.grid(row=3, column=0, sticky="ew", pady=(6,0))

        rowMode = ttk.Frame(left); rowMode.grid(row=4, column=0, sticky="w", pady=(8,0))
        ttk.Label(rowMode, text="Choose pipeline:").pack(side="left", padx=(0,6))
        ttk.Combobox(rowMode, width=10, textvariable=self.procVar, state="readonly", values=["tiny"]).pack(side="left")
        ttk.Checkbutton(left, text="Anti-Cheat (ignore filenames)", variable=self.antiVar)\
            .grid(row=6, column=0, sticky="w", pady=(4,0))

        tutorial_frame = ttk.LabelFrame(left, text="How to use:", padding=6)
        tutorial_frame.grid(row=7, column=0, sticky="ew", pady=(12, 0))
        tutorial_box = tk.Text(tutorial_frame, height=12, width=45, wrap="word", bg="#f9f9f9",
                               relief="flat", font=("Segoe UI", 9))
        tutorial_box.pack(fill="both", expand=True)
        tutorial_text = ("Imagemodel is using all items from preprocessor."
                         "\n 1. Choose pipeline"
                         "\n       - tiny uses standard spectrograms during processing"
                         "\n       -MEL not built yet)"
                         "\n 2. Drag and drop/or choose file"
                         "\n 3. Keep 'Anticheat ON' to avoid filename being used")
        tutorial_box.insert("1.0", tutorial_text)
        tutorial_box.configure(state="disabled")

        pics_dir = self.cfg.get_project_root() / "Gui_Pictures"
        slots = ttk.LabelFrame(left, text='I think this is a …', padding=6)
        slots.grid(row=5, column=0, sticky="ew", pady=(12, 0))

        from PIL import Image, ImageTk
        def load_img(name):
            p = pics_dir / f"{name}.png"
            if not p.exists(): p = pics_dir / f"{name}.jpg"
            if not p.exists(): return None
            img = Image.open(p).resize((96, 96), Image.BILINEAR)
            return ImageTk.PhotoImage(img)

        classes = self.cfg.get_folders()
        label_map = {0: ("motor_good", "Good"),
                     1: ("motor_broken", "Broken"),
                     2: ("motor_heavyload", "Heavyload")}
        self.count_vars = {}
        self.frames = {}
        for i, cname in enumerate(classes):
            key = label_map.get(i, ("", cname))[0]
            title = label_map.get(i, ("", cname))[1]
            fr = ttk.Frame(slots); fr.grid(row=0, column=i, padx=6)
            self.frames[i] = fr
            pic = load_img(key)
            if pic:
                lbl = ttk.Label(fr, image=pic)
                lbl.image = pic
                lbl.pack()
            ttk.Label(fr, text=title, font=("Segoe UI", 10, "bold")).pack()
            cv = tk.IntVar(value=0)
            ttk.Label(fr, textvariable=cv).pack()
            self.count_vars[i] = cv

        right = ttk.Frame(tab)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(2, weight=1)
        right.columnconfigure(0, weight=1)

        ttk.Label(right, text="Drop a WAV from test_cut here").grid(row=0, column=0, sticky="w")

        drop = tk.Label(right, text="Drag & Drop WAV (Not working \n(or click 'Choose file…')",
                        relief="groove", borderwidth=2, width=40, height=4, anchor="center")
        drop.grid(row=1, column=0, sticky="ew", pady=(4,6))

        def choose_file():
            f = filedialog.askopenfilename(title="Choose WAV", filetypes=[("WAV", "*.wav *.WAV")])
            if f: self._on_path_selected(Path(f))
        ttk.Button(right, text="Choose file…", command=choose_file).grid(row=1, column=0, sticky="e", padx=6, pady=(4,6))

        self.fig = Figure(figsize=(7.0, 3.4), dpi=100)
        self.ax1 = self.fig.add_subplot(1, 2, 1)
        self.ax2 = self.fig.add_subplot(1, 2, 2)
        self.ax1.set_title("Spectrogram")
        self.ax2.set_title("Pixel heatmap/ importance")
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().grid(row=2, column=0, sticky="nsew")

        ttk.Label(right, textvariable=self.predVar, font=("Segoe UI", 12, "bold")).grid(row=3, column=0, sticky="w", pady=(6,0))

        self.imodel = ImageModel()
        self.imodel.processing_mode = self.procVar.get().lower()

        # Drag & drop (optional)
        try:
            from tkinterdnd2 import DND_FILES  # type: ignore
            def drop_handler(event):
                raw = event.data.strip()
                if raw.startswith("{") and raw.endswith("}"):
                    raw = raw[1:-1]
                p = Path(raw)
                if p.suffix.lower() in (".wav", ".WAV"):
                    self._on_path_selected(p)
            drop.drop_target_register(DND_FILES)
            drop.dnd_bind("<<Drop>>", drop_handler)
        except Exception:
            pass

        try:
            style = ttk.Style(tab)
            style.configure("Pred.TFrame", borderwidth=2, relief="solid")
        except Exception:
            pass

    def _render_plots(self, spec, sal):
        self.ax1.clear(); self.ax2.clear()
        self.ax1.imshow(spec, origin="upper", aspect="auto", cmap="gray")
        self.ax1.axis("off")
        self.ax2.imshow(spec, origin="upper", aspect="auto", cmap="gray")
        self.ax2.imshow(sal, origin="upper", aspect="auto", alpha=0.45)
        self.ax2.axis("off")
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _bump(self, pred_idx: int):
        cv = self.count_vars.get(pred_idx)
        if cv: cv.set(cv.get() + 1)
        try:
            for k, fr in self.frames.items():
                fr.configure(style="")
            self.frames[pred_idx].configure(style="Pred.TFrame")
        except Exception:
            pass

    def _on_path_selected(self, path: Path):
        try:
            res = self.imodel.predict_wav(path, anti_cheat=bool(self.antiVar.get()))
            probs = res["probs"]
            idx = res["pred_idx"]
            name = res["pred_name"]
            
            self.predVar.set(f"Prediction: {name}  |  probs={np.round(probs,3)}")
            self._render_plots(res["spec"], res["saliency"])
            self._bump(idx)
        except Exception as e:
            self.predVar.set(f"Error: {e}")


#BLOWFIHCONTROLLER
"""
This is just a subprocess for using the prewritten code given by creator of final2. (Blowfish algorithm.)
"""
class BlowfishController:
    
    """
    Blowfishalgorithm was added late to project, adding the helpers into GUI_helpers
    """
    def __init__(self, root: tk.Tk, text_widget: tk.Text, status_var: tk.StringVar, script_var: tk.StringVar):
        self.root = root
        self.text = text_widget
        self.status = status_var
        self.script_var = script_var

        self._proc: subprocess.Popen | None = None
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._running = False

        # schedule queue pump
        self._pump()

    def append(self, s: str):
        self.text.insert("end", s)
        self.text.see("end")

    def _reader(self, proc: subprocess.Popen):
        try:
            for line in iter(proc.stdout.readline, ''):
                if not line:
                    break
                self._queue.put(line)
        except Exception as e:
            self._queue.put(f"\n[Blowfish] Reader error: {e}\n")
        finally:
            self._running = False

    def _pump(self):
        try:
            while True:
                line = self._queue.get_nowait()
                self.append(line)
        except queue.Empty:
            pass

        # If finished
        if self._proc and (self._proc.poll() is not None) and self._running:
            self._running = False
            rc = self._proc.returncode
            self.append(f"\n[Blowfish] Process finished (exit code {rc}).\n")
            self.status.set(f"Done (exit {rc})")
            self._proc = None

        # reschedule pump
        if self.root:
            self.root.after(50, self._pump)

    def start(self):
        if self._running:
            return
        script_path = Path(self.script_var.get()).resolve()

        cmd = [sys.executable, "-u", str(script_path)]

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        env["OMP_NUM_THREADS"] = "1"

        try:
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(script_path.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                bufsize=1,
                env=env
            )
        except Exception as e:
            messagebox.showerror("Blowfish", f"Could not start process:\n{e}")
            self._proc = None
            return

        self._running = True
        self.status.set("Running…")
        self.append(f"[Blowfish] Starter: {' '.join(cmd)}\n\n")

        self._thread = threading.Thread(target=self._reader, args=(self._proc,), daemon=True)
        self._thread.start()

    def stop(self):
        if self._proc and self._running:
            try:
                self._proc.terminate()
            except Exception:
                pass
            self.append("\n[Blowfish] Terminating process…\n")
            self.status.set("Stopping…")

    def clear(self):
        self.text.delete("1.0", "end")

    def save_log(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default = f"blowfish_log_{ts}.txt"
        path = filedialog.asksaveasfilename(
            title="Save log",
            defaultextension=".txt",
            initialfile=default,
            filetypes=[("Text", "*.txt"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            data = self.text.get("1.0", "end-1c")
            Path(path).write_text(data, encoding="utf-8")
            messagebox.showinfo("Blowfish", f"Saved:\n{path}")
        except Exception as e:
            messagebox.showerror("Blowfish", f"Could not save:\n{e}")

    def shutdown(self):
        """Call from GUI.on_close to ensure the process is stopped."""
        try:
            self.stop()
        except Exception:
            pass
   
#BLOWFISH TAB
"""
Simple tab shell that hosts BlowfishController with Start/Stop/Clear/Save-log.
Lets you browse to a different .py file if you want to run another script.
"""
class BlowfishTab:
    
    """
    Wraps the BlowfishController into a tab component.
    """
    def __init__(self, root: tk.Tk, notebook: ttk.Notebook):
        self.root = root
        self.notebook = notebook
        self.controller: BlowfishController | None = None
        self.status = tk.StringVar(value="Idle")
        self.script_var = tk.StringVar(
            value=str((Path(__file__).resolve().parent / "final2.py"))
        )

    def build(self) -> BlowfishController:
        tab = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(tab, text="Blowfish")

        # Top row: controls
        row = ttk.Frame(tab)
        row.grid(row=0, column=0, sticky="ew")
        row.columnconfigure(10, weight=1)

        ttk.Label(row, textvariable=self.status).grid(row=0, column=0, padx=(0, 10))

        # Script picker
        ttk.Label(row, text="Script:").grid(row=0, column=5, padx=(16, 4))
        entry = ttk.Entry(row, width=48, textvariable=self.script_var)
        entry.grid(row=0, column=6, padx=4, sticky="ew")

        def browse():
            path = filedialog.askopenfilename(
                title="Choose Python script",
                filetypes=[("Python", "*.py"), ("All files", "*.*")]
            )
            if path:
                self.script_var.set(path)
        ttk.Button(row, text="…", width=3, command=browse).grid(row=0, column=7, padx=(0, 4))

        # Log area
        wrap = ttk.Frame(tab)
        wrap.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        tab.rowconfigure(1, weight=1)
        tab.columnconfigure(0, weight=1)

        text = tk.Text(wrap, height=24, wrap="word")
        text.grid(row=0, column=0, sticky="nsew")
        yscroll = ttk.Scrollbar(wrap, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=yscroll.set)
        yscroll.grid(row=0, column=1, sticky="ns")
        wrap.rowconfigure(0, weight=1)
        wrap.columnconfigure(0, weight=1)

        self.controller = BlowfishController(
            root=self.root,
            text_widget=text,
            status_var=self.status,
            script_var=self.script_var,
        )

        # Buttons
        ttk.Button(row, text="Start", command=self.controller.start).grid(row=0, column=1, padx=4)
        ttk.Button(row, text="Stop", command=self.controller.stop).grid(row=0, column=2, padx=4)
        ttk.Button(row, text="Clear", command=self.controller.clear).grid(row=0, column=3, padx=4)
        ttk.Button(row, text="Save log", command=self.controller.save_log).grid(row=0, column=4, padx=4)

        return self.controller



class MlpController:
    """
    Controller that manages running the MLP script in a subprocess,
    capturing its stdout/stderr, and updating the GUI console in real time.

    """

    def __init__(self, root: tk.Tk, text_widget: tk.Text,
                 status_var: tk.StringVar, cmd_var: tk.StringVar,
                 cwd_var: tk.StringVar):
        self.root = root
        self.text = text_widget
        self.status = status_var
        self.cmd_var = cmd_var
        self.cwd_var = cwd_var

        # Process + threading state
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._running = False

        # Start queue pump so GUI refreshes asynchronously
        self._pump()


    def append(self, text: str):
        """Append text to the output widget."""
        self.text.insert("end", text)
        self.text.see("end")


    def _reader(self, proc: subprocess.Popen):
        """
        Background thread reading process stdout line by line.
        Thread from console over to interface
        """
        try:
            for line in iter(proc.stdout.readline, ''):
                if not line:
                    break
                self._queue.put(line)
        except Exception as e:
            self._queue.put(f"\n[MLP Reader error]: {e}\n")
        finally:
            self._queue.put("\n[MLP process finished]\n")
            self._running = False


    def _pump(self):
        """
        Periodically check for new output from the queue and update the GUI.
        Important to avoid overload crash
        """
        try:
            while True:
                line = self._queue.get_nowait()
                self.append(line)
            # loop until queue empty
        except queue.Empty:
            pass
        # reschedule next pump
        self.root.after(80, self._pump)

    
    def start(self):
        """
        Launch the MLP subprocess if not already running.
        """
        if self._running:
            self.append("\n[MLP already running]\n")
            return

        cmd = self.cmd_var.get().strip()
        cwd = self.cwd_var.get().strip()

        if not cmd:
            self.append("\n[No command specified!]\n")
            return

        try:
            self.append(f"\n[Starting MLP process: {cmd}]\n")
            self.status.set("Running")
            self._proc = subprocess.Popen(
                cmd,
                cwd=cwd or None,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self._running = True
            self._thread = threading.Thread(target=self._reader, args=(self._proc,), daemon=True)
            self._thread.start()

        except Exception as e:
            self.append(f"\n[Failed to start process: {e}]\n")
            self.status.set("Error")

    
    def stop(self):
        """
        Stop the MLP subprocess if it is running.
        """
        if not self._proc or not self._running:
            self.append("\n[No active MLP process]\n")
            return

        self.append("\n[Stopping MLP process...]\n")
        try:
            self._proc.terminate()
            self._proc.wait(timeout=3)
        except Exception:
            try:
                self._proc.kill()
            except Exception as e:
                self.append(f"[Force kill failed: {e}]\n")

        self._running = False
        self.status.set("Stopped")

    
    def clear(self):
        """Clear the output console."""
        self.text.delete("1.0", "end")
        self.append("[Console cleared]\n")

   
    def save_log(self):
        """Save the current console contents to a text file."""
        content = self.text.get("1.0", "end-1c")
        if not content.strip():
            messagebox.showinfo("Save log", "No content to save.")
            return
        filename = filedialog.asksaveasfilename(
            title="Save MLP log",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
        )
        if filename:
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(content)
                messagebox.showinfo("Save log", f"Saved to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Save log", f"Error saving log:\n{e}")



#MLP Controller

#MLP TAB
class MlpTab:
    """
    Fixed MLP runner tab: always runs the local mlp.py (no file picker).
    Shows the script path in the entry field.
    """
    def __init__(self, root: tk.Tk, notebook: ttk.Notebook):
        self.root = root
        self.notebook = notebook
        self.controller: MlpController | None = None
        self.status = tk.StringVar(value="Status")

        # Determine exact path to mlp.py (same folder as this file)
        script_path = Path(__file__).resolve().parent / "mlp.py"

        # Show this path in GUI and use it for execution
        self.script_var = tk.StringVar()
        self.script_var.set(str(script_path))  # ensure value is visible

        # Create command and working directory for MlpController
        self.cmd_var = tk.StringVar(value=f'"{sys.executable}" -u "{script_path}"')
        self.cwd_var = tk.StringVar(value=str(script_path.parent))

    def build(self) -> MlpController:
        tab = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(tab, text="MLP")

        #Header row 
        row = ttk.Frame(tab)
        row.grid(row=0, column=0, sticky="ew")
        row.columnconfigure(10, weight=1)

        ttk.Label(row, textvariable=self.status, width=8).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(row, text="Start", command=lambda: self.controller.start()).grid(row=0, column=1, padx=4)
        ttk.Button(row, text="Stop", command=lambda: self.controller.stop()).grid(row=0, column=2, padx=4)
        ttk.Button(row, text="Clear", command=lambda: self.controller.clear()).grid(row=0, column=3, padx=4)
        ttk.Button(row, text="Save log", command=lambda: self.controller.save_log()).grid(row=0, column=4, padx=4)

        ttk.Label(row, text="Script:").grid(row=0, column=5, padx=(16, 4))
        entry = ttk.Entry(row, textvariable=self.script_var, width=60, state="readonly")
        entry.grid(row=0, column=6, padx=4, sticky="ew")

        #Console area
        wrap = ttk.Frame(tab)
        wrap.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        tab.rowconfigure(1, weight=1)
        tab.columnconfigure(0, weight=1)

        text = tk.Text(wrap, height=24, wrap="word")
        text.grid(row=0, column=0, sticky="nsew")
        yscroll = ttk.Scrollbar(wrap, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=yscroll.set)
        yscroll.grid(row=0, column=1, sticky="ns")
        wrap.rowconfigure(0, weight=1)
        wrap.columnconfigure(0, weight=1)

        #Controller setup
        self.controller = MlpController(
            root=self.root,
            text_widget=text,
            status_var=self.status,
            cmd_var=self.cmd_var,
            cwd_var=self.cwd_var,
        )

        return self.controller

    
#Helper wrappers sending to GUI
def build_preprocessor_tab(notebook: ttk.Notebook):
    """Previous public API — now delegates to PreprocessorTab class."""
    PreprocessorTab(notebook).build()


def build_image_model_tab(notebook: ttk.Notebook):
    """Previous public API — now delegates to ImageModelTab class."""
    ImageModelTab(notebook).build()


def build_blowfish_tab(root: tk.Tk, notebook: ttk.Notebook):
    """Previous public API — now delegates to BlowfishTab class and returns its controller."""
    return BlowfishTab(root, notebook).build()

def build_mlp_tab(root: tk.Tk, notebook: ttk.Notebook): 
    "Reuse of Blowfish tab to create mlp controller tab"
    return MlpTab(root, notebook).build()

def build_heartbeat_tab(notebook: ttk.Notebook):
    return heartbeatController(root,notebook).build()





