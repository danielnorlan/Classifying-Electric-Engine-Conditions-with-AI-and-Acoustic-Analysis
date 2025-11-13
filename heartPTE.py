"""
Author: Hans Josef Rosland-Borlaug[258139]
Supervisor: Viktória Tóthi Szófia [PTE]

Reflection:
    This was added insanely late into the project, so in order to make it happen, and to be able to writing the whole report by myself(almost) I hardcoded a lot of things into the module.
    This is NOT optimal way of coding, but I really wanted to help the PTE student having something to present. 
    
"""

import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from pathlib import Path

from PIL import Image, ImageTk
import vlc  

# hardcoded configs 

#Person_image_path = r"D:\Artificial intelligence\Group project\AI Group Project alpha ver\Project mastermap\heartbeat\Animation\person.png"
Person_image_path = r"D:\Artificial intelligence\Group project\AI Group Project alpha ver\Project mastermap\heartbeat\Animation\person2.png"
Physio_path1 = r"D:\Artificial intelligence\Group project\AI Group Project alpha ver\Project mastermap\heartbeat\Animation\Physionet.png"
Physio_path2 = r"D:\Artificial intelligence\Group project\AI Group Project alpha ver\Project mastermap\heartbeat\Animation\Physionet2.png"
image_width = 380  

# Video_path = r"D:\Artificial intelligence\Group project\AI Group Project alpha ver\Project mastermap\heartbeat\Animation\Heart Beating Animation - San Jose Mathew (1080p, h264) (2).mp4"
# -------------------------------------------------------------
Text = (
    "The main idea of our project was to build a heartbeat classification system using real heart sounds "
    "from the PhysioNet Challenge 2016 dataset.\n"
    "We wanted to analyze these recordings to detect early signs of heart disease such as murmurs or arrhythmias.\n\n"
    
    "The system was meant to clean the heartbeat audio, remove background noise, and then extract key features — "
    "like the rhythm, frequency spectrum, and amplitude variations — using the Preprocessor module we built.\n\n"
    
    "After preprocessing, these features were supposed to be sent to different models we experimented with:\n\n"
    
    "- ImageModel.py – a small image-based classifier that turns heartbeat sounds into spectrograms "
    "and uses PCA + Logistic Regression for predictions.\n"
    "- mlp.py – a more advanced MLP (Multilayer Perceptron) that works directly on extracted numerical audio features.\n"
    "- final2.py – our planned deep-learning extension using a CNN trained on log-mel spectrograms.\n\n"
    
    "We also integrated Blowfish encryption to safely handle medical data if the system were deployed.\n\n"
    
    "Our goal was to connect all these parts into one pipeline where:\n"
    "- The Preprocessor converts raw heartbeats to spectrogram images.\n"
    "- The MLP or CNN classifies them as normal or abnormal.\n"
    "- The ImageModel could visualize which frequency regions contributed most (like a saliency map).\n\n"
    
    "However, we didn’t fully reach the implementation stage where everything runs end-to-end.\n"
    "We managed to design and test each part separately, but time constraints and integration challenges stopped us "
    "from completing the full connection between preprocessing, classification, and encryption.\n\n"
    
    "Even so, the base structure is in place.\n"
    "The project demonstrates how different machine learning approaches — from simple PCA models to MLPs and CNNs — "
    "can work together to form an early-warning system for heart conditions."
)


class HeartbeatTab:
    """
    A simple page: person image on the left, read-only info textbox on the right.
    """

    def __init__(self, root: tk.Tk, notebook: ttk.Notebook):
        self.root = root
        self.notebook = notebook
        self._tk_img = None  # keep a reference so Tk doesn't GC the image
        # keep references for the PhysioNet images too
        self.physio1 = None
        self.physio2 = None

    def build(self):
        # Tab container
        tab = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(tab, text="Heartbeat")

        # Title
        ttk.Label(tab, text="Heartbeat", font=("Segoe UI", 14, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )

        #Main area
        body = ttk.Frame(tab)
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)   # left: person image
        body.columnconfigure(1, weight=2)   # middle: text
        body.columnconfigure(2, weight=0)   # right: stacked logos
        body.rowconfigure(0, weight=1)

        #Left side
        img_label = ttk.Label(body)
        img_label.grid(row=0, column=0, sticky="nw", padx=(0, 8))

        self._load_person_scaled_into(img_label, image_width)

        #Right side (Holds text)
        info = scrolledtext.ScrolledText(
            body, wrap="word", height=20, width=48, font=("Segoe UI", 10)
        )
        info.grid(row=0, column=1, sticky="nsew")
        info.insert("1.0", Text)
        info.configure(state="disabled")

        # Right side - stacked PhysioNet images
        img_frame = ttk.Frame(body)
        img_frame.grid(row=0, column=2, sticky="ne", padx=(8, 0))

        try:
            image1 = Image.open(Physio_path1).resize((250, 150))
            image2 = Image.open(Physio_path2).resize((250, 150))

            self.physio1 = ImageTk.PhotoImage(image1)
            self.physio2 = ImageTk.PhotoImage(image2)

            ttk.Label(img_frame, image=self.physio1).pack(pady=(0, 6))
            ttk.Label(img_frame, image=self.physio2).pack()
        except Exception as e:
            ttk.Label(img_frame, text=f"Error loading images:\n{e}", foreground="red").pack()

        return self  # return the tab instance

    # Helpers

    def _load_person_scaled_into(self, label: ttk.Label, target_width: int):
        
        
        """
        Load person.png, scale to target width, and set into the given label.
        I am worried I get get stacked, so I make a function to "safeload so I can backtrack if they stack
        
        """
        path = Path(Person_image_path)
        if not path.exists():
            label.configure(text=f"Image not found:\n{path}")
            return

        pil = Image.open(path).convert("RGBA")
        w0, h0 = pil.size
        scale = target_width / float(w0)
        new_w = int(round(w0 * scale))
        new_h = int(round(h0 * scale))
        pil = pil.resize((new_w, new_h), Image.LANCZOS)

        tk_img = ImageTk.PhotoImage(pil)
        self._tk_img = tk_img
        label.configure(image=tk_img)


# Wrapper used by your main GUI to add the tab
def build_heartbeat_tab(root, notebook):
    return HeartbeatTab(root, notebook).build()


#     #Controls
#     controls = ttk.Frame(tab)
#     controls.grid(row=1, column=0, sticky="w", pady=(0, 6))
#     ttk.Button(controls, text="Play", command=self._on_play).pack(side="left")
#     ttk.Button(controls, text="Stop", command=self._on_stop).pack(side="left", padx=(6, 0))

#     # I keep a status label below the controls so I can see the playback state.
#     self.status = tk.StringVar(value="Ready.")
#     ttk.Label(controls, textvariable=self.status).pack(side="left", padx=(12, 0))

#     # Main area: person image + a video panel exactly over the chest box
#     body = ttk.LabelFrame(tab, text="Info", padding=8)
#     body.grid(row=2, column=0, sticky="nsew")
#     body.columnconfigure(0, weight=1)
#     body.rowconfigure(0, weight=1)

#     # Creating border for the person image (I prefer canvas so I can overlay the video panel via create_window)
#     self.canvas = tk.Canvas(body, highlightthickness=0, bg="white")
#     self.canvas.grid(row=0, column=0, sticky="nsew", padx=self.CANVAS_PAD, pady=self.CANVAS_PAD)

#     # Load + draw the person image (scaled to a friendly width with Pillow)
#     self._draw_person_scaled()

#     # The video panel sits on top of the chest window.
#     # I set bg=white so borders blend with the chest box.
#     self.video_panel = tk.Frame(self.canvas, bg="white", bd=0, highlightthickness=0)
#     # keep the canvas item id if I ever want to control z-order explicitly
#     self.video_item = self.canvas.create_window(0, 0)  # temporary; replaced in _place_video_panel
#     self.canvas.delete(self.video_item)                # cleanup temp; _place_video_panel creates the real one
#     self._place_video_panel()

# def _place_video_panel(self):
#     """
#     Convert the relative chest-box (0..1) coords into pixels for the *displayed* image,
#     then place a real Tk Frame there using canvas.create_window().
#       I dont understand this, I know I have to use the relative coordinates to make the draw assertion correct like in C#, 
#      Calculating the image position and pixel width, but I need to look it up from there.
#      Import VLC player and play it inside a box was harder to understand than I thought
#     """
#     disp_w, disp_h = self._img_disp_size
#     assert disp_w > 0 and disp_h > 0, "Image not drawn yet — cannot place video panel."

#     x0r, y0r, x1r, y1r = self.CHEST_BOX_REL
#     x0 = int(x0r * disp_w)
#     y0 = int(y0r * disp_h)
#     x1 = int(x1r * disp_w)
#     y1 = int(y1r * disp_h)
#     w = max(8, x1 - x0)
#     h = max(8, y1 - y0)

#     # Mount the video panel exactly over the chest region
#     # store item id, in case I want to manage z-order later
#     self.video_item = self.canvas.create_window(
#         x0, y0, window=self.video_panel, anchor="nw", width=w, height=h
#     )