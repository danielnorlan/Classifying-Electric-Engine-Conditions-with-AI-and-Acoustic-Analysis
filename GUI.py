"""
Author: Hans Josef Rosland-Borlaug[258139]
startdate: 19.09.25
Last edit (date):

This module defines the top-level GUI class for the Engine Classifier System. It creates
the main window, builds tabs for different parts of the system (Player, Preprocessor,
ImageModel, and Blowfish), and wires up event handlers. Throughout the code you will
find inline comments explaining the purpose of each line to make the logic easier to
follow. The GUI relies on Tkinter for the interface, matplotlib for plotting
spectrograms, and pygame for audio playback.
"""

# GUI.py
import tkinter as tk  # Import Tkinter base classes for GUI creation
from tkinter import ttk  # Import themed Tkinter widgets for modern look and feel
import pygame  # Import pygame for audio playback
from matplotlib.figure import Figure  # Import Figure class to create matplotlib figures
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Embed matplotlib in Tkinter
from heartPTE import build_heartbeat_tab #The function for building the heartbeat build, IT IS NOT under GUI helpers
import vlc #importing this to try to fix the "afterclose" problem (audio still keeps going from video)

from pathlib import Path  # Import Path for path manipulations (not used directly here)

from Gui_Helpers import (
    build_preprocessor_tab, # Function to construct the Preprocessor tab
    build_image_model_tab,  # Function to construct the ImageModel tab
    build_blowfish_tab, #Function to construct the Blowfish tab; returns a controller
    build_mlp_tab,#Building tab for mlp
    
     
)

from player import Player  
from configs import Config


class GUI:
    """
    Reflection and build:
        
        Top-level GUI container that builds the main notebook and its tabs.
    
        This is the "skeleton build that holds everything together. It constructs a Tkinter
        Notebook widget with tabs for playing audio, preprocessing data, training the
        ImageModel, and running the Blowfish script. All heavy lifting is delegated
        to helper functions defined in Gui_Helpers.py, leaving this class to focus on
        layout and event wiring.
        
        I am not confident that this is a good way to split modules, but as I code I need to be able to have control over widget related and functionsrelated GUI parts.
    """

    def initialize(self, root: tk.Tk):
        """
        Initialize the GUI using a provided Tk root window.

        :param root: The Tk root window in which to build the GUI.
        """
        cfg = Config()#Create a configuration object to access folder paths and settings

        # Store the root window as an attribute for later use
        self.root = root
        # Set the title of the main window to identify the application
        self.root.title("ENGINE CLASSIFIER SYSTEM (GROUP 7)")

        # Configure the root grid to expand the notebook properly
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        # Create a Notebook widget; this will contain multiple tabs
        self.notebook = ttk.Notebook(root)
        # Place the notebook in the grid, occupying the entire window
        self.notebook.grid(row=0, column=0, sticky="nsew")

        #Player tab
        # Create a frame for the Player tab and add it to the notebook
        player_tab = ttk.Frame(self.notebook)
        self.notebook.add(player_tab, text="Player")
        player_tab.columnconfigure(0, weight=1)
        player_tab.rowconfigure(0, weight=1)
        
        #####
        "Hearbeat insert"
        self.heartbeat = build_heartbeat_tab(self.root, self.notebook)

        ####
        
        # A wrapper frame inside the player tab for padding and organization
        wrap = ttk.Frame(player_tab, padding=12)
        wrap.grid(row=0, column=0, sticky="nsew")

        # Folder/File controls (pulled from Config)
        ttk.Label(wrap, text="Folder").grid(row=0, column=0, sticky="w") #Label for folder selection
        self.folder_cb = ttk.Combobox(
            wrap, state="readonly", values=cfg.get_folders(), width  =40 #ComboBox listing available class folders
        )
        self.folder_cb.grid(row=1, column=0, sticky="ew", pady=(0, 8)) #Place the folder combobox

        ttk.Label(wrap, text="File").grid(row=2, column=0, sticky="w") #Label for file selection
        self.file_cb = ttk.Combobox(wrap, state="readonly", values=[], width=60)  # ComboBox for WAV files
        self.file_cb.grid(row=3, column=0, sticky="ew", pady=(0, 8)) #Place the file combobox

        # Create a label frame for the live spectrogram view
        live = ttk.LabelFrame(wrap, text="Live View (spectrogram)")
        live.grid(row=5, column=0, sticky="nsew", pady=(10, 0))
        wrap.rowconfigure(5, weight=1)
        wrap.columnconfigure(0, weight=1)

        # Create the matplotlib figure and axes used for real-time spectrogram visualization
        self.fig_view = Figure(figsize=(6, 3), dpi=100)
        self.ax_view = self.fig_view.add_subplot(111)
        self.ax_view.set_title("No view yet", fontsize=10) # Initial title when no file is selected
        self.ax_view.set_xlabel("Time (s)")  # X-axis label
        self.ax_view.set_ylabel("Frequency (Hz)")  # Y-axis label
        # Embed the matplotlib figure inside the Tkinter frame
        self.canvas_view = FigureCanvasTkAgg(self.fig_view, master=live)
        self.canvas_view.get_tk_widget().pack(fill="both", expand=True)

        # Create a row to display computed frequency bounds (lower, center, upper)
        freq_row = ttk.Frame(live)
        freq_row.pack(fill="x", pady=(6, 6))

        # Tkinter StringVars hold dynamic text for frequency labels
        self.lowerVar  = tk.StringVar(value="Lower: –")  
        self.centerVar = tk.StringVar(value="Center: –") 
        self.upperVar  = tk.StringVar(value="Upper: –")  

        # Place labels bound to the StringVars to show frequency information
        ttk.Label(freq_row, textvariable=self.lowerVar).pack(side="left", padx=(0, 16))
        ttk.Label(freq_row, textvariable=self.centerVar).pack(side="left", padx=(0, 16))
        ttk.Label(freq_row, textvariable=self.upperVar).pack(side="left")

        # Instantiate the Player object; it manages audio playback and visualization
        self.player = Player(
            folder_dir=cfg.get_test_root(),# Directory containing test WAV files
            exts=cfg.get_exts(),# Allowed file extensions (e.g., .wav)
            folder_cb=self.folder_cb, # Reference to the folder combobox
            file_cb=self.file_cb,# Reference to the file combobox
            lowerVar=self.lowerVar,# Reference to the lower frequency variable
            centerVar=self.centerVar,# Reference to the center frequency variable
            upperVar=self.upperVar,# Reference to the upper frequency variable
            ax_view=self.ax_view,# Axes object for drawing spectrograms
            fig_view=self.fig_view,# Figure object containing the axes
            canvas_view=self.canvas_view # Canvas object to render the figure in Tkinter
        )

        # Control Buttons
        
        # Create a horizontal frame for playback buttons
        row = ttk.Frame(wrap)
        row.grid(row=4, column=0, sticky="w", pady=(4, 0))
        # Button to start playback of the selected file
        ttk.Button(row, text="PLAY",           command=self.player.play_selected).grid(row=0, column=0, padx=(0, 6))
        # Button to stop audio playback
        ttk.Button(row, text="STOP",           command=self.player.stop_audio).grid(row=0, column=1, padx=(0, 6))
        # Button to reload the file list based on the selected folder
        ttk.Button(row, text="Reset Filepath", command=self.player.reset_files).grid(row=0, column=2)
        # Button to clear the spectrogram view when no audio is playing
        ttk.Button(row, text="Clear view",     command=self.player.clear_view).grid(row=0, column=7, padx=(12, 0))

        # Bind changes in the folder combobox to refresh the file list
        self.folder_cb.bind("<<ComboboxSelected>>", lambda e: self.player.reset_files())

        # Initialization
        
        # Set initial values for the folder combobox
        folders = cfg.get_folders()
        if folders:
            self.folder_cb.set(folders[0])  # Default to the first folder in the list
        # Populate the file combobox based on the selected folder
        self.player.reset_files()

      
        #Global hotkeys and cleanup
        # Create some basic keyboard shortcuts for the player
        root.bind("<Return>", lambda e: self.player.play_selected())  # Enter plays audio
        root.bind("<space>",  lambda e: self.player.stop_audio())    # Space stops audio
        #
        self.root.protocol("WM_DELETE_WINDOW", self.on_close) #Close window

        #Additional tabs
        
        # Construct the Preprocessor tab using helper function
        build_preprocessor_tab(self.notebook)# Preprocessor
        # Construct the ImageModel tab using helper function
        build_image_model_tab(self.notebook) # ImageModel
        # Construct the Blowfish tab; store controller for later shutdown
        self.blowfish = build_blowfish_tab(self.root, self.notebook)  # Blowfish runner (controller returned)
        #Construct the MLPcontroller tab
        self.terminal = build_mlp_tab(self.root, self.notebook)  


   
    # Close ALL (Shutdown all ongoing processes)
    def on_close(self):         
        """
        stops ongiong processes
        note to self: 
            -Does not stop the mp4 vlc player for some reason, adding some extra self-stops, should have had a fallback.
        """
        
        player = vlc.MediaPlayer()
        try:
            self.root.destroy()
            self.heartbeat.shutdown()
            self.pygame.mixer.stop()
          
            self._vlc_player.stop() #These are kinda unnecessary, but trying to kill the vlc player
            self._vlc_player.release()
            self.player.stop_audio()
            self.blowfish.shutdown()
            self.preprocessor.shutdown()
            self.imagemodel.shutdown()
            
          
        except Exception:
            pass
    
