
""" 
Reflection:
I am experimenting with the possibility to create a module that holds all imports. 
I have not seen anyone do it like this, and I am certain there is a reason why people chose to do imports in all modules, but
I created a class to give it a go anyway.

Note to self: Ask Professor Ru Yan if there is a good way to do this. """


class Library:
    import os
    import numpy as np
    import pygame
    import wave
    import time
    import threading
    from scipy.signal import spectrogram
    from tkinter import messagebox
    from pathlib import Path
    import os
    from pathlib import Path
    import tkinter as tk
    import pygame 
    import matplotlib.pyplot as plt


    from tkinter import ttk, messagebox
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from Gui_Helpers import build_preprocessor_tab, build_image_model_tab
    from player import Player  # Functions from Player class
    from configs import Config # Filepaths from Config class
    from pathlib import Path
    import tkinter as tk
    from tkinter import ttk, messagebox
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    from Preprocessor import Preprocessor, parseMelParams, runPreprocess
    from configs import Config
    from PIL import Image, ImageTk 
    from ImageModel import ImageModel
    from pathlib import Path
    from typing import Optional, Dict, List
    import wave
    import numpy as np
    import matplotlib.pyplot as plt

    from configs import Config
    import tkinter as tk
    from GUI import GUI
    
