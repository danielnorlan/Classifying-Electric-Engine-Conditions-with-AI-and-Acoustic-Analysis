"""
Author: Hans Josef Rosland-Borlaug[258139]
startdate: 19.09.25
Last edit (date): 01.11.25

Build and reflection:
    
Creating a standalone feature in the GUI.  
The Player tab is an addition/extension to a prebuilt system originally used to draw “stockcharts” 
with upper, lower, and center value indicators. I believe I can adapt and remake that logic 
to visualize our audio data in real time instead.

The goal is to make this module feel like a live “sound monitor” for the engine signals — 
a place where we can both listen to the WAV files and instantly see how the frequency energy moves. 
By combining playback and visualization, it becomes much easier to understand what the model 
later tries to learn from these signals.

I’m still exploring how best to integrate this into the existing structure — 
the challenge will be syncing the playback with the visualization thread so it updates smoothly 
without freezing the GUI. Once that’s stable, I can tune the spectrogram’s sensitivity and 
scaling to make it resemble the style of the original chart system, but specialized for sound.

Note to self (late): 
    I am getting Visualization error: unknown format: 3 on the "stresstest.wav" files. looking at the "egenskaper" I can see that stresstest files are 24-bit, not 16-bit. 
    Wave does not support 32 bit float. I have to be aware of it, but I dont want to rewrite the Player module at this moment so (Justify it by explaining to supervisor that you are aware of it).

"""
#Imports
import os  
import numpy as np  
import pygame  
import wave  
import time  
import threading  
from scipy.signal import spectrogram  
from tkinter import messagebox  


class Player:
    def __init__(self, folder_dir, exts, folder_cb, file_cb,
                 lowerVar, centerVar, upperVar, ax_view, fig_view, canvas_view):
        """Initialize the Player by storing Tk variables and UI widget references."""
        self.folder_dir = folder_dir  # Root directory where audio folders reside
        self.exts = exts  # Tuple of accepted audio file extensions
        self.folder_cb = folder_cb  # Combobox widget used for selecting the folder/class
        self.file_cb = file_cb  # Combobox widget used for selecting the file within the class
        self.lowerVar = lowerVar  # StringVar for displaying the lower frequency bound
        self.centerVar = centerVar  # StringVar for displaying the center frequency
        self.upperVar = upperVar  # StringVar for displaying the upper frequency bound
        self.ax_view = ax_view  # Matplotlib Axes for plotting the spectrogram
        self.fig_view = fig_view  # Matplotlib Figure containing the axes
        self.canvas_view = canvas_view  # FigureCanvas used to draw the figure in Tkinter

        self.filemap = {}  # Maps filename to its full path within the selected folder
        self.is_playing = False  # Flag indicating whether audio is currently playing
        self.visual_thread = None  # Thread object for running the visualization in the background
        self.visual_running = False  # Flag indicating whether the visualization thread is active
        pygame.mixer.init()  # Initialize the pygame mixer for audio playback
       

    # File management
    def list_wavs(self, subfolder: str):
        """List all WAV files in the given subfolder."""
        folder_path = self.folder_dir / subfolder  # Build the full path to the selected class folder
        if not folder_path.exists():
            return []  # Return an empty list if the folder does not exist
        lower_exts = tuple(e.lower() for e in self.exts)  # Normalize extensions to lowercase for comparison
        return [p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() in lower_exts]  # Collect valid files

    def reset_files(self):
        """Refresh the file list based on the selected folder."""
        sub = self.folder_cb.get()  # Get the selected folder name
        if not sub:
            return  # Do nothing if no folder is selected
        items = self.list_wavs(sub)  # Retrieve all WAV files in that folder
        self.filemap = {p.name: str(p) for p in items}  # Build a mapping of file name to full path
        names = list(self.filemap.keys())  # Extract just the names for the combobox
        self.file_cb["values"] = names  # Update the combobox values
        if names:
            self.file_cb.set(names[0])  # Set the first file as default selection

    # Playback
    def play_selected(self):
        """Play the currently selected WAV file."""
        disp = self.file_cb.get()  # Get the display name from the combobox
        if not disp:
            messagebox.showinfo("No file selected", "Please pick a file first.")  # Warn user if nothing is selected
            return
        full = self.filemap.get(disp)  # Look up the full path in the map
        if not full or not os.path.exists(full):
            messagebox.showerror("File not found")  # Show error if the file does not exist
            return

        # Prevent stacking: if the same file is already playing, do nothing, adding stackpreventions as I go
        if self.is_playing and pygame.mixer.music.get_busy() and hasattr(self, "current_path") and self.current_path == full:
            return

        # Stop any previous audio and visualization before starting new playback
        self.stop_audio()

        try:
            pygame.mixer.music.load(full)# Load the audio file into the mixer
            pygame.mixer.music.play()# Start playback
            self.is_playing = True # Set playing flag
            self.current_path = full # Store the path of the current file
            self.start_visualization(full) # Begin updating the visualization
        except Exception as e:
            self.is_playing = False
            messagebox.showerror("Playback error", str(e))  # Report any errors encountered during playback

    def stop_audio(self):
        """Stop audio playback and visualization."""
        try:
            if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()  # Stop the music if it's playing
        except Exception:
            pass  # Ignore errors during stop
        self.is_playing = False # Reset playing flag
        self.stop_visualization() # Stop the visualization thread

    # Visualization control
    def start_visualization(self, wav_path):
        """Start a background thread that continuously updates the live spectrogram
            while the audio file is playing.This keeps the GUI responsive — the visualization runs on its own thread
           so the main Tkinter event loop doesn’t freeze while we process audio data.
        """
        if self.visual_running:
            self.stop_visualization() # stop it first to prevent two threads from drawing to the same canvas. (Forgot this so visualization stacked causing GUI crash, and me crashing out)

        self.visual_running = True # Set flag "running", system knows its running
        
      #Create a new background thread dedicated to visualization.
        self.visual_thread = threading.Thread(
        target=self._update_visual,  # The method that does the actual drawing loop
        args=(wav_path,),# Arguments passed to method
        daemon=True# Implementing daemon from threading module to make sure it shuts down when exiting the program. Not required, but a smart function to include
    
        )
        self.visual_thread.start() #start loop

    def stop_visualization(self):
        """Stop the live visualization thread.
        """
        self.visual_running = False  # Signal the thread to stop
        if self.visual_thread and self.visual_thread.is_alive():
            self.visual_thread.join(timeout=1.0)  # Wait briefly for the thread to finish


    def clear_view(self):
        """Clear the spectrogram when playback is stopped.
        """
        # If audio is currently playing or visualization is running, do nothing
        if self.is_playing or (hasattr(self, "visual_running") and self.visual_running):
            return

        # Stop visualization if it is still running
        try:
            self.stop_visualization()
        except Exception:
            pass

        # Clear the axes and reset labels to indicate no current view
        self.ax_view.clear()
        self.ax_view.set_title("No current view", fontsize=10)
        self.ax_view.set_xlabel("Time (s)")
        self.ax_view.set_ylabel("Frequency (Hz)")

        # Reset the frequency text labels to default values
        try:
            self.lowerVar.set("Lower: –")
            self.centerVar.set("Center: –")
            self.upperVar.set("Upper: –")
        except Exception:
            pass

        self.fig_view.tight_layout()# Adjust layout
        self.canvas_view.draw_idle()# Redraw the canvas

  
   

    #Helpers
    def _freq_bounds(self, mag_db, freqs):
        """
        Create lower, center, and upper frequencies from the average spectrum.

        Lower corresponds to the 5%% spectral rolloff, upper to the 95%% rolloff,
        and center is the spectral centroid. Returns a tuple (lower, centroid, upper).
        """
        mag_lin = 10.0 ** (mag_db / 20.0)# Convert dB values to linear magnitude
        mean_spec = mag_lin.mean(axis=1)# Average the spectrum over time
        total = mean_spec.sum()  # Sum of magnitudes
        if total <= 0:
            return 0.0, 0.0, 0.0  # Return zeros if the spectrum is empty

        # Calculate the spectral centroid — this is the "center of mass" of the spectrum.
        # It represents where most of the energy is concentrated (a kind of average frequency).
        centroid = float((freqs * mean_spec).sum() / total)
        
        # Compute the cumulative distribution of the spectrum’s energy.
        # This CDF gradually goes from 0 to 1 as we move from low to high frequencies.
        # We'll use it to find where 5% and 95% of the total energy lie.
        cdf = np.cumsum(mean_spec) / total
        
        # Find the index where cumulative energy first reaches 5% (lower rolloff)
        # np.searchsorted() tells us at which position a value would fit in a sorted array,
        # so here it returns the index where our cumulative energy crosses 0.05.
        lower_idx = int(np.searchsorted(cdf, 0.05))
        
        # Same idea for 95% — this gives us the "upper" rolloff frequency,
        # the point where 95% of the energy is below this frequency.
        upper_idx = int(np.searchsorted(cdf, 0.95))
        
        # Make sure both indices are valid (inside array bounds),
        # then look up the actual frequency values for these indices.
        lower = float(freqs[min(max(lower_idx, 0), len(freqs) - 1)])
        upper = float(freqs[min(max(upper_idx, 0), len(freqs) - 1)])
        
        # Return the three key spectral statistics:
        # - lower: frequency at 5% cumulative energy
        # - centroid: weighted average frequency (center of mass)
        # - upper: frequency at 95% cumulative energy
        return lower, centroid, upper


    def _render_plot_and_labels(self, Sxx_db, f, window_time, lower, center, upper):
        """
        Update the spectrogram and the lower, center, and upper labels
        from the Tk main thread.
        """
        # Update the spectrogram plot with new data
        self.ax_view.clear()
        self.ax_view.imshow(
            Sxx_db,
            origin="lower",
            aspect="auto",
            extent=[0, window_time, f[0], f[-1]],
        )
        #Main labels
        self.ax_view.set_title("Live Spectrogram (Playback)")
        self.ax_view.set_xlabel("Time (s)")
        self.ax_view.set_ylabel("Frequency (Hz)")

        # Update the frequency labels using the provided values
        try:
            #Adding to mainlabels
            self.lowerVar.set(f"Lower: {lower:.1f} Hz")
            self.centerVar.set(f"Center: {center:.1f} Hz")
            self.upperVar.set(f"Upper: {upper:.1f} Hz")
        except Exception:
            pass  # Ignore errors when updating labels

        self.fig_view.tight_layout()  # Adjust the figure layout for better fit
        self.canvas_view.draw_idle()  # Request a redraw of the canvas

    def _update_visual(self, wav_path):
        """Live spectrogram updater.
        
        Runs in a background thread while pygame is playing the same file.
        We keep a rolling ~2s buffer of recent audio, compute a spectrogram
        for that window, convert to dB, estimate lower/center/upper
        frequency bounds, and then ask Tkinter (on the main thread) to
        refresh the plot + labels.
        """
        try:
            # Open the exact WAV file that the player is currently playing.
            # We read it ourselves in small chunks to draw a live view.
            with wave.open(wav_path, "rb") as wf:
                sr         = wf.getframerate()# Audio sample rate (e.g., 16000 Hz)
                n_channels = wf.getnchannels()# 1 = mono, 2 = stereo
                sampwidth  = wf.getsampwidth()# Bytes per sample (2 → int16)
    
                chunk       = 4096# How many samples to pull per read
                window_time = 2.0# Length (in seconds) of our rolling display window
                buffer_len  = int(sr * window_time)# Convert seconds → samples
                # This buffer always holds the *latest* audio we want to visualize.
                buffer=np.zeros(buffer_len, dtype=np.float32)
    
                # Map WAV sample width to a NumPy dtype. We stick to int16 here.
                dtype = np.int16 if sampwidth == 2 else np.int16
                # Scale factor to convert raw int16 to floating point in [-1, 1].
                scale = 32768.0
    
                # We must schedule UI updates on Tk's main thread; this gets us the Tk widget.
                tk_widget = self.canvas_view.get_tk_widget()
    
                """
                Keep updating as long as:
                - Our visualization flag is True (we haven't been asked to stop), AND
                - pygame is still playing the file.
                """
                
                while self.visual_running and pygame.mixer.music.get_busy():#Read some raw frames from the WAV file.
                    data = wf.readframes(chunk)
                    if not data:# We've reached the end of the file; stop updating the view.
                        break
    
                    # Convert raw WAV byte data into a floating-point waveform:
                    #   1️ Interpret the bytes as 16-bit signed integers (PCM samples)
                    #   2️ Convert them to float32 for math/plotting
                    #   3️ Divide by 32768.0 to normalize the amplitude to the range [-1.0, 1.]
                    y = np.frombuffer(data, dtype=dtype).astype(np.float32) / scale # This keeps the waveform compatible with NumPy operations and ensures consistent audio levels for further analysis or spectrogram generation.
    
                    # Making sure all is Mono
                    if n_channels == 2:
                        y = y.reshape(-1, 2).mean(axis=1)
    
                    
                    #Keep a rolling 2s window of chosen audio
                    n = len(y)
                    if n > 0:
                        if n >= buffer_len:
                            # If this new chunk is *longer* than our rolling 2-second window,
                            # we cant keep everything — so we only take the *latest* samples.
                            # This ensures the buffer always represents the most recent sound.
                            #Messed this up earlier so it looped
                            buffer[:] = y[-buffer_len:]
                        else:
                            # Otherwise “scroll left” and append the new samples at the end.
                            buffer = np.roll(buffer, -n)
                            buffer[-n:] = y
    
    
                # Now compute a spectrogram of the current rolling 2-second buffer.
                # The spectrogram shows how the frequency content of the sound changes over time.
                # - nperseg: size of each FFT window
                # - noverlap: how much neighboring windows overlap
                # Larger overlap = smoother animation, but more computation.
                    f, t, Sxx = spectrogram(
                        buffer,
                        fs=sr,
                        nperseg=1024,
                        noverlap=512,
                        scaling="density",
                        mode="magnitude",
                    )
    
                    # Convert magnitude values (0 → ∞) to decibels (a log scale in dB).
                    # The +1e-10 prevents log(0), which would produce -inf.
                    Sxx_db = 20 * np.log10(Sxx + 1e-10)
    
                    # Estimate three descriptive frequencies from this spectrum:
                    # - lower:  the 5% spectral rolloff (where the low end contains 5% of total energy)
                    # - center: the spectral centroid (the “balance point” of the frequencies)
                    # - upper:  the 95% spectral rolloff (where the high end contains 5% of total energy)
                    # These give an intuitive sense of the sound’s pitch and range over time and is for the most just aestethic, but informative.
                    lower, center, upper = self._freq_bounds(Sxx_db, f)
    
                        #Ask Tkinter to redraw the plot on the *main thread* (UI thread).
                        # We can’t update Tk widgets directly from a background thread, so we schedule
                        # the update with .after(). The lambda captures the current data snapshot
                        # to avoid late-binding issues where variables might change before drawing.
                   
                    tk_widget.after(
                        0,
                        lambda Sxx_db=Sxx_db, f=f, wt=window_time,
                               lo=lower, ce=center, up=upper:
                            self._render_plot_and_labels(Sxx_db, f, wt, lo, ce, up)
                    )
    
                    # A tiny nap keeps CPU use reasonable and targets ~10 fps.
                    time.sleep(0.01) #This keeps the visualization smoother (play around with sleeptime to see)
    
        except Exception as e:
            # Any unexpected issues get printed to console.
            print("Visualization error:", e)
        finally:
            # Whether we exit by finishing or error, mark the thread as no longer running.
            self.visual_running = False
