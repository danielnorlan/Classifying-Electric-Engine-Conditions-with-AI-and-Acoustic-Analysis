# main.py
import tkinter as tk
try:
    from tkinterdnd2 import TkinterDnD  # <- prøv DnD-root
    root = TkinterDnD.Tk()
except Exception:
    root = tk.Tk()  # fall-back uten DnD

from GUI import GUI
app = GUI()
app.initialize(root)
root.mainloop()

