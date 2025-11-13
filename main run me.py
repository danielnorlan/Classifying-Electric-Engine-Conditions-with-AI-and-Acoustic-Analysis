"""
Author: Hans Josef Rosland-Borlaug[258139]
startdate: 19.09.25
Last edit (date): 
"""
import tkinter as tk
from GUI import GUI


root = tk.Tk()
app = GUI()
app.initialize(root)
root.mainloop()
