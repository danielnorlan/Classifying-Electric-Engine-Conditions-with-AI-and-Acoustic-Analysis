"""

Author: Hans Josef Rosland-Borlaug[258139]
startdate: 19.09.25
Last edit (date): 
"""



#FALLBACK LIBRARIES AND IMPORT

import matplotlib.pyplot as plt 
import numpy as np 
import tkinter as tk 
import wave
import io
import time

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression  
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler 
from configs import Config 

"""
VALIDATION PAGE:
    Reflection:
        I want to create this page for "validating" different functions in the program. Because my build is starting to get bigger and bigger, its important to test functions as we go. 
        I might create a tab that monitors these conditions, but only for fun. 
        Build:
"""            


#Libraries
"""
Library validation
"""

def validate_libraries():
    """
    Originally created because "torch" and "pytorch" created problems when operating alongside certain mathplot versions.(Not sure why). 
    Library system overview is created to make sure all libraries are properly installed (pip).
    I think this is a clever move due to several people working on these projects are using different conda base versions, and even different software. 
    By creating this, everyone can make sure all libraries are installed. 
    
    Last edit: Only me worked on this project, but I still think it was a clever feature.
    """
    required = [
        "FastAPI", #Putting in a "IMPOSTERs" to make sure its properly working
        "numpy",
        "matplotlib",
        "sklearn",
        "PIL",          
        "tkinter",
        "TKinter",#IMPOSTER similar
        "wave",
        "joblib",
        "pathlib",
        "torch",# Mainly from the bubblefish
        "sys",
        "subprocess",
        "threading",
        "queue",
        "os",
        "typing",
        "tqdm",#From MLP module
        "thinterdnd2", #For drag and drop (NOT IMPLEMENTED)
        "physionet",
        "vlc",
        "Pillow",
        "pytorch",#IMPOSTER
    ]

    print("LIBRARY VALIDATION")  # Indicate the start of the library validation phase
    
    for lib in required:
        try:
            __import__(lib)
            print(f"{lib:<15} OK") #:<15 creates a "unified spacing, looks prettier when list is longer"
            
        except ImportError:
            # Print a message if the library could not be imported because it is missing
            print(f"{lib} MISSING")
            
        except Exception:
            # Print a generic error if another exception occurred during import
            print(f"{lib} ERROR")
    print("END----------------------------")  # Indicate the end of the library validation output



validate_libraries()

#%%
#Configs

from pathlib import Path
from configs import Config  # Checking roots


def validate_config():
    """
    The function "validate_config()" creates an object from Config,
    defines a list of getter method names, and uses getattr() (built in function) to retrieve
    them. It checks if each returned path exists
    and prints OK or ERROR for each.
    """
    cfg = Config()  # Create Config object
    getters = [
        "get_project_root",
        "get_dataset_root",
        "get_train_cut",
        "get_test_cut",
        "get_preproc_root",
        "get_false_root", #Imposter root for control of function
    ]

    print("CONFIG VALIDATION ")
    for name in getters:
        try:
            fitems = getattr(cfg, name)# Retrieve method dynamically
            val = fitems()# Call the method

            try:
                # Try converting val to a Path — works if val is str or Path-like
                path = Path(val)
                if path.exists():
                    print(f"{name:<20} OK") #Creating "unified spacing again"
                else:
                    print(f"{name} ERROR")
            except TypeError:
                print(f"{name} ERROR")#Handling Typ-errors if val is not a valid path or string. 

        except Exception: #Handling Exception error
            print(f"{name} ERROR")
            
    print("END-----------------------")

validate_config()


#%% CONTROLLING ARTIFACTS/ITEMS IN PREPROCESSED
from pathlib import Path
from configs import Config

def count_preprocessed_items():
    """
    Goes through the 'preprocessed' folder and counts how many files are
    inside each subfolder. Prints the results and returns a dictionary.
    """
    cfg = Config() #config
    pre = cfg.get_preproc_root()
    
    
    print(f"Path: {pre}") #The actual filepath (Where the folder is placed by user(locally/in system)) (Using this to check items later)
    print(f"Preprocessed folder control:")

    if not pre.exists(): #Using Purepath subclass and function "exists" to validate if the path is there at all.
        print("Cannot find folder (Preprocessed")
        return {}

    results = {}#Will hold each subfolder name and its file count
    total = 0 #Count variable for total items
    
    for sub in sorted(pre.iterdir()):# pre.iterdir() lists all items (files and folders) directly inside pre, sorting them alphabetically.
        if sub.is_dir(): #Checks if the current item is a directory (not a file)
            n = len(list(sub.glob("*.*")))#using sub.glob to find all files of any type inside folder, n = numbers of items len counts items in folder and list turns them into a list. 
            results[sub.name] = n #Count for number of items in path.
            total += n #Total count
            print(f"{sub.name:<20} {n} files") #Unified spacing displaying files found in subfolders.

    print(f"Total: {total} files\n") #Total files found in path (allsubfolders)
    print("END----------------------------------------------")
    return results


count_preprocessed_items()


#%%VALIDATE BIT RATE from "test" folder
import wave
from pathlib import Path


#Hardcoding path for easy testing purposes (And I dont need subfolders, or check any other. i can just replace the name)
path = Path(r"D:\Artificial intelligence\Group project\AI Group Project alpha ver\Project mastermap\IDMT-ISA-ELECTRIC-ENGINE\test\engine3_heavyload\stresstest.wav")

#Testing with 24 bites per sample Wav-file
# path = Path(r"D:\Artificial intelligence\Group project\AI Group Project alpha ver\Project mastermap\IDMT-ISA-ELECTRIC-ENGINE\test\engine3_heavyload\talking_2.wav")

def check_bitrate():
    """
    Simple function to check what type of Wav file (path) is. 
    Returns 24 bits per sample if it is 24 bits. 
    Returns error 3 if it is 32-bit float.
    """
    if not path.exists():
        print(f"Error file not found {path}")
        return

    try:
        with wave.open(str(path),"rb") as wf:
            sampwidth = wf.getsampwidth()
            bit_depth = sampwidth * 8
            print(f"\nFile: {path.name}")
            print(f"Sample width: {sampwidth} bytes")
            print(f"Bit depth:    {bit_depth} bits per sample")
            
    except wave.Error as e:
        print(f"[WAVE ERROR] {e}")
        print("END---------------------------------------")
    except Exception as e:
        print(f"[Unexpected error] {e}")
        print("END---------------------------------------")


        

check_bitrate()

#%%PhysioNet - NOT WORKING

import physionet as pn
"""
Checkint the PhysioNet and what data it contains
Got this from the pypi.org
not working
"""
# Download a dataset
pn.download('ptbdb', 'data/ptbdb')

# List all datasets
pn.list_datasets()

print("END------------------------------------")

#%%

