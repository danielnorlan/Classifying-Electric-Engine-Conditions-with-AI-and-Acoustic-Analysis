"""
Author: Hans Josef Rosland-Borlaug [258139]
"""
from pathlib import Path  # Standard library: path handling

class Config:
    """
    handles all important directories relative to *this* file, which lives in
    the 'Project mastermap' folder. Share the whole folder and paths will
    still work on any machine.
    """
    
    
    def __init__(self):
        """ 
        Creating the correct paths, for projectroot, Datasetroot, and folder navigation.
        """
        # PROJECTROOTS
        self.project_root = Path(__file__).resolve().parent
        # Finds the folder where (configs.py) is located. That folder is now our main "Project mastermap". 
        # All other paths are "subpaths" inside this location. Makes it easier for me to make sure that all systems are unified under same directory/path. 

        # DATASET ROOTS
        self.dataset_root = self.project_root / "IDMT-ISA-ELECTRIC-ENGINE"
        # Dataset provided for this project is located inside the project_root. 
        # This way it makes sure that this path can be used anywhere in the project and I only need to tweak the config settings on the other models. 

        # Datasets
        # These subfolders holds the different datasets provided for the project. Both for modeltraining and modeltesting. Creating getter-functions for these later for easy accsess. 
        self.test_root   = self.dataset_root / "test"
        self.test_cut    = self.dataset_root / "test_cut"
        self.train_root  = self.dataset_root / "train"
        self.train_cut   = self.dataset_root / "train_cut"

        # SUBFOLDERS
        self.folders = ["engine1_good", "engine2_broken", "engine3_heavyload"]
        # The three folders represent the different motor conditions. by creating a unified folderpath, I get easy accsess to the folders inside the different paths above.

        # FILETYPE
        self.exts = (".wav", ".WAV")
        # Accepted filetypes for this project. 
        # Making sure to hedge myself because I used some time debugging (because I used lowecase and had some problems with this),

        # SAVE PREPROCESSED (ARTIFACTS)
        self.preproc_root = self.project_root / "preprocessed"
        # Where preprocessor will store PNG tiles etc.

        # META DIR (for misc files like a manifest)
        self.meta_dir = self.project_root / "_meta"
        # Creating a dedicated metadata folder so get_manifest_path() has a defined location.

    # GETTERS
    """ 
    Simple getter-functions for fetching the correct paths needed
    Note to self: Testing these in the Function validation tab.
    """
    def get_project_root(self):   return self.project_root
    def get_dataset_root(self):   return self.dataset_root
    def get_test_root(self):      return self.test_root
    def get_test_cut(self):       return self.test_cut
    def get_train_root(self):     return self.train_root
    def get_train_cut(self):      return self.train_cut
    def get_folders(self):        return self.folders
    def get_exts(self):           return self.exts
    def get_preproc_root(self):   return self.preproc_root

    def get_manifest_path(self):
        """
        Path to a manifest CSV (if you want to log preprocessing or training runs).
        Ensures meta folder exists when asked for the path.
        """
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        return self.meta_dir / "manifest.csv"
   
######################################################################################################


#NEW GETTERS
"""
Adding new configs to the heartbeat/PTE part of the project 

"""















    