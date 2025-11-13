# Classifying-Electric-Engine-Conditions-with-AI-and-Acoustic-Analysis
Student project for AI &amp; ML. Classifies electric engine sounds as good, broken or heavy load using Python. Pipeline with audio preprocessing, windowing, feature engineering (mel features, MFCC, spectral stats), PCA and an MLP classifier, plus a simple GUI to explore data and results.


# Engine Condition Classifier GUI

**Author:** Daniel Norlan, Hans Josef Rosland-Borlaug 
**Start date:** 19.09.2025  
**Last edit:** 05.11.2025  

This repository contains the code for a desktop GUI that ties together several
machine learning pipelines for classifying electric engine states and experimenting
with heartbeat data. The focus is on integrating different models into a single
interface with a shared configuration system.

> **Note:** The original audio datasets are **not** included in this repository due
> to size and licensing. This repo contains the full codebase and configuration
> logic only.

---

## How to run

1. Make sure you have Python installed (3.x).
2. Install the required libraries  
   Check `Function_validation.py` for the full list of dependencies and simple checks.
3. To start the application, run:

   ```bash
   python main.py
For the heartbeat animation tab you also need VLC installed on your system,
as the GUI can use VLC to play video or animation content.

Main modules
main.py
Entry point that boots the GUI and wires up the different pages.

GUI.py / Gui_Helpers.py
Core window layout and helper functions for buttons, callbacks, navigation,
and status messages.

configs.py
Central config module. Holds paths, parameters and helper functions used to
keep the different models on a common root and directory structure.

Preprocessor.py
Tools for converting raw WAV files into mel-style or “Tiny” spectrogram tiles
for image-based pipelines.

ImageModel.py
Lightweight image-based model that operates on spectrogram tiles.

player.py
Audio player tab with rolling spectrogram view for exploring engine sounds.

heartPTE.py
Experimental heartbeat tab aimed at connecting the project to a medical /
PTE angle.

Documentation and comments are spread across the individual modules.

Reflection
This project is a mix of several people’s code, wrapped in one interface.
My main focus was:

building a GUI that can host multiple ML pipelines,

designing a common config layer, and

making it possible to plug in models that were not originally written
for a shared framework.

That turned out to be harder than expected. Classes and functions were moved
between modules several times, and the structure still reflects that history.
The separation between “base GUI” and “GUI helpers” works, but it is not as
clean or discoverable as I would like for other developers. This is a clear
lesson in designing for maintainability and readability, not just “making it
work”.

I also had to adapt external model code that assumed local hard-coded paths.
To integrate them, I introduced a central configs.py module and gradually
replaced local paths with calls into the config. This took time, but it made
it possible to run all pipelines from one GUI without manually changing paths
in several files.

On the positive side, I learned a lot about:

structuring a multi-module Python project,

using wrappers and config getters to hide path logic,

thinking about users other than myself when designing a GUI,

and debugging the interactions between independent ML pipelines and a shared
interface.

Known limitations and ideas for future work
Some helper functions, like parseMelParams, still exist in more than one
place. Ideally they should all import from configs.py instead of having
local copies.

The drag-and-drop feature in the ImageModel tab exists in code, but is not
fully wired up and tested across different systems.

The MEL/TINY pipeline selection could be exposed as a simple toggle in the
GUI, to make it easier to compare different preprocessing strategies.

The Heartbeat tab is more of a demo right now. A natural next step would be
to connect it to a proper PhysioNet-style dataset and build a small end-to-end
classification pipeline with visual feedback.

A “system status” view and better validation hooks could make it easier for
users to see if all models, paths and dependencies are correctly set before
they start running experiments.

Despite the rough edges, this prototype works as a playground for engine-sound
and heartbeat models and as a practical exercise in gluing together multiple
AI pipelines inside one GUI application.
