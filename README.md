# ASD_MultiModal_MaskGNN

## Install Dependencies

All dependencies should be stored in requirements.txt file. run `pip install -r requirements.txt` to install dependencies properly

## sMRI + fMRI models (ABIDE I data)

Files related to this portion are in ASD2 Folder

To run the fMRI models open Jupyter Notebook `fMRI_MaskGNN.ipynb` and run all --> approximately 5 hours to run in full

To run the sMRI models open Jupyter Notebook `sMRI_MaskGNN.ipynb` and run all --> Approximately 1 hour to run in full

To run the Fusion Model with same structure as fMRI and sMRI open Jupyter Notebook `FusionMaskGNN.ipynb` and run all --> Approximately 2 hours to run in full

To run the Fusion Model with tuned architecture open Jupyter Notebook `FusionMaskGNNTuned.ipynb` and run all --> Approximately 2 hours to run in full

## sMRI + fMRI + DTI (ABIDE II Data)

As the data is too large to add to github, use the link https://fcon_1000.projects.nitrc.org/indi/abide/abide_II.html do download data. Do so through the following sites

  - NYU1
  - NYU2
  - Barrow Neurological Institute
  - SDSU
  - Trinity Centre

Run ScanPreprocessing.py to preprocess the data --> About 8 hours to run
Run GNNBuilding.py to run the models --> About 2 hours to run




