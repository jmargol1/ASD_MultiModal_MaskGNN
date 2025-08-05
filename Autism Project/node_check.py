import numpy as np
from pathlib import Path

# CHANGE THIS to match your real output path:
processed_dir = Path("/Users/joemargolis/Desktop/Autism Project/processed_features")

# find all subject folders
subject_folders = [f for f in processed_dir.iterdir() if f.is_dir() and f.name.isdigit()]

if not subject_folders:
    print("⚠️ No subject folders found. Double-check your processed_dir path!")
else:
    for folder in subject_folders:
        subject_id = folder.name
        struct_path = folder / 'structural_features.npy'
        
        if struct_path.exists():
            struct_data = np.load(struct_path, allow_pickle=True).item()
            num_regions = len(struct_data.keys())
            regions = sorted(struct_data.keys())
            
            print(f"✅ Subject {subject_id} has {num_regions} regions: {regions[:5]} ... {regions[-5:]}")
        else:
            print(f"⚠️ Subject {subject_id} has no structural features file yet.")
