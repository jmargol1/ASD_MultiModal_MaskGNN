import os
import pandas as pd
import urllib.request

# -------------------------------------
# CONFIGURATION
# -------------------------------------

# Path to merged Freesurfer CSV
FREE_CSV = './freesurfer/merged_freesurfer.csv'

# Output folder for ROI time series
OUT_DIR = './fmri_roits_desikan/'

# S3 URL prefix for Desikan-Killiany atlas ROI time series
S3_PREFIX = 'https://fcp-indi.s3.amazonaws.com/data/Projects/ABIDE_Initiative/Outputs/cpac/nofilt_noglobal/rois_dk/'

# -------------------------------------
# LOAD SUBJECT LIST
# -------------------------------------

print("Loading Freesurfer merged CSV...")
df = pd.read_csv(FREE_CSV)

subjects = df['subject'].dropna().unique().tolist()
print(f"Subjects in Freesurfer merged CSV: {len(subjects)}")

# -------------------------------------
# CREATE OUTPUT DIRECTORY
# -------------------------------------

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# -------------------------------------
# DOWNLOAD FILES
# -------------------------------------

downloaded = 0
skipped = 0
missing = 0

for sub in subjects:
    filename = f"{sub}_rois_dk.1D"
    s3_url = S3_PREFIX + filename
    local_path = os.path.join(OUT_DIR, filename)

    if os.path.exists(local_path):
        print(f"✓ Already exists: {filename}")
        skipped += 1
        continue

    try:
        print(f"→ Downloading {filename} ...")
        urllib.request.urlretrieve(s3_url, local_path)
        downloaded += 1
    except Exception as e:
        print(f"!! Error downloading {filename}: {e}")
        missing += 1

print("\nDone!")
print(f"✅ Downloaded: {downloaded}")
print(f"✅ Skipped (already present): {skipped}")
print(f"❌ Missing / errors: {missing}")

