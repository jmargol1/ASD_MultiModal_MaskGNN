import os
import pandas as pd

# ---------------------------------------------
# CONFIGURATION
# ---------------------------------------------

# Path to the freesurfer folder
FREE_PATH = os.path.join(os.path.dirname(__file__), "freesurfer")

# Files to load
files = [
    "aseg_table.tsv",
    "lh.aparc_table_thickness.tsv",
    "rh.aparc_table_thickness.tsv",
    "lh.aparc_table_area.tsv",
    "rh.aparc_table_area.tsv",
    "lh.aparc_table_volume.tsv",
    "rh.aparc_table_volume.tsv"
]

dfs = []

# ---------------------------------------------
# LOAD EACH TSV FILE
# ---------------------------------------------

for f in files:
    path = os.path.join(FREE_PATH, f)
    if os.path.exists(path):
        print(f"Loading {f}...")
        df = pd.read_csv(path, sep="\t")

        # Always rename first column to 'subject'
        first_col = df.columns[0]
        if first_col != "subject":
            df = df.rename(columns={first_col: "subject"})
        
        dfs.append(df)
    else:
        print(f"File {f} not found, skipping...")

# ---------------------------------------------
# MERGE ALL DATAFRAMES
# ---------------------------------------------

if len(dfs) > 0:
    df_merged = dfs[0]
    for df in dfs[1:]:
        df_merged = df_merged.merge(df, on="subject", how="outer")
else:
    print("No tables loaded.")
    exit()

print(f"Shape after merging TSVs: {df_merged.shape}")

# ---------------------------------------------
# LOAD brainstats_abide_with_pial.txt
# ---------------------------------------------

brainvol_path = os.path.join(FREE_PATH, "brainstats_abide_with_pial.txt")

if os.path.exists(brainvol_path):
    print("Loading brainstats_abide_with_pial.txt...")
    # Try reading as tab-separated first
    df_brainvol = pd.read_csv(brainvol_path, sep="\t")

    if df_brainvol.shape[1] == 1:
        # If that fails (only one column), try whitespace
        df_brainvol = pd.read_csv(brainvol_path, delim_whitespace=True)

    # Rename first column if needed
    first_col = df_brainvol.columns[0]
    if first_col != "subject":
        df_brainvol = df_brainvol.rename(columns={first_col: "subject"})

    # Merge brainvol into merged dataframe
    df_merged = df_merged.merge(df_brainvol, on="subject", how="outer")

    print(f"Shape after merging brainvol: {df_merged.shape}")
else:
    print("brainstats_abide_with_pial.txt not found. Skipping brainvol merge.")

# ---------------------------------------------
# SAVE FINAL MERGED CSV
# ---------------------------------------------

outpath = os.path.join(FREE_PATH, "merged_freesurfer.csv")
df_merged.to_csv(outpath, index=False)
print(f"✅ Saved merged Freesurfer table to: {outpath}")

