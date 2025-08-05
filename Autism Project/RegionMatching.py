import pandas as pd
import numpy as np
from nilearn import datasets
import nibabel as nib
import nilearn.image as image

# ----------------------------------------------
# STEP 1 — Load your Power coordinates and AAL labels
# ----------------------------------------------

# Load Power coordinates
power_df = pd.read_csv("Power_coords.csv")
print(f"✅ Loaded Power atlas: {len(power_df)} regions")

# Load AAL labels
aal_labels_df = pd.read_csv("AAL_regions.csv")
print(f"✅ Loaded AAL region list: {len(aal_labels_df)} regions")

# Fetch the AAL atlas image
aal_atlas = datasets.fetch_atlas_aal()
aal_img = nib.load(aal_atlas.maps)
aal_data = aal_img.get_fdata()

# Check unique values
region_values = np.unique(aal_data)
print(f"✅ AAL atlas unique region values (excluding background): {region_values[region_values > 0]}")

# ----------------------------------------------
# STEP 2 — Compute AAL region centroids
# ----------------------------------------------

centroid_records = []

for region_value in region_values:
    if region_value == 0:
        continue

    region_mask = aal_data == region_value
    coords = np.argwhere(region_mask)

    if len(coords) == 0:
        continue

    centroid_voxel = np.mean(coords, axis=0)

    # Convert voxel → MNI mm coordinates
    centroid_mni = nib.affines.apply_affine(aal_img.affine, centroid_voxel)

    region_index = int(region_value) - 1
    if region_index < len(aal_labels_df):
        region_name = aal_labels_df.iloc[region_index]["RegionName"]
    else:
        region_name = f"Region_{region_value}"

    centroid_records.append({
        "region_value": region_value,
        "region_name": region_name,
        "x": centroid_mni[0],
        "y": centroid_mni[1],
        "z": centroid_mni[2]
    })

aal_centroids_df = pd.DataFrame(centroid_records)
aal_centroids_df.to_csv("AAL_centroids.csv", index=False)
print(f"✅ Saved AAL centroids: {len(aal_centroids_df)} regions → AAL_centroids.csv")

# ----------------------------------------------
# STEP 3 — Find nearest AAL region for each Power ROI
# ----------------------------------------------

matches = []

for i, row in power_df.iterrows():
    roi_id = row["roi"]
    roi_coord = np.array([row["x"], row["y"], row["z"]])

    # Compute distances to all AAL centroids
    distances = np.linalg.norm(aal_centroids_df[["x", "y", "z"]].values - roi_coord, axis=1)
    min_idx = np.argmin(distances)

    matched_region = aal_centroids_df.iloc[min_idx]

    matches.append({
        "power_roi": int(roi_id),
        "power_x": roi_coord[0],
        "power_y": roi_coord[1],
        "power_z": roi_coord[2],
        "matched_region_value": matched_region["region_value"],
        "matched_region_name": matched_region["region_name"],
        "matched_region_x": matched_region["x"],
        "matched_region_y": matched_region["y"],
        "matched_region_z": matched_region["z"],
        "distance_mm": distances[min_idx]
    })

matches_df = pd.DataFrame(matches)
matches_df.to_csv("Matched_AAL_Power_Regions.csv", index=False)

print(f"✅ Saved {len(matches_df)} matched regions → Matched_AAL_Power_Regions.csv")

