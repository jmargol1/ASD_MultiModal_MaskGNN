import numpy as np
import pandas as pd
from nilearn import datasets, image
import nibabel as nib

# Load AAL atlas
aal_atlas = datasets.fetch_atlas_aal()
atlas_img = nib.load(aal_atlas.maps)
atlas_data = atlas_img.get_fdata()
aal_labels = aal_atlas.labels

# Check sizes
print(f"Atlas has {len(aal_labels)} labels")
print(f"Max voxel value in atlas: {atlas_data.max()}")

# Find unique region labels
region_labels = np.unique(atlas_data)
print("Unique region values:", region_labels)

# Map label numbers to names
aal_region_names = []
for label_value in region_labels:
    if label_value == 0:
        continue  # skip background
    if 0 < label_value <= len(aal_labels):
        region_name = aal_labels[int(label_value) - 1]
    else:
        region_name = 'Unknown'
    aal_region_names.append(region_name)

print(f"✅ Found {len(aal_region_names)} region names")

# Save to CSV
df = pd.DataFrame({
    'region_id': region_labels[region_labels > 0],
    'region_name': aal_region_names
})

df.to_csv("AAL_regions.csv", index=False)
print("✅ AAL regions saved to AAL_regions.csv")
