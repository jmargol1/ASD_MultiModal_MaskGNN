from nilearn import datasets

# Fetch Power atlas
power = datasets.fetch_coords_power_2011()

# Get the DataFrame of ROI coordinates
coords_df = power["rois"]

# Save to CSV
coords_df.to_csv("Power_coords.csv", index=False)
print("✅ Power coordinates saved to Power_coords.csv")
