import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets, image

class AtlasCheckerGrid:
    def __init__(self):
        self.aal_atlas = datasets.fetch_atlas_aal(version='SPM12')
        self.aal_atlas_img = nib.load(self.aal_atlas.maps)
        self.mni_template = datasets.load_mni152_template()
        self.atlas_cache = {}
        self.modalities = {
            "anat_1": "anat.nii.gz",
            "dti_1": "dti.nii.gz",
            "rest_1": "rest.nii.gz"
        }

    def get_overlay_slice(self, img_path):
        img = nib.load(img_path)
        if img.ndim == 4:
            img = image.index_img(img, 0)

        img_resampled = image.resample_to_img(img, self.mni_template, interpolation='continuous')
        atlas_resampled = image.resample_to_img(self.aal_atlas_img, img_resampled, interpolation='nearest')

        img_data = img_resampled.get_fdata()
        atlas_data = atlas_resampled.get_fdata().astype(np.int16)

        z_mid = img_data.shape[2] // 2
        return img_data[:, :, z_mid].T, atlas_data[:, :, z_mid].T

    def run_batch_grid(self, root_dir="Brain Scans", save_path="overlay_grid.png"):
        site_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        selected_data = []

        for site in site_dirs:
            site_path = os.path.join(root_dir, site)
            checked = {}
            for subject in os.listdir(site_path):
                subj_path = os.path.join(site_path, subject, "session_1")
                if not os.path.isdir(subj_path):
                    continue
                for modality, filename in self.modalities.items():
                    if modality in checked:
                        continue
                    mod_path = os.path.join(subj_path, modality)
                    img_path = os.path.join(mod_path, filename)
                    if os.path.exists(img_path):
                        checked[modality] = img_path
                    if len(checked) == len(self.modalities):
                        break
                if len(checked) == len(self.modalities):
                    break
            if len(checked) == len(self.modalities):
                selected_data.append((site, checked))

        # Plotting
        n_rows = len(selected_data)
        n_cols = len(self.modalities)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

        if n_rows == 1:
            axes = np.expand_dims(axes, 0)

        for row_idx, (site, modality_paths) in enumerate(selected_data):
            for col_idx, (modality, filename) in enumerate(self.modalities.items()):
                ax = axes[row_idx, col_idx]
                anat_slice, atlas_slice = self.get_overlay_slice(modality_paths[modality])
                ax.imshow(anat_slice, cmap='gray', origin='lower')
                ax.imshow(atlas_slice, cmap='tab20', origin='lower', alpha=0.5)
                ax.axis('off')
                if row_idx == 0:
                    ax.set_title(modality.replace('_1', '').upper(), fontsize=14)
                if col_idx == 0:
                    ax.text(-0.1, 0.5, site, transform=ax.transAxes,
                            fontsize=12, va='center', ha='right', rotation=90)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved grid image to {save_path}")

# ---------- RUN ----------
if __name__ == "__main__":
    checker = AtlasCheckerGrid()
    checker.run_batch_grid()

