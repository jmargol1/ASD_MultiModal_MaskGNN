#!/usr/bin/env python3
"""
ABIDE Dataset Optimized Preprocessing Pipeline - COMPLETE FIXED VERSION
Handles MPRAGE, T1w, and FLAIR sequences across different sites with speed optimizations
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import warnings
import psutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

warnings.filterwarnings('ignore')

# Neuroimaging libraries
from nilearn import datasets, connectome, input_data, image
from nilearn.plotting import plot_connectome, plot_glass_brain
import dipy
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel
from dipy.tracking.utils import seeds_from_mask
from dipy.tracking import utils
from dipy.segment.mask import median_otsu

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ABIDEOptimizedPreprocessor:
    """
    Optimized preprocessing pipeline for ABIDE dataset
    Handles MPRAGE, T1w, and FLAIR sequences with site-specific adaptations
    """
    
    def __init__(self, output_dir: str = "./processed_features"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # ✅ Load Power → AAL matched regions
        try:
            matched_regions_df = pd.read_csv("Matched_AAL_Power_Regions.csv")
            self.regions_to_keep = (
                matched_regions_df["matched_region_value"]
                .dropna()
                .unique()
                .astype(int)
                .tolist()
            )
            logger.info(f"✅ Keeping {len(self.regions_to_keep)} AAL regions matching Power ROIs")
        except Exception as e:
            logger.error(f"❌ Failed to load Matched_AAL_Power_Regions.csv: {e}")
            self.regions_to_keep = []

        # ✅ Site-specific information
        self.site_info = {
            'BarrowNeurologicalScans': {
                'primary_contrast': 'MPRAGE',
                'secondary_contrast': 'FLAIR',
                'scanner': 'Siemens/GE',
                'preprocessing_params': {
                    'mprage_fwhm': 3.0,
                    'flair_fwhm': 4.0,
                    'intensity_percentile': 2
                }
            },
            'NYU1Scans': {
                'primary_contrast': 'MPRAGE',
                'scanner': 'Siemens',
                'preprocessing_params': {
                    'mprage_fwhm': 3.0,
                    'intensity_percentile': 2
                }
            },
            'NYU2Scans': {
                'primary_contrast': 'MPRAGE',
                'scanner': 'Siemens',
                'preprocessing_params': {
                    'mprage_fwhm': 3.0,
                    'intensity_percentile': 2
                }
            },
            'SDSUScans': {
                'primary_contrast': 'T1w',
                'scanner': 'GE',
                'preprocessing_params': {
                    'mprage_fwhm': 4.0,
                    'intensity_percentile': 5
                }
            },
            'TrinityCentreScans': {
                'primary_contrast': 'MPRAGE',
                'scanner': 'Siemens',
                'preprocessing_params': {
                    'mprage_fwhm': 3.0,
                    'intensity_percentile': 2
                }
            }
        }
        
        # Load and cache atlases for speed
        self.load_and_cache_atlases_optimized()
        
        # Initialize feature storage
        self.structural_features = {}
        self.functional_features = {}
        self.dti_features = {}
        
        # Initialize processing tracking
        self.processing_log = {
            'atlas_usage': {},
            'contrast_type': {},
            'site_info': {},
            'processing_status': {},
            'processing_times': {},
            'error_log': {}
        }

        
    def load_and_cache_atlases_optimized(self):
        """Load standard brain atlases and cache for speed"""
        logger.info("🚀 Loading and caching brain atlases for speed...")
        
        # Load AAL atlas
        try:
            self.aal_atlas = datasets.fetch_atlas_aal(version='SPM12')
            if hasattr(self.aal_atlas, 'maps'):
                if isinstance(self.aal_atlas.maps, str):
                    self.aal_atlas_img = nib.load(self.aal_atlas.maps)
                else:
                    self.aal_atlas_img = self.aal_atlas.maps
            else:
                raise ValueError("AAL atlas maps not found")
            logger.info("✅ AAL atlas loaded: 90 regions")
        except Exception as e:
            logger.warning(f"Could not load AAL atlas: {e}")
            # Fallback to Harvard-Oxford
            try:
                self.aal_atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
                if hasattr(self.aal_atlas, 'maps'):
                    if isinstance(self.aal_atlas.maps, str):
                        self.aal_atlas_img = nib.load(self.aal_atlas.maps)
                    else:
                        self.aal_atlas_img = self.aal_atlas.maps
                else:
                    raise ValueError("Harvard-Oxford atlas maps not found")
                logger.info("✅ Harvard-Oxford atlas loaded as fallback")
            except Exception as e2:
                logger.error(f"Could not load any atlas: {e2}")
                self.aal_atlas = None
                self.aal_atlas_img = None
        
        # Pre-compute atlas data for common resolutions (SPEED OPTIMIZATION)
        self.atlas_cache = {}
        if self.aal_atlas_img is not None:
            common_shapes = [
                (193, 256, 256),  # MPRAGE common
                (64, 64, 50),     # fMRI common
                (192, 192, 48),   # DTI common
                (91, 109, 91)     # Standard MNI
            ]
            
            for shape in common_shapes:
                try:
                    dummy_affine = np.eye(4)
                    dummy_img = nib.Nifti1Image(np.zeros(shape), dummy_affine)
                    
                    resampled_atlas = image.resample_to_img(
                        self.aal_atlas_img, dummy_img, 
                        interpolation='nearest', copy=False
                    )
                    
                    self.atlas_cache[shape] = resampled_atlas.get_fdata().astype(np.int16)
                    logger.info(f"   ✅ Cached atlas for shape {shape}")
                except:
                    continue
            
            logger.info(f"🚀 Speed optimization: {len(self.atlas_cache)} atlas resolutions cached")
        else:
            self.atlas_cache = {}
        
        # Load coordinates for plotting/analysis
        try:
            self.coordinates = datasets.fetch_coords_power_2011()
            logger.info("✅ Power coordinates loaded for connectivity analysis")
        except:
            logger.warning("Could not load Power coordinates")
    
    def get_cached_atlas_optimized(self, target_shape: tuple) -> np.ndarray:
        """Get pre-cached atlas or resample on demand (MAJOR speed improvement)"""
        if hasattr(self, 'atlas_cache') and target_shape in self.atlas_cache:
            return self.atlas_cache[target_shape]
        
        # Fallback to resampling
        dummy_affine = np.eye(4)
        dummy_img = nib.Nifti1Image(np.zeros(target_shape), dummy_affine)
        
        resampled_atlas = image.resample_to_img(
            self.aal_atlas_img, dummy_img,
            interpolation='nearest', copy=False
        )
        
        atlas_data = resampled_atlas.get_fdata().astype(np.int16)
        
        # Cache for future use (if memory allows)
        if hasattr(self, 'atlas_cache') and len(self.atlas_cache) < 10:
            self.atlas_cache[target_shape] = atlas_data
        
        return atlas_data
    
    def detect_contrast_and_site(self, img_path: str, dataset_folder: str) -> Tuple[str, str]:
        """
        Detect contrast type and site information from path and dataset
        """
        img_path_lower = str(img_path).lower()
        
        # Get site information
        site_info = self.site_info.get(dataset_folder, {})
        primary_contrast = site_info.get('primary_contrast', 'T1w')
        
        # Detect specific contrast from filename
        if any(x in img_path_lower for x in ['mprage', 'mp-rage', 'mp_rage']):
            contrast_type = 'MPRAGE'
        elif any(x in img_path_lower for x in ['flair', 'fluid']):
            contrast_type = 'FLAIR'
        elif any(x in img_path_lower for x in ['t1w', 't1_', 't1-weighted', 'anat']):
            # For SDSU, this is standard T1w; for others, might be MPRAGE labeled as T1
            if dataset_folder == 'SDSUScans':
                contrast_type = 'T1w'
            else:
                contrast_type = 'MPRAGE'  # Likely MPRAGE labeled as anatomical
        else:
            # Use site default
            contrast_type = primary_contrast
        
        logger.info(f"Detected {contrast_type} from {dataset_folder}")
        return contrast_type, dataset_folder
    
    def process_structural_mri(self, t1_path: str, subject_id: str, 
                             dataset_folder: str = None) -> Dict[str, np.ndarray]:
        """
        Process structural MRI with site and contrast-specific optimization
        """
        logger.info(f"Processing structural MRI for subject {subject_id}")
        
        try:
            # Detect contrast and site
            contrast_type, site = self.detect_contrast_and_site(t1_path, dataset_folder)
            
            # Load image
            t1_img = nib.load(t1_path)
            logger.info(f"✅ Loaded {contrast_type} image: {t1_img.shape}")
            
            # Apply contrast and site-specific preprocessing
            preprocessed_img = self._preprocess_with_site_optimization(
                t1_img, contrast_type, site, subject_id
            )
            
            # Extract features using optimized method
            features = self._extract_structural_features_optimized(
                preprocessed_img, subject_id, contrast_type, site
            )
            
            # Store features and metadata
            self.structural_features[subject_id] = features
            self.processing_log['contrast_type'][subject_id] = contrast_type
            self.processing_log['site_info'][subject_id] = site
            
            return features
            
        except Exception as e:
            logger.error(f"Error processing structural MRI for {subject_id}: {e}")
            return {}

    def _preprocess_with_site_optimization(
        self,
        img: nib.Nifti1Image,
        contrast_type: str,
        site: str,
        subject_id: str
    ) -> nib.Nifti1Image:
        """
        Apply site and contrast-specific preprocessing,
        including spatial normalization to MNI.
        """
        # Get site-specific parameters
        site_params = self.site_info.get(site, {}).get('preprocessing_params', {})
        
        # Determine smoothing parameters
        if contrast_type == 'MPRAGE':
            fwhm = site_params.get('mprage_fwhm', 3.0)
            percentile = site_params.get('intensity_percentile', 2)
        elif contrast_type == 'FLAIR':
            fwhm = site_params.get('flair_fwhm', 4.0)
            percentile = site_params.get('intensity_percentile', 5)
        else:  # T1w
            fwhm = site_params.get('mprage_fwhm', 4.0)
            percentile = site_params.get('intensity_percentile', 5)

        
        try:
            mni_template = datasets.load_mni152_template(resolution=2)
            img_mni = image.resample_to_img(
                img,
                target_img=mni_template,
                interpolation="continuous",
                force_resample=True
            )
            logger.info(f"✅ Resampled image to MNI: shape {img_mni.shape}")
        except Exception as e:
            logger.warning(f"⚠️ MNI resampling failed: {e}")
            img_mni = img

        smoothed_img = image.smooth_img(img_mni, fwhm=fwhm)

        # Contrast-specific intensity normalization
        if contrast_type == 'MPRAGE':
            normalized_img = image.math_img(
                f"(img - np.percentile(img[img > 0], {percentile})) / np.percentile(img[img > 0], {100 - percentile})",
                img=smoothed_img
            )
        else:
            normalized_img = image.math_img(
                f"(img - np.percentile(img[img > 0], {percentile})) / np.percentile(img[img > 0], 95)",
                img=smoothed_img
            )
        
        return normalized_img

    
    def _extract_structural_features_optimized(
        self,
        t1_img,
        subject_id: str,
        contrast_type: str,
        site: str
    ) -> Dict[str, np.ndarray]:
        """
        Extract structural features with contrast and site-specific optimizations,
        and filter to only matched AAL regions if regions_to_keep is set.
        """

        max_retries = 3

        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1} for {contrast_type} processing of {subject_id}")

                if self.aal_atlas_img is None:
                    raise ValueError("No valid atlas available")

                # Load T1 data
                t1_data = t1_img.get_fdata()

                # Check if shapes already match
                if t1_img.shape == self.aal_atlas_img.shape:
                    atlas_data = self.aal_atlas_img.get_fdata()
                    logger.info(f"✅ Atlas shape matches T1 image for subject {subject_id}. Skipping resampling.")
                else:
                    # Resample the atlas to the T1 image
                    logger.info(
                        f"🔄 Resampling atlas for subject {subject_id} "
                        f"from shape {self.aal_atlas_img.shape} to {t1_img.shape}"
                    )
                    atlas_img = image.resample_to_img(
                        self.aal_atlas_img,
                        t1_img,
                        interpolation='nearest'
                    )
                    atlas_data = atlas_img.get_fdata()

                # Check unique regions
                unique_regions = np.unique(atlas_data)
                if len(unique_regions) < 10:
                    raise ValueError(
                        f"Atlas appears corrupted: only {len(unique_regions)} regions found"
                    )

                regions = unique_regions[1:]  # exclude background
                regional_features = {}

                # Loop over regions
                for region_id in regions:
                    # ✅ NEW: skip regions not matched to Power
                    if self.regions_to_keep and int(region_id) not in self.regions_to_keep:
                        logger.debug(f"Skipping AAL region {int(region_id)} (not matched to Power)")
                        continue

                    region_mask = (atlas_data == region_id)

                    if np.sum(region_mask) > 0:
                        volume = np.sum(region_mask)
                        mean_intensity = np.mean(t1_data[region_mask])
                        std_intensity = np.std(t1_data[region_mask])

                        # Contrast-specific features
                        if contrast_type == 'MPRAGE':
                            features = {
                                'volume': volume,
                                'mean_intensity': mean_intensity,
                                'std_intensity': std_intensity,
                                'intensity_range': np.ptp(t1_data[region_mask]),
                                'tissue_contrast': mean_intensity / (std_intensity + 1e-8),
                                'high_intensity_fraction': np.sum(
                                    t1_data[region_mask] > np.percentile(t1_data[region_mask], 75)
                                ) / volume
                            }
                        elif contrast_type == 'FLAIR':
                            features = {
                                'volume': volume,
                                'mean_intensity': mean_intensity,
                                'std_intensity': std_intensity,
                                'intensity_range': np.ptp(t1_data[region_mask]),
                                'lesion_indicator': np.sum(
                                    t1_data[region_mask] > np.percentile(t1_data[region_mask], 90)
                                ) / volume,
                                'csf_suppression': np.sum(
                                    t1_data[region_mask] < np.percentile(t1_data[region_mask], 10)
                                ) / volume
                            }
                        else:
                            # Default: T1w SDSU
                            features = {
                                'volume': volume,
                                'mean_intensity': mean_intensity,
                                'std_intensity': std_intensity,
                                'intensity_range': np.ptp(t1_data[region_mask]),
                                'tissue_contrast': mean_intensity / (std_intensity + 1e-8)
                            }

                        regional_features[f'region_{int(region_id):02d}'] = features

                if len(regional_features) < 10:
                    raise ValueError(
                        f"Too few regions extracted: {len(regional_features)}"
                    )

                logger.info(
                    f"✅ Extracted {contrast_type} features for {len(regional_features)} brain regions"
                )

                atlas_name = "AAL" if "aal" in str(self.aal_atlas.description).lower() else "Harvard-Oxford"
                self.processing_log['atlas_usage'][subject_id] = atlas_name

                return regional_features

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {subject_id}: {e}")

                if attempt == max_retries - 1:
                    logger.error(
                        f"All attempts failed for {subject_id}. Using fallback extraction."
                    )
                    return self._extract_fallback_features(
                        t1_img,
                        subject_id,
                        contrast_type,
                        site
                    )

        raise RuntimeError(f"Failed to extract structural features for {subject_id}")



    
    def _extract_fallback_features(self, t1_img, subject_id: str, 
                                 contrast_type: str, site: str) -> Dict[str, np.ndarray]:
        """
        Fallback feature extraction when atlas fails
        """
        logger.info(f"Using fallback extraction for {contrast_type} from {site}")
        
        # Try Harvard-Oxford as backup
        try:
            ho_atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
            
            if isinstance(ho_atlas.maps, str):
                atlas_img = nib.load(ho_atlas.maps)
            else:
                atlas_img = ho_atlas.maps
            
            atlas_img = image.resample_to_img(atlas_img, t1_img, interpolation='nearest')
            atlas_data = atlas_img.get_fdata()
            t1_data = t1_img.get_fdata()
            
            regions = np.unique(atlas_data)[1:]
            regional_features = {}
            
            for region_id in regions:
                region_mask = (atlas_data == region_id)
                
                if np.sum(region_mask) > 0:
                    # Use contrast-specific features even in fallback
                    if contrast_type == 'MPRAGE':
                        features = {
                            'volume': np.sum(region_mask),
                            'mean_intensity': np.mean(t1_data[region_mask]),
                            'std_intensity': np.std(t1_data[region_mask]),
                            'tissue_contrast': np.mean(t1_data[region_mask]) / (np.std(t1_data[region_mask]) + 1e-8)
                        }
                    elif contrast_type == 'FLAIR':
                        features = {
                            'volume': np.sum(region_mask),
                            'mean_intensity': np.mean(t1_data[region_mask]),
                            'std_intensity': np.std(t1_data[region_mask]),
                            'lesion_indicator': np.sum(t1_data[region_mask] > np.percentile(t1_data[region_mask], 90)) / np.sum(region_mask)
                        }
                    else:  # T1w
                        features = {
                            'volume': np.sum(region_mask),
                            'mean_intensity': np.mean(t1_data[region_mask]),
                            'std_intensity': np.std(t1_data[region_mask])
                        }
                    
                    regional_features[f'region_{int(region_id):02d}'] = features
            
            if len(regional_features) >= 30:
                logger.info(f"✅ Harvard-Oxford fallback successful: {len(regional_features)} regions")
                self.processing_log['atlas_usage'][subject_id] = "Harvard-Oxford"
                return regional_features
            
        except Exception as e:
            logger.warning(f"Harvard-Oxford fallback failed: {e}")
        
        # Final fallback: grid-based features
        logger.warning(f"Using grid-based fallback for {subject_id}")
        return self._create_grid_features(t1_img, subject_id, contrast_type)
    
    def _create_grid_features(self, t1_img, subject_id: str, contrast_type: str) -> Dict[str, np.ndarray]:
        """
        Create grid-based features as final fallback
        """
        t1_data = t1_img.get_fdata()
        brain_mask = t1_data > np.percentile(t1_data[t1_data > 0], 10)
        
        # Create consistent number of regions
        regional_features = {}
        x_size, y_size, z_size = t1_data.shape
        
        # 6x5x3 = 90 regions to match AAL
        x_divisions, y_divisions, z_divisions = 6, 5, 3
        region_id = 1
        
        for i in range(x_divisions):
            for j in range(y_divisions):
                for k in range(z_divisions):
                    x_start = i * x_size // x_divisions
                    x_end = (i + 1) * x_size // x_divisions
                    y_start = j * y_size // y_divisions
                    y_end = (j + 1) * y_size // y_divisions
                    z_start = k * z_size // z_divisions
                    z_end = (k + 1) * z_size // z_divisions
                    
                    region_mask = np.zeros_like(brain_mask, dtype=bool)
                    region_mask[x_start:x_end, y_start:y_end, z_start:z_end] = True
                    region_mask = region_mask & brain_mask
                    
                    if np.sum(region_mask) > 10:
                        # Contrast-specific features even in grid
                        if contrast_type == 'MPRAGE':
                            features = {
                                'volume': np.sum(region_mask),
                                'mean_intensity': np.mean(t1_data[region_mask]),
                                'std_intensity': np.std(t1_data[region_mask]),
                                'tissue_contrast': np.mean(t1_data[region_mask]) / (np.std(t1_data[region_mask]) + 1e-8)
                            }
                        else:
                            features = {
                                'volume': np.sum(region_mask),
                                'mean_intensity': np.mean(t1_data[region_mask]),
                                'std_intensity': np.std(t1_data[region_mask])
                            }
                    else:
                        features = {
                            'volume': 0,
                            'mean_intensity': 0,
                            'std_intensity': 0
                        }
                    
                    regional_features[f'region_{region_id:02d}'] = features
                    region_id += 1
        
        logger.info(f"✅ Created {len(regional_features)} grid-based regions")
        self.processing_log['atlas_usage'][subject_id] = "Grid-based"
        return regional_features
    
    def process_functional_mri(self, fmri_path: str, subject_id: str, 
                             tr: float = 2.0, high_pass: float = 0.01) -> Dict[str, np.ndarray]:
        """
        Process resting-state functional MRI
        """
        logger.info(f"Processing functional MRI for subject {subject_id}")
        
        try:
            # Load fMRI data
            fmri_img = nib.load(fmri_path)
            logger.info(f"✅ Loaded fMRI image: {fmri_img.shape}")
            
            # Basic preprocessing
            fmri_img = self._preprocess_fmri(fmri_img, tr, high_pass)
            
            # Extract time series and connectivity
            features = self._extract_functional_features(fmri_img, subject_id)
            
            # Store features
            self.functional_features[subject_id] = features
            
            return features
            
        except Exception as e:
            logger.error(f"Error processing functional MRI for {subject_id}: {e}")
            return {}
    
    def _preprocess_fmri(self, fmri_img, tr: float, high_pass: float):
        """
        Basic fMRI preprocessing including MNI normalization
        """

        try:
            mni_template = datasets.load_mni152_template(resolution=2)
            fmri_img = image.resample_to_img(
                fmri_img,
                target_img=mni_template,
                interpolation="continuous",
                force_resample=True
            )
            logger.info(f"✅ Resampled fMRI to MNI space: {fmri_img.shape}")
        except Exception as e:
            logger.warning(f"⚠️ MNI resampling failed: {e}")
            # leave unchanged

        # Spatial smoothing
        fmri_img = image.smooth_img(fmri_img, fwhm=6)

        # Temporal filtering
        fmri_img = image.clean_img(
            fmri_img,
            detrend=True,
            standardize=True,
            low_pass=0.1,
            high_pass=high_pass,
            t_r=tr
        )

        return fmri_img
    
    def _extract_functional_features(self, fmri_img, subject_id: str) -> Dict[str, np.ndarray]:
        """
        Extract functional connectivity features using AAL regions filtered to match Power.
        Produces time series and connectivity matrix of shape (93, 93).
        """

        # ------------------------------------------------
        # Filter AAL atlas to only your 93 kept regions
        # ------------------------------------------------
        aal_data = self.aal_atlas_img.get_fdata()
        region_voxels = np.isin(aal_data, self.regions_to_keep)
        filtered_aal_data = np.where(region_voxels, aal_data, 0)

        filtered_aal_img = nib.Nifti1Image(
            filtered_aal_data,
            affine=self.aal_atlas_img.affine,
            header=self.aal_atlas_img.header
        )

        # ------------------------------------------------
        # Create the masker for just those regions
        # ------------------------------------------------
        masker = input_data.NiftiLabelsMasker(
            labels_img=filtered_aal_img,
            standardize=True,
            detrend=True,
            verbose=0
        )

        # ------------------------------------------------
        # Extract time series
        # ------------------------------------------------
        time_series = masker.fit_transform(fmri_img)
        logger.info(f"✅ Extracted AAL timeseries for {subject_id}: shape {time_series.shape}")

        n_regions = time_series.shape[1]

        # Compute connectivity
        correlation_matrix = np.corrcoef(time_series.T)

        # Vectorize upper triangle
        connectivity_vector = correlation_matrix[np.triu_indices(n_regions, k=1)]

        # Network metrics
        network_features = self._compute_network_measures(correlation_matrix)

        features = {
            'time_series': time_series,
            'connectivity_vector': connectivity_vector,
            'correlation_matrix': correlation_matrix,
            'network_measures': network_features
        }

        return features


    
    def _compute_network_measures(self, correlation_matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute graph theory network measures"""
        
        # Threshold matrix to create binary network
        threshold = 0.3  # Keep top 30% of connections
        binary_matrix = np.abs(correlation_matrix) > threshold
        
        # Node-level measures
        node_degree = np.sum(binary_matrix, axis=1)
        node_strength = np.sum(np.abs(correlation_matrix), axis=1)
        
        # Clustering coefficient (simplified version)
        clustering = np.zeros(correlation_matrix.shape[0])
        for i in range(correlation_matrix.shape[0]):
            neighbors = np.where(binary_matrix[i, :])[0]
            if len(neighbors) > 1:
                subgraph = binary_matrix[np.ix_(neighbors, neighbors)]
                clustering[i] = np.sum(subgraph) / (len(neighbors) * (len(neighbors) - 1))
        
        return {
            'node_degree': node_degree,
            'node_strength': node_strength,
            'clustering_coefficient': clustering
        }
    
    def process_dti(self, dti_path: str, bval_path: str, bvec_path: str, 
                   subject_id: str, mask_path: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Process Diffusion Tensor Imaging data - FIXED for volume mismatch
        """
        logger.info(f"Processing DTI for subject {subject_id}")
        
        try:
            # Load DTI data
            dti_img = nib.load(dti_path)
            dti_data = dti_img.get_fdata()
            logger.info(f"✅ Loaded DTI image: {dti_data.shape}")
            
            # Load gradient information
            bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
            logger.info(f"✅ Loaded gradients: {len(bvals)} directions")
            
            # FIX: Handle volume mismatch for Barrow data
            n_volumes = dti_data.shape[-1] if len(dti_data.shape) == 4 else 1
            n_gradients = len(bvals)
            
            if n_volumes != n_gradients:
                logger.warning(f"⚠️  Volume mismatch detected: DTI has {n_volumes} volumes, gradients have {n_gradients} directions")
                
                if n_volumes == n_gradients + 1:
                    # Conservative fix: analyze which volume to remove
                    logger.info("🔍 Analyzing volumes to safely remove extra volume...")
                    
                    # Get first and last volumes for comparison
                    first_vol = dti_data[..., 0]
                    last_vol = dti_data[..., -1]
                    
                    # Check which one looks more like a b=0 (should have higher intensity)
                    first_mean = np.mean(first_vol[first_vol > 0])
                    last_mean = np.mean(last_vol[last_vol > 0])
                    
                    logger.info(f"   First volume mean intensity: {first_mean:.2f}")
                    logger.info(f"   Last volume mean intensity: {last_mean:.2f}")
                    
                    # The b=0 volume should have higher intensity
                    # Remove the volume that looks like a duplicate b=0
                    if bvals[0] == 0:  # First gradient is b=0
                        if last_mean > first_mean * 0.9:  # Last volume looks like b=0 too
                            logger.info("   Removing last volume (appears to be duplicate b=0)")
                            dti_data = dti_data[..., :-1]
                        else:
                            logger.warning("   Last volume doesn't look like b=0, keeping all volumes and skipping DTI")
                            return {}
                    else:
                        logger.warning("   Unexpected gradient structure, skipping DTI processing")
                        return {}
                
                elif n_volumes == n_gradients - 1:
                    logger.error(f"❌ DTI missing volume - cannot process safely")
                    return {}
                else:
                    logger.error(f"❌ Major volume mismatch ({n_volumes} vs {n_gradients}) - cannot process safely")
                    return {}
                
                # Verify fix worked
                final_volumes = dti_data.shape[-1] if len(dti_data.shape) == 4 else 1
                if final_volumes != n_gradients:
                    logger.error(f"❌ Fix failed: still have {final_volumes} vs {n_gradients}")
                    return {}
                else:
                    logger.info(f"✅ Volume mismatch resolved: {final_volumes} volumes now match {n_gradients} gradients")
            
            # Create gradient table
            gtab = gradient_table(bvals, bvecs)
            
            # Create or load brain mask
            if mask_path and os.path.exists(mask_path):
                mask_img = nib.load(mask_path)
                mask = mask_img.get_fdata().astype(bool)
            else:
                # Create mask using median_otsu on b=0 volume
                b0_idx = 0 if bvals[0] == 0 else np.where(bvals == 0)[0][0]
                b0_mask, mask = median_otsu(dti_data[:, :, :, b0_idx], median_radius=2, numpass=1)
                logger.info("✅ Created brain mask using median_otsu")
            
            # Fit DTI model
            features = self._extract_dti_features(dti_data, gtab, mask, subject_id)
            
            # Store features
            self.dti_features[subject_id] = features
            
            return features
            
        except Exception as e:
            logger.error(f"Error processing DTI for {subject_id}: {e}")
            return {}
    
    def _extract_dti_features(self, dti_data: np.ndarray, gtab, mask: np.ndarray, 
                             subject_id: str) -> Dict[str, np.ndarray]:
        """Extract DTI-derived features"""
        
        # Fit tensor model
        tenmodel = TensorModel(gtab)
        tenfit = tenmodel.fit(dti_data, mask=mask)
        
        # Compute DTI metrics
        fa = tenfit.fa  # Fractional Anisotropy
        md = tenfit.md  # Mean Diffusivity
        rd = tenfit.rd  # Radial Diffusivity
        ad = tenfit.ad  # Axial Diffusivity
        
        logger.info("✅ Computed DTI metrics (FA, MD, RD, AD)")
        
        # Extract regional DTI values using atlas
        regional_dti = self._extract_regional_dti(fa, md, rd, ad, mask)
        
        # Compute white matter integrity measures
        wm_features = self._compute_wm_features(fa, md, mask)
        
        features = {
            'fa_map': fa,
            'md_map': md,
            'rd_map': rd,
            'ad_map': ad,
            'regional_dti': regional_dti,
            'wm_features': wm_features
        }
        
        return features
    
    def _extract_regional_dti(self, fa: np.ndarray, md: np.ndarray, 
                             rd: np.ndarray, ad: np.ndarray, 
                             mask: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Extract DTI metrics for atlas-defined regions"""
        
        # Global and hemispheric measures
        left_hemisphere = mask.copy()
        left_hemisphere[:, mask.shape[1]//2:, :] = False
        
        right_hemisphere = mask.copy()
        right_hemisphere[:, :mask.shape[1]//2, :] = False
        
        regional_features = {}
        
        # Global measures
        regional_features['global'] = {
            'fa_mean': np.mean(fa[mask]),
            'fa_std': np.std(fa[mask]),
            'md_mean': np.mean(md[mask]),
            'md_std': np.std(md[mask]),
            'rd_mean': np.mean(rd[mask]),
            'ad_mean': np.mean(ad[mask])
        }
        
        # Hemispheric measures
        for hemi_name, hemi_mask in [('left', left_hemisphere), ('right', right_hemisphere)]:
            if np.sum(hemi_mask) > 0:
                regional_features[hemi_name] = {
                    'fa_mean': np.mean(fa[hemi_mask]),
                    'md_mean': np.mean(md[hemi_mask]),
                    'rd_mean': np.mean(rd[hemi_mask]),
                    'ad_mean': np.mean(ad[hemi_mask])
                }
        
        logger.info(f"✅ Extracted regional DTI features for {len(regional_features)} regions")
        
        return regional_features
    
    def _compute_wm_features(self, fa: np.ndarray, md: np.ndarray, 
                            mask: np.ndarray) -> Dict[str, float]:
        """Compute white matter integrity features"""
        
        # White matter mask (high FA regions)
        wm_mask = (fa > 0.2) & mask
        
        features = {
            'wm_volume': np.sum(wm_mask),
            'wm_fa_mean': np.mean(fa[wm_mask]) if np.sum(wm_mask) > 0 else 0,
            'wm_md_mean': np.mean(md[wm_mask]) if np.sum(wm_mask) > 0 else 0,
            'fa_histogram_peak': np.argmax(np.histogram(fa[mask], bins=50)[0]) / 50.0
        }
        
        return features
    
    def save_features(self, subject_id: str):
        """Save extracted features to disk"""
        
        subject_dir = self.output_dir / subject_id
        subject_dir.mkdir(exist_ok=True)
        
        # Save structural features
        if subject_id in self.structural_features:
            np.save(subject_dir / 'structural_features.npy', 
                    self.structural_features[subject_id])
            
            # ➡️ DEBUGGING LINE HERE:
            features = self.structural_features[subject_id]
            num_regions = len(features)
            print(f"✅ [DEBUG] Structural for {subject_id}: {num_regions} regions saved.")
            if num_regions > 0:
                first_region = list(features.keys())[0]
                print(f"    Sample region: {first_region} → {features[first_region]}")
        
        # Save functional features
        if subject_id in self.functional_features:
            np.save(subject_dir / 'functional_features.npy', 
                    self.functional_features[subject_id])
            
            # ➡️ DEBUGGING LINE HERE:
            func = self.functional_features[subject_id]
            matrix_shape = func['correlation_matrix'].shape
            print(f"✅ [DEBUG] Functional for {subject_id}: correlation matrix shape {matrix_shape}")
        
        # Save DTI features
        if subject_id in self.dti_features:
            np.save(subject_dir / 'dti_features.npy', 
                    self.dti_features[subject_id])
            
            # ➡️ DEBUGGING LINE HERE:
            dti = self.dti_features[subject_id]
            wm_feats = dti.get('wm_features', None)
            regional = dti.get('regional_dti', None)
            print(f"✅ [DEBUG] DTI for {subject_id}: wm_features → {type(wm_feats)}, regional_dti → {type(regional)}")

    
    def save_processing_log(self):
        """Save detailed processing log"""
        import json
        
        log_file = self.output_dir / 'processing_log.json'
        
        # Convert any numpy types to regular Python types for JSON serialization
        clean_log = {}
        for key, value in self.processing_log.items():
            if isinstance(value, dict):
                clean_log[key] = {str(k): str(v) for k, v in value.items()}
            else:
                clean_log[key] = value
        
        with open(log_file, 'w') as f:
            json.dump(clean_log, f, indent=2)
        
        logger.info(f"✅ Saved processing log to {log_file}")
    
    def create_comprehensive_summary(self) -> pd.DataFrame:
        """
        Create comprehensive summary including contrast and site information
        """
        summary_data = []
        
        for subject_id in set(list(self.structural_features.keys()) + 
                            list(self.functional_features.keys()) + 
                            list(self.dti_features.keys())):
            
            summary = {'subject_id': subject_id}
            
            # Structural summary with contrast info
            if subject_id in self.structural_features:
                struct_feat = self.structural_features[subject_id]
                summary['n_structural_regions'] = len(struct_feat)
                summary['has_structural'] = True
                summary['atlas_used'] = self.processing_log['atlas_usage'].get(subject_id, 'Unknown')
                summary['contrast_type'] = self.processing_log['contrast_type'].get(subject_id, 'Unknown')
                summary['site'] = self.processing_log['site_info'].get(subject_id, 'Unknown')
            else:
                summary['has_structural'] = False
                summary['atlas_used'] = 'None'
                summary['contrast_type'] = 'None'
                summary['site'] = 'Unknown'
            
            # Functional summary
            if subject_id in self.functional_features:
                func_feat = self.functional_features[subject_id]
                summary['n_functional_connections'] = len(func_feat['connectivity_vector'])
                summary['n_timepoints'] = func_feat['time_series'].shape[0]
                summary['has_functional'] = True
            else:
                summary['has_functional'] = False
            
            # DTI summary
            if subject_id in self.dti_features:
                dti_feat = self.dti_features[subject_id]
                summary['n_dti_regions'] = len(dti_feat['regional_dti'])
                summary['has_dti'] = True
            else:
                summary['has_dti'] = False
            
            summary['complete_multimodal'] = (summary.get('has_structural', False) and 
                                            summary.get('has_functional', False) and 
                                            summary.get('has_dti', False))
            
            summary_data.append(summary)
        
        summary_df = pd.DataFrame(summary_data)
        
        if len(summary_df) == 0:
            logger.warning("No subjects found - creating empty summary")
            summary_df = pd.DataFrame(columns=[
                'subject_id', 'has_structural', 'has_functional', 'has_dti', 
                'complete_multimodal', 'atlas_used', 'contrast_type', 'site'
            ])
        
        # Save summary
        summary_df.to_csv(self.output_dir / 'comprehensive_summary.csv', index=False)
        
        # Print detailed summary
        self.print_comprehensive_summary(summary_df)
        
        return summary_df
    
    def print_comprehensive_summary(self, summary_df: pd.DataFrame):
        """
        Print detailed summary including contrast and site breakdowns
        """
        if len(summary_df) == 0:
            print("No subjects processed")
            return
        
        print("\n" + "="*80)
        print("COMPREHENSIVE PROCESSING SUMMARY")
        print("="*80)
        
        # Overall statistics
        total_subjects = len(summary_df)
        complete_subjects = summary_df['complete_multimodal'].sum()
        
        print(f"Total subjects processed: {total_subjects}")
        print(f"Complete multimodal subjects: {complete_subjects}")
        print(f"Subjects with structural data: {summary_df['has_structural'].sum()}")
        print(f"Subjects with functional data: {summary_df['has_functional'].sum()}")
        print(f"Subjects with DTI data: {summary_df['has_dti'].sum()}")
        
        # Site breakdown
        print(f"\n{'SITE BREAKDOWN':^80}")
        print("-" * 80)
        site_counts = summary_df['site'].value_counts()
        for site, count in site_counts.items():
            percentage = (count / total_subjects) * 100
            print(f"  {site:<25}: {count:>3} subjects ({percentage:>5.1f}%)")
        
        # Contrast breakdown
        print(f"\n{'CONTRAST TYPE BREAKDOWN':^80}")
        print("-" * 80)
        contrast_counts = summary_df['contrast_type'].value_counts()
        for contrast, count in contrast_counts.items():
            percentage = (count / total_subjects) * 100
            print(f"  {contrast:<25}: {count:>3} subjects ({percentage:>5.1f}%)")
        
        # Atlas usage
        print(f"\n{'ATLAS USAGE':^80}")
        print("-" * 80)
        atlas_counts = summary_df['atlas_used'].value_counts()
        for atlas, count in atlas_counts.items():
            percentage = (count / total_subjects) * 100
            print(f"  {atlas:<25}: {count:>3} subjects ({percentage:>5.1f}%)")
        
        print("="*80)

# Integration with existing functions
def find_subject_files(base_dir: str, subject_id: str) -> Dict[str, str]:
    """
    Enhanced file finder that also detects dataset folder
    """
    base_path = Path(base_dir)
    dataset_folders = ['BarrowNeurologicalScans', 'NYU1Scans', 'NYU2Scans', 
                      'SDSUScans', 'TrinityCentreScans']
    
    files_found = {}
    
    for dataset_folder in dataset_folders:
        dataset_path = base_path / dataset_folder
        if not dataset_path.exists():
            continue
            
        subject_path = dataset_path / subject_id
        if not subject_path.exists():
            continue
            
        session_path = subject_path / "session_1"
        if not session_path.exists():
            continue
            
        logger.info(f"Found subject {subject_id} in {dataset_folder}")
        
        modality_folders = {
            'anat': 'anat_1',
            'rest': 'rest_1', 
            'dti': 'dti_1'
        }
        
        for modality, folder_name in modality_folders.items():
            modality_path = session_path / folder_name
            if modality_path.exists():
                files_in_folder = list(modality_path.glob("*"))
                
                if modality == 'anat':
                    # Look for structural files
                    anat_files = [f for f in files_in_folder if f.suffix in ['.nii', '.gz'] 
                                and any(x in f.name.lower() for x in ['t1', 'anat', 'structural', 'mprage', 'flair'])]
                    if anat_files:
                        files_found['t1'] = str(anat_files[0])
                        
                elif modality == 'rest':
                    # Look for functional files
                    func_files = [f for f in files_in_folder if f.suffix in ['.nii', '.gz']
                                and any(x in f.name.lower() for x in ['rest', 'func', 'bold'])]
                    if func_files:
                        files_found['fmri'] = str(func_files[0])
                        
                elif modality == 'dti':
                    # Look for DTI files
                    dti_files = [f for f in files_in_folder if f.suffix in ['.nii', '.gz']
                               and any(x in f.name.lower() for x in ['dti', 'dwi', 'diffusion'])]
                    bval_files = [f for f in files_in_folder if f.suffix == '.bval']
                    bvec_files = [f for f in files_in_folder if f.suffix == '.bvec']
                    
                    if dti_files:
                        files_found['dti'] = str(dti_files[0])
                    if bval_files:
                        files_found['bval'] = str(bval_files[0])
                    if bvec_files:
                        files_found['bvec'] = str(bvec_files[0])
        
        # If we found files for this subject, we can stop searching other datasets
        if files_found:
            files_found['dataset'] = dataset_folder
            break
    
    return files_found

def process_abide_subject_optimized(base_dir: str, subject_id: str, 
                                  processor: ABIDEOptimizedPreprocessor):
    """
    Process a single ABIDE subject with optimized pipeline
    """
    logger.info(f"Processing subject {subject_id}")
    
    # Find files for this subject
    files = find_subject_files(base_dir, subject_id)
    
    if not files:
        logger.warning(f"No files found for subject {subject_id}")
        return
    
    dataset_folder = files.get('dataset', 'Unknown')
    logger.info(f"Subject {subject_id} found in dataset: {dataset_folder}")
    
    # Process each modality if files exist
    if 't1' in files:
        logger.info(f"Processing structural MRI: {files['t1']}")
        processor.process_structural_mri(files['t1'], subject_id, dataset_folder)
    else:
        logger.warning(f"T1 file not found for subject {subject_id}")
    
    if 'fmri' in files:
        logger.info(f"Processing functional MRI: {files['fmri']}")
        processor.process_functional_mri(files['fmri'], subject_id)
    else:
        logger.warning(f"fMRI file not found for subject {subject_id}")
    
    if all(key in files for key in ['dti', 'bval', 'bvec']):
        logger.info(f"Processing DTI: {files['dti']}")
        processor.process_dti(files['dti'], files['bval'], files['bvec'], subject_id)
    else:
        missing = [key for key in ['dti', 'bval', 'bvec'] if key not in files]
        logger.warning(f"DTI files not complete for subject {subject_id}. Missing: {missing}")
    
    # Save features
    processor.save_features(subject_id)

# PARALLEL PROCESSING FUNCTIONS
def process_subject_wrapper_optimized(args):
    """Wrapper for parallel processing"""
    base_dir, subject_id, output_dir = args
    
    try:
        # Create processor with speed optimizations
        processor = ABIDEOptimizedPreprocessor(output_dir=output_dir)
        
        # Use existing processing function
        process_abide_subject_optimized(base_dir, subject_id, processor)
        
        return subject_id, "success"
        
    except Exception as e:
        logger.error(f"Failed to process {subject_id}: {e}")
        return subject_id, "failed"

def run_parallel_preprocessing(base_dir: str, output_dir: str = "./abide_optimized_features"):
    """Run existing pipeline in parallel for massive speedup"""
    
    # Get optimal number of workers
    n_jobs = 4
    logger.info(f"🚀 Starting parallel preprocessing with {n_jobs} workers")
    logger.info(f"💻 Available CPUs: {psutil.cpu_count()}")
    logger.info(f"💾 Available RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    # Find all subjects
    base_path = Path(base_dir)
    dataset_folders = ['BarrowNeurologicalScans', 'NYU1Scans', 'NYU2Scans', 
                      'SDSUScans', 'TrinityCentreScans']
    
    all_subjects = set()
    for dataset_folder in dataset_folders:
        dataset_path = base_path / dataset_folder
        if dataset_path.exists():
            subject_folders = [d.name for d in dataset_path.iterdir() 
                             if d.is_dir() and d.name.isdigit()]
            all_subjects.update(subject_folders)
    
    all_subjects = sorted(list(all_subjects))
    logger.info(f"📊 Found {len(all_subjects)} subjects to process")
    
    # Prepare arguments for parallel processing
    args_list = [(base_dir, subject_id, output_dir) for subject_id in all_subjects]
    
    # Process in parallel
    successful = 0
    failed = 0
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Submit all jobs
        future_to_subject = {
            executor.submit(process_subject_wrapper_optimized, args): args[1] 
            for args in args_list
        }
        
        # Process results as they complete
        for i, future in enumerate(as_completed(future_to_subject), 1):
            subject_id, status = future.result()
            
            if status == "success":
                successful += 1
                logger.info(f"✅ [{i}/{len(all_subjects)}] {subject_id}: SUCCESS")
            else:
                failed += 1
                logger.info(f"❌ [{i}/{len(all_subjects)}] {subject_id}: FAILED")
            
            # Progress update every 10 subjects
            if i % 10 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                eta = (len(all_subjects) - i) / rate if i < len(all_subjects) else 0
                logger.info(f"📈 Progress: {i}/{len(all_subjects)} ({rate:.1f} subjects/min, ETA: {eta/60:.1f} min)")
    
    total_time = time.time() - start_time
    
    logger.info(f"\n{'='*60}")
    logger.info(f"PARALLEL PROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Processing rate: {len(all_subjects)/total_time*60:.1f} subjects/hour")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success rate: {successful/len(all_subjects)*100:.1f}%")
    
    return successful, failed

if __name__ == "__main__":
    # Your base directory containing all dataset folders
    base_directory = "/Users/joemargolis/Desktop/Autism Project/Brain Scans"
    
    # Run with parallel processing for massive speedup
    print("🚀 Starting speed-optimized parallel preprocessing...")
    print("This should complete in 30-45 minutes instead of hours!")
    
    successful, failed = run_parallel_preprocessing(
        base_directory, 
        output_dir="./abide_speed_optimized_features"
    )
    
    # Create comprehensive summary
    print("\n📊 Generating comprehensive feature summary...")
    processor = ABIDEOptimizedPreprocessor(output_dir="./abide_speed_optimized_features")
    
    # Load results from processed subjects for summary
    summary = processor.create_comprehensive_summary()
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"Successfully processed: {successful} subjects")
    print(f"Failed to process: {failed} subjects")
    print(f"Success rate: {successful/(successful+failed)*100:.1f}%")
    
    print(f"\n" + "="*80)
    print("RECOMMENDATIONS FOR MASK GNN")
    print("="*80)
    print("1. Feature Standardization:")
    print("   - MPRAGE and T1w features should be highly comparable")
    print("   - Consider separate normalization for FLAIR features")
    print("   - Use site as a covariate in your GNN if needed")
    print()
    print("2. Quality Control:")
    print("   - Check that >90% of subjects use consistent atlas")
    print("   - Verify feature distributions are similar across sites")
    print("   - Consider excluding subjects with grid-based fallback")
    print()
    print("3. GNN Input Considerations:")
    print("   - Use contrast_type as node feature if needed")
    print("   - Site information can be used for domain adaptation")
    print("   - MPRAGE features may be more reliable than T1w")
    print("="*80)