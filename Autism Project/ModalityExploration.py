#!/usr/bin/env python3
"""
Robust Brain Feature Visualizations - Fixed Version
Handles data inconsistencies and provides better debugging
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

logger = logging.getLogger(__name__)

class RobustBrainVisualizer:
    """
    Robust brain feature visualization with better error handling
    """
    
    def __init__(self, processed_dir="./abide_speed_optimized_features"):
        self.processed_dir = Path(processed_dir)
        self.summary_df = None
        self.load_summary()
        
    def load_summary(self):
        """Load summary data"""
        summary_file = self.processed_dir / 'comprehensive_summary.csv'
        if summary_file.exists():
            self.summary_df = pd.read_csv(summary_file)
            logger.info(f"✅ Loaded summary: {len(self.summary_df)} subjects")
        else:
            logger.error("Summary file not found!")
    
    def debug_subject_data(self, subject_id, modality='all'):
        """Debug a specific subject's data"""
        
        print(f"\n🔍 DEBUGGING SUBJECT {subject_id}")
        print("=" * 50)
        
        subject_dir = self.processed_dir / str(subject_id)
        
        if modality in ['all', 'structural']:
            struct_file = subject_dir / 'structural_features.npy'
            if struct_file.exists():
                try:
                    struct_data = np.load(struct_file, allow_pickle=True).item()
                    print(f"✅ Structural: {len(struct_data)} regions")
                    sample_region = list(struct_data.keys())[0]
                    print(f"   Sample region: {sample_region}")
                    print(f"   Features: {list(struct_data[sample_region].keys())}")
                except Exception as e:
                    print(f"❌ Structural error: {e}")
            else:
                print(f"❌ Structural file not found")
        
        if modality in ['all', 'functional']:
            func_file = subject_dir / 'functional_features.npy'
            if func_file.exists():
                try:
                    func_data = np.load(func_file, allow_pickle=True).item()
                    print(f"✅ Functional data keys: {list(func_data.keys())}")
                    
                    if 'correlation_matrix' in func_data:
                        corr_matrix = func_data['correlation_matrix']
                        print(f"   Correlation matrix shape: {corr_matrix.shape}")
                        print(f"   Matrix range: [{np.nanmin(corr_matrix):.3f}, {np.nanmax(corr_matrix):.3f}]")
                        print(f"   Has NaN values: {np.isnan(corr_matrix).any()}")
                        print(f"   Has infinite values: {np.isinf(corr_matrix).any()}")
                    
                    if 'time_series' in func_data:
                        ts = func_data['time_series']
                        print(f"   Time series shape: {ts.shape}")
                        
                except Exception as e:
                    print(f"❌ Functional error: {e}")
            else:
                print(f"❌ Functional file not found")
        
        if modality in ['all', 'dti']:
            dti_file = subject_dir / 'dti_features.npy'
            if dti_file.exists():
                try:
                    dti_data = np.load(dti_file, allow_pickle=True).item()
                    print(f"✅ DTI data keys: {list(dti_data.keys())}")
                    
                    if 'regional_dti' in dti_data:
                        regional = dti_data['regional_dti']
                        print(f"   DTI regions: {list(regional.keys())}")
                        if 'global' in regional:
                            global_dti = regional['global']
                            print(f"   Global DTI features: {list(global_dti.keys())}")
                            
                except Exception as e:
                    print(f"❌ DTI error: {e}")
            else:
                print(f"❌ DTI file not found")
    
    def plot_structural_features_robust(self, max_subjects=231):
        """Robust structural feature plotting"""
        
        structural_subjects = self.summary_df[
            self.summary_df['has_structural'] == True
        ]['subject_id'].tolist()[:max_subjects]
        
        print(f"📊 Processing {len(structural_subjects)} subjects for structural analysis")
        
        # Collect all features
        all_features = {}
        successful_subjects = []
        
        for subject_id in structural_subjects:
            try:
                struct_file = self.processed_dir / str(subject_id) / 'structural_features.npy'
                if struct_file.exists():
                    struct_data = np.load(struct_file, allow_pickle=True).item()
                    
                    for region_id, region_data in struct_data.items():
                        if region_id not in all_features:
                            all_features[region_id] = {}
                        
                        for feat_name, feat_value in region_data.items():
                            if feat_name not in all_features[region_id]:
                                all_features[region_id][feat_name] = []
                            
                            # Only add valid numerical values
                            if isinstance(feat_value, (int, float)) and not np.isnan(feat_value) and not np.isinf(feat_value):
                                all_features[region_id][feat_name].append(feat_value)
                    
                    successful_subjects.append(subject_id)
                    
            except Exception as e:
                print(f"   ⚠️ Error with subject {subject_id}: {e}")
                continue
        
        print(f"✅ Successfully loaded {len(successful_subjects)} subjects")
        print(f"✅ Found {len(all_features)} brain regions")
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('🧠 Structural Brain Features (Robust Analysis)', fontsize=16, fontweight='bold')
        
        feature_types = ['volume', 'mean_intensity', 'std_intensity', 'intensity_range', 'tissue_contrast']
        
        for idx, feature_type in enumerate(feature_types):
            if idx >= 6:  # Only 6 subplots
                break
                
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            # Collect feature data
            plot_data = []
            for region_id, region_features in all_features.items():
                if feature_type in region_features and len(region_features[feature_type]) > 5:
                    mean_val = np.mean(region_features[feature_type])
                    std_val = np.std(region_features[feature_type])
                    plot_data.append({
                        'region': region_id,
                        'mean': mean_val,
                        'std': std_val,
                        'n_subjects': len(region_features[feature_type])
                    })
            
            if plot_data:
                plot_df = pd.DataFrame(plot_data)
                plot_df = plot_df.sort_values('mean', ascending=False).head(10)  # Top 10 regions
                
                ax.bar(range(len(plot_df)), plot_df['mean'], yerr=plot_df['std'], capsize=5)
                ax.set_title(f'{feature_type.replace("_", " ").title()}')
                ax.set_xlabel('Brain Region (Top 10)')
                ax.set_ylabel('Feature Value')
                ax.set_xticks(range(len(plot_df)))
                ax.set_xticklabels([r.split('_')[-1] for r in plot_df['region']], rotation=45)
        
        # Remove empty subplot
        if len(feature_types) == 5:
            axes[1, 2].remove()
        
        plt.tight_layout()
        plt.show()
        
        return successful_subjects, all_features
    
    def plot_functional_features_robust(self, max_subjects=231):
        """Robust functional connectivity plotting"""
        
        functional_subjects = self.summary_df[
            self.summary_df['has_functional'] == True
        ]['subject_id'].tolist()[:max_subjects]
        
        print(f"📊 Processing {len(functional_subjects)} subjects for functional analysis")
        
        valid_matrices = []
        connectivity_values = []
        successful_subjects = []
        
        for subject_id in functional_subjects:
            try:
                func_file = self.processed_dir / str(subject_id) / 'functional_features.npy'
                if func_file.exists():
                    func_data = np.load(func_file, allow_pickle=True).item()
                    
                    if 'correlation_matrix' in func_data:
                        corr_matrix = func_data['correlation_matrix']
                        
                        # Check if matrix is valid
                        if (isinstance(corr_matrix, np.ndarray) and 
                            corr_matrix.shape[0] == corr_matrix.shape[1] and
                            corr_matrix.shape[0] > 5 and
                            not np.isnan(corr_matrix).all() and
                            not np.isinf(corr_matrix).any()):
                            
                            # Clean the matrix
                            corr_matrix_clean = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
                            
                            # Extract upper triangular
                            mask = np.triu(np.ones_like(corr_matrix_clean, dtype=bool), k=1)
                            conn_values = corr_matrix_clean[mask]
                            
                            # Only keep reasonable correlation values
                            valid_conn = conn_values[(conn_values >= -1) & (conn_values <= 1)]
                            
                            if len(valid_conn) > 0:
                                valid_matrices.append(corr_matrix_clean)
                                connectivity_values.extend(valid_conn)
                                successful_subjects.append(subject_id)
                    
            except Exception as e:
                print(f"   ⚠️ Error with subject {subject_id}: {e}")
                continue
        
        print(f"✅ Successfully loaded {len(successful_subjects)} functional datasets")
        
        if len(valid_matrices) == 0:
            print("❌ No valid functional connectivity matrices found")
            return
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🔗 Functional Connectivity (Robust Analysis)', fontsize=16, fontweight='bold')
        
        # 1. Average connectivity (only use matrices of same size)
        matrix_shapes = [m.shape for m in valid_matrices]
        from collections import Counter
        most_common_shape = Counter(matrix_shapes).most_common(1)[0][0]
        
        same_size_matrices = [m for m in valid_matrices if m.shape == most_common_shape]
        
        if same_size_matrices:
            avg_matrix = np.mean(same_size_matrices, axis=0)
            
            im = axes[0, 0].imshow(avg_matrix, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
            axes[0, 0].set_title(f'Average Connectivity Matrix\n({len(same_size_matrices)} subjects)')
            plt.colorbar(im, ax=axes[0, 0])
        
        # 2. Connectivity distribution
        if connectivity_values:
            axes[0, 1].hist(connectivity_values, bins=50, alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Connectivity Strength Distribution')
            axes[0, 1].set_xlabel('Correlation Coefficient')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.7)
        
        # 3. Matrix size distribution
        size_counts = Counter([f"{s[0]}x{s[1]}" for s in matrix_shapes])
        sizes, counts = zip(*size_counts.items())
        
        axes[1, 0].bar(sizes, counts)
        axes[1, 0].set_title('Connectivity Matrix Sizes')
        axes[1, 0].set_xlabel('Matrix Size')
        axes[1, 0].set_ylabel('Number of Subjects')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Connectivity strength statistics
        if same_size_matrices:
            subject_strengths = []
            for matrix in same_size_matrices:
                mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
                strength = np.mean(np.abs(matrix[mask]))
                subject_strengths.append(strength)
            
            axes[1, 1].hist(subject_strengths, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('Subject-wise Connectivity Strength')
            axes[1, 1].set_xlabel('Mean Absolute Connectivity')
            axes[1, 1].set_ylabel('Number of Subjects')
        
        plt.tight_layout()
        plt.show()
        
        # Summary
        if connectivity_values:
            print(f"\n🔗 FUNCTIONAL CONNECTIVITY SUMMARY:")
            print(f"   Valid subjects: {len(successful_subjects)}")
            print(f"   Most common matrix size: {most_common_shape}")
            print(f"   Average connectivity: {np.mean(connectivity_values):.3f}")
            print(f"   Connectivity std: {np.std(connectivity_values):.3f}")
            print(f"   Connectivity range: [{np.min(connectivity_values):.3f}, {np.max(connectivity_values):.3f}]")
        
        return successful_subjects, valid_matrices
    
    def plot_dti_features_robust(self, max_subjects=231):
        """Robust DTI feature plotting"""
        
        dti_subjects = self.summary_df[
            self.summary_df['has_dti'] == True
        ]['subject_id'].tolist()[:max_subjects]
        
        print(f"📊 Processing {len(dti_subjects)} subjects for DTI analysis")
        
        fa_global = []
        md_global = []
        fa_left = []
        fa_right = []
        wm_volumes = []
        successful_subjects = []
        
        for subject_id in dti_subjects:
            try:
                dti_file = self.processed_dir / str(subject_id) / 'dti_features.npy'
                if dti_file.exists():
                    dti_data = np.load(dti_file, allow_pickle=True).item()
                    
                    # Extract regional DTI
                    if 'regional_dti' in dti_data:
                        regional = dti_data['regional_dti']
                        
                        if 'global' in regional:
                            global_data = regional['global']
                            if 'fa_mean' in global_data and isinstance(global_data['fa_mean'], (int, float)):
                                if not np.isnan(global_data['fa_mean']) and 0 <= global_data['fa_mean'] <= 1:
                                    fa_global.append(global_data['fa_mean'])
                            
                            if 'md_mean' in global_data and isinstance(global_data['md_mean'], (int, float)):
                                if not np.isnan(global_data['md_mean']) and global_data['md_mean'] > 0:
                                    md_global.append(global_data['md_mean'])
                        
                        # Hemispheric FA
                        if 'left' in regional and 'fa_mean' in regional['left']:
                            fa_val = regional['left']['fa_mean']
                            if isinstance(fa_val, (int, float)) and not np.isnan(fa_val) and 0 <= fa_val <= 1:
                                fa_left.append(fa_val)
                        
                        if 'right' in regional and 'fa_mean' in regional['right']:
                            fa_val = regional['right']['fa_mean']
                            if isinstance(fa_val, (int, float)) and not np.isnan(fa_val) and 0 <= fa_val <= 1:
                                fa_right.append(fa_val)
                    
                    # White matter volume
                    if 'wm_features' in dti_data and 'wm_volume' in dti_data['wm_features']:
                        wm_vol = dti_data['wm_features']['wm_volume']
                        if isinstance(wm_vol, (int, float)) and not np.isnan(wm_vol) and wm_vol > 0:
                            wm_volumes.append(wm_vol)
                    
                    successful_subjects.append(subject_id)
                    
            except Exception as e:
                print(f"   ⚠️ Error with subject {subject_id}: {e}")
                continue
        
        print(f"✅ Successfully loaded {len(successful_subjects)} DTI datasets")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🧲 DTI White Matter Features (Robust Analysis)', fontsize=16, fontweight='bold')
        
        # 1. FA by region
        fa_data = []
        if fa_global:
            fa_data.extend([('Global', fa) for fa in fa_global])
        if fa_left:
            fa_data.extend([('Left', fa) for fa in fa_left])
        if fa_right:
            fa_data.extend([('Right', fa) for fa in fa_right])
        
        if fa_data:
            fa_df = pd.DataFrame(fa_data, columns=['Region', 'FA'])
            sns.boxplot(data=fa_df, x='Region', y='FA', ax=axes[0, 0])
            axes[0, 0].set_title('Fractional Anisotropy by Region')
            axes[0, 0].set_ylabel('FA Value')
        
        # 2. FA distribution
        if fa_global:
            axes[0, 1].hist(fa_global, bins=30, alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Global FA Distribution')
            axes[0, 1].set_xlabel('FA Value')
            axes[0, 1].set_ylabel('Frequency')
        
        # 3. MD distribution
        if md_global:
            axes[1, 0].hist(md_global, bins=30, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Global MD Distribution')
            axes[1, 0].set_xlabel('MD Value')
            axes[1, 0].set_ylabel('Frequency')
        
        # 4. WM Volume distribution
        if wm_volumes:
            axes[1, 1].hist(wm_volumes, bins=30, alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('White Matter Volume Distribution')
            axes[1, 1].set_xlabel('WM Volume (voxels)')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
        
        # Summary
        print(f"\n🧲 DTI SUMMARY:")
        print(f"   Valid subjects: {len(successful_subjects)}")
        if fa_global:
            print(f"   Global FA: {np.mean(fa_global):.3f} ± {np.std(fa_global):.3f}")
        if md_global:
            print(f"   Global MD: {np.mean(md_global):.6f} ± {np.std(md_global):.6f}")
        if wm_volumes:
            print(f"   WM Volume: {np.mean(wm_volumes):.0f} ± {np.std(wm_volumes):.0f} voxels")
        
        return successful_subjects
    
    def create_robust_report(self, max_subjects=231):
        """Create robust visualization report with better error handling"""
        
        print("🎨 Creating Robust Brain Feature Visualization Report...")
        print("=" * 70)
        
        # 1. Structural analysis
        print("\n1️⃣ Robust Structural Analysis...")
        struct_subjects, struct_features = self.plot_structural_features_robust(max_subjects)
        
        # 2. Functional analysis
        print("\n2️⃣ Robust Functional Analysis...")
        func_subjects, func_matrices = self.plot_functional_features_robust(max_subjects//2)
        
        # 3. DTI analysis
        print("\n3️⃣ Robust DTI Analysis...")
        dti_subjects = self.plot_dti_features_robust(max_subjects)
        
        # Summary
        print(f"\n✅ ROBUST ANALYSIS COMPLETE!")
        print(f"   Structural: {len(struct_subjects) if struct_subjects else 0} subjects")
        print(f"   Functional: {len(func_subjects) if func_subjects else 0} subjects")
        print(f"   DTI: {len(dti_subjects) if dti_subjects else 0} subjects")

if __name__ == "__main__":
    # Create robust visualizer
    viz = RobustBrainVisualizer()
    
    # Create robust report
    viz.create_robust_report(max_subjects=231)
    
    print("\n💡 Debug tools available:")
    print("   viz.debug_subject_data('29114')  # Debug specific subject")
    print("   viz.debug_subject_data('29114', 'functional')  # Debug specific modality")