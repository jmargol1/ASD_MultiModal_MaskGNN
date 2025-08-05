#!/usr/bin/env python3
"""
Fixed ABIDE Data Visualization and Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class FixedABIDEViz:
    def __init__(self, processed_dir="./abide_speed_optimized_features"):
        self.processed_dir = Path(processed_dir)
        self.summary_df = None
        self.load_data()
    
    def load_data(self):
        """Load or generate summary data"""
        summary_file = self.processed_dir / 'comprehensive_summary.csv'
        
        if summary_file.exists():
            self.summary_df = pd.read_csv(summary_file)
            logger.info(f"✅ Loaded summary: {len(self.summary_df)} subjects")
        else:
            logger.warning("Summary file not found. Generating from processed data...")
            self.summary_df = generate_summary_from_processed_data(str(self.processed_dir))
        
        if self.summary_df.empty:
            logger.error("No data found!")
            return
        
        logger.info(f"📊 Columns: {list(self.summary_df.columns)}")
    
    def show_dataset_overview(self):
        """Show dataset overview with safety checks"""
        if self.summary_df is None or self.summary_df.empty:
            print("❌ No data available for overview")
            return {}
        
        total = len(self.summary_df)
        if total == 0:
            print("❌ No subjects found")
            return {}
        
        structural = self.summary_df['has_structural'].sum()
        functional = self.summary_df['has_functional'].sum()
        dti = self.summary_df['has_dti'].sum()
        complete = self.summary_df['complete_multimodal'].sum()
        
        print("📈 DATASET STATISTICS:")
        print(f"   Total subjects: {total}")
        print(f"   Structural data: {structural} ({structural/total*100:.1f}%)")
        print(f"   Functional data: {functional} ({functional/total*100:.1f}%)")
        print(f"   DTI data: {dti} ({dti/total*100:.1f}%)")
        print(f"   Complete multimodal: {complete} ({complete/total*100:.1f}%)")
        
        return {
            'total': total,
            'structural': structural,
            'functional': functional,
            'dti': dti,
            'complete': complete
        }
    
    def quick_analysis(self):
        """Quick analysis with error handling"""
        print("🚀 Starting Quick Analysis...")
        print("="*70)
        print("🧠 ABIDE DATASET OVERVIEW")
        print("="*70)
        
        summary = self.show_dataset_overview()
        
        if not summary:
            return {}, {}
        
        # Site breakdown
        if 'site' in self.summary_df.columns:
            print("\n🌍 SITE BREAKDOWN:")
            site_stats = self.summary_df.groupby('site').agg({
                'has_structural': 'sum',
                'has_functional': 'sum', 
                'has_dti': 'sum',
                'complete_multimodal': 'sum'
            })
            site_counts = self.summary_df['site'].value_counts()
            
            for site in site_stats.index:
                count = site_counts[site]
                complete = site_stats.loc[site, 'complete_multimodal']
                print(f"   {site}: {count} subjects ({complete} complete)")
        
        # Contrast breakdown
        if 'contrast_type' in self.summary_df.columns:
            print("\n🔬 CONTRAST TYPE BREAKDOWN:")
            contrast_counts = self.summary_df['contrast_type'].value_counts()
            for contrast, count in contrast_counts.items():
                percentage = (count / len(self.summary_df)) * 100
                print(f"   {contrast}: {count} subjects ({percentage:.1f}%)")
        
        print("="*70)
        
        return summary, self.summary_df

if __name__ == "__main__":
    # Import the summary generator
    from pathlib import Path
    import sys
    
    # Run the fixed analysis
    viz = FixedABIDEViz()
    summary, sample_data = viz.quick_analysis()
    
    print("\n🎯 Analysis complete!")
