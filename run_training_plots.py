#!/usr/bin/env python3
"""
Script to generate training and validation plots for Alzheimer's Detection Research Paper
Run this script to create publication-quality graphs
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from training_plots import create_training_plots, create_learning_curves_analysis, create_model_performance_summary

def main():
    """
    Main function to generate all training plots
    """
    print("=" * 60)
    print("ALZHEIMER'S DISEASE DETECTION - TRAINING PLOTS GENERATOR")
    print("=" * 60)
    
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    print("\n1. Generating Training and Validation Plots...")
    try:
        # Generate main training plots
        fig1 = create_training_plots(save_path='results/training_validation_plots.png')
        print("   ✅ Training and validation plots saved to: results/training_validation_plots.png")
    except Exception as e:
        print(f"   ❌ Error generating training plots: {e}")
    
    print("\n2. Generating Learning Curves Analysis...")
    try:
        # Generate learning curves analysis
        fig2 = create_learning_curves_analysis(save_path='results/learning_curves_analysis.png')
        print("   ✅ Learning curves analysis saved to: results/learning_curves_analysis.png")
    except Exception as e:
        print(f"   ❌ Error generating learning curves: {e}")
    
    print("\n3. Generating Performance Summary Table...")
    try:
        # Generate performance summary
        fig3 = create_model_performance_summary()
        print("   ✅ Performance summary saved to: model_performance_summary.png")
    except Exception as e:
        print(f"   ❌ Error generating performance summary: {e}")
    
    print("\n" + "=" * 60)
    print("PLOT GENERATION COMPLETE!")
    print("=" * 60)
    print("\nGenerated Files:")
    print("📊 results/training_validation_plots.png - Main accuracy/loss plots")
    print("📈 results/learning_curves_analysis.png - Learning curves with convergence analysis")
    print("📋 model_performance_summary.png - Performance metrics table")
    print("\nThese plots are ready for your research paper!")
    
    # Show the plots
    plt.show()

if __name__ == "__main__":
    main() 