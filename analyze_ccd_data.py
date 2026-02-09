#!/usr/bin/env python3
"""
Beta-Carotene CCD Data Analysis Script
=======================================
Comprehensive analysis of Central Composite Design experimental data
for H. pluvialis beta-carotene optimization.

Authors: S. Shakthi Bhavanee, T. P. Rajarajan
Institution: St. Peter's College of Engineering and Technology
Date: 2024-2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class BetaCaroteneAnalyzer:
    """
    Comprehensive analyzer for beta-carotene CCD experimental data
    """
    
    def __init__(self, filepath):
        """Initialize with data file path"""
        self.df = pd.read_csv(filepath)
        self.features = [col for col in self.df.columns if col != 'BetacaroteneDCM']
        self.target = 'BetacaroteneDCM'
        
    def data_summary(self):
        """Print comprehensive data summary"""
        print("="*70)
        print("BETA-CAROTENE CCD DATA ANALYSIS SUMMARY")
        print("="*70)
        print(f"\nDataset Shape: {self.df.shape[0]} experiments × {self.df.shape[1]} variables")
        print(f"Target Variable: {self.target}")
        print(f"Number of Features: {len(self.features)}")
        
        print("\n" + "-"*70)
        print("EXPERIMENTAL DESIGN STRUCTURE")
        print("-"*70)
        
        # Identify design points
        factorial_points = self.df[
            self.df[self.features].isin([0, 1, -1]).all(axis=1)
        ]
        axial_points = self.df[
            (self.df[self.features].abs() > 1.5).any(axis=1)
        ]
        center_points = self.df[
            (self.df[self.features] == 0.5).all(axis=1)
        ]
        
        print(f"Factorial Points (±1):      {len(factorial_points)}")
        print(f"Axial Points (±α=1.69):     {len(axial_points)}")
        print(f"Center Points (0):          {len(center_points)}")
        print(f"Total Experiments:          {len(self.df)}")
        
        print("\n" + "-"*70)
        print("RESPONSE VARIABLE STATISTICS")
        print("-"*70)
        
        stats_df = self.df[self.target].describe()
        print(f"\nBeta-Carotene (DCM Extract) [mg/L]:")
        print(f"  Mean:        {stats_df['mean']:.3f} mg/L")
        print(f"  Std Dev:     {stats_df['std']:.3f} mg/L")
        print(f"  Min:         {stats_df['min']:.3f} mg/L")
        print(f"  Max:         {stats_df['max']:.3f} mg/L")
        print(f"  Range:       {stats_df['max'] - stats_df['min']:.3f} mg/L")
        print(f"  Median:      {stats_df['50%']:.3f} mg/L")
        print(f"  CV:          {(stats_df['std']/stats_df['mean']*100):.2f}%")
        
        return stats_df
    
    def correlation_analysis(self):
        """Analyze correlations between features and target"""
        print("\n" + "="*70)
        print("CORRELATION ANALYSIS")
        print("="*70)
        
        correlations = self.df[self.features].corrwith(self.df[self.target])
        correlations = correlations.sort_values(ascending=False)
        
        print("\nPearson Correlation with Beta-Carotene Yield:")
        print("-"*70)
        for feature, corr in correlations.items():
            significance = "***" if abs(corr) > 0.5 else "**" if abs(corr) > 0.3 else "*" if abs(corr) > 0.1 else ""
            print(f"  {feature:20s}: {corr:7.4f}  {significance}")
        
        print("\n*** |r| > 0.5 (Strong), ** |r| > 0.3 (Moderate), * |r| > 0.1 (Weak)")
        
        return correlations
    
    def feature_importance_analysis(self):
        """Calculate feature importance using linear regression"""
        print("\n" + "="*70)
        print("FEATURE IMPORTANCE (LINEAR REGRESSION)")
        print("="*70)
        
        X = self.df[self.features]
        y = self.df[self.target]
        
        # Standardize features for fair comparison
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit model
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Get coefficients
        importance_df = pd.DataFrame({
            'Feature': self.features,
            'Coefficient': model.coef_,
            'Abs_Coefficient': np.abs(model.coef_)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        print("\nStandardized Regression Coefficients:")
        print("-"*70)
        print(f"{'Feature':<20} {'Coefficient':>12} {'Effect':>10}")
        print("-"*70)
        
        for _, row in importance_df.iterrows():
            effect = "Positive" if row['Coefficient'] > 0 else "Negative"
            print(f"{row['Feature']:<20} {row['Coefficient']:>12.4f} {effect:>10}")
        
        # Model performance
        y_pred = model.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        print(f"\nModel Performance:")
        print(f"  R² Score:    {r2:.4f}")
        print(f"  MAE:         {mae:.4f} mg/L")
        print(f"  RMSE:        {rmse:.4f} mg/L")
        
        return importance_df, model
    
    def identify_optimal_conditions(self):
        """Identify experiments with highest beta-carotene yield"""
        print("\n" + "="*70)
        print("TOP PERFORMING EXPERIMENTS")
        print("="*70)
        
        top_n = 5
        top_experiments = self.df.nlargest(top_n, self.target)
        
        print(f"\nTop {top_n} Experiments by Beta-Carotene Yield:")
        print("-"*70)
        
        for idx, (i, row) in enumerate(top_experiments.iterrows(), 1):
            print(f"\n{idx}. Experiment {i+1}: {row[self.target]:.3f} mg/L")
            print("   Active Factors:")
            for feature in self.features:
                if row[feature] > 0.5:  # Positive level
                    print(f"      • {feature}: {row[feature]:.2f}")
        
        # Statistical comparison with mean
        mean_yield = self.df[self.target].mean()
        top_mean = top_experiments[self.target].mean()
        improvement = ((top_mean - mean_yield) / mean_yield) * 100
        
        print(f"\nAverage of Top {top_n}:     {top_mean:.3f} mg/L")
        print(f"Overall Average:      {mean_yield:.3f} mg/L")
        print(f"Improvement:          {improvement:.1f}%")
        
        return top_experiments
    
    def generate_visualizations(self, save_path='./'):
        """Generate comprehensive visualizations"""
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        
        # 1. Distribution of Beta-Carotene Yield
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Histogram
        axes[0, 0].hist(self.df[self.target], bins=15, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(self.df[self.target].mean(), color='red', 
                          linestyle='--', linewidth=2, label='Mean')
        axes[0, 0].set_xlabel('β-Carotene (mg/L)', fontsize=11)
        axes[0, 0].set_ylabel('Frequency', fontsize=11)
        axes[0, 0].set_title('Distribution of β-Carotene Yield', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Box plot
        axes[0, 1].boxplot(self.df[self.target], vert=True)
        axes[0, 1].set_ylabel('β-Carotene (mg/L)', fontsize=11)
        axes[0, 1].set_title('Box Plot: Yield Variability', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Correlation heatmap
        correlations = self.df[self.features].corrwith(self.df[self.target])
        colors = ['red' if x < 0 else 'green' for x in correlations.values]
        axes[1, 0].barh(range(len(correlations)), correlations.values, color=colors, alpha=0.7)
        axes[1, 0].set_yticks(range(len(correlations)))
        axes[1, 0].set_yticklabels(correlations.index, fontsize=9)
        axes[1, 0].set_xlabel('Correlation Coefficient', fontsize=11)
        axes[1, 0].set_title('Feature Correlations with β-Carotene', 
                            fontsize=12, fontweight='bold')
        axes[1, 0].axvline(0, color='black', linewidth=0.8)
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # Predicted vs Actual (from linear model)
        X = self.df[self.features]
        y = self.df[self.target]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LinearRegression()
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        
        axes[1, 1].scatter(y, y_pred, alpha=0.6, s=50)
        axes[1, 1].plot([y.min(), y.max()], [y.min(), y.max()], 
                       'r--', linewidth=2, label='Perfect Fit')
        axes[1, 1].set_xlabel('Actual β-Carotene (mg/L)', fontsize=11)
        axes[1, 1].set_ylabel('Predicted β-Carotene (mg/L)', fontsize=11)
        axes[1, 1].set_title(f'Predicted vs Actual (R²={r2_score(y, y_pred):.3f})', 
                            fontsize=12, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Beta-Carotene CCD Analysis Dashboard', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f'{save_path}analysis_dashboard.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: analysis_dashboard.png")
        plt.close()
        
        # 2. Feature Importance Bar Chart
        importance_df, _ = self.feature_importance_analysis()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors_imp = ['green' if x > 0 else 'red' for x in importance_df['Coefficient']]
        bars = ax.barh(importance_df['Feature'], importance_df['Coefficient'], 
                      color=colors_imp, alpha=0.7)
        ax.set_xlabel('Standardized Coefficient', fontsize=12)
        ax.set_title('Feature Importance for β-Carotene Production', 
                    fontsize=14, fontweight='bold')
        ax.axvline(0, color='black', linewidth=1)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, importance_df['Coefficient'])):
            ax.text(val, i, f' {val:.3f}', va='center', 
                   ha='left' if val > 0 else 'right', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}feature_importance.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: feature_importance.png")
        plt.close()
        
        print("\n✓ All visualizations generated successfully!")
    
    def export_results(self, filepath='analysis_results.txt'):
        """Export analysis results to text file"""
        import sys
        from io import StringIO
        
        # Capture print output
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        
        # Run all analyses
        self.data_summary()
        self.correlation_analysis()
        self.feature_importance_analysis()
        self.identify_optimal_conditions()
        
        # Get output
        output = mystdout.getvalue()
        sys.stdout = old_stdout
        
        # Write to file
        with open(filepath, 'w') as f:
            f.write(output)
        
        print(f"\n✓ Results exported to: {filepath}")
        
        return output


def main():
    """Main execution function"""
    print("\n" + "="*70)
    print(" BETA-CAROTENE OPTIMIZATION ANALYSIS")
    print(" H. pluvialis - Central Composite Design")
    print("="*70 + "\n")
    
    # Initialize analyzer
    filepath = "CCD_PredictionFinal_DCM.csv"
    analyzer = BetaCaroteneAnalyzer(filepath)
    
    # Run complete analysis
    print("Running comprehensive analysis...\n")
    
    # 1. Data Summary
    analyzer.data_summary()
    
    # 2. Correlation Analysis
    analyzer.correlation_analysis()
    
    # 3. Feature Importance
    analyzer.feature_importance_analysis()
    
    # 4. Optimal Conditions
    analyzer.identify_optimal_conditions()
    
    # 5. Generate Visualizations
    analyzer.generate_visualizations()
    
    # 6. Export Results
    analyzer.export_results('CCD_analysis_results.txt')
    
    print("\n" + "="*70)
    print(" ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated Files:")
    print("  • analysis_dashboard.png")
    print("  • feature_importance.png")
    print("  • CCD_analysis_results.txt")
    print("\n")


if __name__ == "__main__":
    main()
