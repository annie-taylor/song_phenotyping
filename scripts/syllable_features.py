"""
Syllable Feature Analysis Module

Comprehensive tools for analyzing acoustic features in bird syllable databases.
Includes exploratory analysis, statistical testing, visualization, and feature selection.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew, kurtosis
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from syllable_database import SyllableDatabase


class SyllableFeatureAnalyzer:
    """
    Main class for analyzing syllable acoustic features and their relationship to manual labels.
    """

    def __init__(self, bird_path: str, exclude_labels: List[str] = None):
        """
        Initialize the analyzer.

        Args:
            bird_path: Path to bird directory
            exclude_labels: Manual labels to exclude from analysis (default: ['None'])
        """
        self.bird_path = Path(bird_path)
        self.bird_name = self.bird_path.name
        self.exclude_labels = exclude_labels or ['None']

        # Load database
        self.db = SyllableDatabase(bird_path)
        self.df = self.db.load_database()

        if self.df.empty:
            raise ValueError(f"No syllable database found for {bird_path}")

        # Get acoustic features (excluding metadata and list-based features)
        self.acoustic_features = self._identify_acoustic_features()

        logging.info(f"Initialized analyzer for {self.bird_name}: "
                     f"{len(self.df)} syllables, {len(self.acoustic_features)} acoustic features")

    def _identify_acoustic_features(self) -> List[str]:
        """Identify scalar acoustic features suitable for analysis."""
        exclude_cols = {
            'bird_name', 'hash_id', 'song_file', 'syllable_index', 'manual_label',
            'clustering_labels', 'position_in_song', 'song_length_syllables',
            'start_time_ms', 'end_time_ms', 'mfcc_means', 'mfcc_stds'
        }

        # Get columns that don't start with 'cluster_' and aren't in exclude list
        candidate_features = [
            col for col in self.df.columns
            if col not in exclude_cols and not col.startswith('cluster_')
        ]

        # Filter out list-based features
        scalar_features = []
        for feature in candidate_features:
            if feature in self.df.columns:
                sample_values = self.df[feature].dropna().head(10)
                if len(sample_values) > 0:
                    # Check if any values are lists/arrays
                    is_list_feature = any(isinstance(val, (list, np.ndarray)) for val in sample_values)
                    if not is_list_feature:
                        scalar_features.append(feature)

        return scalar_features

    def explore_database_structure(self) -> Dict[str, Any]:
        """Get comprehensive overview of database structure."""
        print("=== Database Overview ===")
        print(f"Bird: {self.bird_name}")
        print(f"Total syllables: {len(self.df)}")
        print(f"Total columns: {len(self.df.columns)}")

        # Manual labels analysis
        manual_labels = self.df['manual_label'].value_counts()
        print(f"\nManual label distribution:")
        for label, count in manual_labels.items():
            print(f"  {label}: {count} syllables ({count / len(self.df) * 100:.1f}%)")

        print(f"\nAcoustic features available: {len(self.acoustic_features)}")

        # Check for missing data
        missing_data = self.df[self.acoustic_features].isnull().sum()
        features_with_missing = missing_data[missing_data > 0]
        if len(features_with_missing) > 0:
            print(f"\nFeatures with missing data:")
            for feature, count in features_with_missing.items():
                print(f"  {feature}: {count} missing ({count / len(self.df) * 100:.1f}%)")
        else:
            print("\nNo missing data in acoustic features!")

        return {
            'bird_name': self.bird_name,
            'total_syllables': len(self.df),
            'manual_labels': manual_labels.to_dict(),
            'acoustic_features': self.acoustic_features,
            'n_acoustic_features': len(self.acoustic_features),
            'missing_data': features_with_missing.to_dict() if len(features_with_missing) > 0 else {}
        }

    def analyze_single_feature(self, feature: str, figsize: Tuple[int, int] = (12, 8)) -> pd.DataFrame:
        """
        Analyze distribution of a single feature across manual labels.

        Args:
            feature: Feature column name to analyze
            figsize: Figure size for plots

        Returns:
            DataFrame with summary statistics by label
        """
        # Filter data
        analysis_df = self.df[~self.df['manual_label'].isin(self.exclude_labels)].copy()
        analysis_df = analysis_df.dropna(subset=[feature])

        if analysis_df.empty:
            print(f"No valid data for feature {feature}")
            return pd.DataFrame()

        print(f"\n=== Analysis for {feature} ===")
        print(f"Valid syllables: {len(analysis_df)}")

        # Summary statistics by label
        summary_stats = analysis_df.groupby('manual_label')[feature].agg([
            'count', 'mean', 'std', 'min', 'max', 'median'
        ]).round(3)
        print(f"\nSummary statistics by manual label:")
        print(summary_stats)

        # Statistical tests (ANOVA)
        labels = analysis_df['manual_label'].unique()
        if len(labels) > 1:
            groups = [analysis_df[analysis_df['manual_label'] == label][feature].values
                      for label in labels]
            groups = [group for group in groups if len(group) > 0]

            if len(groups) > 1:
                f_stat, p_value = stats.f_oneway(*groups)
                print(f"\nANOVA F-statistic: {f_stat:.3f}, p-value: {p_value:.6f}")
                if p_value < 0.001:
                    print("*** Highly significant differences between labels")
                elif p_value < 0.01:
                    print("** Significant differences between labels")
                elif p_value < 0.05:
                    print("* Marginally significant differences between labels")
                else:
                    print("No significant differences between labels")

        # Create visualizations
        self._plot_single_feature_analysis(analysis_df, feature, summary_stats, figsize)

        return summary_stats


    def _plot_single_feature_analysis(self, analysis_df: pd.DataFrame, feature: str,
                                      summary_stats: pd.DataFrame, figsize: Tuple[int, int]):
        """Create comprehensive plots for single feature analysis."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Feature Analysis: {feature}', fontsize=16)

        labels = analysis_df['manual_label'].unique()

        # 1. Box plot
        sns.boxplot(data=analysis_df, x='manual_label', y=feature, ax=axes[0, 0])
        axes[0, 0].set_title('Distribution by Manual Label')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # 2. Violin plot
        sns.violinplot(data=analysis_df, x='manual_label', y=feature, ax=axes[0, 1])
        axes[0, 1].set_title('Density Distribution by Manual Label')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # 3. Histogram with overlays
        for label in labels:
            label_data = analysis_df[analysis_df['manual_label'] == label][feature]
            axes[1, 0].hist(label_data, alpha=0.6, label=f'{label} (n={len(label_data)})', bins=20)
        axes[1, 0].set_xlabel(feature)
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Overlapping Histograms')
        axes[1, 0].legend()

        # 4. Mean values with error bars
        means = summary_stats['mean']
        stds = summary_stats['std']
        axes[1, 1].bar(range(len(means)), means.values, yerr=stds.values,
                       capsize=5, alpha=0.7, color='skyblue')
        axes[1, 1].set_xticks(range(len(means)))
        axes[1, 1].set_xticklabels(means.index, rotation=45)
        axes[1, 1].set_ylabel(feature)
        axes[1, 1].set_title('Mean ± SD by Label')

        plt.tight_layout()
        plt.show()


    def analyze_all_features(self, save_plots: bool = False,
                             plot_dir: str = 'feature_plots') -> pd.DataFrame:
        """
        Analyze all features and identify which show strongest differences between labels.

        Args:
            save_plots: Whether to save individual plots
            plot_dir: Directory to save plots

        Returns:
            DataFrame with analysis results for all features
        """
        analysis_df = self.df[~self.df['manual_label'].isin(self.exclude_labels)].copy()

        if save_plots:
            os.makedirs(plot_dir, exist_ok=True)

        results = []
        print("Analyzing all features...")

        for feature in tqdm(self.acoustic_features):
            try:
                # Remove NaN values
                feature_df = analysis_df.dropna(subset=[feature])

                if len(feature_df) == 0:
                    continue

                # Get unique labels for this feature
                labels = feature_df['manual_label'].unique()

                if len(labels) < 2:
                    continue

                # ANOVA test
                groups = [feature_df[feature_df['manual_label'] == label][feature].values
                          for label in labels]
                groups = [group for group in groups if len(group) > 0]

                if len(groups) < 2:
                    continue

                f_stat, p_value = stats.f_oneway(*groups)

                # Calculate effect size (eta-squared)
                overall_mean = feature_df[feature].mean()
                ss_total = ((feature_df[feature] - overall_mean) ** 2).sum()

                ss_between = 0
                for label in labels:
                    group_data = feature_df[feature_df['manual_label'] == label][feature]
                    if len(group_data) > 0:
                        group_mean = group_data.mean()
                        ss_between += len(group_data) * (group_mean - overall_mean) ** 2

                eta_squared = ss_between / ss_total if ss_total > 0 else 0

                # Summary statistics
                summary_stats = feature_df.groupby('manual_label')[feature].agg([
                    'count', 'mean', 'std'
                ])

                results.append({
                    'feature': feature,
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'eta_squared': eta_squared,
                    'n_syllables': len(feature_df),
                    'n_labels': len(labels),
                    'mean_by_label': summary_stats['mean'].to_dict(),
                    'std_by_label': summary_stats['std'].to_dict()
                })

                # Create and save plot if requested
                if save_plots:
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(data=feature_df, x='manual_label', y=feature)
                    plt.title(f'{feature}\nF={f_stat:.3f}, p={p_value:.6f}, η²={eta_squared:.3f}')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(os.path.join(plot_dir, f'{feature}_boxplot.png'),
                                dpi=300, bbox_inches='tight')
                    plt.close()

            except Exception as e:
                logging.warning(f"Error analyzing {feature}: {e}")
                continue

        # Convert to DataFrame and sort by effect size
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('eta_squared', ascending=False)

        return results_df


    def create_feature_overview_heatmap(self, features: List[str] = None,
                                        figsize: Tuple[int, int] = (15, 10)) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create heatmap showing mean values of features by manual label.

        Args:
            features: List of features to include (None = all acoustic features)
            figsize: Figure size

        Returns:
            Tuple of (feature_means, standardized_feature_means)
        """
        # Filter data
        analysis_df = self.df[~self.df['manual_label'].isin(self.exclude_labels)].copy()

        if features is None:
            features = self.acoustic_features

        # Calculate mean values by label
        feature_means = analysis_df.groupby('manual_label')[features].mean()

        # Standardize features (z-score) for better visualization
        feature_means_standardized = feature_means.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

        # Create heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(feature_means_standardized.T,
                    annot=False,
                    cmap='RdBu_r',
                    center=0,
                    cbar_kws={'label': 'Standardized Mean Value'})

        plt.title('Feature Means by Manual Label (Standardized)', fontsize=16)
        plt.xlabel('Manual Label')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.show()

        return feature_means, feature_means_standardized


    def analyze_mfcc_features(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Special analysis for MFCC features which are stored as lists.

        Returns:
            Tuple of (mfcc_results, mfcc_means_df, mfcc_stds_df)
        """
        # Filter data
        analysis_df = self.df[~self.df['manual_label'].isin(self.exclude_labels)].copy()
        analysis_df = analysis_df.dropna(subset=['mfcc_means', 'mfcc_stds'])

        if analysis_df.empty:
            print("No MFCC data available")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # Convert MFCC lists to separate columns
        mfcc_means_df = pd.DataFrame(analysis_df['mfcc_means'].tolist(),
                                     columns=[f'mfcc_mean_{i + 1}' for i in range(13)])
        mfcc_stds_df = pd.DataFrame(analysis_df['mfcc_stds'].tolist(),
                                    columns=[f'mfcc_std_{i + 1}' for i in range(13)])

        # Add manual labels
        mfcc_means_df['manual_label'] = analysis_df['manual_label'].values
        mfcc_stds_df['manual_label'] = analysis_df['manual_label'].values

        # Analyze each MFCC coefficient
        print("=== MFCC Means Analysis ===")
        mfcc_results = []

        for i in range(1, 14):
            feature = f'mfcc_mean_{i}'

            # ANOVA test
            labels = mfcc_means_df['manual_label'].unique()
            groups = [mfcc_means_df[mfcc_means_df['manual_label'] == label][feature].values
                      for label in labels]

            f_stat, p_value = stats.f_oneway(*groups)

            # Effect size
            overall_mean = mfcc_means_df[feature].mean()
            ss_total = ((mfcc_means_df[feature] - overall_mean) ** 2).sum()

            ss_between = 0
            for label in labels:
                group_data = mfcc_means_df[mfcc_means_df['manual_label'] == label][feature]
                group_mean = group_data.mean()
                ss_between += len(group_data) * (group_mean - overall_mean) ** 2

            eta_squared = ss_between / ss_total if ss_total > 0 else 0

            mfcc_results.append({
                'coefficient': i,
                'f_statistic': f_stat,
                'p_value': p_value,
                'eta_squared': eta_squared
            })

            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"MFCC {i:2d}: F={f_stat:6.3f}, p={p_value:.6f}, η²={eta_squared:.3f} {significance}")

        # Create visualization
        self._plot_mfcc_analysis(mfcc_results, mfcc_means_df)

        mfcc_results_df = pd.DataFrame(mfcc_results)
        return mfcc_results_df, mfcc_means_df, mfcc_stds_df


    def _plot_mfcc_analysis(self, mfcc_results: List[Dict], mfcc_means_df: pd.DataFrame):
        """Create MFCC analysis visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. MFCC means heatmap
        mfcc_features = [f'mfcc_mean_{i + 1}' for i in range(13)]
        mfcc_means_by_label = mfcc_means_df.groupby('manual_label')[mfcc_features].mean()
        sns.heatmap(mfcc_means_by_label.T, annot=True, fmt='.2f', cmap='RdBu_r', ax=axes[0, 0])
        axes[0, 0].set_title('MFCC Means by Manual Label')
        axes[0, 0].set_xlabel('Manual Label')
        axes[0, 0].set_ylabel('MFCC Coefficient')

        # 2. Effect sizes
        mfcc_results_df = pd.DataFrame(mfcc_results)
        axes[0, 1].bar(mfcc_results_df['coefficient'], mfcc_results_df['eta_squared'])
        axes[0, 1].set_title('MFCC Discrimination Power (η²)')
        axes[0, 1].set_xlabel('MFCC Coefficient')
        axes[0, 1].set_ylabel('Effect Size (η²)')

        # 3. P-values
        axes[1, 0].bar(mfcc_results_df['coefficient'], -np.log10(mfcc_results_df['p_value']))
        axes[1, 0].axhline(y=-np.log10(0.05), color='r', linestyle='--', label='p=0.05')
        axes[1, 0].axhline(y=-np.log10(0.01), color='orange', linestyle='--', label='p=0.01')
        axes[1, 0].axhline(y=-np.log10(0.001), color='red', linestyle='--', label='p=0.001')
        axes[1, 0].set_title('MFCC Statistical Significance')
        axes[1, 0].set_xlabel('MFCC Coefficient')
        axes[1, 0].set_ylabel('-log10(p-value)')
        axes[1, 0].legend()

        # 4. Box plot of most discriminative MFCC
        best_mfcc_idx = mfcc_results_df.loc[mfcc_results_df['eta_squared'].idxmax(), 'coefficient']
        best_mfcc_feature = f'mfcc_mean_{best_mfcc_idx}'
        sns.boxplot(data=mfcc_means_df, x='manual_label', y=best_mfcc_feature, ax=axes[1, 1])
        axes[1, 1].set_title(f'Most Discriminative: MFCC {best_mfcc_idx}')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()


    def apply_multiple_comparisons_correction(self, feature_results: pd.DataFrame,
                                              method: str = 'fdr_bh') -> pd.DataFrame:
        """
        Apply multiple comparisons correction to feature analysis results.

        Args:
            feature_results: DataFrame from analyze_all_features()
            method: Correction method ('fdr_bh', 'bonferroni', etc.)

        Returns:
            DataFrame with corrected p-values
        """
        if len(feature_results) == 0:
            return feature_results

        rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
            feature_results['p_value'], method=method
        )

        feature_results = feature_results.copy()
        feature_results['p_corrected'] = p_corrected
        feature_results['significant_corrected'] = rejected

        print(f"Multiple comparisons correction ({method}):")
        print(f" Original significant features (p<0.05): {sum(feature_results['p_value'] < 0.05)}")
        print(f" Corrected significant features: {sum(rejected)}")
        print(f" Correction factor: {len(feature_results)} tests")

        return feature_results.sort_values('p_corrected')


    def analyze_feature_correlations(self, features: List[str] = None,
                                     correlation_threshold: float = 0.8) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Analyze correlations between features and identify redundant ones.

        Args:
            features: List of features to analyze (None = all acoustic features)
            correlation_threshold: Threshold for identifying high correlations

        Returns:
            Tuple of (correlation_matrix, high_correlation_pairs)
        """
        if features is None:
            features = self.acoustic_features

        # Filter data
        analysis_df = self.df[~self.df['manual_label'].isin(self.exclude_labels)].copy()

        # Only keep numeric features
        numeric_features = []
        for feature in features:
            if feature in analysis_df.columns:
                sample_values = analysis_df[feature].dropna().head(10)
                if len(sample_values) > 0:
                    is_list_feature = any(isinstance(val, (list, np.ndarray)) for val in sample_values)
                    if not is_list_feature:
                        numeric_features.append(feature)

        if len(numeric_features) == 0:
            print("No numeric features available for correlation analysis")
            return pd.DataFrame(), []

        print(f"Analyzing correlations for {len(numeric_features)} numeric features")

        # Calculate correlation matrix
        try:
            corr_matrix = analysis_df[numeric_features].corr()
        except Exception as e:
            print(f"Error calculating correlation matrix: {e}")
            return pd.DataFrame(), []

        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if not np.isnan(corr_val) and corr_val > correlation_threshold:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })

        # Visualization
        self._plot_correlation_analysis(corr_matrix, high_corr_pairs, correlation_threshold)

        print(f"Found {len(high_corr_pairs)} feature pairs with correlation > {correlation_threshold}")

        return corr_matrix, high_corr_pairs


    def _plot_correlation_analysis(self, corr_matrix: pd.DataFrame,
                                   high_corr_pairs: List[Dict], threshold: float):
        """Create correlation analysis visualization."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Correlation heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r',
                    center=0, ax=axes[0])
        axes[0].set_title(f'Feature Correlation Matrix ({len(corr_matrix)} features)')

        # High correlation pairs
        if high_corr_pairs:
            high_corr_df = pd.DataFrame(high_corr_pairs)
            high_corr_df = high_corr_df.sort_values('correlation', ascending=False)

            # Limit to top 15 for readability
            top_pairs = high_corr_df.head(15)
            y_pos = range(len(top_pairs))
            axes[1].barh(y_pos, top_pairs['correlation'])
            axes[1].set_yticks(y_pos)
            axes[1].set_yticklabels([f"{row['feature1'][:15]}\nvs\n{row['feature2'][:15]}"
                                     for _, row in top_pairs.iterrows()], fontsize=8)
            axes[1].set_xlabel('Absolute Correlation')
            axes[1].set_title(f'Top High Correlations (r>{threshold})')
            axes[1].axvline(x=threshold, color='red', linestyle='--')
        else:
            axes[1].text(0.5, 0.5, f'No correlations > {threshold}',
                         ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('High Correlations (None Found)')

        plt.tight_layout()
        plt.show()


    def compare_manual_vs_clustering(self, clustering_method: str = None) -> Dict[str, Any]:
        """
        Compare how well manual labels vs automatic clustering separate acoustic space.

        Args:
            clustering_method: Name of clustering column to compare (auto-detected if None)

        Returns:
            Dictionary with comparison results
        """
        # Find clustering method if not specified
        if clustering_method is None:
            clustering_cols = [col for col in self.df.columns if col.startswith('cluster_rank')]
            if not clustering_cols:
                print("No clustering methods found")
                return {}
            clustering_method = clustering_cols[0]  # Use rank 0 (best) method

        print(f"\n=== MANUAL vs AUTOMATIC CLUSTERING COMPARISON ===")
        print(f"Automatic method: {clustering_method}")

        # Filter data
        comparison_df = self.df[
            (self.df['manual_label'].notna()) &
            (~self.df['manual_label'].isin(self.exclude_labels)) &
            (self.df[clustering_method].notna()) &
            (self.df[clustering_method] != -1)
            ].copy()

        if len(comparison_df) == 0:
            print("No data available for comparison")
            return {}

        print(f"Comparing {len(comparison_df)} syllables")
        print(f"Manual labels: {sorted(comparison_df['manual_label'].unique())}")
        print(f"Automatic clusters: {sorted(comparison_df[clustering_method].unique())}")

        # Compare discrimination power
        manual_scores = []
        auto_scores = []

        for feature in self.acoustic_features:
            feature_df = comparison_df.dropna(subset=[feature])

            if len(feature_df) < 10:  # Skip if too few samples
                continue

            # Manual labels
            manual_labels = feature_df['manual_label'].unique()
            if len(manual_labels) > 1:
                manual_groups = [feature_df[feature_df['manual_label'] == label][feature].values
                                 for label in manual_labels]
                manual_f, manual_p = stats.f_oneway(*manual_groups)
                manual_scores.append({'feature': feature, 'f_stat': manual_f, 'p_value': manual_p})

            # Automatic clusters
            auto_labels = feature_df[clustering_method].unique()
            if len(auto_labels) > 1:
                auto_groups = [feature_df[feature_df[clustering_method] == label][feature].values
                               for label in auto_labels]
                auto_f, auto_p = stats.f_oneway(*auto_groups)
                auto_scores.append({'feature': feature, 'f_stat': auto_f, 'p_value': auto_p})

        # Convert to DataFrames
        manual_df = pd.DataFrame(manual_scores).sort_values('f_stat', ascending=False)
        auto_df = pd.DataFrame(auto_scores).sort_values('f_stat', ascending=False)

        # Merge for comparison
        comparison = manual_df.merge(auto_df, on='feature', suffixes=('_manual', '_auto'))
        comparison['manual_better'] = comparison['f_stat_manual'] > comparison['f_stat_auto']
        comparison['ratio'] = comparison['f_stat_auto'] / comparison['f_stat_manual']

        # Calculate agreement metrics
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        ari = adjusted_rand_score(comparison_df['manual_label'], comparison_df[clustering_method])
        nmi = normalized_mutual_info_score(comparison_df['manual_label'], comparison_df[clustering_method])

        # Summary
        print(f"\nFeatures where manual labels discriminate better: {comparison['manual_better'].sum()}")
        print(f"Features where automatic clustering discriminates better: {(~comparison['manual_better']).sum()}")
        print(f"\nAgreement between manual and automatic labels:")
        print(f" Adjusted Rand Index: {ari:.3f} (1.0 = perfect agreement, 0.0 = random)")
        print(f" Normalized Mutual Information: {nmi:.3f} (1.0 = perfect agreement)")

        # Visualization
        self._plot_manual_vs_clustering_comparison(comparison, comparison_df, clustering_method)

        return {
            'clustering_method': clustering_method,
            'comparison_results': comparison,
            'manual_discrimination': manual_df,
            'auto_discrimination': auto_df,
            'agreement_metrics': {'ari': ari, 'nmi': nmi},
            'n_syllables': len(comparison_df)
        }


    def _plot_manual_vs_clustering_comparison(self, comparison: pd.DataFrame,
                                              comparison_df: pd.DataFrame, clustering_method: str):
        """Create visualization for manual vs clustering comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Scatter plot: Manual vs Auto F-statistics
        axes[0, 0].scatter(comparison['f_stat_manual'], comparison['f_stat_auto'], alpha=0.6)
        max_val = max(comparison[['f_stat_manual', 'f_stat_auto']].max())
        axes[0, 0].plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Equal performance')
        axes[0, 0].set_xlabel('Manual Labels F-statistic')
        axes[0, 0].set_ylabel('Automatic Clustering F-statistic')
        axes[0, 0].set_title('Discrimination Power Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Ratio histogram
        axes[0, 1].hist(comparison['ratio'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(x=1, color='red', linestyle='--', label='Equal performance')
        axes[0, 1].set_xlabel('Ratio (Auto F-stat / Manual F-stat)')
        axes[0, 1].set_ylabel('Number of Features')
        axes[0, 1].set_title('Performance Ratio Distribution')
        axes[0, 1].legend()

        # 3. Side-by-side comparison of best discriminating feature
        if len(comparison) > 0:
            best_feature = comparison.loc[comparison['f_stat_manual'].idxmax(), 'feature']

            # Manual labels
            sns.boxplot(data=comparison_df, x='manual_label', y=best_feature, ax=axes[1, 0])
            axes[1, 0].set_title(f'Manual Labels: {best_feature}')
            axes[1, 0].tick_params(axis='x', rotation=45)

            # Automatic clusters
            sns.boxplot(data=comparison_df, x=clustering_method, y=best_feature, ax=axes[1, 1])
            axes[1, 1].set_title(f'Automatic Clusters: {best_feature}')
            axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()


    def generate_comprehensive_summary(self, top_n: int = 15) -> Dict[str, Any]:
        """
        Generate comprehensive summary of most discriminative features.

        Args:
            top_n: Number of top features to analyze in detail

        Returns:
            Dictionary with comprehensive analysis results
        """
        print("=== COMPREHENSIVE SYLLABLE ANALYSIS SUMMARY ===\n")

        # Basic database info
        structure_info = self.explore_database_structure()

        # Run comprehensive feature analysis
        print("\nAnalyzing feature discrimination power...")
        feature_results = self.analyze_all_features()

        # Apply multiple comparisons correction
        corrected_results = self.apply_multiple_comparisons_correction(feature_results)

        # MFCC analysis
        print("\nAnalyzing MFCC coefficients...")
        mfcc_results, mfcc_means_df, mfcc_stds_df = self.analyze_mfcc_features()

        # Top discriminative features
        print(f"\n=== TOP {top_n} MOST DISCRIMINATIVE FEATURES ===")
        print("(Features that best separate manual labels)\n")

        top_features = corrected_results.head(top_n)

        for i, (idx, row) in enumerate(top_features.iterrows(), 1):
            # Significance markers
            if row['p_corrected'] < 0.001:
                sig = "***"
            elif row['p_corrected'] < 0.01:
                sig = "**"
            elif row['p_corrected'] < 0.05:
                sig = "*"
            else:
                sig = ""

            # Effect size interpretation
            if row['eta_squared'] >= 0.14:
                effect = "Large"
            elif row['eta_squared'] >= 0.06:
                effect = "Medium"
            elif row['eta_squared'] >= 0.01:
                effect = "Small"
            else:
                effect = "Negligible"

            print(f"{i:2d}. {row['feature']:<30}")
            print(f"    Effect Size (η²): {row['eta_squared']:.4f} ({effect})")
            print(f"    Significance: p = {row['p_corrected']:.6f} {sig}")
            print(f"    Sample size: {row['n_syllables']} syllables across {row['n_labels']} labels")

            # Show means by label
            means = row['mean_by_label']
            print(f"    Means by label: ", end="")
            for label, mean_val in means.items():
                print(f"{label}={mean_val:.3f} ", end="")
            print("\n")

        # Summary statistics
        significant_features = corrected_results[corrected_results['significant_corrected']]
        highly_significant = corrected_results[corrected_results['p_corrected'] < 0.001]
        large_effect = corrected_results[corrected_results['eta_squared'] >= 0.14]
        medium_effect = corrected_results[corrected_results['eta_squared'] >= 0.06]

        summary_stats = {
            'total_features_analyzed': len(corrected_results),
            'significant_features': len(significant_features),
            'highly_significant_features': len(highly_significant),
            'large_effect_features': len(large_effect),
            'medium_effect_features': len(medium_effect),
            'significance_percentage': len(significant_features) / len(corrected_results) * 100,
            'mean_effect_size': corrected_results['eta_squared'].mean(),
            'median_effect_size': corrected_results['eta_squared'].median(),
            'max_effect_size': corrected_results['eta_squared'].max(),
            'best_discriminating_feature': corrected_results.iloc[0]['feature'],
            'best_feature_effect_size': corrected_results.iloc[0]['eta_squared']
        }

        print("=== SUMMARY STATISTICS ===")
        print(f"Total features analyzed: {summary_stats['total_features_analyzed']}")
        print(
            f"Statistically significant (corrected p < 0.05): {summary_stats['significant_features']} ({summary_stats['significance_percentage']:.1f}%)")
        print(f"Highly significant (corrected p < 0.001): {summary_stats['highly_significant_features']}")
        print(f"Large effect size (η² ≥ 0.14): {summary_stats['large_effect_features']}")
        print(f"Medium effect size (η² ≥ 0.06): {summary_stats['medium_effect_features']}")
        print(
            f"Best discriminating feature: {summary_stats['best_discriminating_feature']} (η²={summary_stats['best_feature_effect_size']:.4f})")
        print(f"Mean effect size: {summary_stats['mean_effect_size']:.4f}")

        # Create comprehensive visualization
        self._create_comprehensive_visualization(corrected_results, top_features, mfcc_results, summary_stats)

        return {
            'structure_info': structure_info,
            'feature_results': corrected_results,
            'top_features': top_features,
            'mfcc_results': mfcc_results,
            'summary_stats': summary_stats,
            'significant_features': significant_features,
            'large_effect_features': large_effect,
            'mfcc_means_df': mfcc_means_df,
            'mfcc_stds_df': mfcc_stds_df
        }


    def _create_comprehensive_visualization(self, feature_results: pd.DataFrame,
                                            top_features: pd.DataFrame, mfcc_results: pd.DataFrame,
                                            summary_stats: Dict[str, Any]):
        """Create comprehensive visualization of analysis results."""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

        # Filter data for visualization
        filtered_df = self.df[~self.df['manual_label'].isin(self.exclude_labels)].copy()

        # 1. Feature overview heatmap (top row, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        viz_features = [f for f in self.acoustic_features if f in top_features['feature'].values][:15]

        if len(viz_features) > 0:
            feature_means_viz = filtered_df.groupby('manual_label')[viz_features].mean()
            feature_means_std_viz = feature_means_viz.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

            sns.heatmap(feature_means_std_viz.T, annot=False, cmap='RdBu_r', center=0, ax=ax1)
            ax1.set_title('Top 15 Features by Manual Label (Standardized)', fontsize=12)
            ax1.set_xlabel('Manual Label')
            ax1.set_ylabel('Features')
        else:
            ax1.text(0.5, 0.5, 'No suitable features for heatmap',
                     ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Feature Heatmap (No Data)')

        # 2. Effect sizes (top row, right)
        ax2 = fig.add_subplot(gs[0, 2:])
        top_10 = feature_results.head(10)
        ax2.barh(range(len(top_10)), top_10['eta_squared'].values)
        ax2.set_yticks(range(len(top_10)))
        ax2.set_yticklabels([f.replace('_', '\n') for f in top_10['feature'].values], fontsize=8)
        ax2.set_xlabel('Effect Size (η²)')
        ax2.set_title('Top 10 Features: Effect Sizes')
        ax2.grid(axis='x', alpha=0.3)

        # 3. Best discriminating feature (second row, left)
        ax3 = fig.add_subplot(gs[1, :2])
        best_feature = feature_results.iloc[0]['feature']
        best_feature_data = filtered_df.dropna(subset=[best_feature])
        if len(best_feature_data) > 0:
            sns.boxplot(data=best_feature_data, x='manual_label', y=best_feature, ax=ax3)
            ax3.set_title(f'Best Discriminating Feature: {best_feature}')
            ax3.tick_params(axis='x', rotation=45)
        else:
            ax3.text(0.5, 0.5, 'No data for best feature',
                     ha='center', va='center', transform=ax3.transAxes)

        # 4. MFCC discrimination power (second row, right)
        ax4 = fig.add_subplot(gs[1, 2:])
        if len(mfcc_results) > 0:
            ax4.bar(mfcc_results['coefficient'], mfcc_results['eta_squared'])
            ax4.set_xlabel('MFCC Coefficient')
            ax4.set_ylabel('Effect Size (η²)')
            ax4.set_title('MFCC Discrimination Power')
        else:
            ax4.text(0.5, 0.5, 'No MFCC data available',
                     ha='center', va='center', transform=ax4.transAxes)

        # 5. Manual label distribution (third row, left)
        ax5 = fig.add_subplot(gs[2, 0])
        label_counts = filtered_df['manual_label'].value_counts()
        ax5.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%')
        ax5.set_title('Manual Label Distribution')

        # 6. Feature significance (third row, center-left)
        ax6 = fig.add_subplot(gs[2, 1])
        significance_bins = ['p≥0.05', '0.01≤p<0.05', '0.001≤p<0.01', 'p<0.001']
        significance_counts = [
            len(feature_results[feature_results['p_corrected'] >= 0.05]),
            len(feature_results[(feature_results['p_corrected'] >= 0.01) & (feature_results['p_corrected'] < 0.05)]),
            len(feature_results[(feature_results['p_corrected'] >= 0.001) & (feature_results['p_corrected'] < 0.01)]),
            len(feature_results[feature_results['p_corrected'] < 0.001])
        ]
        ax6.bar(significance_bins, significance_counts)
        ax6.set_title('Feature Significance Distribution')
        ax6.tick_params(axis='x', rotation=45)

        # 7. Effect size distribution (third row, center-right)
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.hist(feature_results['eta_squared'], bins=20, alpha=0.7, edgecolor='black')
        ax7.axvline(x=0.01, color='green', linestyle='--', label='Small (0.01)')
        ax7.axvline(x=0.06, color='orange', linestyle='--', label='Medium (0.06)')
        ax7.axvline(x=0.14, color='red', linestyle='--', label='Large (0.14)')
        ax7.set_xlabel('Effect Size (η²)')
        ax7.set_ylabel('Number of Features')
        ax7.set_title('Effect Size Distribution')
        ax7.legend()

        # 8. Feature category breakdown (third row, right)
        ax8 = fig.add_subplot(gs[2, 3])
        feature_categories = {
            'Spectral': ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'spectral_contrast'],
            'F0/Pitch': ['f0_mean', 'f0_std', 'f0_min', 'f0_max', 'f0_range', 'f0_voiced_fraction'],
            'Energy': ['rms_energy', 'onset_strength'],
            'Temporal': ['duration_ms', 'tempo_estimate', 'zero_crossing_rate'],
            'Context': ['prev_syllable_gap', 'next_syllable_gap']
        }

        category_counts = {}
        for category, keywords in feature_categories.items():
            count = sum(1 for feature in top_features['feature']
                        if any(keyword in feature for keyword in keywords))
            if count > 0:
                category_counts[category] = count

        # Add "Other" category
        total_categorized = sum(category_counts.values())
        if total_categorized < len(top_features):
            category_counts['Other'] = len(top_features) - total_categorized

        if category_counts:
            ax8.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%')
            ax8.set_title(f'Feature Categories in Top {len(top_features)}')
        else:
            ax8.text(0.5, 0.5, 'No categorized features',
                     ha='center', va='center', transform=ax8.transAxes)
            ax8.set_title('Feature Categories')

        # 9. Summary text (bottom row, spans 2 columns)
        ax9 = fig.add_subplot(gs[3, :2])
        ax9.axis('off')
        summary_text = f"""
    ANALYSIS SUMMARY
    
    Bird: {self.bird_name}
    Total Syllables: {len(self.df):,}
    Manual Labels: {len(filtered_df['manual_label'].unique())}
    
    Feature Analysis:
    • {summary_stats['total_features_analyzed']} features analyzed
    • {summary_stats['significant_features']} significant (corrected p<0.05)
    • {summary_stats['large_effect_features']} with large effect size
    • Best feature: {summary_stats['best_discriminating_feature'][:20]}...
    • Max η²: {summary_stats['best_feature_effect_size']:.4f}
    
    Quality Assessment:
    • {summary_stats['significance_percentage']:.1f}% features significant
    • Mean effect size: {summary_stats['mean_effect_size']:.4f}
    • Manual labels show {'good' if summary_stats['mean_effect_size'] > 0.1 else 'moderate' if summary_stats['mean_effect_size'] > 0.05 else 'weak'} discrimination
    """

        ax9.text(0.1, 0.9, summary_text, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))

        # 10. Feature importance comparison (bottom row, right)
        ax10 = fig.add_subplot(gs[3, 2:])
        if len(top_features) >= 10:
            # Show top 10 features with their ranks
            top_10_for_comparison = top_features.head(10)
            y_pos = range(len(top_10_for_comparison))

            bars = ax10.barh(y_pos, top_10_for_comparison['eta_squared'])
            ax10.set_yticks(y_pos)
            ax10.set_yticklabels([f.replace('_', '\n') for f in top_10_for_comparison['feature']], fontsize=8)
            ax10.set_xlabel('Effect Size (η²)')
            ax10.set_title('Top 10 Features: Detailed View')

            # Color bars by significance
            for i, (_, row) in enumerate(top_10_for_comparison.iterrows()):
                if row['p_corrected'] < 0.001:
                    bars[i].set_color('red')
                elif row['p_corrected'] < 0.01:
                    bars[i].set_color('orange')
                elif row['p_corrected'] < 0.05:
                    bars[i].set_color('yellow')
                else:
                    bars[i].set_color('lightblue')
        else:
            ax10.text(0.5, 0.5, f'Only {len(top_features)} features available',
                      ha='center', va='center', transform=ax10.transAxes)
            ax10.set_title('Feature Comparison')

        plt.suptitle(f'Comprehensive Syllable Analysis: {self.bird_name}', fontsize=16, y=0.98)
        plt.show()


    def save_analysis_results(self, results: Dict[str, Any], output_dir: str = 'analysis_results'):
        """
        Save comprehensive analysis results to files.

        Args:
            results: Results dictionary from generate_comprehensive_summary()
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        bird_name = self.bird_name

        try:
            # Save feature results as CSV
            results['feature_results'].to_csv(
                output_path / f'{bird_name}_feature_analysis.csv', index=False
            )

            # Save top features
            results['top_features'].to_csv(
                output_path / f'{bird_name}_top_features.csv', index=False
            )

            # Save MFCC results
            if len(results['mfcc_results']) > 0:
                results['mfcc_results'].to_csv(
                    output_path / f'{bird_name}_mfcc_analysis.csv', index=False
                )
                results['mfcc_means_df'].to_csv(
                    output_path / f'{bird_name}_mfcc_means_by_label.csv', index=False
                )

            # Save summary statistics
            summary_df = pd.DataFrame([results['summary_stats']])
            summary_df.to_csv(output_path / f'{bird_name}_summary_stats.csv', index=False)

            # Save complete results as JSON
            json_results = self._prepare_json_results(results)
            with open(output_path / f'{bird_name}_complete_analysis.json', 'w') as f:
                json.dump(json_results, f, indent=2)

            print(f"Analysis results saved to: {output_path}")

        except Exception as e:
            print(f"Error saving results: {e}")


    def _prepare_json_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare results for JSON serialization."""
        def convert_for_json(obj):
            if obj is None:
                return None
            elif isinstance(obj, (bool, np.bool_)):
                return bool(obj)
            elif isinstance(obj, (int, np.integer)):
                return int(obj)
            elif isinstance(obj, (float, np.floating)):
                return float(obj)
            elif isinstance(obj, str):
                return obj
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, pd.Series):
                return obj.to_list()
            elif isinstance(obj, dict):
                return {str(k): convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_for_json(item) for item in obj]
            else:
                return str(obj)  # Convert everything else to string

        return convert_for_json(results)


    # Convenience methods for common analyses
    def quick_analysis(self, top_n: int = 10) -> pd.DataFrame:
        """
        Run a quick analysis of the most discriminative features.

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with top discriminative features
        """
        feature_results = self.analyze_all_features()
        corrected_results = self.apply_multiple_comparisons_correction(feature_results)
        return corrected_results.head(top_n)


    def analyze_feature_group(self, group_name: str, feature_keywords: List[str]) -> pd.DataFrame:
        """
        Analyze a specific group of related features.

        Args:
            group_name: Name of the feature group
            feature_keywords: Keywords to identify features in this group

        Returns:
            DataFrame with analysis results for the feature group
        """
        group_features = [
            feature for feature in self.acoustic_features
            if any(keyword in feature for keyword in feature_keywords)
        ]

        if not group_features:
            print(f"No features found for group '{group_name}'")
            return pd.DataFrame()

        print(f"\n=== {group_name.upper()} FEATURES ===")
        print(f"Found {len(group_features)} features: {group_features}")

        # Analyze each feature in the group
        analysis_df = self.df[~self.df['manual_label'].isin(self.exclude_labels)].copy()
        group_results = []

        for feature in group_features:
            feature_df = analysis_df.dropna(subset=[feature])
            if len(feature_df) == 0:
                continue

            labels = feature_df['manual_label'].unique()
            if len(labels) < 2:
                continue

            # ANOVA test
            groups = [feature_df[feature_df['manual_label'] == label][feature].values
                      for label in labels]
            groups = [group for group in groups if len(group) > 0]

            if len(groups) < 2:
                continue

            f_stat, p_value = stats.f_oneway(*groups)

            # Calculate effect size
            overall_mean = feature_df[feature].mean()
            ss_total = ((feature_df[feature] - overall_mean) ** 2).sum()

            ss_between = 0
            for label in labels:
                group_data = feature_df[feature_df['manual_label'] == label][feature]
                if len(group_data) > 0:
                    group_mean = group_data.mean()
                    ss_between += len(group_data) * (group_mean - overall_mean) ** 2

            eta_squared = ss_between / ss_total if ss_total > 0 else 0

            group_results.append({
                'feature': feature,
                'f_statistic': f_stat,
                'p_value': p_value,
                'eta_squared': eta_squared,
                'significant': p_value < 0.05
            })

        # Sort by F-statistic
        group_results.sort(key=lambda x: x['f_statistic'], reverse=True)

        print(f"\nResults for {group_name} features:")
        for result in group_results:
            sig_marker = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result[
                                                                                                                  'p_value'] < 0.05 else ""
            print(
                f"  {result['feature']:<25} F={result['f_statistic']:6.2f}, p={result['p_value']:.4f}, η²={result['eta_squared']:.3f} {sig_marker}")

        return pd.DataFrame(group_results)


# Utility functions (can be used as standalone functions as well)
def explore_analysis_files(bird_path: str):
    """
    Quick exploration of analysis JSON files to see what's in them.

    Args:
        bird_path: Path to bird directory
    """
    database_dir = Path(bird_path) / 'data' / 'syllable_database'

    if not database_dir.exists():
        print(f"No database directory found at {database_dir}")
        return

    # Find all JSON files
    json_files = list(database_dir.glob("*.json"))

    if not json_files:
        print("No JSON analysis files found")
        return

    print(f"Found {len(json_files)} analysis files:")

    for json_file in json_files:
        print(f"\n{'=' * 50}")
        print(f"File: {json_file.name}")
        print(f"{'=' * 50}")

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Show the top-level structure
            print("Top-level keys:")
            for key in data.keys():
                print(f"  - {key}: {type(data[key])}")

            # If it's a clustering analysis, show key metrics
            if 'clustering_quality' in data:
                quality = data['clustering_quality']
                if 'cluster_separation' in quality:
                    sep = quality['cluster_separation']
                    print(f"\nClustering Quality Summary:")
                    print(f"  Method: {quality.get('clustering_method', 'unknown')}")
                    print(f"  Clusters: {quality.get('n_clusters', 'unknown')}")
                    print(f"  Syllables: {quality.get('n_syllables', 'unknown')}")
                    print(f"  Mean separation ratio: {sep.get('mean_separation_ratio', 'unknown'):.3f}")

            # If it has manual comparison
            if 'manual_comparison' in data:
                comp = data['manual_comparison']
                if 'comparison_summary' in comp:
                    summary = comp['comparison_summary']
                    print(f"\nManual vs Clustering:")
                    print(f"  Manual separation: {summary.get('manual_mean_separation', 'unknown'):.3f}")
                    print(f"  Clustering separation: {summary.get('clustering_mean_separation', 'unknown'):.3f}")
                    print(f"  Relative quality: {summary.get('relative_quality', 'unknown'):.3f}")

            # If it's a database summary
            if 'total_syllables' in data:
                print(f"\nDatabase Summary:")
                print(f"  Bird: {data.get('bird_name', 'unknown')}")
                print(f"  Total syllables: {data.get('total_syllables', 'unknown')}")
                if 'manual_labels' in data:
                    ml = data['manual_labels']
                    print(
                        f"  Manual labels: {ml.get('n_labeled_syllables', 0)} syllables, {ml.get('n_unique_labels', 0)} unique labels")
                if 'clustering_methods' in data:
                    print(f"  Clustering methods: {len(data['clustering_methods'])}")

        except Exception as e:
            print(f"Error reading {json_file.name}: {e}")


def run_complete_analysis(bird_path: str, exclude_labels: List[str] = None,
                          save_results: bool = True, output_dir: str = 'analysis_results') -> Dict[str, Any]:
    """
    Run complete syllable analysis pipeline.

    Args:
        bird_path: Path to bird directory
        exclude_labels: Labels to exclude from analysis
        save_results: Whether to save results to files
        output_dir: Directory to save results

    Returns:
        Dictionary with all analysis results
    """
    print("=" * 60)
    print("COMPLETE SYLLABLE DATABASE ANALYSIS")
    print("=" * 60)

    try:
        # Initialize analyzer
        analyzer = SyllableFeatureAnalyzer(bird_path, exclude_labels)

        # Run comprehensive analysis
        results = analyzer.generate_comprehensive_summary(top_n=15)

        # Add additional analyses
        print("\n" + "=" * 40)
        print("FEATURE GROUP ANALYSIS")
        print("=" * 40)

        # Analyze feature groups
        spectral_analysis = analyzer.analyze_feature_group(
            "Spectral", ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'spectral_contrast']
        )
        f0_analysis = analyzer.analyze_feature_group(
            "Fundamental Frequency", ['f0_mean', 'f0_std', 'f0_min', 'f0_max', 'f0_range', 'f0_voiced_fraction']
        )
        energy_analysis = analyzer.analyze_feature_group(
            "Energy", ['rms_energy', 'onset_strength']
        )
        temporal_analysis = analyzer.analyze_feature_group(
            "Temporal", ['duration_ms', 'tempo_estimate', 'zero_crossing_rate']
        )

        # Add group analyses to results
        results['feature_group_analysis'] = {
            'spectral': spectral_analysis,
            'f0': f0_analysis,
            'energy': energy_analysis,
            'temporal': temporal_analysis
        }

        # Correlation analysis
        print("\n" + "=" * 40)
        print("CORRELATION ANALYSIS")
        print("=" * 40)

        corr_matrix, high_corr_pairs = analyzer.analyze_feature_correlations()
        results['correlation_analysis'] = {
            'correlation_matrix': corr_matrix,
            'high_correlations': high_corr_pairs
        }

        # Manual vs clustering comparison
        print("\n" + "=" * 40)
        print("MANUAL vs AUTOMATIC CLUSTERING")
        print("=" * 40)

        clustering_comparison = analyzer.compare_manual_vs_clustering()
        if clustering_comparison:
            results['clustering_comparison'] = clustering_comparison

        # Save results if requested
        if save_results:
            analyzer.save_analysis_results(results, output_dir)

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)

        return results

    except Exception as e:
        print(f"Error during analysis: {e}")
        raise

# Clean utility functions (standalone, not trying to mimic class methods)
def quick_feature_analysis(bird_path: str, feature: str, exclude_labels: List[str] = None) -> pd.DataFrame:
    """
    Quick analysis of a single feature (convenience function).

    Args:
        bird_path: Path to bird directory
        feature: Feature to analyze
        exclude_labels: Labels to exclude

    Returns:
        Summary statistics DataFrame
    """
    analyzer = SyllableFeatureAnalyzer(bird_path, exclude_labels)
    return analyzer.analyze_single_feature(feature)


def compare_birds_on_feature(bird_paths: List[str], feature: str, exclude_labels: List[str] = None) -> pd.DataFrame:
    """
    Compare multiple birds on a single feature.

    Args:
        bird_paths: List of paths to bird directories
        feature: Feature to compare
        exclude_labels: Labels to exclude

    Returns:
        Comparison results DataFrame
    """
    results = []

    for bird_path in bird_paths:
        try:
            analyzer = SyllableFeatureAnalyzer(bird_path, exclude_labels)
            bird_results = analyzer.analyze_all_features()

            if feature in bird_results['feature'].values:
                feature_row = bird_results[bird_results['feature'] == feature].iloc[0]
                results.append({
                    'bird': analyzer.bird_name,
                    'feature': feature,
                    'eta_squared': feature_row['eta_squared'],
                    'f_statistic': feature_row['f_statistic'],
                    'p_value': feature_row['p_value'],
                    'n_syllables': feature_row['n_syllables']
                })
        except Exception as e:
            print(f"Error analyzing {Path(bird_path).name}: {e}")

    return pd.DataFrame(results).sort_values('eta_squared', ascending=False)


if __name__ == "__main__":
    # Example usage
    #import sys

    test_paths = [
        # os.path.join('/Volumes', 'Extreme SSD', 'wseg test', 'bu85bu97'),
        # os.path.join('/Volumes', 'Extreme SSD', 'evsong test', 'or18or24'),
        os.path.join('E:/', 'xfosters')
    ]

    #if len(sys.argv) > 1:
    for bird_path in test_paths:
        #bird_path = sys.argv[1]
        print(f"Running analysis for: {bird_path}")

        # Run complete analysis
        results = run_complete_analysis(bird_path, save_results=True)

        print(f"\nAnalysis completed for {Path(bird_path).name}")
        print(f"Top discriminating feature: {results['summary_stats']['best_discriminating_feature']}")
        print(f"Effect size: {results['summary_stats']['best_feature_effect_size']:.4f}")

        # Example of using the analyzer directly for custom analysis
        analyzer = SyllableFeatureAnalyzer(bird_path)

        # Quick analysis of specific features
        spectral_results = analyzer.analyze_single_feature('spectral_centroid_mean')
        duration_results = analyzer.analyze_single_feature('duration_ms')

    else:
        print("Usage: python syllable_feature_analyzer.py <bird_path>")
        print("Example: python syllable_feature_analyzer.py '/path/to/bird/directory'")
        print("\nFor programmatic use:")
        print("from syllable_feature_analyzer import SyllableFeatureAnalyzer")
        print("analyzer = SyllableFeatureAnalyzer('/path/to/bird')")
        print("results = analyzer.generate_comprehensive_summary()")


