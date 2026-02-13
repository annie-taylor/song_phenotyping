"""
Advanced Feature Analysis Methods

Advanced machine learning and statistical methods for syllable feature analysis.
Includes PCA, LDA, Random Forest, correlation-aware analysis, and feature selection.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from syllable_features import SyllableFeatureAnalyzer


class AdvancedFeatureMethods:
    """
    Advanced machine learning methods for syllable feature analysis.
    """

    def __init__(self, analyzer: SyllableFeatureAnalyzer):
        """
        Initialize with a SyllableFeatureAnalyzer instance.

        Args:
            analyzer: Initialized SyllableFeatureAnalyzer
        """
        self.analyzer = analyzer
        self.df = analyzer.df
        self.acoustic_features = analyzer.acoustic_features
        self.exclude_labels = analyzer.exclude_labels
        self.bird_name = analyzer.bird_name

    def pca_analysis(self, n_components: int = None) -> Dict[str, Any]:
        """
        Use PCA to find orthogonal components that discriminate between labels.

        Args:
            n_components: Number of components (None = auto-select up to 10)

        Returns:
            Dictionary with PCA results
        """
        # Filter and prepare data
        analysis_df = self.df[~self.df['manual_label'].isin(self.exclude_labels)].copy()
        feature_data = analysis_df[self.acoustic_features].dropna()

        if len(feature_data) == 0:
            print("No valid data for PCA analysis")
            return {}

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feature_data)

        # PCA
        if n_components is None:
            n_components = min(10, len(self.acoustic_features))

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        # Create PC DataFrame
        pc_columns = [f'PC{i + 1}' for i in range(n_components)]
        pca_df = pd.DataFrame(X_pca, columns=pc_columns, index=feature_data.index)
        pca_df['manual_label'] = analysis_df.loc[feature_data.index, 'manual_label']

        # Analyze PC discrimination
        pc_results = []
        for pc in pc_columns:
            labels = pca_df['manual_label'].unique()
            groups = [pca_df[pca_df['manual_label'] == label][pc].values for label in labels]
            f_stat, p_value = stats.f_oneway(*groups)

            pc_results.append({
                'component': pc,
                'explained_variance_ratio': pca.explained_variance_ratio_[int(pc[2:]) - 1],
                'f_statistic': f_stat,
                'p_value': p_value
            })

        pc_results_df = pd.DataFrame(pc_results)

        # Feature loadings
        loadings = pd.DataFrame(
            pca.components_[:n_components].T,
            columns=pc_columns,
            index=self.acoustic_features
        )

        # Visualization
        self._plot_pca_analysis(pca, pca_df, pc_results_df, loadings, n_components)

        # Print results
        print("=== PCA ANALYSIS RESULTS ===")
        print(f"Total variance explained by {n_components} components: {pca.explained_variance_ratio_.sum():.1%}")
        print(f"\nComponent discrimination power:")
        for _, row in pc_results_df.iterrows():
            sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row[
                                                                                                     'p_value'] < 0.05 else ""
            print(
                f"  {row['component']}: {row['explained_variance_ratio']:.1%} variance, F={row['f_statistic']:.2f} {sig}")

        # Show strongest feature loadings for top PCs
        print(f"\nStrongest feature contributions:")
        for pc_idx in range(min(3, n_components)):
            pc_name = f'PC{pc_idx + 1}'
            loadings_pc = pd.Series(pca.components_[pc_idx], index=self.acoustic_features)
            top_positive = loadings_pc.nlargest(3)
            top_negative = loadings_pc.nsmallest(3)

            print(f"\n{pc_name} ({pca.explained_variance_ratio_[pc_idx]:.1%} variance):")
            print("  Positive loadings:", ", ".join([f"{idx}({val:.2f})" for idx, val in top_positive.items()]))
            print("  Negative loadings:", ", ".join([f"{idx}({val:.2f})" for idx, val in top_negative.items()]))

        return {
            'pca_model': pca,
            'pca_data': pca_df,
            'pc_results': pc_results_df,
            'loadings': loadings,
            'scaler': scaler,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'total_variance_explained': pca.explained_variance_ratio_.sum()
        }

    def _plot_pca_analysis(self, pca, pca_df: pd.DataFrame, pc_results_df: pd.DataFrame,
                           loadings: pd.DataFrame, n_components: int):
        """Create PCA analysis visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Explained variance
        axes[0, 0].bar(range(1, n_components + 1), pca.explained_variance_ratio_)
        axes[0, 0].set_xlabel('Principal Component')
        axes[0, 0].set_ylabel('Explained Variance Ratio')
        axes[0, 0].set_title('PCA Explained Variance')

        # 2. PC discrimination power
        axes[0, 1].bar(range(1, n_components + 1), pc_results_df['f_statistic'])
        axes[0, 1].set_xlabel('Principal Component')
        axes[0, 1].set_ylabel('F-statistic')
        axes[0, 1].set_title('PC Discrimination Power')

        # 3. PC1 vs PC2 scatter
        for label in pca_df['manual_label'].unique():
            label_data = pca_df[pca_df['manual_label'] == label]
            axes[1, 0].scatter(label_data['PC1'], label_data['PC2'],
                               label=f'{label} (n={len(label_data)})', alpha=0.6)
        axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        axes[1, 0].set_title('PC1 vs PC2 by Manual Label')
        axes[1, 0].legend()

        # 4. Feature loadings for PC1 and PC2
        axes[1, 1].scatter(loadings['PC1'], loadings['PC2'], alpha=0.7)
        for i, (feature, row) in enumerate(loadings.iterrows()):
            if abs(row['PC1']) > 0.3 or abs(row['PC2']) > 0.3:  # Only label strong loadings
                axes[1, 1].annotate(feature.replace('_', '\n'),
                                    (row['PC1'], row['PC2']),
                                    fontsize=8, alpha=0.8)

        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1, 1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
        axes[1, 1].set_xlabel('PC1 Loading')
        axes[1, 1].set_ylabel('PC2 Loading')
        axes[1, 1].set_title('Feature Loadings (PC1 vs PC2)')

        plt.tight_layout()
        plt.show()


    def lda_analysis(self) -> Dict[str, Any]:
        """
        Use LDA to find linear combinations that best discriminate between labels.

        Returns:
            Dictionary with LDA results
        """
        # Filter and prepare data
        analysis_df = self.df[~self.df['manual_label'].isin(self.exclude_labels)].copy()
        feature_data = analysis_df[self.acoustic_features].dropna()

        if len(feature_data) == 0:
            print("No valid data for LDA analysis")
            return {}

        # Get labels and check class distribution
        labels = analysis_df.loc[feature_data.index, 'manual_label']
        label_counts = labels.value_counts()

        # Filter out classes with insufficient samples
        min_samples_per_class = 2
        valid_labels = label_counts[label_counts >= min_samples_per_class].index

        print(f"Original classes: {len(label_counts)}")
        print(f"Classes with ≥{min_samples_per_class} samples: {len(valid_labels)}")

        if len(valid_labels) < 2:
            print("Insufficient classes for LDA analysis")
            return {}

        # Filter data to only include valid classes
        valid_mask = labels.isin(valid_labels)
        feature_data = feature_data[valid_mask]
        labels = labels[valid_mask]

        n_classes = len(labels.unique())
        max_discriminants = min(n_classes - 1, len(self.acoustic_features))

        print(f"LDA Setup: {len(feature_data)} syllables, {n_classes} classes, {len(self.acoustic_features)} features")
        print(f"Will create {max_discriminants} discriminants")

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feature_data)

        # LDA
        lda = LinearDiscriminantAnalysis(n_components=max_discriminants)

        try:
            X_lda = lda.fit_transform(X_scaled, labels)
        except Exception as e:
            print(f"LDA fitting failed: {e}")
            return {}

        # Create LDA DataFrame
        n_components = X_lda.shape[1]
        ld_columns = [f'LD{i + 1}' for i in range(n_components)]
        lda_df = pd.DataFrame(X_lda, columns=ld_columns, index=feature_data.index)
        lda_df['manual_label'] = labels

        # Calculate discrimination metrics
        ld_results = []
        for i, ld in enumerate(ld_columns):
            explained_var = lda.explained_variance_ratio_[i] if hasattr(lda, 'explained_variance_ratio_') else None

            # F-statistic for this discriminant
            groups = [lda_df[lda_df['manual_label'] == label][ld].values for label in labels.unique()]
            groups = [group for group in groups if len(group) > 0]

            if len(groups) > 1:
                try:
                    f_stat, p_value = stats.f_oneway(*groups)
                except:
                    f_stat, p_value = 0, 1
            else:
                f_stat, p_value = 0, 1

            ld_results.append({
                'discriminant': ld,
                'explained_variance_ratio': explained_var,
                'f_statistic': f_stat,
                'p_value': p_value
            })

        ld_results_df = pd.DataFrame(ld_results)

        # Feature importance (discriminant loadings)
        feature_importance = pd.DataFrame()
        try:
            coef_matrix = lda.coef_
            if coef_matrix.shape == (n_components, len(self.acoustic_features)):
                coef_data = coef_matrix.T
            else:
                min_features = min(coef_matrix.shape[1], len(self.acoustic_features))
                min_components = min(coef_matrix.shape[0], n_components)
                coef_data = coef_matrix[:min_components, :min_features].T

            feature_importance = pd.DataFrame(
                coef_data,
                columns=ld_columns,
                index=self.acoustic_features
            )
        except Exception as e:
            print(f"Error creating feature importance DataFrame: {e}")
            feature_importance = pd.DataFrame(index=self.acoustic_features, columns=ld_columns)

        # Cross-validation
        cv_mean, cv_std = np.nan, np.nan
        try:
            min_class_size = labels.value_counts().min()
            cv_folds = min(5, min_class_size)

            if cv_folds >= 2:
                cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                cv_scores = cross_val_score(lda, X_scaled, labels, cv=cv_strategy)
                cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
        except Exception as e:
            print(f"Cross-validation error: {e}")

        # Visualization
        self._plot_lda_analysis(lda_df, ld_results_df, feature_importance, n_components, cv_mean, cv_std)

        # Print results
        print("=== LINEAR DISCRIMINANT ANALYSIS RESULTS ===")
        print(f"Number of classes: {n_classes}")
        print(f"Features used: {len(self.acoustic_features)}")
        print(f"Number of discriminants: {n_components}")

        if not pd.isna(cv_mean):
            print(f"Cross-validation accuracy: {cv_mean:.3f} ± {cv_std:.3f}")

        if hasattr(lda, 'explained_variance_ratio_') and len(lda.explained_variance_ratio_) > 0:
            print(f"Total variance explained: {lda.explained_variance_ratio_.sum():.1%}")
            for i, ratio in enumerate(lda.explained_variance_ratio_):
                print(f"  LD{i + 1}: {ratio:.1%}")

        # Show most discriminating features
        if not feature_importance.empty and len(feature_importance.columns) > 0:
            print(f"\nMost discriminating features:")
            for i, ld in enumerate(ld_columns):
                if ld in feature_importance.columns:
                    print(f"\n{ld}:")
                    try:
                        feature_coeffs = feature_importance[ld].dropna()
                        if len(feature_coeffs) > 0:
                            feature_coeffs_abs = feature_coeffs.abs().sort_values(ascending=False)
                            top_5 = feature_coeffs_abs.head(5)

                            for feature, abs_coeff in top_5.items():
                                actual_coeff = feature_importance.loc[feature, ld]
                                direction = "+" if actual_coeff > 0 else "-"
                                print(f"  {direction} {feature:<30} |coeff|={abs_coeff:.3f}")
                    except Exception as e:
                        print(f"  Error processing {ld}: {e}")
        return {
                    'lda_model': lda,
                    'lda_data': lda_df,
                    'ld_results': ld_results_df,
                    'feature_importance': feature_importance,
                    'scaler': scaler,
                    'cv_accuracy': cv_mean,
                    'cv_std': cv_std,
                    'n_components': n_components,
                    'n_classes': n_classes,
                    'explained_variance_ratio': lda.explained_variance_ratio_ if hasattr(lda, 'explained_variance_ratio_') else None
                }


    def _plot_lda_analysis(self, lda_df: pd.DataFrame, ld_results_df: pd.DataFrame,
                           feature_importance: pd.DataFrame, n_components: int, cv_mean: float, cv_std: float):
        """Create LDA analysis visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Discriminant power
        if len(ld_results_df) > 0:
            axes[0, 0].bar(range(1, len(ld_results_df) + 1), ld_results_df['f_statistic'])
            axes[0, 0].set_xlabel('Linear Discriminant')
            axes[0, 0].set_ylabel('F-statistic')
            axes[0, 0].set_title('LDA Discrimination Power')

        # 2. Classification accuracy
        if not pd.isna(cv_mean):
            axes[0, 1].bar(['CV Score'], [cv_mean])
            axes[0, 1].errorbar(['CV Score'], [cv_mean], yerr=[cv_std],
                                capsize=5, color='black')
            axes[0, 1].set_ylabel('Classification Accuracy')
            axes[0, 1].set_title(f'LDA Cross-Validation\n{cv_mean:.3f} ± {cv_std:.3f}')
            axes[0, 1].set_ylim(0, 1)
        else:
            axes[0, 1].text(0.5, 0.5, 'Cross-validation\nnot available',
                            ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Cross-Validation (Unavailable)')

        # 3. LD1 vs LD2 scatter (if available)
        if n_components >= 2:
            for label in lda_df['manual_label'].unique():
                label_data = lda_df[lda_df['manual_label'] == label]
                axes[1, 0].scatter(label_data['LD1'], label_data['LD2'],
                                   label=f'{label} (n={len(label_data)})', alpha=0.6)
            axes[1, 0].set_xlabel('LD1')
            axes[1, 0].set_ylabel('LD2')
            axes[1, 0].set_title('Linear Discriminants by Manual Label')
            axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # Single discriminant - show as histogram
            for label in lda_df['manual_label'].unique():
                label_data = lda_df[lda_df['manual_label'] == label]
                axes[1, 0].hist(label_data['LD1'], alpha=0.6, label=f'{label}', bins=15)
            axes[1, 0].set_xlabel('LD1')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_title('LD1 Distribution by Manual Label')
            axes[1, 0].legend()

        # 4. Feature importance
        if not feature_importance.empty and len(feature_importance.columns) > 0:
            try:
                if n_components >= 2 and 'LD2' in feature_importance.columns:
                    # Scatter plot of LD1 vs LD2 coefficients
                    ld1_vals = feature_importance['LD1'].dropna()
                    ld2_vals = feature_importance['LD2'].dropna()

                    if len(ld1_vals) > 0 and len(ld2_vals) > 0:
                        axes[1, 1].scatter(ld1_vals, ld2_vals, alpha=0.7)

                        # Label most important features
                        for feature in feature_importance.index:
                            ld1_coef = feature_importance.loc[feature, 'LD1']
                            ld2_coef = feature_importance.loc[feature, 'LD2']
                            if not (pd.isna(ld1_coef) or pd.isna(ld2_coef)):
                                if abs(ld1_coef) > 0.5 or abs(ld2_coef) > 0.5:
                                    axes[1, 1].annotate(feature.replace('_', '\n'),
                                                        (ld1_coef, ld2_coef),
                                                        fontsize=8, alpha=0.8)

                        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
                        axes[1, 1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
                        axes[1, 1].set_xlabel('LD1 Coefficient')
                        axes[1, 1].set_ylabel('LD2 Coefficient')
                        axes[1, 1].set_title('Feature Importance (LD1 vs LD2)')
                else:
                    # Single discriminant - show as bar plot
                    if 'LD1' in feature_importance.columns:
                        ld1_coeffs = feature_importance['LD1'].dropna()
                        if len(ld1_coeffs) > 0:
                            top_features = ld1_coeffs.abs().nlargest(10)
                            if len(top_features) > 0:
                                y_pos = range(len(top_features))
                                coefficients = [feature_importance.loc[feat, 'LD1'] for feat in top_features.index]

                                axes[1, 1].barh(y_pos, coefficients)
                                axes[1, 1].set_yticks(y_pos)
                                axes[1, 1].set_yticklabels([f.replace('_', '\n') for f in top_features.index], fontsize=8)
                                axes[1, 1].set_xlabel('LD1 Coefficient')
                                axes[1, 1].set_title('Top 10 Most Important Features')
                                axes[1, 1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
            except Exception as e:
                axes[1, 1].text(0.5, 0.5, f'Error plotting\nfeature importance:\n{str(e)[:30]}...',
                                ha='center', va='center', transform=axes[1, 1].transAxes)
        else:
            axes[1, 1].text(0.5, 0.5, 'Feature importance\nunavailable',
                            ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Feature Importance (Unavailable)')

        plt.tight_layout()
        plt.show()


    def random_forest_analysis(self, n_estimators: int = 100, random_state: int = 42) -> Dict[str, Any]:
        """
        Use Random Forest to identify feature importance while accounting for interactions.

        Args:
            n_estimators: Number of trees in the forest
            random_state: Random state for reproducibility

        Returns:
            Dictionary with Random Forest results
        """
        # Filter and prepare data
        analysis_df = self.df[~self.df['manual_label'].isin(self.exclude_labels)].copy()
        feature_data = analysis_df[self.acoustic_features].dropna()

        if len(feature_data) == 0:
            print("No valid data for Random Forest analysis")
            return {}

        # Get labels
        labels = analysis_df.loc[feature_data.index, 'manual_label']
        n_classes = len(labels.unique())

        if n_classes < 2:
            print("Need at least 2 classes for Random Forest")
            return {}

        print(
            f"Random Forest setup: {len(feature_data)} syllables, {n_classes} classes, {len(self.acoustic_features)} features")

        # Random Forest
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        rf.fit(feature_data, labels)

        # Feature importance
        importance_df = pd.DataFrame({
            'feature': self.acoustic_features,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        # Cross-validation score
        try:
            cv_scores = cross_val_score(rf, feature_data, labels, cv=5)
            cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
        except Exception as e:
            print(f"Cross-validation error: {e}")
            cv_scores = np.array([])
            cv_mean, cv_std = np.nan, np.nan

        # Get univariate results for comparison
        univariate_results = self.analyzer.analyze_all_features()

        # Merge with RF importance
        comparison = importance_df.merge(
            univariate_results[['feature', 'f_statistic']],
            on='feature', how='inner'
        )

        # Visualization
        self._plot_random_forest_analysis(importance_df, cv_mean, cv_std, comparison)

        # Print results
        print("=== RANDOM FOREST FEATURE IMPORTANCE RESULTS ===")
        print(f"Number of classes: {n_classes}")
        if not pd.isna(cv_mean):
            print(f"Cross-validation accuracy: {cv_mean:.3f} ± {cv_std:.3f}")

        # Calculate features needed for different importance thresholds
        cumulative_importance = importance_df['importance'].cumsum()
        n_80 = (cumulative_importance >= 0.8).idxmax() + 1 if (cumulative_importance >= 0.8).any() else len(importance_df)
        n_90 = (cumulative_importance >= 0.9).idxmax() + 1 if (cumulative_importance >= 0.9).any() else len(importance_df)

        print(f"Features needed for 80% importance: {n_80}")
        print(f"Features needed for 90% importance: {n_90}")

        print(f"\nTop 15 most important features:")
        for i, (_, row) in enumerate(importance_df.head(15).iterrows(), 1):
            print(f"{i:2d}. {row['feature']:<30} importance={row['importance']:.4f}")

        return {
            'rf_model': rf,
            'feature_importance': importance_df,
            'cv_scores': cv_scores,
            'cv_accuracy': cv_mean,
            'cv_std': cv_std,
            'comparison_with_univariate': comparison,
            'n_classes': n_classes,
            'features_for_80_percent': n_80,
            'features_for_90_percent': n_90
        }


    def _plot_random_forest_analysis(self, importance_df: pd.DataFrame, cv_mean: float,
                                     cv_std: float, comparison: pd.DataFrame):
        """Create Random Forest analysis visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Feature importance bar plot
        top_20 = importance_df.head(20)
        axes[0, 0].barh(range(len(top_20)), top_20['importance'])
        axes[0, 0].set_yticks(range(len(top_20)))
        axes[0, 0].set_yticklabels([f.replace('_', '\n') for f in top_20['feature']], fontsize=8)
        axes[0, 0].set_xlabel('Feature Importance')
        axes[0, 0].set_title('Top 20 Random Forest Feature Importances')

        # 2. Cumulative importance
        cumulative_importance = importance_df['importance'].cumsum()
        axes[0, 1].plot(range(1, len(cumulative_importance) + 1), cumulative_importance)
        axes[0, 1].axhline(y=0.8, color='r', linestyle='--', label='80% threshold')
        axes[0, 1].axhline(y=0.9, color='orange', linestyle='--', label='90% threshold')
        axes[0, 1].set_xlabel('Number of Features')
        axes[0, 1].set_ylabel('Cumulative Importance')
        axes[0, 1].set_title('Cumulative Feature Importance')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Find number of features for 80% and 90% importance
        n_80 = (cumulative_importance >= 0.8).idxmax() + 1 if (cumulative_importance >= 0.8).any() else len(
            cumulative_importance)
        n_90 = (cumulative_importance >= 0.9).idxmax() + 1 if (cumulative_importance >= 0.9).any() else len(
            cumulative_importance)
        axes[0, 1].axvline(x=n_80, color='r', linestyle=':', alpha=0.7)
        axes[0, 1].axvline(x=n_90, color='orange', linestyle=':', alpha=0.7)

        # 3. Cross-validation scores
        if not pd.isna(cv_mean):
            axes[1, 0].bar(['RF Accuracy'], [cv_mean])
            axes[1, 0].errorbar(['RF Accuracy'], [cv_mean], yerr=[cv_std],
                                capsize=5, color='black')
            axes[1, 0].set_ylabel('Classification Accuracy')
            axes[1, 0].set_title(f'Random Forest Cross-Validation\n{cv_mean:.3f} ± {cv_std:.3f}')
            axes[1, 0].set_ylim(0, 1)
        else:
            axes[1, 0].text(0.5, 0.5, 'Cross-validation\nnot available',
                            ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Cross-Validation (Unavailable)')

            # 4. Feature importance vs univariate F-statistic comparison
            if len(comparison) > 0:
                axes[1, 1].scatter(comparison['f_statistic'], comparison['importance'], alpha=0.6)

                # Label top features
                top_rf = comparison.nlargest(5, 'importance')
                for _, row in top_rf.iterrows():
                    axes[1, 1].annotate(row['feature'].replace('_', '\n'),
                                        (row['f_statistic'], row['importance']),
                                        fontsize=8, alpha=0.8)

                axes[1, 1].set_xlabel('Univariate F-statistic')
                axes[1, 1].set_ylabel('Random Forest Importance')
                axes[1, 1].set_title('RF Importance vs Univariate Analysis')
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'No comparison\ndata available',
                                ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Comparison (Unavailable)')

            plt.tight_layout()
            plt.show()

    def comprehensive_correlation_analysis(self, correlation_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Run comprehensive analysis that accounts for feature correlations.

        Args:
            correlation_threshold: Threshold for identifying high correlations

        Returns:
            Dictionary with comprehensive correlation analysis results
        """
        print("=== COMPREHENSIVE CORRELATION-AWARE ANALYSIS ===")

        # 1. Correlation analysis
        print("\n1. CORRELATION ANALYSIS")
        print("=" * 40)
        corr_matrix, high_corr_pairs = self.analyzer.analyze_feature_correlations(
            correlation_threshold=correlation_threshold
        )

        # 2. Multiple comparisons correction
        print("\n2. MULTIPLE COMPARISONS CORRECTION")
        print("=" * 40)
        feature_results = self.analyzer.analyze_all_features()
        corrected_results = self.analyzer.apply_multiple_comparisons_correction(
            feature_results, method='fdr_bh'
        )

        # 3. PCA analysis
        print("\n3. PRINCIPAL COMPONENT ANALYSIS")
        print("=" * 40)
        pca_results = self.pca_analysis(n_components=10)

        # 4. LDA analysis
        print("\n4. LINEAR DISCRIMINANT ANALYSIS")
        print("=" * 40)
        lda_results = self.lda_analysis()

        # 5. Random Forest analysis
        print("\n5. RANDOM FOREST FEATURE IMPORTANCE")
        print("=" * 40)
        rf_results = self.random_forest_analysis()

        # 6. Create summary comparison
        print("\n6. METHOD COMPARISON SUMMARY")
        print("=" * 40)

        # Combine results from different methods
        summary_data = []

        # Univariate results (corrected)
        for i, (_, row) in enumerate(corrected_results.head(10).iterrows(), 1):
            summary_data.append({
                'method': 'Univariate (corrected)',
                'feature': row['feature'],
                'rank': i,
                'score': row['eta_squared'],
                'p_value': row['p_corrected']
            })

        # Random Forest results
        if rf_results and 'feature_importance' in rf_results:
            rf_df = rf_results['feature_importance']
            for i, (_, row) in enumerate(rf_df.head(10).iterrows(), 1):
                summary_data.append({
                    'method': 'Random Forest',
                    'feature': row['feature'],
                    'rank': i,
                    'score': row['importance'],
                    'p_value': None
                })

        # LDA results (feature importance)
        if lda_results and 'feature_importance' in lda_results:
            feature_importance = lda_results['feature_importance']
            if not feature_importance.empty and len(feature_importance.columns) > 0:
                ld1_importance = feature_importance.iloc[:, 0].abs().sort_values(ascending=False)
                for i, (feature, importance) in enumerate(ld1_importance.head(10).items(), 1):
                    summary_data.append({
                        'method': 'LDA (LD1)',
                        'feature': feature,
                        'rank': i,
                        'score': importance,
                        'p_value': None
                    })

        # Create comparison DataFrame
        summary_df = pd.DataFrame(summary_data)

        # Find consensus features
        consensus_features = pd.Series(dtype=int)
        if len(summary_df) > 0:
            feature_method_counts = summary_df.groupby('feature').size()
            consensus_features = feature_method_counts[feature_method_counts >= 2].sort_values(ascending=False)

        # Visualization of method agreement
        self._plot_comprehensive_analysis(summary_df, consensus_features, high_corr_pairs,
                                          corrected_results, rf_results, lda_results, correlation_threshold)

        # Generate recommendations
        self._print_analysis_recommendations(consensus_features, high_corr_pairs, corrected_results,
                                             rf_results, lda_results)

        return {
            'correlation_results': (corr_matrix, high_corr_pairs),
            'corrected_univariate': corrected_results,
            'pca_results': pca_results,
            'lda_results': lda_results,
            'rf_results': rf_results,
            'summary_comparison': summary_df,
            'consensus_features': consensus_features if len(consensus_features) > 0 else None,
        }


    def _plot_comprehensive_analysis(self, summary_df: pd.DataFrame, consensus_features: pd.Series,
                                     high_corr_pairs: List[Dict], corrected_results: pd.DataFrame,
                                     rf_results: Dict, lda_results: Dict, correlation_threshold: float):
        """Create comprehensive analysis visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Top features by method consensus
        if len(consensus_features) > 0:
            axes[0, 0].barh(range(len(consensus_features)), consensus_features.values)
            axes[0, 0].set_yticks(range(len(consensus_features)))
            axes[0, 0].set_yticklabels([f.replace('_', '\n') for f in consensus_features.index], fontsize=8)
            axes[0, 0].set_xlabel('Number of Methods Ranking in Top 10')
            axes[0, 0].set_title('Feature Consensus Across Methods')
        else:
            axes[0, 0].text(0.5, 0.5, 'No consensus features\nfound',
                            ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Feature Consensus (None Found)')

        # 2. Correlation network visualization
        if high_corr_pairs:
            try:
                import networkx as nx

                G = nx.Graph()
                for pair in high_corr_pairs[:20]:  # Top 20 correlations
                    G.add_edge(pair['feature1'], pair['feature2'], weight=pair['correlation'])

                if len(G.nodes()) > 0:
                    pos = nx.spring_layout(G, k=1, iterations=50)
                    nx.draw(G, pos, ax=axes[0, 1], with_labels=True,
                            node_color='lightblue', node_size=500,
                            font_size=6, font_weight='bold')
                    axes[0, 1].set_title(f'High Correlation Network (r>{correlation_threshold})')
                else:
                    axes[0, 1].text(0.5, 0.5, 'No network to display',
                                    ha='center', va='center', transform=axes[0, 1].transAxes)
            except ImportError:
                # Fallback if networkx not available
                axes[0, 1].text(0.5, 0.5, 'NetworkX not available\nfor correlation network',
                                ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Correlation Network (NetworkX required)')
        else:
            axes[0, 1].text(0.5, 0.5, f'No high correlations\n(r>{correlation_threshold}) found',
                            ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Correlation Network (None Found)')

        # 3. Method performance comparison
        performance_data = []

        # Add univariate performance (mean effect size)
        mean_eta_squared = corrected_results['eta_squared'].mean()
        performance_data.append({'Method': 'Univariate', 'Score': mean_eta_squared, 'Metric': 'Mean η²'})

        # Add Random Forest performance
        if rf_results and 'cv_accuracy' in rf_results and not pd.isna(rf_results['cv_accuracy']):
            performance_data.append(
                {'Method': 'Random Forest', 'Score': rf_results['cv_accuracy'], 'Metric': 'CV Accuracy'})

        # Add LDA performance
        if lda_results and 'cv_accuracy' in lda_results and not pd.isna(lda_results['cv_accuracy']):
            performance_data.append({'Method': 'LDA', 'Score': lda_results['cv_accuracy'], 'Metric': 'CV Accuracy'})

        if performance_data:
            # Separate accuracy and effect size metrics
            accuracy_data = [p for p in performance_data if p['Metric'] == 'CV Accuracy']
            effect_data = [p for p in performance_data if p['Metric'] == 'Mean η²']

            if accuracy_data:
                methods = [p['Method'] for p in accuracy_data]
                scores = [p['Score'] for p in accuracy_data]
                bars = axes[1, 0].bar(methods, scores)
                axes[1, 0].set_ylabel('Cross-Validation Accuracy')
                axes[1, 0].set_title('Method Performance Comparison')
                axes[1, 0].set_ylim(0, 1)

                # Add value labels on bars
                for bar, score in zip(bars, scores):
                    axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                                    f'{score:.3f}', ha='center', va='bottom')

            if effect_data:
                # Add effect size as text annotation
                effect_score = effect_data[0]['Score']
                axes[1, 0].text(0.02, 0.98, f'Mean Effect Size (η²): {effect_score:.3f}',
                                transform=axes[1, 0].transAxes, va='top',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        else:
            axes[1, 0].text(0.5, 0.5, 'No performance\ndata available',
                            ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Method Performance (Unavailable)')

        # 4. Feature redundancy analysis
        if high_corr_pairs:
            redundancy_counts = {}
            for pair in high_corr_pairs:
                for feature in [pair['feature1'], pair['feature2']]:
                    redundancy_counts[feature] = redundancy_counts.get(feature, 0) + 1

            # Show most redundant features
            redundant_features = sorted(redundancy_counts.items(), key=lambda x: x[1], reverse=True)[:15]

            if redundant_features:
                features_list, counts = zip(*redundant_features)
                axes[1, 1].barh(range(len(features_list)), counts)
                axes[1, 1].set_yticks(range(len(features_list)))
                axes[1, 1].set_yticklabels([f.replace('_', '\n') for f in features_list], fontsize=8)
                axes[1, 1].set_xlabel('Number of High Correlations')
                axes[1, 1].set_title('Most Redundant Features')
            else:
                axes[1, 1].text(0.5, 0.5, 'No redundant\nfeatures found',
                                ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Feature Redundancy (None Found)')
        else:
            axes[1, 1].text(0.5, 0.5, 'No correlation\ndata available',
                            ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Feature Redundancy (No Data)')

        plt.tight_layout()
        plt.show()

    def _print_analysis_recommendations(self, consensus_features: pd.Series, high_corr_pairs: List[Dict],
                                        corrected_results: pd.DataFrame, rf_results: Dict, lda_results: Dict):
        """Print analysis recommendations."""
        print("\n7. ANALYSIS RECOMMENDATIONS")
        print("=" * 40)

        # Significant features after correction
        significant_corrected = corrected_results[corrected_results['significant_corrected']]
        print(f"Features significant after multiple comparisons correction: {len(significant_corrected)}")

        # Consensus features
        if len(consensus_features) > 0:
            print(f"Features identified by multiple methods: {len(consensus_features)}")
            print("\nTop consensus features:")
            for i, (feature, count) in enumerate(consensus_features.head(10).items(), 1):
                print(f"  {i:2d}. {feature:<30} (identified by {count} methods)")
        else:
            print("No features identified by multiple methods")

        # Redundancy recommendations
        if high_corr_pairs:
            print(f"\nHigh correlations found: {len(high_corr_pairs)}")
            print("Consider removing redundant features to reduce multicollinearity.")

            # Suggest features to potentially remove
            print("\nMost redundant features (consider removing):")
            feature_corr_counts = {}
            for pair in high_corr_pairs:
                feature_corr_counts[pair['feature1']] = feature_corr_counts.get(pair['feature1'], 0) + 1
                feature_corr_counts[pair['feature2']] = feature_corr_counts.get(pair['feature2'], 0) + 1

            sorted_redundant = sorted(feature_corr_counts.items(), key=lambda x: x[1], reverse=True)
            for i, (feature, count) in enumerate(sorted_redundant[:10], 1):
                print(f"  {i:2d}. {feature:<30} ({count} high correlations)")
        else:
            print("\nNo high correlations found - features appear independent")

        # Method-specific insights
        print(f"\nMethod-specific insights:")
        if rf_results and 'cv_accuracy' in rf_results and not pd.isna(rf_results['cv_accuracy']):
            rf_accuracy = rf_results['cv_accuracy']
            print(f"  Random Forest accuracy: {rf_accuracy:.3f} (accounts for feature interactions)")

        if lda_results and 'cv_accuracy' in lda_results and not pd.isna(lda_results['cv_accuracy']):
            lda_accuracy = lda_results['cv_accuracy']
            print(f"  LDA accuracy: {lda_accuracy:.3f} (finds optimal linear combinations)")

        univariate_significant = len(significant_corrected)
        print(f"  Univariate analysis: {univariate_significant} features significant after correction")


    def recommend_feature_subset(self, max_features: int = 15, correlation_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Recommend a subset of features that balances discrimination power with low redundancy.

        Args:
            max_features: Maximum number of features to recommend
            correlation_threshold: Correlation threshold for identifying redundancy

        Returns:
            Dictionary with recommended features and analysis
        """
        print("=== FEATURE SUBSET RECOMMENDATIONS ===")

        # Run comprehensive analysis
        results = self.comprehensive_correlation_analysis(correlation_threshold)

        # Get key results
        corrected_results = results['corrected_univariate']
        high_corr_pairs = results['correlation_results'][1]
        rf_results = results['rf_results']
        consensus_features = results['consensus_features']

        # Initialize recommendation tracking
        recommended_features = []
        excluded_features = set()
        feature_scores = {}

        print(f"\nBuilding feature subset (max {max_features} features)...")

        # Method 1: Start with consensus features
        if consensus_features is not None and len(consensus_features) > 0:
            print(f"\nStep 1: Adding consensus features...")
            for feature in consensus_features.index:
                if len(recommended_features) >= max_features:
                    break
                if feature not in excluded_features:
                    recommended_features.append(feature)
                    feature_scores[feature] = {
                        'reason': 'consensus',
                        'methods': consensus_features[feature],
                        'redundancy_score': 0
                    }
                    print(f"  Added: {feature} (consensus from {consensus_features[feature]} methods)")

        # Method 2: Add top univariate features (corrected p-values)
        if len(recommended_features) < max_features:
            print(f"\nStep 2: Adding top univariate features...")
            significant_features = corrected_results[corrected_results['significant_corrected']]

            for _, row in significant_features.iterrows():
                if len(recommended_features) >= max_features:
                    break

                feature = row['feature']
                if feature not in recommended_features and feature not in excluded_features:

                    # Check for high correlations with already selected features
                    is_redundant = False
                    redundant_with = []

                    for pair in high_corr_pairs:
                        if ((pair['feature1'] == feature and pair['feature2'] in recommended_features) or
                                (pair['feature2'] == feature and pair['feature1'] in recommended_features)):
                            is_redundant = True
                            other_feature = pair['feature2'] if pair['feature1'] == feature else pair['feature1']
                            redundant_with.append((other_feature, pair['correlation']))

                    if not is_redundant:
                        recommended_features.append(feature)
                        feature_scores[feature] = {
                            'reason': 'univariate_significant',
                            'eta_squared': row['eta_squared'],
                            'p_corrected': row['p_corrected'],
                            'redundancy_score': 0
                        }
                        print(f"  Added: {feature} (η²={row['eta_squared']:.3f}, p={row['p_corrected']:.4f})")
                    else:
                        excluded_features.add(feature)
                        print(f"  Skipped: {feature} (redundant with {redundant_with[0][0]}, r={redundant_with[0][1]:.3f})")

        # Method 3: Add top Random Forest features (if not redundant)
        if len(recommended_features) < max_features and rf_results and 'feature_importance' in rf_results:
            print(f"\nStep 3: Adding Random Forest features...")
            rf_importance = rf_results['feature_importance']

            for _, row in rf_importance.iterrows():
                if len(recommended_features) >= max_features:
                    break

                feature = row['feature']
                if feature not in recommended_features and feature not in excluded_features:

                    # Check for redundancy
                    is_redundant = False
                    for pair in high_corr_pairs:
                        if ((pair['feature1'] == feature and pair['feature2'] in recommended_features) or
                                (pair['feature2'] == feature and pair['feature1'] in recommended_features)):
                            is_redundant = True
                            break

                        if not is_redundant:
                            recommended_features.append(feature)
                            feature_scores[feature] = {
                                'reason': 'random_forest',
                                'rf_importance': row['importance'],
                                'redundancy_score': 0
                            }


                            print(f"  Added: {feature} (RF importance={row['importance']:.4f})")

        # Calculate redundancy scores for final set
        print(f"\nStep 4: Calculating redundancy scores...")
        for feature in recommended_features:
            redundancy_count = 0
            for pair in high_corr_pairs:
                if pair['feature1'] == feature or pair['feature2'] == feature:
                    other_feature = pair['feature2'] if pair['feature1'] == feature else pair['feature1']
                    if other_feature in recommended_features:
                        redundancy_count += 1

            feature_scores[feature]['redundancy_score'] = redundancy_count
            if redundancy_count > 0:
                print(f"  {feature}: {redundancy_count} high correlations within selected set")

        # Create final analysis
        print(f"\n=== FINAL FEATURE SUBSET ===")
        print(f"Selected {len(recommended_features)} features:")

        # Group by selection reason
        by_reason = {}
        for feature, score_info in feature_scores.items():
            reason = score_info['reason']
            if reason not in by_reason:
                by_reason[reason] = []
            by_reason[reason].append((feature, score_info))

        for reason, features_list in by_reason.items():
            print(f"\n{reason.replace('_', ' ').title()} features ({len(features_list)}):")
            for i, (feature, score_info) in enumerate(features_list, 1):
                redundancy = score_info['redundancy_score']
                redundancy_str = f" [!{redundancy} corr]" if redundancy > 0 else ""
                print(f"  {i}. {feature}{redundancy_str}")

        # Performance evaluation of selected subset
        print(f"\n=== SUBSET PERFORMANCE EVALUATION ===")

        # Test with selected features
        subset_analyzer = AdvancedFeatureMethods(self.analyzer)
        subset_analyzer.acoustic_features = recommended_features

        # Test LDA with selected features
        try:
            lda_subset = subset_analyzer.lda_analysis()
            if lda_subset and 'cv_accuracy' in lda_subset:
                print(f"LDA with selected features: {lda_subset['cv_accuracy']:.3f} accuracy")
            else:
                print(f"LDA with selected features: Available but no CV score")
        except Exception as e:
            print(f"LDA with selected features: Error - {str(e)[:50]}...")

        # Test Random Forest with selected features
        try:
            rf_subset = subset_analyzer.random_forest_analysis()
            if rf_subset and 'cv_accuracy' in rf_subset:
                rf_accuracy_subset = rf_subset['cv_accuracy']
                print(f"Random Forest with selected features: {rf_accuracy_subset:.3f} accuracy")

                # Compare with full feature set performance
                if rf_results and 'cv_accuracy' in rf_results and not pd.isna(rf_results['cv_accuracy']):
                    full_accuracy = rf_results['cv_accuracy']
                    accuracy_change = rf_accuracy_subset - full_accuracy
                    print(
                        f"Accuracy change: {accuracy_change:+.3f} ({len(recommended_features)} vs {len(self.acoustic_features)} features)")
            else:
                print(f"Random Forest with selected features: Available but no CV score")
        except Exception as e:
            print(f"Random Forest with selected features: Error - {str(e)[:50]}...")

        # Feature category analysis
        print(f"\n=== FEATURE CATEGORY BREAKDOWN ===")
        categories = {
            'Spectral': ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'spectral_contrast'],
            'F0/Pitch': ['f0_mean', 'f0_std', 'f0_min', 'f0_max', 'f0_range', 'f0_voiced_fraction'],
            'Energy': ['rms_energy', 'onset_strength'],
            'Temporal': ['duration_ms', 'tempo_estimate', 'zero_crossing_rate'],
            'Context': ['prev_syllable_gap', 'next_syllable_gap']
        }

        category_counts = {}
        for category, keywords in categories.items():
            count = sum(1 for feature in recommended_features
                        if any(keyword in feature for keyword in keywords))
            if count > 0:
                category_counts[category] = count

        for category, count in category_counts.items():
            print(f"  {category}: {count} features")

        # Final recommendations
        print(f"\n=== RECOMMENDATIONS ===")
        total_redundancy = sum(score_info['redundancy_score'] for score_info in feature_scores.values())

        if total_redundancy == 0:
            print("✓ Selected features have minimal redundancy")
        else:
            print(f"⚠ Selected features have {total_redundancy} internal correlations")
            print("  Consider removing features with highest redundancy scores")

        if len(recommended_features) == max_features:
            print(f"✓ Selected maximum requested features ({max_features})")
        else:
            print(f"• Selected {len(recommended_features)} features (< {max_features} requested)")
            print("  Consider lowering correlation threshold or including more methods")

        # Create summary visualization
        self._plot_feature_subset_summary(recommended_features, feature_scores, category_counts,
                                          by_reason, rf_results)

        # Return comprehensive results
        return {
            'recommended_features': recommended_features,
            'feature_scores': feature_scores,
            'excluded_features': list(excluded_features),
            'category_breakdown': category_counts,
            'method_breakdown': {reason: len(features) for reason, features in by_reason.items()},
            'total_redundancy': total_redundancy,
            'comprehensive_results': results
        }


    def _plot_feature_subset_summary(self, recommended_features: List[str], feature_scores: Dict,
                                     category_counts: Dict, by_reason: Dict, 
                                     
                                     
                                     
                                     
                                     
                                     
                                     
                                     
                                     
                                     
                                     
                                     
                                     
                                     
                                     
                                     
                                     
                                     
                                     rf_results: Dict):
        """Create summary visualization for feature subset."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Selected features by category
        if category_counts:
            axes[0, 0].pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%')
            axes[0, 0].set_title('Selected Features by Category')
        else:
            axes[0, 0].text(0.5, 0.5, 'No categorized\nfeatures',
                            ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Feature Categories')

        # 2. Selection method breakdown
        method_counts = {reason: len(features) for reason, features in by_reason.items()}
        if method_counts:
            method_labels = [r.replace('_', ' ').title() for r in method_counts.keys()]
            axes[0, 1].bar(method_labels, method_counts.values())
            axes[0, 1].set_title('Features by Selection Method')
            axes[0, 1].tick_params(axis='x', rotation=45)
        else:
            axes[0, 1].text(0.5, 0.5, 'No method data',
                            ha='center', va='center', transform=axes[0, 1].transAxes)

        # 3. Redundancy scores
        redundancy_scores = [score_info['redundancy_score'] for score_info in feature_scores.values()]
        if redundancy_scores:
            max_redundancy = max(redundancy_scores) if redundancy_scores else 1
            bins = max(1, max_redundancy + 1)
            axes[1, 0].hist(redundancy_scores, bins=bins, alpha=0.7, edgecolor='black')
            axes[1, 0].set_xlabel('Number of High Correlations')
            axes[1, 0].set_ylabel('Number of Features')
            axes[1, 0].set_title('Redundancy Distribution')
        else:
            axes[1, 0].text(0.5, 0.5, 'No redundancy data',
                            ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Redundancy Distribution')

        # 4. Feature importance comparison (if available)
        if rf_results and 'feature_importance' in rf_results:
            rf_importance_full = rf_results['feature_importance']
            selected_importance = rf_importance_full[rf_importance_full['feature'].isin(recommended_features)]

            if len(selected_importance) > 0:
                # Sort by importance for better visualization
                selected_importance = selected_importance.sort_values('importance', ascending=True)
                axes[1, 1].barh(range(len(selected_importance)), selected_importance['importance'])
                axes[1, 1].set_yticks(range(len(selected_importance)))
                axes[1, 1].set_yticklabels([f.replace('_', '\n') for f in selected_importance['feature']], fontsize=8)
                axes[1, 1].set_xlabel('Random Forest Importance')
                axes[1, 1].set_title('Selected Features: RF Importance')
            else:
                axes[1, 1].text(0.5, 0.5, 'No RF importance\ndata for selected features',
                                ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('RF Importance (No Data)')
        else:
            axes[1, 1].text(0.5, 0.5, 'No RF results\navailable',
                            ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('RF Importance (Unavailable)')

        plt.tight_layout()
        plt.show()


# Convenience functions for the advanced methods module
def run_advanced_analysis(bird_path: str, exclude_labels: List[str] = None,
                          correlation_threshold: float = 0.8, max_features: int = 15) -> Dict[str, Any]:
    """
    Run complete advanced analysis pipeline.

    Args:
        bird_path: Path to bird directory
        exclude_labels: Labels to exclude from analysis
        correlation_threshold: Threshold for correlation analysis
        max_features: Maximum features for subset recommendation

    Returns:
        Dictionary with all advanced analysis results
    """
    print("=" * 60)
    print("ADVANCED SYLLABLE FEATURE ANALYSIS")
    print("=" * 60)

    try:
        # Initialize base analyzer
        from syllable_feature_analyzer import SyllableFeatureAnalyzer
        base_analyzer = SyllableFeatureAnalyzer(bird_path, exclude_labels)

        # Initialize advanced methods
        advanced = AdvancedFeatureMethods(base_analyzer)

        # Run comprehensive correlation-aware analysis
        print("\n" + "=" * 40)
        print("COMPREHENSIVE CORRELATION ANALYSIS")
        print("=" * 40)
        comprehensive_results = advanced.comprehensive_correlation_analysis(correlation_threshold)

        # Feature subset recommendation
        print("\n" + "=" * 40)
        print("FEATURE SUBSET RECOMMENDATION")
        print("=" * 40)
        subset_results = advanced.recommend_feature_subset(max_features, correlation_threshold)

        # Combine all results
        results = {
            'bird_name': base_analyzer.bird_name,
            'bird_path': bird_path,
            'comprehensive_analysis': comprehensive_results,
            'feature_subset_recommendation': subset_results,
            'analysis_parameters': {
                'correlation_threshold': correlation_threshold,
                'max_features': max_features,
                'exclude_labels': exclude_labels
            }
        }

        print("\n" + "=" * 60)
        print("ADVANCED ANALYSIS COMPLETE!")
        print("=" * 60)

        # Print summary
        if subset_results['recommended_features']:
            print(f"\nRecommended features ({len(subset_results['recommended_features'])}):")
            for i, feature in enumerate(subset_results['recommended_features'], 1):
                print(f"  {i:2d}. {feature}")

        return results

    except Exception as e:
        print(f"Error during advanced analysis: {e}")
        raise


def compare_analysis_methods(bird_path: str, exclude_labels: List[str] = None) -> pd.DataFrame:
    """
    Compare different analysis methods on the same dataset.

    Args:
        bird_path: Path to bird directory
        exclude_labels: Labels to exclude from analysis

    Returns:
        DataFrame comparing method performance
    """
    from syllable_feature_analyzer import SyllableFeatureAnalyzer

    # Initialize analyzers
    base_analyzer = SyllableFeatureAnalyzer(bird_path, exclude_labels)
    advanced = AdvancedFeatureMethods(base_analyzer)

    results = []

    # Univariate analysis
    try:
        feature_results = base_analyzer.analyze_all_features()
        corrected_results = base_analyzer.apply_multiple_comparisons_correction(feature_results)
        significant_features = len(corrected_results[corrected_results['significant_corrected']])
        mean_effect_size = corrected_results['eta_squared'].mean()

        results.append({
            'Method': 'Univariate (corrected)',
            'Significant_Features': significant_features,
            'Mean_Effect_Size': mean_effect_size,
            'CV_Accuracy': None,
            'Notes': f'{significant_features} significant after correction'
        })
    except Exception as e:
        print(f"Univariate analysis failed: {e}")

    # PCA analysis
    try:
        pca_results = advanced.pca_analysis()
        if pca_results:
            variance_explained = pca_results['total_variance_explained']
            results.append({
                'Method': 'PCA',
                'Significant_Features': None,
                'Mean_Effect_Size': None,
                'CV_Accuracy': None,
                'Notes': f'{variance_explained:.1%} variance explained'
            })
    except Exception as e:
        print(f"PCA analysis failed: {e}")

    # LDA analysis
    try:
        lda_results = advanced.lda_analysis()
        if lda_results and 'cv_accuracy' in lda_results:
            cv_acc = lda_results['cv_accuracy']
            n_components = lda_results.get('n_components', 'Unknown')
            results.append({
                'Method': 'LDA',
                'Significant_Features': None,
                'Mean_Effect_Size': None,
                'CV_Accuracy': cv_acc if not pd.isna(cv_acc) else None,
                'Notes': f'{n_components} discriminants'
            })
    except Exception as e:
        print(f"LDA analysis failed: {e}")

    # Random Forest analysis
    try:
        rf_results = advanced.random_forest_analysis()
        if rf_results and 'cv_accuracy' in rf_results:
            cv_acc = rf_results['cv_accuracy']
            n_features_80 = rf_results.get('features_for_80_percent', 'Unknown')
            results.append({
                'Method': 'Random Forest',
                'Significant_Features': None,
                'Mean_Effect_Size': None,
                'CV_Accuracy': cv_acc if not pd.isna(cv_acc) else None,
                'Notes': f'{n_features_80} features for 80% importance'
            })
    except Exception as e:
        print(f"Random Forest analysis failed: {e}")

    # Convert to DataFrame and display
    comparison_df = pd.DataFrame(results)

    print("=== METHOD COMPARISON SUMMARY ===")
    print(comparison_df.to_string(index=False))

    return comparison_df


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        bird_path = sys.argv[1]
        print(f"Running advanced analysis for: {bird_path}")

        # Run complete advanced analysis
        results = run_advanced_analysis(bird_path, correlation_threshold=0.8, max_features=12)

        # Compare methods
        print("\n" + "=" * 60)
        print("METHOD COMPARISON")
        print("=" * 60)
        comparison = compare_analysis_methods(bird_path)

        print(f"\nAdvanced analysis completed for {Path(bird_path).name}")

        # Show recommended features
        recommended = results['feature_subset_recommendation']['recommended_features']
        if recommended:
            print(f"\nTop recommended features:")
            for i, feature in enumerate(recommended[:5], 1):
                print(f"  {i}. {feature}")
    else:
        print("Usage: python advanced_feature_methods.py <bird_path>")
        print("Example: python advanced_feature_methods.py '/path/to/bird/directory'")
        print("\nFor programmatic use:")
        print("from syllable_feature_analyzer import SyllableFeatureAnalyzer")
        print("from advanced_feature_methods import AdvancedFeatureMethods")
        print("base_analyzer = SyllableFeatureAnalyzer('/path/to/bird')")
        print("advanced = AdvancedFeatureMethods(base_analyzer)")
        print("results = advanced.comprehensive_correlation_analysis()")