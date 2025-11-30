"""
Machine Learning analysis engine for ECRAN Customer Insights.
Performs clustering, regression, and decision tree analysis.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from typing import Dict, Tuple, List, Optional


class InsightAnalyzer:
    """
    ML-driven analyzer for customer segmentation and purchase drivers.
    Validates the de-seasonalization hypothesis through data science.
    """
    def __init__(self, random_state: int = 42):
        """
        Initialize the analyzer.

        Args:
            random_state: Seed for reproducibility.
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.cluster_model = None
        self.regression_model = None
        self.tree_model = None
        self.feature_names = []

    def perform_clustering(
        self,
        df: pd.DataFrame,
        n_clusters: int = 3
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Perform K-Means clustering to identify customer tribes.

        Args:
            df: Survey data DataFrame.
            n_clusters: Number of clusters (default 3 for archetypes).

        Returns:
            Tuple of (DataFrame with cluster_id, clustering insights dict).
        """
        print("\n🔍 CLUSTERING ANALYSIS: Identifying Customer Tribes")
        print("=" * 60)

        # Select features for clustering
        cluster_features = [
            'concern_uvb',
            'concern_blue_light',
            'texture_sensitivity',
            'concern_pollution'
        ]

        X = df[cluster_features].values

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Perform K-Means
        self.cluster_model = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10
        )

        clusters = self.cluster_model.fit_predict(X_scaled)

        # Add cluster to dataframe
        df_clustered = df.copy()
        df_clustered['cluster_id'] = clusters

        # Analyze clusters
        insights = {}
        for i in range(n_clusters):
            cluster_data = df_clustered[df_clustered['cluster_id'] == i]

            insights[f'cluster_{i}'] = {
                'size': len(cluster_data),
                'size_pct': len(cluster_data) / len(df) * 100,
                'avg_uvb': cluster_data['concern_uvb'].mean(),
                'avg_blue_light': cluster_data['concern_blue_light'].mean(),
                'avg_texture': cluster_data['texture_sensitivity'].mean(),
                'avg_pollution': cluster_data['concern_pollution'].mean(),
                'buy_rate': cluster_data['willingness_to_buy_winter'].mean(),
                'avg_age': cluster_data['age'].mean()
            }

        # Identify the "Smart Ager" cluster (highest buy rate)
        smart_ager_cluster = max(
            range(n_clusters),
            key=lambda i: insights[f'cluster_{i}']['buy_rate']
        )

        df_clustered['is_smart_ager'] = (
            df_clustered['cluster_id'] == smart_ager_cluster
        ).astype(int)

        insights['smart_ager_cluster_id'] = smart_ager_cluster

        # Print insights
        print(f"\n✓ Identified {n_clusters} customer tribes:")
        for i in range(n_clusters):
            cluster_info = insights[f'cluster_{i}']
            marker = "🎯 SMART AGER" if i == smart_ager_cluster else ""
            print(f"\n  Cluster {i} {marker}")
            print(f"    Size: {cluster_info['size']} ({cluster_info['size_pct']:.1f}%)")
            print(f"    UVB Concern: {cluster_info['avg_uvb']:.1f}")
            print(f"    Blue Light: {cluster_info['avg_blue_light']:.1f}")
            print(f"    Texture Sens: {cluster_info['avg_texture']:.1f}")
            print(f"    Pollution Concern: {cluster_info['avg_pollution']:.1f}")
            print(f"    Winter Buy Rate: {cluster_info['buy_rate']*100:.1f}%")

        return df_clustered, insights

    def perform_regression(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Perform Logistic Regression to identify purchase drivers.

        Args:
            df: Survey data DataFrame with cluster information.

        Returns:
            Tuple of (coefficient DataFrame, regression insights dict).
        """
        print("\n📈 REGRESSION ANALYSIS: What Drives Winter Purchase?")
        print("=" * 60)

        # Select features
        feature_cols = [
            'concern_uvb',
            'concern_uva',
            'concern_blue_light',
            'concern_pollution',
            'texture_sensitivity',
            'age'
        ]

        self.feature_names = feature_cols

        X = df[feature_cols].values
        y = df['willingness_to_buy_winter'].values

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=self.random_state, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Fit logistic regression
        self.regression_model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000
        )
        self.regression_model.fit(X_train_scaled, y_train)

        # Predictions and accuracy
        y_pred = self.regression_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        # Extract coefficients
        coefficients = self.regression_model.coef_[0]

        coef_df = pd.DataFrame({
            'Feature': feature_cols,
            'Coefficient': coefficients,
            'Abs_Coefficient': np.abs(coefficients)
        }).sort_values('Abs_Coefficient', ascending=False)

        # Insights
        insights = {
            'accuracy': accuracy,
            'top_driver': coef_df.iloc[0]['Feature'],
            'top_driver_coef': coef_df.iloc[0]['Coefficient'],
            'coefficients': coef_df
        }

        # Print insights
        print(f"\n✓ Model Accuracy: {accuracy*100:.1f}%")
        print(f"\n  Top Purchase Drivers:")
        for idx, row in coef_df.head(3).iterrows():
            direction = "📈 Positive" if row['Coefficient'] > 0 else "📉 Negative"
            print(f"    {direction} - {row['Feature']}: {row['Coefficient']:.3f}")

        return coef_df, insights

    def perform_decision_tree(
        self,
        df: pd.DataFrame,
        max_depth: int = 3
    ) -> Tuple[DecisionTreeClassifier, Dict]:
        """
        Build decision tree to identify the golden path to conversion.

        Args:
            df: Survey data DataFrame.
            max_depth: Maximum tree depth for interpretability.

        Returns:
            Tuple of (fitted tree model, tree insights dict).
        """
        print("\n🌳 DECISION TREE: The Golden Path to Conversion")
        print("=" * 60)

        # Use same features as regression
        feature_cols = self.feature_names or [
            'concern_uvb',
            'concern_uva',
            'concern_blue_light',
            'concern_pollution',
            'texture_sensitivity',
            'age'
        ]

        X = df[feature_cols].values
        y = df['willingness_to_buy_winter'].values

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=self.random_state, stratify=y
        )

        # Fit decision tree with adaptive parameters based on dataset size
        # Scale min_samples based on training set size
        n_samples = len(X_train)
        min_split = max(2, int(n_samples * 0.05))  # 5% of samples
        min_leaf = max(1, int(n_samples * 0.02))   # 2% of samples

        self.tree_model = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=self.random_state,
            min_samples_split=min_split,
            min_samples_leaf=min_leaf
        )
        self.tree_model.fit(X_train, y_train)

        # Predictions
        y_pred = self.tree_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Calculate and normalize feature importances
        importances = self.tree_model.feature_importances_
        total = sum(importances)
        if total > 0:  # Avoid division by zero
            importances = [imp / total for imp in importances]

        # Create DataFrame with actual feature importances
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importances
        }).sort_values('Importance', ascending=False)


        insights = {
            'accuracy': accuracy,
            'max_depth': self.tree_model.get_depth(),
            'n_leaves': self.tree_model.get_n_leaves(),
            'feature_importance': importance_df,
            'top_feature': importance_df.iloc[0]['Feature']
        }

        # Print insights
        print(f"\n✓ Tree Accuracy: {accuracy*100:.1f}%")
        print(f"  Tree Depth: {insights['max_depth']}")
        print(f"  Number of Leaves: {insights['n_leaves']}")
        print(f"\n  Top Splitting Features:")
        for idx, row in importance_df.head(3).iterrows():
            if row['Importance'] > 0:
                print(f"    {row['Feature']}: {row['Importance']:.3f}")

        return self.tree_model, insights
