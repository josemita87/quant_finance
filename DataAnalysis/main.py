"""
Executive script for generating ECRAN Strategic Consulting visualizations.
Streamlined to focus on core strategic narratives only.
"""

from data import CaseData
from src import PlotEngine, PlotConfig


from src.ml_engine import InsightAnalyzer


def generate_strategic_visualizations(engine: PlotEngine, data: CaseData):
    """Generate the three core strategic visualizations (Phase 1)."""
    print("\n🎯 STRATEGIC VISUALIZATIONS (PHASE 1)")
    print("=" * 60)

    # 1. THE BLUE OCEAN - Seasonality Calendar Heatmap
    print("\n📅 Generating 'Blue Ocean' Seasonality Analysis...")
    months, sales = data.get_seasonality_arrays()
    engine.calendar_heatmap(
        monthly_data=sales,
        months_labels=months,
        title="THE 'BLUE OCEAN': 250 DAYS OF UNTAPPED OPPORTUNITY",
        filename='01_blue_ocean_seasonality'
    )

    # 2. THE AGGRESSION SHIFT - Summer vs. Winter Radar
    print("🎯 Generating 'Aggression Shift' Radar Chart...")
    categories, summer, winter = data.get_radar_arrays()
    engine.comparison_radar(
        categories=categories,
        profile_a=summer,
        profile_b=winter,
        label_a="Summer Profile (Current Focus)",
        label_b="Winter Profile (The Gap)",
        title="THE AGGRESSION SHIFT: WHY WINTER NEEDS PROTECTION",
        filename='02_aggression_shift_radar'
    )

    # 3. STRATEGIC POSITIONING MAP
    print("🗺️  Generating Strategic Positioning Map...")
    positioning = data.get_positioning_dict()
    engine.strategic_positioning_map(
        brand_data=positioning,
        title="STRATEGIC POSITIONING: THE MIGRATION TO WINNING ZONE",
        xlabel="COSMETIC APPEAL (Texture/Scent/Elegance)",
        ylabel="CLINICAL AUTHORITY (Protection/Science)",
        filename='03_strategic_positioning'
    )


def generate_ml_insights(engine: PlotEngine, data: CaseData):
    """Generate ML-driven insights and visualizations (Phase 2)."""
    print("\n" + "=" * 70)
    print("🤖 STEP 2: MACHINE LEARNING ANALYSIS (PHASE 2)")
    print("=" * 70)

    # Initialize analyzer
    analyzer = InsightAnalyzer(random_state=42)

    # Get survey data
    survey_data = data.get_survey_data()
    if survey_data.empty:
        print("❌ Error: No survey data available for Phase 2 analysis.")
        return

    # 1.1 Clustering Analysis
    print("\n🔍 Performing Clustering Analysis...")
    survey_data_clustered, cluster_insights = analyzer.perform_clustering(
        survey_data,
        n_clusters=3
    )

    # 1.2 Regression Analysis
    print("📉 Performing Regression Analysis...")
    coef_df, regression_insights = analyzer.perform_regression(
        survey_data_clustered
    )

    # 1.3 Decision Tree Analysis
    print("🌳 Performing Decision Tree Analysis...")
    tree_model, tree_insights = analyzer.perform_decision_tree(
        survey_data_clustered,
        max_depth=3
    )

    print("\n" + "=" * 70)
    print("📈 STEP 3: ADVANCED VISUALIZATIONS")
    print("=" * 70)

    # 3.1 Cluster Tribes Visualization
    print("\n🎯 Generating Cluster Tribes Chart...")
    smart_ager_cluster = cluster_insights['smart_ager_cluster_id']
    engine.plot_cluster_tribes(
        df=survey_data_clustered,
        smart_ager_cluster=smart_ager_cluster,
        filename='04_cluster_tribes_smart_agers'
    )

    # 3.2 Purchase Drivers Visualization
    print("📊 Generating Purchase Drivers Chart...")
    engine.plot_purchase_drivers(
        coef_df=coef_df,
        filename='05_purchase_drivers'
    )

    # 3.3 Decision Tree Visualization
    print("🌳 Generating Decision Tree Chart...")
    engine.plot_decision_tree_rules(
        tree_model=tree_model,
        feature_names=analyzer.feature_names,
        filename='06_decision_tree_golden_path'
    )

    # Bonus: Feature Importance
    print("📉 Generating Feature Importance Chart...")
    engine.plot_feature_importance(
        importance_df=tree_insights['feature_importance'],
        title="Feature Importance for Winter Purchase Prediction",
        filename='07_feature_importance'
    )

    # Regional Analysis - Spain Map
    print("🗺️  Generating Spain Regional Map...")
    engine.plot_spain_map_regional_profiles(
        df=survey_data_clustered,
        smart_ager_cluster=smart_ager_cluster,
        filename='08_spain_regional_map'
    )


def print_case_summary(data: CaseData):
    """Print a summary of the case data."""
    summary = data.summary()

    print("\n" + "=" * 60)
    print("📋 CASE DATA SUMMARY")
    print("=" * 60)
    print(f"\n✓ Total Datasets Available: {summary['total_datasets']}")
    print(f"✓ Key Strategic Insight: {summary['key_insight']}")
    print(f"✓ Strategic Focus: {summary['strategic_focus']}")


def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("🌞 ECRAN STRATEGIC CONSULTING - VISUALIZATION SUITE")
    print("   AC Marca Case Study Analysis")
    print("=" * 70)

    # Initialize components
    print("\n🔧 Initializing framework...")
    data = CaseData()
    engine = PlotEngine(output_dir='output')

    print("✓ Data loaded: ECRAN case study digitized")
    print("✓ Brand identity: Professional Blue-Tone")
    print("✓ Output directory: output/")

    # Print case summary
    print_case_summary(data)

    # Generate core strategic visualizations only
    try:
        # Phase 1
        generate_strategic_visualizations(engine, data)

        # Phase 2
        generate_ml_insights(engine, data)

        print("\n" + "=" * 70)
        print("✅ SUCCESS: All strategic visualizations generated!")
        print("=" * 70)
        print("\n📁 Generated strategic narrative assets:")
        print("\n   PHASE 1:")
        print("   1. Blue Ocean Seasonality (250-day winter opportunity)")
        print("   2. Aggression Shift Radar (Summer vs. Winter stressors)")
        print("   3. Strategic Positioning Map (ECRAN migration path)")
        print("\n   PHASE 2:")
        print("   4. Cluster Tribes (Smart Agers)")
        print("   5. Purchase Drivers (Regression)")
        print("   6. Decision Tree (Golden Path)")
        print("   7. Feature Importance")
        print("   8. Spain Regional Map")
        print("\n🎯 All charts are presentation-ready at 300 DPI.")
        print("   Professional blue-tone with transparent backgrounds.\n")

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
