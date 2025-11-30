"""
Central repository for digitized case numbers from the ECRAN/AC Marca case study.
All data extracted from the actual case study PDF.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any


class CaseData:
    """
    Centralized data repository for the ECRAN sunscreen case study.
    Contains actual digitized metrics from the case study document.
    """

    def __init__(self):
        """Initialize the case data repository."""
        self._load_data()

    def _load_data(self):
        """Load all case study data into structured formats."""

        # ========== CSV DATA LOADING ==========
        try:
            self.survey_data = pd.read_csv('data/survey_data.csv')
            print("✓ Loaded survey_data.csv")
        except FileNotFoundError:
            print("⚠ Warning: survey_data.csv not found")
            self.survey_data = pd.DataFrame()

        try:
            self.upload_ready_responses = pd.read_csv('data/upload_ready_responses.csv')
            print("✓ Loaded upload_ready_responses.csv")
        except FileNotFoundError:
            print("⚠ Warning: upload_ready_responses.csv not found")
            self.upload_ready_responses = pd.DataFrame()

        # ========== REAL CASE STUDY DATA ==========

        # Seasonality Sales Index (Page 13 of case study)
        # Shows extreme summer concentration - 250 untapped winter days
        self.seasonality = pd.DataFrame({
            'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            'Sales_Index': [15.7, 20.5, 40.4, 66.3, 79.0, 130.0,
                           139.1, 88.1, 29.9, 18.6, 14.4, 16.9],
            'Season': ['Winter', 'Winter', 'Spring', 'Spring', 'Spring', 'Summer',
                      'Summer', 'Summer', 'Fall', 'Fall', 'Winter', 'Winter']
        })

        # Skin Stressor Profiles (Pages 20-21)
        # Summer vs. Winter aggression differences
        self.stressor_radar = pd.DataFrame({
            'Stressor': ['UVB (Burning)', 'UVA (Aging)', 'Blue Light',
                        'Pollution', 'Dryness'],
            'Summer_Score': [10, 9, 7, 7, 4],  # High burn risk
            'Winter_Score': [2, 8, 9, 9, 9]    # "Hidden" winter risks
        })

        # Competitive Positioning Matrix (Page 17 & strategic analysis)
        # X-axis: Cosmetic Appeal (texture, scent, elegance)
        # Y-axis: Clinical Authority (protection, science)
        self.positioning = pd.DataFrame({
            'Brand': ['NIVEA', 'GARNIER', 'ISDIN', 'ECRAN (Current)', 'ECRAN (Target)'],
            'Cosmetic_Appeal': [8, 7, 6, 3, 7],
            'Clinical_Authority': [4, 3, 9, 7, 8],
            'Type': ['competitor', 'competitor', 'competitor', 'current', 'target']
        })

        # Market share data (from case context)
        self.market_share = pd.DataFrame({
            'Brand': ['NIVEA', 'ISDIN', 'GARNIER', 'ECRAN', 'La Roche-Posay', 'Others'],
            'Share_Percent': [28.5, 18.3, 16.7, 12.4, 8.9, 15.2],
            'Segment': ['Mass Premium', 'Dermocosmetic', 'Mass Market',
                       'Clinical Mass', 'Pharmacy', 'Various']
        })

        # Product portfolio focus areas
        self.product_lines = pd.DataFrame({
            'Category': ['Body Sun Care', 'Facial Sun Care', 'After Sun',
                        'Kids Protection', 'Sport/Water Resistant'],
            'Current_Share': [45, 15, 12, 18, 10],
            'Growth_Potential': [3, 25, 5, 8, 12],
            'Margin': [18, 32, 15, 20, 22]
        })

        # Channel distribution (Spanish market context)
        self.channels = pd.DataFrame({
            'Channel': ['Hypermarkets', 'Supermarkets', 'Pharmacies',
                       'Perfumeries', 'Online', 'Others'],
            'Sales_Share': [32.1, 28.4, 18.5, 10.2, 8.3, 2.5],
            'ECRAN_Index': [125, 118, 75, 65, 95, 88]  # 100 = market average
        })

        # Strategic opportunity sizing
        self.opportunity_matrix = pd.DataFrame({
            'Opportunity': ['Winter Protection', 'Facial Premiumization',
                          'Digital Commerce', 'Pharmacy Channel'],
            'Market_Size_M': [85, 120, 65, 95],  # Millions €
            'ECRAN_Share': [8, 5, 12, 15],  # Current %
            'Potential_Share': [18, 15, 20, 25]  # Target %
        })

        # Consumer insights (from case study)
        self.consumer_barriers = pd.DataFrame({
            'Barrier': ['Greasy Texture', 'White Residue', 'Strong Smell',
                       'Not Daily Use', 'Only Summer Need'],
            'Mention_Rate': [42, 38, 28, 55, 61],  # % of consumers citing
            'ECRAN_Association': [35, 32, 18, 48, 58]  # % associating with ECRAN
        })

    # ========== DATA ACCESS METHODS ==========

    def get_seasonality(self) -> pd.DataFrame:
        """Return monthly seasonality data."""
        return self.seasonality.copy()

    def get_stressor_radar(self) -> pd.DataFrame:
        """Return skin stressor comparison data."""
        return self.stressor_radar.copy()

    def get_positioning(self) -> pd.DataFrame:
        """Return competitive positioning matrix."""
        return self.positioning.copy()

    def get_market_share(self) -> pd.DataFrame:
        """Return market share data."""
        return self.market_share.copy()

    def get_product_lines(self) -> pd.DataFrame:
        """Return product portfolio data."""
        return self.product_lines.copy()

    def get_channels(self) -> pd.DataFrame:
        """Return channel performance data."""
        return self.channels.copy()

    def get_opportunity_matrix(self) -> pd.DataFrame:
        """Return strategic opportunity sizing."""
        return self.opportunity_matrix.copy()

    def get_consumer_barriers(self) -> pd.DataFrame:
        """Return consumer barrier analysis."""
        return self.consumer_barriers.copy()

    def get_survey_data(self) -> pd.DataFrame:
        """Return raw survey data."""
        return self.survey_data.copy()

    def get_upload_ready_responses(self) -> pd.DataFrame:
        """Return upload ready responses."""
        return self.upload_ready_responses.copy()

    def get_seasonality_arrays(self) -> tuple:
        """
        Return seasonality data as arrays for plotting.
        Returns: (months, sales_values)
        """
        return (
            self.seasonality['Month'].tolist(),
            self.seasonality['Sales_Index'].tolist()
        )

    def get_radar_arrays(self) -> tuple:
        """
        Return radar data as arrays for plotting.
        Returns: (categories, summer_scores, winter_scores)
        """
        return (
            self.stressor_radar['Stressor'].tolist(),
            self.stressor_radar['Summer_Score'].tolist(),
            self.stressor_radar['Winter_Score'].tolist()
        )

    def get_positioning_dict(self) -> Dict[str, Dict]:
        """
        Return positioning data as dictionary for scatter plotting.
        Returns: {brand: {coords: (x,y), type: str}}
        """
        result = {}
        for _, row in self.positioning.iterrows():
            result[row['Brand']] = {
                'coords': (row['Cosmetic_Appeal'], row['Clinical_Authority']),
                'type': row['Type']
            }
        return result

    def summary(self) -> Dict[str, Any]:
        """Return a summary of all available data."""
        return {
            'datasets': [
                'seasonality',
                'stressor_radar',
                'positioning',
                'market_share',
                'product_lines',
                'channels',
                'opportunity_matrix',
                'consumer_barriers'
            ],
            'total_datasets': 8,
            'key_insight': '250 untapped winter days opportunity',
            'strategic_focus': 'Facial premiumization + winter protection'
        }
