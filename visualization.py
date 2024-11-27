"""
Data visualization module using Plotly for interactive charts.

This module handles:
- Time series visualizations
- Category distributions
- Monthly comparisons
- Interactive chart generation
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs import Scatter
import pandas as pd


class Visualizer:
    """
    Creates interactive visualizations for data analysis.

    Key responsibilities:
    - Generate time series plots
    - Create category distribution charts
    - Display monthly comparisons
    - Handle data aggregation for visualization
    """

    def __init__(self, data_processor):
        """
        Initialize Visualizer with a data processor.

        Args:
            data_processor: Instance of DataProcessor containing the data to visualize
        """
        self.data_processor = data_processor

    def create_time_series(self, numeric_col):
        """Create a time series visualization for a numeric column.

        Args:
            numeric_col: Name of the numeric column to plot

        Returns:
            plotly.Figure: Time series plot or None if no date column exists"""
        if not self.data_processor.date_column:
            return None

        fig = px.line(
            self.data_processor.df,
            x=self.data_processor.date_column,
            y=numeric_col,
            title=f"{numeric_col} Over Time",
        )
        return fig

    def create_category_distribution(self, numeric_col, category_col):
        """
        Create a pie chart showing distribution across categories.

        Args:
            numeric_col: Name of the numeric column for values
            category_col: Name of the categorical column for grouping

        Returns:
            plotly.Figure: Pie chart showing category distribution
        """
        fig = px.pie(
            self.data_processor.df.groupby(category_col)[numeric_col]
            .sum()
            .reset_index(),
            values=numeric_col,
            names=category_col,
            title=f"{numeric_col} Distribution by {category_col}",
        )
        return fig

    def create_monthly_comparison(self, numeric_col):
        """Create a combined bar and line chart for monthly analysis.

        Shows:
        - Monthly totals as bars
        - Monthly averages as a line

        Args:
            numeric_col: Name of the numeric column to analyze

        Returns:
            plotly.Figure: Combined bar and line chart or None if no month data"""
        if "month" not in self.data_processor.df.columns:
            return None

        monthly_data = self.data_processor.df.groupby("month")[numeric_col].agg(
            ["sum", "mean"]
        )

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=monthly_data.index, y=monthly_data["sum"], name=f"Total {numeric_col}"
            )
        )
        fig.add_trace(
            Scatter(
                x=monthly_data.index,
                y=monthly_data["mean"],
                name=f"Average {numeric_col}",
            )
        )
        fig.update_layout(
            title=f"Monthly {numeric_col} Analysis",
            xaxis_title="Month",
            yaxis_title=numeric_col,
        )
        return fig
