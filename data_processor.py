import pandas as pd
from datetime import datetime
import numpy as np
from utils.logger import Logger


class DataProcessor:
    def __init__(self):
        self.df = None
        self.date_column = None
        self.numeric_columns = []
        self.categorical_columns = []
        self.logger = Logger("data_processor")

    def load_data(self, file):
        try:
            self.df = pd.read_csv(file)
            self._infer_column_types()
            self._process_dates()
            return True, "Data loaded successfully"
        except Exception as e:
            return False, f"Error loading data: {str(e)}"

    def _infer_column_types(self):
        # Reset columns
        self.date_column = None
        self.numeric_columns = []
        self.categorical_columns = []

        # Identify date columns
        for col in self.df.columns:
            try:
                pd.to_datetime(self.df[col])
                self.date_column = col
                break
            except:
                continue

        # Identify numeric and categorical columns
        for col in self.df.columns:
            if col != self.date_column:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    self.numeric_columns.append(col)
                else:
                    self.categorical_columns.append(col)

        # Ensure numeric columns are properly formatted
        for col in self.numeric_columns:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

    def _process_dates(self):
        if self.date_column:
            self.df[self.date_column] = pd.to_datetime(self.df[self.date_column])
            self.df["year"] = self.df[self.date_column].dt.year
            self.df["month"] = self.df[self.date_column].dt.month
            self.df["quarter"] = self.df[self.date_column].dt.quarter

    def _aggregate_by_project(self, col):
        """Aggregate values by project if applicable."""
        if "project_id" in self.df.columns:
            return self.df.groupby("project_id")[col].sum().reset_index()[col]
        return self.df[col]

    def get_summary_metrics(self):
        """Get summary metrics with consistent project-level aggregation."""
        if self.df is None:
            return {}

        metrics = {}

        # Calculate summary statistics for numeric columns
        for col in self.numeric_columns:
            # For budget, amount, or monetary columns, calculate by project first
            if any(
                term in col.lower()
                for term in ["budget", "amount", "cost", "price", "revenue"]
            ):
                series = self._aggregate_by_project(col)
            else:
                series = self.df[col]

            metrics[f"total_{col}"] = float(series.sum())
            metrics[f"average_{col}"] = float(series.mean())
            metrics[f"max_{col}"] = float(series.max())
            metrics[f"min_{col}"] = float(series.min())

        return metrics

    def get_column_info(self):
        """Get information about the columns in the DataFrame."""
        if self.df is None:
            return {
                "date_column": None,
                "numeric_columns": [],
                "categorical_columns": [],
            }

        return {
            "date_column": self.date_column,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
        }

    def get_aggregated_data(self, column, group_by=None):
        """Get consistently aggregated data for visualizations and analysis."""
        if column not in self.df.columns:
            return None

        if group_by and group_by in self.df.columns:
            # For monetary columns, aggregate by project first if applicable
            if any(
                term in column.lower()
                for term in ["budget", "amount", "cost", "price", "revenue"]
            ):
                if "project_id" in self.df.columns:
                    # First aggregate by project
                    project_data = (
                        self.df.groupby(["project_id", group_by])[column]
                        .sum()
                        .reset_index()
                    )
                    # Then aggregate by the requested group
                    return project_data.groupby(group_by)[column].agg(
                        ["sum", "mean", "max", "min"]
                    )

            # For non-monetary columns or when no project_id is available
            return self.df.groupby(group_by)[column].agg(["sum", "mean", "max", "min"])

        # If no grouping, still aggregate by project for monetary columns
        if any(
            term in column.lower()
            for term in ["budget", "amount", "cost", "price", "revenue"]
        ):
            return self._aggregate_by_project(column)

        return self.df[column]

    def prepare_data_context(self):
        """Prepare data context for AI analysis with consistent aggregation."""
        context = {}

        # Add basic dataset info
        context["total_rows"] = len(self.df)
        context["columns"] = list(self.df.columns)

        # Add aggregated metrics for monetary columns
        for col in self.numeric_columns:
            if any(
                term in col.lower()
                for term in ["budget", "amount", "cost", "price", "revenue"]
            ):
                series = self._aggregate_by_project(col)
                context[f"{col}_metrics"] = {
                    "total": float(series.sum()),
                    "average": float(series.mean()),
                    "max": float(series.max()),
                    "min": float(series.min()),
                }

        return context
