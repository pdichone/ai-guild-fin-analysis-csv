from fpdf import FPDF
import pandas as pd
import streamlit as st


class ReportGenerator:
    def __init__(self):
        self.pdf = FPDF()
        # Enable UTF-8 encoding support
        self.pdf.set_auto_page_break(auto=True, margin=15)

    def _clean_text(self, text):
        """Clean text to handle Unicode characters"""
        # Replace problematic characters with their ASCII equivalents
        replacements = {
            '"': '"',
            '"': '"',
            """: "'",
            """: "'",
            "–": "-",
            "—": "-",
            "…": "...",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def generate_report(
        self, summary_metrics, insights, filename="financial_report.pdf"
    ):
        # Generate PDF
        self.pdf.add_page()

        # Add title
        self.pdf.set_font("Arial", "B", 16)
        self.pdf.cell(0, 10, "Financial Analysis Report", ln=True, align="C")

        # Add summary metrics
        self.pdf.set_font("Arial", "B", 12)
        self.pdf.cell(0, 10, "Summary Metrics:", ln=True)
        self.pdf.set_font("Arial", "", 10)
        for metric, value in summary_metrics.items():
            metric_text = self._clean_text(
                f'{metric.replace("_", " ").title()}: ${value:,.2f}'
            )
            self.pdf.cell(0, 10, metric_text, ln=True)

        # Add AI insights
        self.pdf.set_font("Arial", "B", 12)
        self.pdf.cell(0, 10, "AI-Generated Insights:", ln=True)
        self.pdf.set_font("Arial", "", 10)
        cleaned_insights = self._clean_text(insights)
        self.pdf.multi_cell(0, 10, cleaned_insights)

        # Save PDF
        self.pdf.output(filename)

        # Display report in UI with enhanced styling
        st.markdown(
            """
            <style>
                .report-container {
                    background-color: #ffffff;
                    padding: 2rem;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin: 1rem 0;
                }
                .report-title {
                    color: #1f1f1f;
                    text-align: center;
                    margin-bottom: 2rem;
                }
                .section-title {
                    color: #2c5282;
                    margin: 1.5rem 0 1rem 0;
                }
                .metric-container {
                    background-color: #f7fafc;
                    padding: 1rem;
                    border-radius: 5px;
                    margin-bottom: 1rem;
                }
                .insights-container {
                    background-color: #f7fafc;
                    padding: 1.5rem;
                    border-radius: 5px;
                    margin: 1rem 0;
                    white-space: pre-line;
                }
            </style>
        """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="report-container">', unsafe_allow_html=True)
        st.markdown(
            '<h2 class="report-title">Financial Analysis Report</h2>',
            unsafe_allow_html=True,
        )

        # Summary Metrics with enhanced display
        st.markdown(
            '<h3 class="section-title">Summary Metrics</h3>', unsafe_allow_html=True
        )
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, (metric, value) in enumerate(summary_metrics.items()):
            cols[i % 2].metric(metric.replace("_", " ").title(), f"${value:,.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

        # AI Insights with enhanced display
        st.markdown(
            '<h3 class="section-title">AI-Generated Insights</h3>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="insights-container">{insights}</div>', unsafe_allow_html=True
        )

        # Download button with enhanced styling
        st.markdown(
            """
            <style>
                .stDownloadButton button {
                    background-color: #2c5282;
                    color: white;
                    border: none;
                    padding: 0.5rem 1rem;
                    border-radius: 5px;
                    margin-top: 1rem;
                }
                .stDownloadButton button:hover {
                    background-color: #2a4365;
                }
            </style>
        """,
            unsafe_allow_html=True,
        )

        with open(filename, "rb") as pdf_file:
            st.download_button(
                label="Download PDF Report",
                data=pdf_file,
                file_name=filename,
                mime="application/pdf",
            )

        st.markdown("</div>", unsafe_allow_html=True)
