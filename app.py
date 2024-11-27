import streamlit as st
import pandas as pd
from data_processor import DataProcessor
from visualization import Visualizer
from data_embedder import DataEmbedder
from conversation_handler import ConversationHandler
from report_generator import ReportGenerator
from utils.logger import Logger
from models.model_manager import ModelManager


def initialize_session_state():
    """Initialize all session state variables."""
    try:
        # Initialize all required session state variables
        st.session_state.initialized = True
        st.session_state.data_processors = {}
        st.session_state.data_embedder = DataEmbedder()
        st.session_state.model_manager = ModelManager()
        st.session_state.conversation_handler = ConversationHandler(
            st.session_state.data_embedder, st.session_state.model_manager
        )
        st.session_state.report_generator = ReportGenerator()
        st.session_state.logger = Logger("streamlit_app")
        st.session_state.messages = []
        st.session_state.uploaded_files = set()
        st.session_state.current_file = None  # Track the most recently uploaded file
    except Exception as e:
        st.error(f"Error initializing application: {str(e)}")
        st.session_state.initialized = False


def process_uploaded_file(uploaded_file):
    """Process a single uploaded file."""
    try:
        # Create a new data processor for this file
        data_processor = DataProcessor()
        success, message = data_processor.load_data(uploaded_file)

        if success:
            # Store the processor with the filename as key
            st.session_state.data_processors[uploaded_file.name] = data_processor

            # Always embed the data for new files
            column_info = data_processor.get_column_info()
            embedding_success = st.session_state.data_embedder.embed_data(
                data_processor.df, column_info, source_name=uploaded_file.name
            )

            if embedding_success:
                st.session_state.uploaded_files.add(uploaded_file.name)
                st.session_state.current_file = (
                    uploaded_file.name
                )  # Update current file
                st.success(f"Successfully processed and embedded {uploaded_file.name}")
            else:
                st.error(f"Failed to embed data from {uploaded_file.name}")
        else:
            st.error(message)

        return success
    except Exception as e:
        st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
        return False


def main():
    st.set_page_config(page_title="CSV Data Analysis Tool", layout="wide")
    st.title("CSV Data Analysis Tool")

    # Initialize session state first
    initialize_session_state()

    if not st.session_state.initialized:
        st.error(
            "Failed to initialize application. Please refresh the page or contact support."
        )
        return

    # Model Selection and Database Reset in sidebar
    with st.sidebar:
        st.header("Settings")

        # Model Settings
        st.subheader("Model Settings")
        if st.session_state.model_manager:
            available_models = st.session_state.model_manager.get_available_models()
            current_model = st.session_state.conversation_handler.get_current_model()

            selected_model = st.selectbox(
                "Select AI Model",
                available_models,
                index=(
                    available_models.index(current_model)
                    if current_model in available_models
                    else 0
                ),
                key="model_selector",
                help="Choose between OpenAI GPT-4 (default) or Local Llama2",
            )

            # Update model if changed
            if selected_model != current_model:
                success, message = st.session_state.conversation_handler.set_model(
                    selected_model
                )
                if success:
                    st.success(f"Switched to {selected_model}")
                else:
                    st.error(message)

        # Database Reset
        st.subheader("Database Management")
        if st.button(
            "Reset Database",
            type="primary",
            help="Clear all embedded data and start fresh",
        ):
            try:
                st.session_state.data_embedder.clear_data()
                st.session_state.uploaded_files.clear()
                st.session_state.data_processors.clear()
                st.session_state.messages.clear()
                st.session_state.current_file = None
                st.success("Database successfully reset!")
                st.rerun()
            except Exception as e:
                st.error(f"Error resetting database: {str(e)}")

    # File upload section
    st.header("Data Upload")
    uploaded_files = st.file_uploader(
        "Upload your CSV files", type=["csv"], accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.data_processors:
                process_uploaded_file(uploaded_file)

        # Display currently available datasets
        st.header("Available Datasets")
        st.info(
            f"Currently loaded datasets: {', '.join(st.session_state.uploaded_files)}"
        )
        if st.session_state.current_file:
            st.success(
                f"Currently viewing analysis for: {st.session_state.current_file}"
            )

        # Display data analysis for the current file
        if st.session_state.current_file:
            data_processor = st.session_state.data_processors[
                st.session_state.current_file
            ]

            with st.expander(
                f"Analysis for {st.session_state.current_file}", expanded=True
            ):
                # Initialize visualizer for this dataset
                visualizer = Visualizer(data_processor)

                # Summary metrics
                st.subheader("Summary Metrics")
                metrics = data_processor.get_summary_metrics()
                display_metrics(metrics)

                # Generate Report button
                if st.button(
                    f"Generate Report for {st.session_state.current_file}",
                    key=f"report_{st.session_state.current_file}",
                ):
                    with st.spinner("Generating report..."):
                        # Get AI-generated insights
                        if st.session_state.conversation_handler:
                            insights = st.session_state.conversation_handler.process_query(
                                f"Provide a comprehensive analysis of the data in {st.session_state.current_file}. "
                                "Include key trends, notable patterns, and important observations."
                            )
                        else:
                            insights = "AI analysis not available. Please ensure the conversation handler is properly initialized."

                        # Generate and display the report
                        st.session_state.report_generator.generate_report(
                            metrics,
                            insights,
                            filename=f"report_{st.session_state.current_file.replace('.csv', '')}.pdf",
                        )

                # Visualization section
                st.subheader("Visualizations")

                numeric_cols = data_processor.numeric_columns
                if numeric_cols:
                    numeric_col = st.selectbox(
                        f"Select numeric column for analysis ({st.session_state.current_file})",
                        numeric_cols,
                        key=f"numeric_{st.session_state.current_file}",
                    )

                    if numeric_col:
                        tabs = st.tabs(
                            ["Time Series", "Category Distribution", "Monthly Analysis"]
                        )

                        with tabs[0]:
                            if data_processor.date_column:
                                with st.spinner(
                                    "Creating time series visualization..."
                                ):
                                    fig = visualizer.create_time_series(numeric_col)
                                    if fig:
                                        st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info(
                                    "No date column detected for time series visualization"
                                )

                        with tabs[1]:
                            if data_processor.categorical_columns:
                                category_col = st.selectbox(
                                    "Select category column",
                                    data_processor.categorical_columns,
                                    key=f"category_{st.session_state.current_file}",
                                )
                                if category_col:
                                    with st.spinner(
                                        "Creating category distribution..."
                                    ):
                                        fig = visualizer.create_category_distribution(
                                            numeric_col, category_col
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info(
                                    "No categorical columns detected for distribution analysis"
                                )

                        with tabs[2]:
                            if "month" in data_processor.df.columns:
                                with st.spinner("Creating monthly comparison..."):
                                    fig = visualizer.create_monthly_comparison(
                                        numeric_col
                                    )
                                    if fig:
                                        st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("No monthly data available for comparison")

        # Chat interface
        st.markdown("---")
        st.header("Chat with Your Data")

        chat_container = st.container()

        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("Ask about your data"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Analyzing your question..."):
                        try:
                            response = (
                                st.session_state.conversation_handler.process_query(
                                    prompt
                                )
                            )
                            print(f"====>{response}")
                            st.markdown(response)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": response}
                            )
                        except Exception as e:
                            error_msg = "Sorry, I couldn't process your question. Please try again."
                            st.error(error_msg)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": error_msg}
                            )


def display_metrics(metrics):
    num_metrics = len(metrics)
    num_rows = (num_metrics + 1) // 2

    for row in range(num_rows):
        col1, col2 = st.columns(2)

        idx = row * 2
        if idx < num_metrics:
            metric_name = list(metrics.keys())[idx]
            value = metrics[metric_name]
            with col1:
                st.metric(
                    label=metric_name.replace("_", " ").title(),
                    value=(
                        f"${value:,.2f}"
                        if "amount" in metric_name.lower()
                        or "budget" in metric_name.lower()
                        else f"{value:,.2f}"
                    ),
                    help=f"Full value: {value:,.2f}",
                )

        idx = row * 2 + 1
        if idx < num_metrics:
            metric_name = list(metrics.keys())[idx]
            value = metrics[metric_name]
            with col2:
                st.metric(
                    label=metric_name.replace("_", " ").title(),
                    value=(
                        f"${value:,.2f}"
                        if "amount" in metric_name.lower()
                        or "budget" in metric_name.lower()
                        else f"{value:,.2f}"
                    ),
                    help=f"Full value: {value:,.2f}",
                )


if __name__ == "__main__":
    main()
