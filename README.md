# Financial Data Analysis Tool

A powerful Python-based application for analyzing financial and business data through interactive visualizations, AI-powered insights, and automated reporting.

## Features

- 📊 Interactive data visualizations
- 🤖 AI-powered data analysis
- 📝 Automated report generation
- 💬 Natural language querying
- 📈 Multi-file data processing
- 🔄 Real-time data updates

## Prerequisites

1. **Python Environment**
   - Python 3.8 or higher
   - pip package manager

2. **OpenAI API Key**
   - Required for default GPT-4 model
   - Get it from [OpenAI Platform](https://platform.openai.com)

3. **Ollama (Optional)**
   - Required for local Llama3.2 model
   - Follow installation steps below

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/financial-data-analysis.git
   cd financial-data-analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

4. **Install Ollama (Optional, for local Llama2)**
   ```bash
   # macOS or Linux
   curl https://ollama.ai/install.sh | sh

   # Windows
   # Download from https://ollama.ai/download
   ```

5. **Pull Llama3.2 model (Optional)**
   ```bash
   ollama pull llama3.2
   ```

## Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Upload CSV files**
   - Click "Browse files" button
   - Select one or more CSV files
   - Wait for processing to complete

3. **Explore visualizations**
   - View time series plots
   - Analyze category distributions
   - Compare monthly trends

4. **Chat with your data**
   - Ask questions in natural language
   - Get AI-powered insights
   - Generate detailed reports

## File Structure

```
financial-data-analysis/
├── app.py                 # Main Streamlit application
├── config.py             # Configuration and environment settings
├── requirements.txt      # Python dependencies
├── .env                 # Environment variables
├── data_processor.py    # Data processing and metrics calculation
├── visualization.py     # Plotly-based data visualization
├── data_embedder.py     # ChromaDB vector database management
├── conversation_handler.py # AI chat and query processing
├── report_generator.py  # PDF report generation with FPDF
├── models/
│   └── model_manager.py # AI model management (OpenAI/Llama2)
└── utils/
    ├── logger.py        # Logging system
    ├── validators.py    # Data validation
    └── cache_manager.py # Result caching
```

## Data Format Requirements

### CSV Structure
- Must include at least one date column
- Must include numeric columns for analysis
- Optional categorical columns for grouping

### Example Format
```csv
date,amount,category,type
2023-01-15,5000.00,Sales,revenue
2023-01-20,1200.00,Office Supplies,expense
```

### Column Types
1. **Date Columns**
   - ISO format (YYYY-MM-DD)
   - Consistent formatting

2. **Numeric Columns**
   - Decimal numbers
   - No currency symbols
   - Percentages as decimals

3. **Categorical Columns**
   - Text data
   - Consistent naming
   - Case-sensitive

## Features in Detail

### 1. Data Processing
- Automatic type inference
- Missing value handling
- Data validation
- Aggregation management

### 2. Visualizations
- Time series analysis
- Category distributions
- Monthly comparisons
- Interactive charts

### 3. AI Integration
- Natural language processing
- Context-aware responses
- Multi-model support
- Automated insights

### 4. Report Generation
- PDF format
- Executive summaries
- Key metrics
- Downloadable reports

## Model Options

### 1. OpenAI GPT-4 (Default)
- Requires API key
- Cloud-based processing
- Best for complex analysis

### 2. Local Llama2 (Optional)
- Runs locally
- No API key needed
- Privacy-focused option

## Performance Optimization

1. **Data Management**
- Batch processing
- Efficient storage
- Memory optimization
- Cache management

2. **Query Processing**
- Indexed searches
- Result caching
- Parallel processing
- Optimized aggregations

## Security

1. **API Security**
- Secure key storage
- Regular rotation
- Access control
- Rate limiting

2. **Data Privacy**
- Local processing
- No external storage
- Session isolation
- Secure cleanup

## Troubleshooting

### Common Issues

1. **Database Connection**
   ```
   ValueError: Could not connect to tenant default_tenant
   ```
   Solution: Reset the database using the "Reset Database" button

2. **Model Switching**
   ```
   Error: Failed to switch model
   ```
   Solution: Ensure Ollama is running for local models

3. **CSV Processing**
   ```
   Error: Invalid file format
   ```
   Solution: Verify CSV format matches requirements

### Best Practices

1. **Data Preparation**
   - Clean data before upload
   - Verify column formats
   - Remove duplicates
   - Handle missing values

2. **Model Selection**
   - Use GPT-4 for complex analysis
   - Use Llama2 for privacy needs
   - Monitor API usage
   - Test both options

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for GPT-4 API
- Ollama team for Llama2 integration
- Streamlit for the UI framework
- ChromaDB for vector storage

## Support

For support, please:
1. Check the documentation
2. Search existing issues
3. Create a new issue if needed
4. Contact the maintainers

## Roadmap

- [ ] Additional visualization types
- [ ] Enhanced report templates
- [ ] More AI model options
- [ ] Advanced data processing
- [ ] Improved error handling