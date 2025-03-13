# AI Research Assistant Workshop Stubs

This directory contains the necessary files to get started with the AI Research Assistant workshop.

## Quickstart Guide

### Prerequisites
- Python 3.8 or higher
- Git
- Anthropic API key (get one at https://console.anthropic.com/)

### Quick Setup (5 minutes)
```bash
# 1. Clone and enter directory
git clone <repository-url>
cd workshop_stubs

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.template .env
# Edit .env with your Anthropic API key
```

### Quick Test
```bash
# Run the Streamlit app
streamlit run app.py
```

## Detailed Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd workshop_stubs
   ```

2. **Create and Activate Virtual Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**
   ```bash
   # Copy the template file
   cp .env.template .env
   
   # Edit .env and add your Anthropic API key
   # You can get an API key from: https://console.anthropic.com/
   ```

5. **Verify Installation**
   ```bash
   # Run tests to verify setup
   pytest
   ```

## Project Structure

```
workshop_stubs/
├── .env.template          # Template for environment variables
├── .gitignore            # Git ignore rules
├── README.md             # This file
└── requirements.txt      # Python dependencies
```

## Getting Help

If you encounter any issues during setup:

1. Check the main workshop README for detailed instructions
2. Ensure your Python version is 3.8 or higher
3. Verify that your Anthropic API key is valid
4. Make sure all dependencies are installed correctly

## Next Steps

After completing the setup:

1. Follow the workshop instructions in the main README
2. Create the necessary Python files (agents.py, tasks.py, crew.py, app.py)
3. Run the Streamlit application with `streamlit run app.py` 