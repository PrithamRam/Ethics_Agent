# Medical Ethics AI Assistant - Setup Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Elasticsearch (for paper database)
- Git (for version control)

## Elasticsearch Setup

1. The application includes Elasticsearch 8.12.1. To start Elasticsearch:
```bash
# On macOS/Linux
./elasticsearch-8.12.1/bin/elasticsearch

# On Windows
.\elasticsearch-8.12.1\bin\elasticsearch.bat
```

2. Verify Elasticsearch is running:
```bash
curl http://localhost:9200
```

3. Important Elasticsearch Notes:
- The server may take a few moments to start
- Default configuration files are in `elasticsearch-8.12.1/config/`
- Logs are available in `elasticsearch-8.12.1/logs/`
- Data is stored in `elasticsearch-8.12.1/data/`

## Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Ethics_Asst
```

2. Create and activate a virtual environment:
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
.\venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
pip install elasticsearch  # Additional required package
```

## Configuration

1. Create a `.env` file in the root directory with the following variables:
```
PUBMED_EMAIL=your.email@example.com
PUBMED_API_KEY=your_api_key
```

2. Ensure Elasticsearch is running:
- The application expects Elasticsearch to be running on `localhost:9200`
- If you're using a different Elasticsearch configuration, update the settings in `src/ethics_db.py`

## Running the Application

1. Make sure your virtual environment is activated:
```bash
source venv/bin/activate  # On macOS/Linux
.\venv\Scripts\activate   # On Windows
```

2. Start the application:
```bash
python3 main.py
```

3. Access the application:
- Open your web browser and navigate to: `http://127.0.0.1:8005`
- The application will initialize the database with medical ethics papers on first run

## Application Structure

The application consists of several key components:

1. **FastAPI Backend** (`main.py`, `src/api.py`)
   - Handles HTTP requests
   - Manages application lifecycle
   - Coordinates between components

2. **Ethics Database** (`src/ethics_db.py`)
   - Manages paper storage and retrieval
   - Interfaces with Elasticsearch
   - Handles paper indexing and search

3. **Medical Ethics Assistant** (`src/medical_ethics_assistant.py`)
   - Processes ethical queries
   - Integrates with PubMed
   - Generates AI-powered responses

4. **Frontend** (`static/index.html`)
   - User interface for interacting with the system
   - Displays results and analyses

## Troubleshooting

1. **Missing Dependencies**
   - If you encounter missing package errors, install them using:
   ```bash
   pip install <package-name>
   ```

2. **Elasticsearch Connection Issues**
   - Ensure Elasticsearch is running: `curl http://localhost:9200`
   - Check Elasticsearch logs for potential issues
   - Verify your Elasticsearch configuration matches the application settings

3. **Database Initialization Issues**
   - Check the `data/pubmed_papers.json` file exists
   - Ensure sufficient disk space for database creation
   - Verify Elasticsearch has appropriate permissions

## Additional Notes

- The application uses port 8005 by default
- Database initialization may take several minutes on first run
- The system requires an active internet connection for PubMed integration
- For production deployment, consider setting up proper security measures and SSL certificates