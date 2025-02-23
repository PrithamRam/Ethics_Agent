# Medical Ethics AI Assistant

A sophisticated AI system that helps medical professionals analyze ethical dilemmas by combining literature review with AI-powered analysis.

## System Overview

The Medical Ethics AI Assistant processes ethical queries through three main components:
1. Literature Search - Finding relevant medical ethics papers
2. AI Analysis - Providing ethical guidance based on general principles
3. Literature Analysis - Synthesizing findings from relevant papers

### Architecture Diagram 

### Key Components

1. **Ethics Database** (`src/ethics_database.py`)
   - SQLite database with FTS5 search
   - Stores medical ethics papers with metadata
   - Intelligent search term extraction using GPT
   - Weighted relevance scoring

2. **Response Parser** (`src/response_parser.py`)
   - Generates structured analysis using GPT-4
   - Combines literature findings with ethical principles
   - Formats responses for clear presentation

3. **API Layer** (`src/api.py`)
   - FastAPI-based REST interface
   - Handles query processing and response generation
   - Manages conversation context

4. **Frontend** (`static/index.html`)
   - Clean, responsive interface
   - Three-section display (AI Analysis, Papers, Literature Analysis)
   - Async request handling

### Flow Diagram 