# AI Article Assistant

An intelligent article crawler and RAG (Retrieval-Augmented Generation) system that crawls the web for articles, stores them in a vector database, and allows you to ask questions about the collected content.

## Features

- **Article Crawling**: Search and crawl articles from the web based on keywords
- **Intelligent Summarization**: Automatically generate summaries using Azure OpenAI
- **Vector Storage**: Store articles in ChromaDB for efficient similarity search
- **RAG Q&A**: Ask questions and get answers based on your collected articles
- **Modern UI**: Next.js frontend with a clean, responsive interface

## Project Structure

```
news-crawler/
├── app/                    # Next.js frontend
├── backend/               # FastAPI backend
│   ├── main.py           # Main FastAPI application
│   ├── config.json       # Configuration settings
│   ├── requirements.txt  # Python dependencies
│   └── .env             # Environment variables (secrets)
├── components/           # Shared UI components
└── scripts/             # Setup and utility scripts
```

## Prerequisites

- Python 3.11+
- Node.js 18+
- Conda (recommended) or Python venv
- Azure OpenAI API access

## Setup

1. **Clone and setup the project**:
   ```bash
   chmod +x scripts/setup.sh
   ./scripts/setup.sh
   ```

2. **Configure environment variables**:
   Create `backend/.env` with your Azure OpenAI credentials:
   ```
   AZURE_OPENAI_KEY=your_azure_openai_key_here
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   ```

3. **Activate the conda environment**:
   ```bash
   cd backend && conda activate ./backend-env
   cd ..  # Return to project root
   ```

## Running the Application

### Start the Backend (from project root)
```bash
uvicorn backend.main:app --reload
```

The backend will be available at: http://localhost:8000

### Start the Frontend (in a new terminal)
```bash
npm run dev
```

The frontend will be available at: http://localhost:3000

## API Endpoints

- `POST /crawl` - Crawl articles based on keywords
- `POST /ask` - Ask questions about collected articles
- `GET /articles` - Get all stored articles
- `DELETE /articles/{id}` - Delete a specific article
- `GET /health` - Health check

## Configuration

Edit `backend/config.json` to customize:
- Search keywords
- Crawl settings
- Azure OpenAI deployment names
- Embedding settings

## Usage

1. Open the web interface at http://localhost:3000
2. Use the crawl feature to collect articles based on your interests
3. Ask questions about the collected articles using the Q&A interface
4. Browse and manage your article collection

## Tech Stack

- **Frontend**: Next.js, React, TypeScript, Tailwind CSS
- **Backend**: FastAPI, Python
- **Vector Database**: ChromaDB
- **AI**: Azure OpenAI (GPT-4, text-embedding-ada-002)
- **Web Scraping**: aiohttp, BeautifulSoup4

## Development

The application supports hot reload for both frontend and backend development.

- Frontend changes are automatically reflected
- Backend changes trigger automatic restart with `--reload` flag
