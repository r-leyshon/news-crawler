"""
Vercel serverless function entry point for FastAPI backend.
"""
import sys
from pathlib import Path

# Add backend directory to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

# Import the FastAPI app
from main import app

# Vercel expects the app to be named 'app' or 'handler'
handler = app
