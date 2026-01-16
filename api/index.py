"""
Vercel serverless function entry point for FastAPI backend.
"""
import sys
import os

# Add backend directory to path so we can import main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

# Import the FastAPI app and Mangum handler from backend
from main import app, handler

# Export both for Vercel's ASGI/Lambda support
