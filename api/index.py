import sys
import os

# Add the backend directory to the path for imports
backend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend")
sys.path.insert(0, backend_dir)

# Change to backend directory so relative paths work (for config.json etc)
os.chdir(backend_dir)

from main import app

# Vercel serverless function handler using Mangum
try:
    from mangum import Mangum
    handler = Mangum(app, lifespan="off")
except ImportError:
    handler = app
