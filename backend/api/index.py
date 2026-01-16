import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app

# Export the FastAPI app as 'app' for Vercel's ASGI support
# This tells Vercel to treat this as an ASGI application
