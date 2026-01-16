import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

from main import app

# Vercel serverless function handler using Mangum
try:
    from mangum import Mangum
    handler = Mangum(app)
except ImportError:
    handler = app
