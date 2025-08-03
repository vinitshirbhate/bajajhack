# api.py - The entry point for Netlify Functions

from mangum import Mangum
# Import your existing FastAPI app instance from your main file
from main import app

# Create a handler which Netlify will use to run your app
handler = Mangum(app, lifespan="off")
