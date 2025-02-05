"""Main entry point for SEO Ninja."""
import os
import logging
from dotenv import load_dotenv
from seoninja.api.app import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure data directories exist
os.makedirs('data/cache', exist_ok=True)

if __name__ == '__main__':
    load_dotenv()
    app.run(debug=True)