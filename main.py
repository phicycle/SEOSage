import os
from dotenv import load_dotenv
from agent.seo_agent import SEOAgent
import json
import argparse
import logging
import warnings
import asyncio

warnings.filterwarnings("ignore", category=DeprecationWarning)

def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('seo_agent.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_env_variables():
    """Load environment variables"""
    load_dotenv()
    
    required_vars = ['MOZ_ACCESS_ID', 'MOZ_SECRET_KEY', 'OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
    return os.getenv('MOZ_ACCESS_ID'), os.getenv('MOZ_SECRET_KEY')

async def run_seo_agent(url: str, mode: str = "analyze", articles: int = 2) -> None:
    """
    Run the SEO agent with specified parameters.
    """
    logger = setup_logging()
    logger.info(f"Starting SEO Agent for {url}")
    
    try:
        # Load credentials
        access_id, secret_key = load_env_variables()
        
        # Initialize agent
        logger.info("Initializing SEO Agent...")
        agent = SEOAgent(access_id, secret_key)
        
        if mode == "analyze":
            logger.info("Starting website analysis...")
            results = await agent.analyze_website(url)
            print("\n📊 Analysis Results:")
            print(json.dumps(results, indent=2))
        elif mode == "generate":
            logger.info("Starting content generation...")
            results = await agent.generate_content_batch(url, articles)
            print("\n📝 Generated Content:")
            print(json.dumps(results, indent=2))
            
    except Exception as e:
        logger.error(f"Error running SEO agent: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        raise

def main():
    parser = argparse.ArgumentParser(description='SEO Agent - Automated SEO Analysis and Content Generation')
    parser.add_argument('url', help='Website URL to analyze')
    parser.add_argument('--mode', choices=['analyze', 'generate'], 
                       default='analyze', help='Operation mode')
    parser.add_argument('--articles', type=int, default=5,
                       help='Number of articles to generate')
    
    args = parser.parse_args()
    
    # Run async function
    asyncio.run(run_seo_agent(args.url, args.mode, args.articles))

if __name__ == "__main__":
    main() 