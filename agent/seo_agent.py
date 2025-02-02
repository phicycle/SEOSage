# Standard library imports
import json
import os
import pickle
from collections import defaultdict
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

# Third-party imports
from langchain.agents import Tool, initialize_agent
from langchain_openai import ChatOpenAI

# Local imports
from modules.crawler import WebsiteCrawler
from modules.keyword_research import KeywordResearch
from modules.content_generator import ContentGenerator
from modules.seo_optimizer import SEOOptimizer

class SEOAgent:
    """
    Agent that orchestrates SEO analysis, content generation, and optimization.
    
    Attributes:
        llm: Language model for content generation
        crawler: Website crawler instance
        keyword_research: Keyword research tool
        content_generator: Content generation tool
        seo_optimizer: SEO optimization tool
        cache_dir: Directory for caching analysis results
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize SEO agent with configuration."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing SEO Agent...")
        
        try:
            self.llm = ChatOpenAI(
                api_key=os.getenv('OPENAI_API_KEY'),
                temperature=0.7,
                model="gpt-4"
            )
            self.logger.info("LLM initialized successfully")
            
            # Initialize components
            self.crawler = None
            self.keyword_research = KeywordResearch(
                api_token=config.get('moz_token')
            )
            self.content_generator = ContentGenerator(self.llm)
            self.seo_optimizer = SEOOptimizer(config)
            self.cache_dir = "cache"
            
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Setup tools with error handling
            self.logger.info("Setting up tools...")
            self.tools = self._setup_tools()
            
            # Initialize agent with warning suppression
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.agent = self._initialize_agent()
            
            self.logger.info("SEO Agent initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing SEO Agent: {str(e)}")
            self.logger.error("Stack trace:", exc_info=True)
            raise

    def _initialize_agent(self) -> Any:
        """Initialize the LangChain agent with tools."""
        return initialize_agent(
            self.tools,
            self.llm,
            agent="zero-shot-react-description",
            verbose=True
        )

    def _setup_tools(self) -> List[Tool]:
        """Set up the available tools for the agent."""
        return [
            Tool(
                name="Crawler",
                func=self.crawl_website,
                description="Crawls website to analyze structure and content"
            ),
            Tool(
                name="KeywordResearch",
                func=self.research_keywords,
                description="Finds relevant keywords for the website"
            ),
            Tool(
                name="ContentGenerator",
                func=self.generate_content,
                description="Generates SEO-optimized content"
            ),
            Tool(
                name="SEOOptimizer",
                func=self.optimize_content,
                description="Applies SEO best practices to content"
            )
        ]

    def _get_cache_path(self, domain: str) -> str:
        """Get path for domain-specific cache file."""
        return os.path.join(self.cache_dir, f"{domain.replace('.', '_')}.pkl")

    def _load_cache(self, domain: str) -> Optional[Dict[str, Any]]:
        """Load cached analysis data for domain."""
        cache_path = self._get_cache_path(domain)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None

    def _save_cache(self, domain: str, data: Any) -> None:
        """Save analysis results to cache."""
        try:
            # Create cache directory if it doesn't exist
            os.makedirs('cache', exist_ok=True)
            
            # Sanitize domain for filename
            safe_domain = domain.replace('://', '_').replace('/', '_')
            cache_path = f'cache/{safe_domain}.pkl'
            
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            self.logger.debug(f"Saved cache for {domain}")
        except Exception as e:
            self.logger.error(f"Error saving cache: {str(e)}")

    async def analyze_website(self, domain: str) -> Dict[str, Any]:
        """
        Perform step-by-step analysis of website structure and keywords.
        
        Args:
            domain: Website domain to analyze
        
        Returns:
            Dict containing analysis results
        """
        self.logger.info(f"Starting analysis for domain: {domain}")
        
        try:
            print("\n" + "="*50)
            print(f"🔍 Starting analysis for domain: {domain}")
            print("="*50)
            
            # Step 1: Initialize and validate domain
            print("\n📊 Step 1: Initializing analysis...")
            if not domain:
                raise ValueError("Domain cannot be empty")
            
            # Step 2: Website Structure Analysis
            print("\n🌐 Step 2: Analyzing website structure...")
            structure_analysis = await self._analyze_structure(domain)
            if not structure_analysis:
                raise ValueError("Failed to analyze website structure")
            
            # Step 3: Keyword Research
            print("\n🔍 Step 3: Performing keyword research...")
            keyword_analysis = await self._analyze_keywords(domain)
            if not keyword_analysis:
                raise ValueError("Failed to perform keyword research")
            
            # Step 4: Content Analysis
            print("\n📝 Step 4: Analyzing existing content...")
            content_analysis = await self._analyze_content(domain)
            
            # Step 5: Competitor Analysis
            print("\n🏢 Step 5: Analyzing competitors...")
            competitor_analysis = await self._analyze_competitors(domain, keyword_analysis)
            
            # Combine all analyses
            analysis_results = {
                'structure': structure_analysis,
                'keywords': keyword_analysis,
                'content': content_analysis,
                'competitors': competitor_analysis,
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache the results
            self._save_cache(domain, analysis_results)
            
            self.logger.info("Analysis completed successfully")
            print("\n✅ Analysis completed successfully!")
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error analyzing website: {str(e)}")
            self.logger.error("Stack trace:", exc_info=True)
            raise

    async def _analyze_structure(self, domain: str) -> Dict[str, Any]:
        """Analyze website structure."""
        try:
            self.logger.info("Initializing crawler...")
            self.crawler = WebsiteCrawler(domain)
            
            self.logger.info("Starting crawl...")
            await self.crawler.crawl_site()
            
            self.logger.info("Analyzing crawled content...")
            analysis = self.crawler.analyze_content()
            self.logger.info(f"Analysis: {analysis}")
            if not analysis:
                raise ValueError("No structure data available")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Structure analysis failed: {str(e)}")
            return self.crawler._get_empty_analysis() if self.crawler else {}

    async def _analyze_keywords(self, domain: str) -> Dict[str, Any]:
        """Analyze keyword opportunities for domain."""
        try:
            print("  • Getting ranking opportunities...")
            opportunities = self.keyword_research.get_ranking_opportunities(domain)
            
            if not opportunities:
                print("  ⚠️ No keyword opportunities found")
                return {}
            
            print(f"  ✓ Found {len(opportunities)} keyword opportunities")
            
            # Find new keyword opportunities
            print("  • Finding additional keyword opportunities...")
            new_keywords = self.keyword_research.find_new_keywords(
                domain=domain,
                limit=10
            )
            
            if new_keywords:
                print(f"  ✓ Found {len(new_keywords)} additional keyword opportunities")
            
            return {
                "ranking_opportunities": {
                    "total": len(opportunities),
                    "keywords": opportunities
                },
                "new_opportunities": {
                    "total": len(new_keywords),
                    "keywords": new_keywords
                }
            }
            
        except Exception as e:
            self.logger.error(f"Keyword analysis failed: {str(e)}")
            print(f"  ❌ Keyword analysis failed: {str(e)}")
            return {}

    async def _analyze_content(self, domain: str) -> Dict[str, Any]:
        """Analyze existing content."""
        try:
            if not self.crawler or not self.crawler.content_data:
                return {}
            
            content_analysis = {
                'total_pages': len(self.crawler.content_data),
                'avg_word_count': 0,
                'content_issues': []
            }
            
            total_words = 0
            for url, content in self.crawler.content_data.items():
                word_count = len(content.split())
                total_words += word_count
                
                # Check content against SEO rules
                issues = self.seo_optimizer.apply_rules(content)
                if issues:
                    content_analysis['content_issues'].append({
                        'url': url,
                        'issues': issues
                    })
                
            if content_analysis['total_pages'] > 0:
                content_analysis['avg_word_count'] = total_words / content_analysis['total_pages']
            
            return content_analysis
            
        except Exception as e:
            self.logger.error(f"Content analysis failed: {str(e)}")
            return {}

    async def _analyze_competitors(self, domain: str, keyword_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze competitors for top keywords."""
        try:
            competitor_data = {}
            selected_keywords = keyword_analysis.get('selected_targets', {}).get('keywords', [])[:5]
            
            if not selected_keywords:
                return {}
                
            print(f"  • Analyzing competitors for top {len(selected_keywords)} keywords...")
            
            for keyword_data in selected_keywords:
                keyword = keyword_data['keyword']
                analysis = await self.analyze_competitors(domain, keyword)
                if 'error' not in analysis:
                    competitor_data[keyword] = analysis
                    print(f"  ✓ Analyzed competitors for '{keyword}'")
                
            return {
                "analyzed_keywords": len(competitor_data),
                "competitor_data": competitor_data
            }
            
        except Exception as e:
            self.logger.error(f"Competitor analysis failed: {str(e)}")
            print(f"  ❌ Competitor analysis failed: {str(e)}")
            return {}

    async def analyze_competitors(self, domain: str, keyword: str) -> Dict[str, Any]:
        """
        Analyze competitor content for a specific keyword
        """
        try:
            # Get SERP analysis for the keyword
            serp_data = self.keyword_research.get_serp_analysis(keyword)
            if not serp_data:
                return {'error': 'No SERP data available'}
                
            competitor_analysis = []
            organic_results = serp_data.get('organic_results', [])
            
            for result in organic_results[:10]:  # Analyze top 10 results
                if domain not in result.get('url', ''):  # Skip own domain
                    content = await self.crawler.get_page_content(result['url'])
                    analysis = {
                        'url': result.get('url', ''),
                        'position': result.get('position', 0),
                        'content_length': len(content.split()) if content else 0,
                        'page_score': await self.seo_optimizer.analyze_page(result['url'])
                    }
                    competitor_analysis.append(analysis)
            
            return {
                'keyword': keyword,
                'competitor_count': len(competitor_analysis),
                'avg_content_length': sum(a['content_length'] for a in competitor_analysis) / len(competitor_analysis) if competitor_analysis else 0,
                'competitors': competitor_analysis
            }
        except Exception as e:
            self.logger.error(f"Error analyzing competitors: {str(e)}")
            return {'error': str(e)}

    async def generate_content_batch(self, domain: str, num_articles: int = 2) -> List[Dict[str, Any]]:
        """
        Generate multiple pieces of content using cached analysis.
        
        Args:
            domain: Website domain
            num_articles: Number of articles to generate
            
        Returns:
            List of generated content with SEO analysis
        """
        print("\n" + "="*50)
        print(f"📝 Starting batch content generation for {domain}")
        print(f"Target: {num_articles} articles")
        print("="*50)
        
        analysis_data = self._load_cache(domain)
        if not analysis_data:
            print("❌ No analysis data found. Please run analyze_website first.")
            return []
        
        keywords = analysis_data.get('keywords', {})  # Remove json.loads since it's already a dict
        selected_keywords = keywords.get('selected_targets', {}).get('keywords', [])[:num_articles]
        
        results = []
        for idx, keyword_data in enumerate(selected_keywords, 1):
            result = await self._generate_single_content(idx, num_articles, keyword_data)
            results.append(result)
                
        print("\n" + "="*50)
        print(f"✨ Batch generation complete! Generated {len(results)} articles")
        print("="*50)
        
        return results

    async def _generate_single_content(self, idx: int, total: int, keyword_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate and optimize a single piece of content."""
        print(f"\n📌 Generating content {idx}/{total}")
        print(f"Keyword: {keyword_data['keyword']}")
        print(f"Search Volume: {keyword_data.get('search_volume', 'N/A')}")
        print(f"Difficulty: {keyword_data.get('difficulty', 'N/A')}")
        
        # Generate content
        print("🤖 Generating blog post...")
        content = await self.content_generator.generate_blog_post(keyword_data)
        
        # Optimize content
        print("🔍 Checking SEO optimization...")
        issues = await self.optimize_content(content)
        
        # Save content
        filename = await self.content_generator.save_as_markdown(content, keyword_data['keyword'])
        print(f"✅ Content saved to: {filename}")
        
        result = {
            'keyword': keyword_data['keyword'],
            'content': content,
            'issues': issues if issues else []
        }
        
        await self._print_optimization_results(result['issues'])
        return result

    def _print_optimization_results(self, issues: List[str]) -> None:
        """Print SEO optimization results."""
        if issues:
            print("\n⚠️ SEO Issues found:")
            for issue in issues:
                print(f"- {issue}")
        else:
            print("✅ Content is fully optimized")

    def crawl_website(self, domain: str) -> str:
        """Tool method for crawler"""
        try:
            if not self.crawler or self.crawler.domain != domain:
                self.crawler = WebsiteCrawler(domain)
            self.crawler.crawl()
            return json.dumps(self.crawler.analyze_content(), indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
        
    def research_keywords(self, domain: str) -> str:
        """Tool method for keyword research"""
        try:
            print(f"\n🔍 Researching keywords for {domain}...")
            opportunities = self.keyword_research.get_ranking_opportunities(domain)
            if not opportunities:
                print("❌ No keyword opportunities found")
                return json.dumps({"error": "No keyword opportunities found"})
            
            print(f"📊 Found {len(opportunities)} keyword opportunities")
            
            # Group by rank positions
            position_groups = defaultdict(list)
            for kw in opportunities:
                rank = kw.get('rank_position', 0)
                if 4 <= rank <= 10:
                    position_groups['top_10'].append(kw)
                elif 11 <= rank <= 20:
                    position_groups['11_20'].append(kw)
                else:
                    position_groups['21_30'].append(kw)
            
            print("\n📈 Keyword Opportunities Summary:")
            print(f"Total Opportunities: {len(opportunities)}")
            for pos, kws in position_groups.items():
                print(f"Position {pos}: {len(kws)} keywords")
            
            result = {
                "total_opportunities": len(opportunities),
                "opportunities_by_position": {
                    pos: len(kws) for pos, kws in position_groups.items()
                },
                "top_opportunities": opportunities[:10]  # Top 10 opportunities
            }
            
            return json.dumps(result, indent=2)
        except Exception as e:
            print(f"❌ Error in keyword research: {str(e)}")
            return json.dumps({"error": str(e)})
            
    def generate_content(self, keyword_data):
        """Tool method for content generation"""
        content = self.content_generator.generate_blog_post(keyword_data)
        filename = self.content_generator.save_as_markdown(content, keyword_data['keyword'])
        print(f"Blog saved as: {filename}")
        return content
        
    def optimize_content(self, content):
        """Tool method for content optimization"""
        return json.dumps(self.seo_optimizer.apply_rules(content), indent=2)
        
    def track_content_performance(self, domain: str, days: int = 30) -> Dict[str, Any]:
        """
        Track performance of generated content
        """
        analysis_data = self._load_cache(domain)
        if not analysis_data:
            raise ValueError("No analysis data found. Run analyze_website first.")
            
        # Get all generated content
        blog_dir = "blogs"
        performance_data = defaultdict(list)
        
        for keyword_dir in os.listdir(blog_dir):
            keyword_path = os.path.join(blog_dir, keyword_dir)
            if os.path.isdir(keyword_path):
                versions = sorted([f for f in os.listdir(keyword_path) if f.endswith('.md')])
                if versions:
                    latest_version = versions[-1]
                    with open(os.path.join(keyword_path, latest_version)) as f:
                        content = f.read()
                        
                    # Get current keyword position
                    try:
                        new_position = self.keyword_research.get_keyword_position(domain, keyword_dir.replace('_', ' '))
                        performance_data[keyword_dir].append({
                            'version': latest_version,
                            'current_position': new_position,
                            'improvement': analysis_data.get('original_position', 0) - new_position
                        })
                    except Exception as e:
                        self.logger.error(f"Error tracking {keyword_dir}: {str(e)}")
                        
        return dict(performance_data) 