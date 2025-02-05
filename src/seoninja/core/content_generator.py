"""Core content generation functionality."""
import os
from datetime import datetime
from typing import Dict, Any, Optional, Set, List, Tuple
import json
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import numpy as np
from collections import Counter
import re
import random

class ContentGenerator:
    """
    Enhanced content generator with advanced NLP and AI capabilities.
    
    Attributes:
        llm: Language model instance
        template: Base template for content generation
        website_context: Dictionary storing website context from crawler
        internal_links: Dictionary mapping topics to relevant internal URLs
        vectorizer: TF-IDF vectorizer for topic modeling
        lemmatizer: WordNet lemmatizer for text processing
    """
    
    def __init__(self, llm: Any) -> None:
        """Initialize content generator with language model."""
        self.llm = llm
        self.template = self._get_base_template()
        self.website_context = {}
        self.internal_links = {}
        self.cache_dir = "data/cache/content"
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLP components
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3)
        )
        self.lemmatizer = WordNetLemmatizer()
        
        # Create necessary directories
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs("data/output/blogs", exist_ok=True)
        
    def _get_base_template(self) -> str:
        """Get the enhanced base template for content generation."""
        return """Write a comprehensive, SEO-optimized blog post about {keyword} in Markdown format.

Structure:
# {title}

## Executive Summary
[A compelling 150-word summary of the key points]

## Introduction
[Engaging introduction with hook, context, and value proposition]

{main_sections}

## Expert Insights
[Include expert quotes and analysis]

## Practical Applications
[Real-world examples and use cases]

## Best Practices & Tips
[Actionable advice and recommendations]

## Industry Trends
[Current and future trends]

## Common Challenges & Solutions
[Address pain points and provide solutions]

## Conclusion
[Summarize key points and include call to action]

## Frequently Asked Questions
{faq_section}

Requirements:
- Primary keyword usage: {keyword_requirements}
- Secondary keywords: {secondary_keywords}
- Internal linking opportunities: {internal_linking}
- Content depth: {depth_requirements}
- User intent: {user_intent}
- Target audience: {audience}

Blog Post:"""

    def _get_intent_template(self, intent: str) -> str:
        """Get enhanced template based on search intent."""
        intent_templates = {
            'informational': """Create an in-depth educational guide about {keyword}.
            
            Focus areas:
            1. Comprehensive explanation of concepts
            2. Step-by-step tutorials
            3. Visual aids and diagrams
            4. Expert insights and research
            5. Common misconceptions
            6. Further reading resources
            
            Style: Educational, thorough, well-researched""",
            
            'commercial': """Create a detailed comparison and analysis of {keyword}.
            
            Focus areas:
            1. Feature comparison matrix
            2. Pricing analysis
            3. Pros and cons
            4. Use case scenarios
            5. ROI calculations
            6. Expert recommendations
            
            Style: Analytical, data-driven, objective""",
            
            'transactional': """Create a comprehensive buying guide for {keyword}.
            
            Focus areas:
            1. Product specifications
            2. Pricing tiers
            3. Purchase process
            4. Warranty information
            5. Customer support details
            6. Post-purchase tips
            
            Style: Clear, actionable, persuasive""",
            
            'navigational': """Create a detailed overview of {keyword}.
            
            Focus areas:
            1. Brand/Product history
            2. Key features
            3. Use cases
            4. Customer testimonials
            5. Support resources
            6. Contact information
            
            Style: Informative, straightforward, helpful"""
        }
        return intent_templates.get(intent, self.template)

    def set_website_context(self, crawler_data: Dict[str, Any]) -> None:
        """
        Set website context from crawler data.
        
        Args:
            crawler_data: Dictionary containing crawled website data
        """
        self.website_context = crawler_data
        self._build_internal_link_index()
        
    def _build_internal_link_index(self) -> None:
        """Build an index of topics to URLs for internal linking."""
        for url, page_data in self.website_context.items():
            # Extract topics from title and headers
            topics = set()
            if 'title' in page_data:
                topics.update(self._extract_topics(page_data['title']))
            
            if 'headers' in page_data:
                for level in ['h1', 'h2', 'h3']:
                    for header in page_data['headers'].get(level, []):
                        topics.update(self._extract_topics(header))
            
            # Add URL to each topic's list of relevant pages
            for topic in topics:
                if topic not in self.internal_links:
                    self.internal_links[topic] = []
                self.internal_links[topic].append({
                    'url': url,
                    'title': page_data.get('title', ''),
                    'relevance': self._calculate_relevance(topic, page_data)
                })
                
    def _extract_topics(self, text: str) -> Set[str]:
        """Extract main topics from text."""
        # Remove common stop words and special characters
        words = text.lower().split()
        topics = set()
        
        # Extract single word topics
        topics.update(words)
        
        # Extract multi-word topics (up to 3 words)
        for i in range(len(words)-1):
            topics.add(f"{words[i]} {words[i+1]}")
            if i < len(words)-2:
                topics.add(f"{words[i]} {words[i+1]} {words[i+2]}")
                
        return topics
        
    def _calculate_relevance(self, topic: str, page_data: Dict[str, Any]) -> float:
        """Calculate relevance score of a page for a topic."""
        score = 0.0
        
        try:
            # Check title
            title = str(page_data.get('title', ''))
            if topic in title.lower():
                score += 3.0
            
            # Check headers
            headers = page_data.get('headers', {})
            if isinstance(headers, dict):
                for level, header_list in headers.items():
                    if isinstance(header_list, list):
                        weight = 4 - int(level[-1]) if level[-1].isdigit() else 1  # h1=3, h2=2, h3=1
                        for header in header_list:
                            if isinstance(header, str) and topic in header.lower():
                                score += weight
            
            # Check content
            content = page_data.get('content', '')
            if isinstance(content, str):
                score += content.lower().count(topic) * 0.1
            elif isinstance(content, dict) and 'text' in content:
                score += str(content['text']).lower().count(topic) * 0.1
            
        except Exception as e:
            self.logger.error(f"Error calculating relevance for topic {topic}: {str(e)}")
            return 0.0
            
        return score

    def _get_cache_path(self, keyword: str) -> str:
        """Get cache file path for a keyword."""
        safe_keyword = keyword.replace(' ', '_').replace('/', '_')
        return os.path.join(self.cache_dir, f"{safe_keyword}.json")
        
    def _load_from_cache(self, keyword: str) -> Optional[str]:
        """Load content from cache if available."""
        cache_path = self._get_cache_path(keyword)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading cache for {keyword}: {str(e)}")
        return None
        
    def _save_to_cache(self, keyword: str, content: str) -> None:
        """Save content to cache."""
        try:
            cache_path = self._get_cache_path(keyword)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(content, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving cache for {keyword}: {str(e)}")

    async def generate_blog_post(self, keyword_data: Dict[str, Any]) -> str:
        """Enhanced blog post generation with advanced features."""
        keyword = keyword_data['keyword']
        
        # Check cache first
        cached_content = self._load_from_cache(keyword)
        if cached_content:
            self.logger.info("Using cached content")
            return cached_content
            
        self.logger.info("Generating new content with enhanced features...")
        
        # Analyze keyword and gather insights
        keyword_insights = self._analyze_keyword(keyword)
        intent = keyword_data.get('intent', self._determine_intent(keyword))
        audience = self._identify_target_audience(keyword)
        
        # Generate content structure
        content_structure = await self._generate_content_structure(keyword, intent)
        
        # Get relevant context and competitor insights
        context = self._get_enhanced_context(keyword)
        competitor_insights = self._analyze_competitor_content(keyword)
        
        # Prepare template variables
        template_vars = {
            'keyword': keyword,
            'title': self._generate_seo_title(keyword, intent),
            'main_sections': content_structure,
            'faq_section': await self._generate_faqs(keyword),
            'keyword_requirements': self._get_keyword_requirements(keyword_insights),
            'secondary_keywords': ', '.join(keyword_insights['secondary_keywords']),
            'internal_linking': self._get_internal_linking_opportunities(keyword),
            'depth_requirements': self._get_depth_requirements(intent),
            'user_intent': intent,
            'audience': audience
        }
        
        # Generate initial content
        prompt = PromptTemplate(
            input_variables=list(template_vars.keys()),
            template=self._get_intent_template(intent)
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        content = await chain.arun(**template_vars)
        
        # Enhance content
        content = await self._enhance_content(content, keyword_data, competitor_insights)
        
        # Save to cache
        self._save_to_cache(keyword, content)
        
        return content

    def _analyze_keyword(self, keyword: str) -> Dict[str, Any]:
        """Perform in-depth keyword analysis."""
        analysis = {
            'main_topic': self._extract_main_topic(keyword),
            'secondary_keywords': self._generate_secondary_keywords(keyword),
            'semantic_field': self._get_semantic_field(keyword),
            'complexity_level': self._analyze_complexity(keyword),
            'search_volume_category': self._categorize_search_volume(keyword),
            'competition_level': self._analyze_competition_level(keyword)
        }
        return analysis

    def _enhance_content(self, content: str, keyword_data: Dict[str, Any], competitor_insights: Dict[str, Any]) -> str:
        """Apply advanced content enhancements."""
        enhanced = content
        
        # Apply semantic enhancements
        enhanced = self._add_semantic_enhancements(enhanced, keyword_data)
        
        # Add LSI keywords
        enhanced = self._add_lsi_keywords(enhanced, keyword_data['keyword'])
        
        # Improve readability
        enhanced = self._improve_readability(enhanced)
        
        # Add schema markup
        enhanced = self._add_schema_markup(enhanced, keyword_data)
        
        # Optimize for featured snippets
        enhanced = self._optimize_for_featured_snippets(enhanced)
        
        # Add competitor insights
        enhanced = self._incorporate_competitor_insights(enhanced, competitor_insights)
        
        return enhanced

    def _analyze_competitor_content(self, keyword: str) -> Dict[str, Any]:
        """Analyze competitor content for insights."""
        # Implementation would include actual competitor analysis logic
        return {
            'avg_word_count': 2000,
            'common_headings': ['Benefits', 'Features', 'How to'],
            'content_gaps': ['Technical details', 'Case studies'],
            'key_topics': ['implementation', 'best practices', 'tools']
        }

    def _improve_readability(self, content: str) -> str:
        """Improve content readability."""
        sentences = sent_tokenize(content)
        improved_sentences = []
        
        for sentence in sentences:
            # Analyze sentence complexity
            blob = TextBlob(sentence)
            if blob.sentiment.polarity < -0.2:
                # Make negative sentences more constructive
                sentence = self._rephrase_constructively(sentence)
            
            if len(sentence.split()) > 25:
                # Break down long sentences
                sentence = self._simplify_sentence(sentence)
            
            improved_sentences.append(sentence)
        
        return ' '.join(improved_sentences)

    def _add_schema_markup(self, content: str, keyword_data: Dict[str, Any]) -> str:
        """Add schema markup for better SEO."""
        # Add article schema
        schema = {
            "@context": "https://schema.org",
            "@type": "Article",
            "headline": self._extract_title(content),
            "keywords": keyword_data['keyword'],
            "articleBody": content,
            "datePublished": datetime.now().isoformat()
        }
        
        # Add schema at the end of content
        return f"{content}\n\n<!-- Schema.org Markup -->\n<script type=\"application/ld+json\">\n{json.dumps(schema, indent=2)}\n</script>"

    def _optimize_for_featured_snippets(self, content: str) -> str:
        """Optimize content for featured snippets."""
        # Add definition-style paragraphs
        if not re.search(r'^[A-Z][^.!?]+ is [^.!?]+[.!?]', content, re.MULTILINE):
            title = self._extract_title(content)
            definition = f"{title} is a comprehensive guide that explains everything you need to know about this topic."
            content = f"{definition}\n\n{content}"
        
        # Add list-style content
        if "## Steps" not in content and "## How to" not in content:
            steps_section = "\n## Steps to Implement\n1. First step\n2. Second step\n3. Third step\n"
            content += steps_section
        
        return content

    def _extract_title(self, content: str) -> str:
        """Extract title from content."""
        match = re.search(r'^# (.+)$', content, re.MULTILINE)
        return match.group(1) if match else ""

    def _simplify_sentence(self, sentence: str) -> str:
        """Break down complex sentences."""
        # Implementation would include sentence simplification logic
        return sentence

    def _rephrase_constructively(self, sentence: str) -> str:
        """Rephrase negative sentences constructively."""
        # Implementation would include rephrasing logic
        return sentence

    def _get_semantic_field(self, keyword: str) -> Set[str]:
        """Get semantic field for a keyword."""
        # Implementation would include semantic field analysis
        return set()

    def _analyze_complexity(self, keyword: str) -> str:
        """Analyze keyword complexity."""
        words = word_tokenize(keyword)
        return 'high' if len(words) > 3 else 'medium' if len(words) > 1 else 'low'

    def _categorize_search_volume(self, keyword: str) -> str:
        """Categorize search volume."""
        # Implementation would include actual search volume analysis
        return 'medium'

    def _analyze_competition_level(self, keyword: str) -> str:
        """Analyze competition level."""
        # Implementation would include actual competition analysis
        return 'medium'

    def _get_depth_requirements(self, intent: str) -> str:
        """Get content depth requirements based on intent."""
        depth_map = {
            'informational': 'Comprehensive coverage with 2000+ words',
            'commercial': 'Detailed analysis with 1500+ words',
            'transactional': 'Clear and concise with 1000+ words',
            'navigational': 'Essential information with 800+ words'
        }
        return depth_map.get(intent, 'Standard depth with 1200+ words')

    async def _generate_faqs(self, keyword: str) -> str:
        """Generate relevant FAQs using AI."""
        # Implementation would include FAQ generation logic
        return "1. What is {keyword}?\n2. How does {keyword} work?\n3. Why is {keyword} important?"

    def _generate_seo_title(self, keyword: str, intent: str) -> str:
        """Generate SEO-optimized title."""
        templates = {
            'informational': f"The Complete Guide to {keyword}: Everything You Need to Know",
            'commercial': f"{keyword} Review: Features, Pricing, and Alternatives",
            'transactional': f"How to Buy {keyword}: Prices, Options & Best Deals",
            'navigational': f"{keyword}: Official Guide & Documentation"
        }
        return templates.get(intent, f"{keyword}: A Comprehensive Guide")

    def _get_related_terms(self, keyword: str) -> Set[str]:
        """Get related terms for a keyword using topic extraction."""
        related_terms = set()
        
        # Add the original keyword
        related_terms.add(keyword.lower())
        
        # Extract topics from the keyword itself
        topics = self._extract_topics(keyword)
        related_terms.update(topics)
        
        # Get related terms from internal links if available
        for topic in topics:
            if topic in self.internal_links:
                for link in self.internal_links[topic]:
                    if 'title' in link:
                        related_terms.update(self._extract_topics(link['title']))
        
        # Remove single character terms and the original keyword
        related_terms = {term for term in related_terms 
                        if len(term) > 1 and term != keyword.lower()}
        
        return related_terms

    def _optimize_keyword_usage(self, content: str, keyword_data: Dict[str, Any]) -> str:
        """Optimize keyword usage in content."""
        if not isinstance(content, str):
            return str(content)
            
        try:
            keyword = keyword_data['keyword']
            # Get related terms safely
            try:
                related_terms = self._get_related_terms(keyword)
            except Exception as e:
                self.logger.warning(f"Could not get related terms: {str(e)}")
                related_terms = set()
            
            # Ensure keyword appears in important sections
            paragraphs = content.split('\n\n')
            
            # Optimize title
            if paragraphs and paragraphs[0].startswith('#'):
                if keyword.lower() not in paragraphs[0].lower():
                    paragraphs[0] = f"{paragraphs[0]} - {keyword}"
            
            # Optimize first paragraph (introduction)
            intro_idx = next((i for i, p in enumerate(paragraphs) 
                            if not p.startswith('#') and len(p) > 50), 1)
            if intro_idx < len(paragraphs):
                intro = paragraphs[intro_idx]
                if keyword.lower() not in intro.lower():
                    paragraphs[intro_idx] = f"{intro} {keyword}."
            
            # Add related terms throughout the content
            if related_terms:
                content_words = set(content.lower().split())
                missing_terms = related_terms - content_words
                
                if missing_terms:
                    # Find suitable paragraphs to add terms
                    for term in missing_terms:
                        for i, para in enumerate(paragraphs[2:-1], 2):
                            if not para.startswith('#') and len(para) > 100:
                                paragraphs[i] = f"{para} {term}."
                                break
            
            return '\n\n'.join(paragraphs)
            
        except Exception as e:
            self.logger.error(f"Error optimizing keyword usage: {str(e)}")
            return content

    def _add_intent_sections(self, content: str, intent: str) -> str:
        """Add intent-specific sections to content."""
        if intent == 'commercial':
            if '## Conclusion' in content:
                content = content.replace(
                    '## Conclusion',
                    '## Comparison Table\n\n|Feature|Details|\n|---|---|\n\n## Conclusion'
                )
        elif intent == 'transactional':
            if '## Conclusion' in content:
                content = content.replace(
                    '## Conclusion',
                    '## Where to Buy\n\n## Pricing\n\n## Conclusion'
                )
        return content

    def _optimize_keyword_density(self, content: str, keyword: str) -> str:
        """Optimize keyword density without over-optimization."""
        word_count = len(content.split())
        keyword_count = content.lower().count(keyword.lower())
        density = (keyword_count / word_count) * 100
        
        if density < 1:
            sections = content.split('\n')
            for i, section in enumerate(sections):
                if section.startswith('## ') and keyword.lower() not in section.lower():
                    sections[i] = f"{section} - {keyword}"
            content = '\n'.join(sections)
        
        return content

    async def save_as_markdown(self, content: str, keyword: str, output_dir: str = "data/output/blogs") -> str:
        """
        Save content with versioning and metadata.
        
        Args:
            content: Blog post content
            keyword: Target keyword
            output_dir: Output directory for blog posts
            
        Returns:
            str: Path to saved file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create keyword-specific directory
        keyword_dir = os.path.join(output_dir, keyword.replace(' ', '_'))
        os.makedirs(keyword_dir, exist_ok=True)
        
        # Get version number
        existing_versions = [f for f in os.listdir(keyword_dir) if f.endswith('.md')]
        version = len(existing_versions) + 1
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{keyword_dir}/v{version}_{timestamp}.md"
        
        # Add metadata
        metadata = {
            'keyword': keyword,
            'version': version,
            'timestamp': timestamp,
            'word_count': len(content.split()),
        }
        
        content_with_metadata = (
            "---\n" + 
            "\n".join(f"{k}: {v}" for k, v in metadata.items()) +
            "\n---\n\n" +
            content
        )
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content_with_metadata)
            
        return filename 

    def _add_semantic_enhancements(self, content: str, keyword_data: Dict[str, Any]) -> str:
        """Add semantic enhancements to content."""
        try:
            # Extract main sections
            sections = re.split(r'\n##? ', content)
            enhanced_sections = []
            
            for section in sections:
                if not section.strip():
                    continue
                    
                # Analyze section sentiment and complexity
                blob = TextBlob(section)
                
                # Enhance sections with negative sentiment
                if blob.sentiment.polarity < 0:
                    section = self._enhance_negative_section(section)
                
                # Add semantic context to complex sections
                if len(section.split()) > 100:
                    section = self._add_semantic_context(section, keyword_data)
                
                enhanced_sections.append(section)
            
            return '## '.join(enhanced_sections)
            
        except Exception as e:
            self.logger.error(f"Error adding semantic enhancements: {str(e)}")
            return content
            
    def _enhance_negative_section(self, section: str) -> str:
        """Enhance sections with negative sentiment."""
        lines = section.split('\n')
        enhanced_lines = []
        
        for line in lines:
            if line.strip():
                blob = TextBlob(line)
                if blob.sentiment.polarity < -0.2:
                    # Add constructive alternatives
                    enhanced_lines.append(line)
                    enhanced_lines.append("\nHowever, there are several solutions available:")
                    enhanced_lines.append("1. Consider implementing best practices")
                    enhanced_lines.append("2. Follow industry standards")
                    enhanced_lines.append("3. Adopt proven methodologies\n")
                else:
                    enhanced_lines.append(line)
                    
        return '\n'.join(enhanced_lines)
        
    def _add_semantic_context(self, section: str, keyword_data: Dict[str, Any]) -> str:
        """Add semantic context to complex sections."""
        # Add relevant examples
        if 'examples' not in section.lower():
            section += "\n\nFor example:\n"
            section += "- Use case 1\n"
            section += "- Use case 2\n"
            section += "- Use case 3\n"
        
        # Add practical tips
        if 'tip' not in section.lower() and 'hint' not in section.lower():
            section += "\n\n💡 Pro Tip: "
            section += f"When implementing {keyword_data['keyword']}, "
            section += "focus on practical applications and measurable results."
        
        return section
        
    def _add_lsi_keywords(self, content: str, keyword: str) -> str:
        """Add LSI (Latent Semantic Indexing) keywords to content."""
        try:
            # Generate LSI keywords
            lsi_keywords = self._generate_lsi_keywords(keyword)
            
            # Find suitable places to add LSI keywords
            paragraphs = content.split('\n\n')
            enhanced_paragraphs = []
            
            for i, para in enumerate(paragraphs):
                if not para.strip() or para.startswith('#'):
                    enhanced_paragraphs.append(para)
                    continue
                
                # Add LSI keywords naturally
                if len(para.split()) > 50 and i < len(lsi_keywords):
                    enhanced_para = self._incorporate_lsi_keyword(
                        para, lsi_keywords[i]
                    )
                    enhanced_paragraphs.append(enhanced_para)
                else:
                    enhanced_paragraphs.append(para)
            
            return '\n\n'.join(enhanced_paragraphs)
            
        except Exception as e:
            self.logger.error(f"Error adding LSI keywords: {str(e)}")
            return content
            
    def _generate_lsi_keywords(self, keyword: str) -> List[str]:
        """Generate LSI keywords for the main keyword."""
        lsi_keywords = []
        
        # Add common word combinations
        words = keyword.split()
        if len(words) > 1:
            lsi_keywords.extend([
                f"best {keyword}",
                f"top {keyword}",
                f"{keyword} guide",
                f"{keyword} tutorial",
                f"{keyword} examples"
            ])
        
        # Add industry-specific variations
        lsi_keywords.extend([
            f"{keyword} solutions",
            f"{keyword} tools",
            f"{keyword} strategies",
            f"{keyword} best practices",
            f"{keyword} tips"
        ])
        
        return lsi_keywords
        
    def _incorporate_lsi_keyword(self, paragraph: str, lsi_keyword: str) -> str:
        """Incorporate LSI keyword naturally into a paragraph."""
        sentences = sent_tokenize(paragraph)
        
        if len(sentences) < 2:
            return f"{paragraph} This is particularly important when considering {lsi_keyword}."
            
        # Insert keyword in the middle of the paragraph
        mid_point = len(sentences) // 2
        connector_phrases = [
            f"When it comes to {lsi_keyword},",
            f"In terms of {lsi_keyword},",
            f"Considering {lsi_keyword},",
            f"Speaking of {lsi_keyword},"
        ]
        
        sentences.insert(
            mid_point,
            f"{random.choice(connector_phrases)} this becomes even more relevant."
        )
        
        return ' '.join(sentences)
        
    async def _generate_content_structure(self, keyword: str, intent: str) -> str:
        """Generate content structure based on keyword and intent."""
        try:
            # Base structure
            structure_templates = {
                'informational': [
                    "## Understanding {keyword}",
                    "## Key Components",
                    "## How It Works",
                    "## Benefits and Advantages",
                    "## Common Challenges",
                    "## Implementation Guide"
                ],
                'commercial': [
                    "## Product Overview",
                    "## Features and Specifications",
                    "## Pricing Plans",
                    "## Comparison with Alternatives",
                    "## Customer Reviews",
                    "## Purchase Guide"
                ],
                'transactional': [
                    "## Product Details",
                    "## Available Options",
                    "## Pricing Information",
                    "## How to Order",
                    "## Shipping and Delivery",
                    "## Return Policy"
                ],
                'navigational': [
                    "## Quick Overview",
                    "## Main Features",
                    "## Getting Started",
                    "## Support Resources",
                    "## Contact Information"
                ]
            }
            
            # Get base structure
            base_sections = structure_templates.get(
                intent,
                structure_templates['informational']
            )
            
            # Customize sections for the keyword
            sections = []
            for section in base_sections:
                sections.append(section.format(keyword=keyword))
            
            # Add subsections based on complexity
            enhanced_sections = []
            for section in sections:
                enhanced_sections.append(section)
                
                # Add relevant subsections
                subsections = await self._generate_subsections(section, keyword)
                enhanced_sections.extend(subsections)
            
            return '\n\n'.join(enhanced_sections)
            
        except Exception as e:
            self.logger.error(f"Error generating content structure: {str(e)}")
            return "\n\n".join([
                "## Introduction",
                "## Main Content",
                "## Conclusion"
            ])
            
    async def _generate_subsections(self, section: str, keyword: str) -> List[str]:
        """Generate relevant subsections for a main section."""
        # Extract section type
        section_type = section.replace('#', '').strip().lower()
        
        if 'overview' in section_type:
            return [
                "### What is {keyword}",
                "### Why {keyword} Matters",
                "### Key Benefits"
            ]
        elif 'features' in section_type:
            return [
                "### Core Features",
                "### Advanced Capabilities",
                "### Technical Specifications"
            ]
        elif 'guide' in section_type:
            return [
                "### Step-by-Step Instructions",
                "### Best Practices",
                "### Common Pitfalls to Avoid"
            ]
        elif 'comparison' in section_type:
            return [
                "### Feature Comparison",
                "### Price Comparison",
                "### Pros and Cons"
            ]
        
        return []  # No subsections for other section types
        
    def _extract_main_topic(self, keyword: str) -> str:
        """Extract the main topic from a keyword phrase."""
        # Remove common modifiers
        modifiers = ['best', 'top', 'guide', 'tutorial', 'how to', 'what is']
        main_topic = keyword.lower()
        
        for modifier in modifiers:
            main_topic = main_topic.replace(modifier, '').strip()
        
        # Get the most significant words
        words = main_topic.split()
        if len(words) <= 2:
            return main_topic
            
        # Use noun phrases for longer keywords
        doc = self.nlp(main_topic)
        noun_phrases = list(doc.noun_chunks)
        
        return str(noun_phrases[0]) if noun_phrases else main_topic
        
    def _generate_secondary_keywords(self, keyword: str) -> List[str]:
        """Generate secondary keywords based on the main keyword."""
        main_topic = self._extract_main_topic(keyword)
        secondary_keywords = set()
        
        # Add common variations
        variations = [
            f"best {main_topic}",
            f"{main_topic} tutorial",
            f"{main_topic} guide",
            f"how to use {main_topic}",
            f"{main_topic} examples",
            f"{main_topic} tips",
            f"{main_topic} benefits",
            f"{main_topic} features",
            f"{main_topic} alternatives",
            f"{main_topic} comparison"
        ]
        secondary_keywords.update(variations)
        
        # Add semantic variations using NLP
        doc = self.nlp(keyword)
        for token in doc:
            if token.has_vector:
                # Find similar words using word vectors
                similar_words = token.vocab.vectors.most_similar(
                    token.vector.reshape(1, -1),
                    n=3
                )
                for word in similar_words:
                    if word != token.text:
                        secondary_keywords.add(
                            keyword.replace(token.text, word)
                        )
        
        return list(secondary_keywords)
        
    def _get_internal_linking_opportunities(self, keyword: str) -> str:
        """Get internal linking opportunities for the keyword."""
        opportunities = []
        
        # Check existing content
        if self.website_context:
            related_pages = []
            topics = self._extract_topics(keyword)
            
            # Find relevant pages
            for topic in topics:
                if topic in self.internal_links:
                    related_pages.extend(self.internal_links[topic])
            
            # Sort by relevance and get top 5
            related_pages.sort(key=lambda x: x['relevance'], reverse=True)
            for page in related_pages[:5]:
                opportunities.append(
                    f"- Link to: {page['title']} ({page['url']})"
                )
        
        if not opportunities:
            opportunities = [
                "- No existing content found for internal linking",
                "- Consider creating supporting content for:",
                f"  - {keyword} tutorial",
                f"  - {keyword} examples",
                f"  - {keyword} best practices"
            ]
        
        return '\n'.join(opportunities)
        
    def _determine_intent(self, keyword: str) -> str:
        """Determine search intent from keyword."""
        keyword_lower = keyword.lower()
        
        # Check for informational intent
        if any(word in keyword_lower for word in ['how', 'what', 'why', 'guide', 'tutorial']):
            return 'informational'
            
        # Check for commercial intent
        if any(word in keyword_lower for word in ['best', 'top', 'review', 'vs', 'compare']):
            return 'commercial'
            
        # Check for transactional intent
        if any(word in keyword_lower for word in ['buy', 'price', 'cost', 'shop', 'purchase']):
            return 'transactional'
            
        # Check for navigational intent
        if any(word in keyword_lower for word in ['login', 'sign in', 'download', 'website']):
            return 'navigational'
            
        # Default to informational
        return 'informational'
        
    def _identify_target_audience(self, keyword: str) -> str:
        """Identify target audience from keyword."""
        keyword_lower = keyword.lower()
        
        # Check for technical audience
        if any(term in keyword_lower for term in ['api', 'code', 'developer', 'programming']):
            return 'Technical professionals and developers'
            
        # Check for business audience
        if any(term in keyword_lower for term in ['business', 'enterprise', 'company', 'professional']):
            return 'Business professionals and decision makers'
            
        # Check for beginners
        if any(term in keyword_lower for term in ['beginner', 'basic', 'introduction', 'start']):
            return 'Beginners and newcomers'
            
        # Check for advanced users
        if any(term in keyword_lower for term in ['advanced', 'expert', 'professional']):
            return 'Advanced users and experts'
            
        # Default to general audience
        return 'General audience interested in {keyword}'
        
    def _get_keyword_requirements(self, keyword_insights: Dict[str, Any]) -> str:
        """Get keyword usage requirements based on insights."""
        requirements = []
        
        # Add basic requirements
        requirements.extend([
            f"Primary keyword: {keyword_insights['main_topic']}",
            "- Use in title, meta description, and first paragraph",
            "- Include in at least one H2 heading",
            "- Maintain natural keyword density (1-2%)"
        ])
        
        # Add complexity-based requirements
        if keyword_insights['complexity_level'] == 'high':
            requirements.extend([
                "- Break down complex concepts",
                "- Include technical definitions",
                "- Add explanatory examples"
            ])
        
        # Add competition-based requirements
        if keyword_insights['competition_level'] == 'high':
            requirements.extend([
                "- Include comprehensive coverage",
                "- Add expert insights",
                "- Provide unique value propositions"
            ])
        
        # Add volume-based requirements
        if keyword_insights['search_volume_category'] == 'high':
            requirements.extend([
                "- Address common user questions",
                "- Include FAQ section",
                "- Optimize for featured snippets"
            ])
        
        return '\n'.join(requirements) 