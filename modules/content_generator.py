# Standard library imports
import os
from datetime import datetime
from typing import Dict, Any, Optional

# Third-party imports
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class ContentGenerator:
    """
    Generates SEO-optimized content using LLM.
    
    Attributes:
        llm: Language model instance
        template: Base template for content generation
    """
    
    def __init__(self, llm: Any) -> None:
        """Initialize content generator with language model."""
        self.llm = llm
        self.template = self._get_base_template()
        
    def _get_base_template(self) -> str:
        """Get the base template for content generation."""
        return """Write a comprehensive, SEO-optimized blog post about {keyword} in Markdown format that follows all best practices.
        Structure:
        # Title
        ## Introduction (100-150 words)
        ## Main Section 1
        ## Main Section 2
        ## Main Section 3
        ## Conclusion with call to action
        ### FAQ
        - Question 1
        - Question 2
        - Question 3
        
        Include:
        - Primary keyword in title, first 100 words, and conclusion
        - Secondary keywords naturally throughout
        - Proper heading hierarchy (#, ##, ###)
        - Internal links in Markdown format [link text](url)
        - Images with alt text in Markdown format ![alt text](image_url)
        
        Blog Post:"""

    def _get_intent_template(self, intent: str) -> str:
        """Get template based on search intent."""
        intent_templates = {
            'informational': """Write a comprehensive guide about {keyword} that educates the reader...
            Include how-to sections and detailed explanations...""",
            
            'commercial': """Create a detailed comparison/review about {keyword}...
            Include pros and cons, pricing information...""",
            
            'transactional': """Write a buying guide for {keyword}...
            Include pricing information, where to buy, and key features...""",
            
            'navigational': """Create an overview of {keyword}...
            Include key features, benefits, and use cases..."""
        }
        return intent_templates.get(intent, self.template)

    def generate_blog_post(self, keyword_data: Dict[str, Any]) -> str:
        """
        Generate blog post using keyword data and intent.
        
        Args:
            keyword_data: Dictionary containing keyword and its metrics
            
        Returns:
            str: Generated blog post content
        """
        intent = keyword_data.get('intent', 'informational')
        keyword = keyword_data['keyword']
        
        template = self._get_intent_template(intent)
        prompt = PromptTemplate(
            input_variables=["keyword"],
            template=template
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        content = chain.run(keyword)
        return self.optimize_content(content, keyword_data)

    def optimize_content(self, content: str, keyword_data: Dict[str, Any]) -> str:
        """
        Optimize content based on keyword data and metrics.
        
        Args:
            content: Generated content
            keyword_data: Keyword information and metrics
            
        Returns:
            str: Optimized content
        """
        keyword = keyword_data['keyword']
        rank = keyword_data.get('rank', 0)
        
        # Ensure proper heading structure
        if not content.startswith('# '):
            content = f"# {keyword}\n\n{content}"
        
        # Add ranking-specific optimizations
        if 4 <= rank <= 10:
            content = self._optimize_competitive_content(content, keyword_data)
        elif 11 <= rank <= 30:
            content = self._optimize_opportunity_content(content, keyword_data)
        
        # Optimize keyword density and related terms
        content = self._optimize_keyword_usage(content, keyword_data)
        
        return content

    def _optimize_competitive_content(self, content: str, keyword_data: Dict[str, Any]) -> str:
        """Optimize content for competitive keywords (rank 4-10)."""
        # Add comprehensive sections
        sections = [
            "## Key Takeaways",
            "## Detailed Analysis",
            "## Expert Insights",
            "## Frequently Asked Questions",
            "## Conclusion"
        ]
        
        for section in sections:
            if section not in content:
                content += f"\n\n{section}\n\nAdd detailed content for this section."
        
        return content

    def _optimize_keyword_usage(self, content: str, keyword_data: Dict[str, Any]) -> str:
        """Optimize keyword usage and related terms."""
        keyword = keyword_data['keyword']
        
        # Add LSI keywords and related terms
        related_terms = self._get_related_terms(keyword)
        content += "\n\n## Related Topics\n\n"
        for term in related_terms:
            content += f"- {term}\n"
        
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

    def save_as_markdown(self, content: str, keyword: str, output_dir: str = "blogs") -> str:
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