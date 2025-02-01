import re
from typing import Dict, List, Any, Tuple, Optional
import logging

class SEOOptimizer:
    """
    Applies SEO optimization rules to content.
    
    Attributes:
        rules: Dictionary of SEO rules and thresholds
        logger: Logger instance for tracking issues
    """
    
    def __init__(self) -> None:
        """Initialize SEO optimizer with rules and logger."""
        self.rules = self.load_seo_rules()
        self.logger = logging.getLogger(__name__)
        
    def load_seo_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load SEO optimization rules and thresholds."""
        return {
            'title': {
                'length': (50, 60),
                'keyword_inclusion': True
            },
            'headers': {
                'structure': '# > ## > ###',
                'keyword_inclusion': True,
                'min_length': 20,
                'max_length': 70
            },
            'content': {
                'keyword_density': (1.0, 3.0),
                'min_word_count': 300,
                'readability_score': 60,
                'internal_links': 2,
                'external_links': 1
            },
            'images': {
                'alt_text': True,
                'optimized': True,
                'min_count': 1
            }
        }
        
    def apply_rules(self, content: str, keyword_data: Dict[str, Any]) -> List[str]:
        """
        Apply SEO rules to content based on keyword data and intent.
        
        Args:
            content: Content to analyze
            keyword_data: Keyword information and metrics
            
        Returns:
            List of SEO issues found
        """
        try:
            keyword = keyword_data['keyword']
            intent = keyword_data.get('intent', 'informational')
            issues = []
            
            # Apply base SEO rules
            issues.extend(self._check_title_rules(content, keyword))
            issues.extend(self._check_header_rules(content, keyword))
            issues.extend(self._check_content_rules(content, keyword))
            issues.extend(self._check_image_rules(content))
            
            # Apply intent-specific rules
            issues.extend(self._check_intent_rules(content, keyword_data))
            
            return issues
        except Exception as e:
            self.logger.error(f"Error applying SEO rules: {str(e)}")
            return [f"Error analyzing content: {str(e)}"]
            
    def _check_title_rules(self, content: str, keyword: str) -> List[str]:
        """Check title for SEO compliance."""
        issues = []
        title_match = re.search(r'^# (.+)$', content, re.MULTILINE)
        
        if title_match:
            title = title_match.group(1)
            title_len = len(title)
            min_len, max_len = self.rules['title']['length']
            
            if title_len < min_len:
                issues.append(f"Title too short ({title_len} chars, min {min_len})")
            elif title_len > max_len:
                issues.append(f"Title too long ({title_len} chars, max {max_len})")
                
            if keyword.lower() not in title.lower():
                issues.append("Primary keyword missing from title")
        else:
            issues.append("Missing main title (# heading)")
            
        return issues
        
    def _check_header_rules(self, content: str, keyword: str) -> List[str]:
        """Check headers for SEO compliance."""
        issues = []
        headers = re.findall(r'^(#{2,3})\s+(.+)$', content, re.MULTILINE)
        
        if not headers:
            issues.append("Missing subheadings (## or ###)")
            return issues
            
        keyword_in_header = False
        for level, header in headers:
            header_len = len(header)
            
            if header_len < self.rules['headers']['min_length']:
                issues.append(f"Header too short: '{header}'")
            elif header_len > self.rules['headers']['max_length']:
                issues.append(f"Header too long: '{header}'")
                
            if keyword.lower() in header.lower():
                keyword_in_header = True
                
        if not keyword_in_header:
            issues.append("Primary keyword missing from all subheadings")
            
        return issues
        
    def _check_content_rules(self, content: str, keyword: str) -> List[str]:
        """Check content body for SEO compliance."""
        issues = []
        words = content.split()
        word_count = len(words)
        
        # Check word count
        if word_count < self.rules['content']['min_word_count']:
            issues.append(f"Content too short ({word_count} words)")
            
        # Check keyword density
        keyword_count = content.lower().count(keyword.lower())
        density = (keyword_count / word_count) * 100 if word_count > 0 else 0
        min_density, max_density = self.rules['content']['keyword_density']
        
        if density < min_density:
            issues.append(f"Keyword density too low ({density:.1f}%)")
        elif density > max_density:
            issues.append(f"Keyword density too high ({density:.1f}%)")
            
        # Check links
        internal_links = len(re.findall(r'\[([^\]]+)\]\((?!http)[^)]+\)', content))
        external_links = len(re.findall(r'\[([^\]]+)\]\(http[s]?://[^)]+\)', content))
        
        if internal_links < self.rules['content']['internal_links']:
            issues.append(f"Not enough internal links ({internal_links})")
        if external_links < self.rules['content']['external_links']:
            issues.append(f"Not enough external links ({external_links})")
            
        return issues
        
    def _check_image_rules(self, content: str) -> List[str]:
        """Check images for SEO compliance."""
        issues = []
        images = re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', content)
        
        if len(images) < self.rules['images']['min_count']:
            issues.append("Not enough images")
            return issues
            
        for alt_text, src in images:
            if not alt_text:
                issues.append(f"Missing alt text for image: {src}")
            elif len(alt_text) < 10:
                issues.append(f"Alt text too short for image: {src}")
                
        return issues
        
    def _check_intent_rules(self, content: str, keyword_data: Dict[str, Any]) -> List[str]:
        """Check intent-specific SEO rules."""
        issues = []
        intent = keyword_data.get('intent', 'informational')
        
        intent_rules = {
            'informational': {
                'min_word_count': 1500,
                'required_sections': ['What is', 'How to', 'FAQ'],
                'internal_links': 3
            },
            'commercial': {
                'min_word_count': 2000,
                'required_sections': ['Comparison', 'Pros and Cons', 'Verdict'],
                'internal_links': 4
            },
            'transactional': {
                'min_word_count': 1000,
                'required_sections': ['Features', 'Pricing', 'Where to Buy'],
                'internal_links': 2
            }
        }
        
        if intent in intent_rules:
            rules = intent_rules[intent]
            word_count = len(content.split())
            
            if word_count < rules['min_word_count']:
                issues.append(
                    f"Content length ({word_count} words) below recommended "
                    f"minimum ({rules['min_word_count']}) for {intent} content"
                )
            
            headers = re.findall(r'^##\s+(.+)$', content, re.MULTILINE)
            for required in rules['required_sections']:
                if not any(required.lower() in h.lower() for h in headers):
                    issues.append(f"Missing required section for {intent} content: {required}")
                    
        return issues 