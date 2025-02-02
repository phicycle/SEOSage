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
    
    def __init__(self, config: Dict[str, Any] = None) -> None:
        """Initialize SEO optimizer with optional configuration."""
        self.config = config or {}  # Make config optional with default empty dict
        self.logger = logging.getLogger(__name__)
        self.rules = self.config.get('seo_rules', {})
        
        # Update intent rules based on new keyword data structure
        self.intent_rules = {
            'informational': {
                'min_word_count': 1500,
                'required_sections': ['What is', 'How to', 'FAQ'],
                'internal_links': 3,
                'keyword_density': (0.01, 0.02)  # 1-2%
            },
            'commercial': {
                'min_word_count': 2000,
                'required_sections': ['Comparison', 'Pros and Cons', 'Verdict'],
                'internal_links': 4,
                'keyword_density': (0.005, 0.015)  # 0.5-1.5%
            },
            'transactional': {
                'min_word_count': 1000,
                'required_sections': ['Features', 'Pricing', 'Where to Buy'],
                'internal_links': 2,
                'keyword_density': (0.01, 0.025)  # 1-2.5%
            },
            'navigational': {  # Added for new intent type
                'min_word_count': 800,
                'required_sections': ['Overview', 'Contact', 'Location'],
                'internal_links': 2,
                'keyword_density': (0.01, 0.02)  # 1-2%
            }
        }
        
    def apply_rules(self, content: str, keyword_data: Dict[str, Any]) -> List[str]:
        """
        Apply SEO rules to content based on keyword data and metrics.
        
        Args:
            content: Content to analyze
            keyword_data: Keyword information and metrics
            
        Returns:
            List of SEO issues found
        """
        try:
            keyword = keyword_data['keyword']
            rank = keyword_data.get('rank', 0)
            issues = []
            
            # Apply base SEO rules
            issues.extend(self._check_title_rules(content, keyword))
            issues.extend(self._check_header_rules(content, keyword))
            issues.extend(self._check_content_rules(content, keyword))
            issues.extend(self._check_image_rules(content))
            
            # Add ranking-specific recommendations
            if 4 <= rank <= 10:
                issues.extend(self._check_competitive_rules(content, keyword_data))
            
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
        
        if intent in self.intent_rules:
            rules = self.intent_rules[intent]
            word_count = len(content.split())
            
            # Check word count
            if word_count < rules['min_word_count']:
                issues.append(
                    f"Content length ({word_count} words) below recommended "
                    f"minimum ({rules['min_word_count']}) for {intent} content"
                )
            
            # Check required sections
            headers = re.findall(r'^##\s+(.+)$', content, re.MULTILINE)
            for required in rules['required_sections']:
                if not any(required.lower() in h.lower() for h in headers):
                    issues.append(f"Missing recommended section for {intent} content: {required}")
            
            # Check keyword density
            keyword = keyword_data['keyword'].lower()
            keyword_count = len(re.findall(rf'\b{re.escape(keyword)}\b', content.lower()))
            density = keyword_count / word_count if word_count > 0 else 0
            min_density, max_density = rules['keyword_density']
            
            if density < min_density:
                issues.append(
                    f"Keyword density ({density:.2%}) below recommended "
                    f"minimum ({min_density:.2%}) for {intent} content"
                )
            elif density > max_density:
                issues.append(
                    f"Keyword density ({density:.2%}) above recommended "
                    f"maximum ({max_density:.2%}) for {intent} content"
                )
        
        return issues
        
    def _check_competitive_rules(self, content: str, keyword_data: Dict[str, Any]) -> List[str]:
        """Check rules for highly competitive keywords."""
        issues = []
        word_count = len(content.split())
        
        # For competitive keywords (rank 4-10), content should be comprehensive
        if word_count < 2000:
            issues.append(
                f"Content length ({word_count} words) may be insufficient "
                "for competitive keyword ranking"
            )
        
        # Check for LSI keywords and related terms
        keyword = keyword_data['keyword'].lower()
        if not self._has_related_terms(content, keyword):
            issues.append("Content may lack sufficient related terms and LSI keywords")
        
        return issues
        
    def _has_related_terms(self, content: str, keyword: str) -> bool:
        """Check for presence of related terms."""
        # This is a simplified check - in practice, you'd want to use
        # a more sophisticated LSI keyword analysis
        content_lower = content.lower()
        related_terms = self._get_related_terms(keyword)
        
        found_terms = sum(1 for term in related_terms if term in content_lower)
        return found_terms >= len(related_terms) // 2 