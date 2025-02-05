"""NLP utilities for advanced content generation."""
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from typing import List, Set, Dict, Any, Tuple
import spacy
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import Counter
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

class NLPProcessor:
    """Advanced NLP processing for content generation."""
    
    def __init__(self):
        """Initialize NLP components."""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.nlp = spacy.load('en_core_web_sm')
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
    def analyze_text_complexity(self, text: str) -> Dict[str, Any]:
        """Analyze text complexity metrics."""
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalnum()]
        
        # Calculate basic metrics
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
        
        # Calculate readability scores
        blob = TextBlob(text)
        
        return {
            'avg_sentence_length': avg_sentence_length,
            'avg_word_length': avg_word_length,
            'sentiment_polarity': blob.sentiment.polarity,
            'sentiment_subjectivity': blob.sentiment.subjectivity,
            'complexity_score': (avg_sentence_length * 0.6 + avg_word_length * 0.4)
        }
    
    def extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases using spaCy."""
        doc = self.nlp(text)
        key_phrases = []
        
        # Extract noun phrases
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Limit to 3 words
                key_phrases.append(chunk.text.lower())
        
        # Extract named entities
        for ent in doc.ents:
            if len(ent.text.split()) <= 3:
                key_phrases.append(ent.text.lower())
        
        # Remove duplicates and sort by length
        key_phrases = list(set(key_phrases))
        key_phrases.sort(key=len, reverse=True)
        
        return key_phrases
    
    def generate_semantic_keywords(self, keyword: str, text_context: str = "") -> List[str]:
        """Generate semantically related keywords."""
        # Get synonyms from WordNet
        synonyms = set()
        for syn in wordnet.synsets(keyword):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().lower())
        
        # Get related terms from text context if provided
        related_terms = set()
        if text_context:
            doc = self.nlp(text_context)
            # Find words with similar vector representations
            keyword_vector = self.nlp(keyword).vector
            for token in doc:
                if token.has_vector and token.text.lower() not in self.stop_words:
                    similarity = np.dot(token.vector, keyword_vector) / (
                        np.linalg.norm(token.vector) * np.linalg.norm(keyword_vector)
                    )
                    if similarity > 0.5:
                        related_terms.add(token.text.lower())
        
        # Combine and sort by relevance
        all_keywords = list(synonyms.union(related_terms))
        all_keywords.sort(key=lambda x: len(x.split()))
        
        return all_keywords[:10]  # Return top 10 related keywords
    
    def improve_readability(self, text: str) -> str:
        """Improve text readability."""
        sentences = sent_tokenize(text)
        improved_sentences = []
        
        for sentence in sentences:
            # Analyze sentence
            analysis = self._analyze_sentence(sentence)
            
            # Apply improvements based on analysis
            if analysis['needs_simplification']:
                sentence = self._simplify_sentence(sentence)
            if analysis['needs_active_voice']:
                sentence = self._convert_to_active_voice(sentence)
            if analysis['needs_clarity']:
                sentence = self._improve_clarity(sentence)
                
            improved_sentences.append(sentence)
        
        return ' '.join(improved_sentences)
    
    def _analyze_sentence(self, sentence: str) -> Dict[str, bool]:
        """Analyze sentence for potential improvements."""
        words = word_tokenize(sentence)
        
        return {
            'needs_simplification': len(words) > 25,
            'needs_active_voice': self._is_passive_voice(sentence),
            'needs_clarity': self._needs_clarity_improvement(sentence)
        }
    
    def _is_passive_voice(self, sentence: str) -> bool:
        """Check if sentence is in passive voice."""
        doc = self.nlp(sentence)
        for token in doc:
            if token.dep_ == "nsubjpass":
                return True
        return False
    
    def _needs_clarity_improvement(self, sentence: str) -> bool:
        """Check if sentence needs clarity improvement."""
        # Check for common clarity issues
        clarity_issues = [
            'this', 'that', 'these', 'those',  # Vague pronouns
            'etc', 'and so on',  # Incomplete expressions
            'very', 'really', 'quite'  # Weak modifiers
        ]
        return any(issue in sentence.lower() for issue in clarity_issues)
    
    def _simplify_sentence(self, sentence: str) -> str:
        """Simplify a complex sentence."""
        doc = self.nlp(sentence)
        
        # Break into clauses
        clauses = []
        current_clause = []
        
        for token in doc:
            current_clause.append(token.text)
            if token.dep_ in ['punct'] and token.text in [',', ';']:
                if len(current_clause) > 3:  # Minimum clause length
                    clauses.append(' '.join(current_clause))
                current_clause = []
        
        if current_clause:
            clauses.append(' '.join(current_clause))
        
        # Reconstruct as separate sentences
        return '. '.join(clause.strip() for clause in clauses)
    
    def _convert_to_active_voice(self, sentence: str) -> str:
        """Convert passive voice to active voice."""
        doc = self.nlp(sentence)
        
        # This is a simplified conversion
        # For a production system, you'd want a more sophisticated approach
        if self._is_passive_voice(sentence):
            # Extract subject and object
            subject = None
            verb = None
            obj = None
            
            for token in doc:
                if token.dep_ == "nsubjpass":
                    subject = token.text
                elif token.dep_ == "ROOT" and token.pos_ == "VERB":
                    verb = token.text
                elif token.dep_ in ["dobj", "pobj"]:
                    obj = token.text
            
            if all([subject, verb, obj]):
                return f"{obj} {verb} {subject}"
        
        return sentence
    
    def _improve_clarity(self, sentence: str) -> str:
        """Improve sentence clarity."""
        # Replace vague pronouns with specific nouns
        doc = self.nlp(sentence)
        improved = sentence
        
        for token in doc:
            if token.text.lower() in ['this', 'that', 'these', 'those']:
                # Look for the nearest noun before this pronoun
                for prev_token in reversed(list(token.doc[:token.i])):
                    if prev_token.pos_ == "NOUN":
                        improved = improved.replace(token.text, prev_token.text, 1)
                        break
        
        # Remove weak modifiers
        weak_modifiers = ['very', 'really', 'quite', 'rather']
        for modifier in weak_modifiers:
            improved = re.sub(rf'\b{modifier}\b\s+', '', improved, flags=re.IGNORECASE)
        
        return improved
    
    def generate_summary(self, text: str, max_length: int = 150) -> str:
        """Generate a concise summary of the text."""
        try:
            summary = self.summarizer(text, max_length=max_length, min_length=30, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            # Fallback to extractive summarization if transformer fails
            sentences = sent_tokenize(text)
            doc = self.nlp(text)
            
            # Score sentences based on keyword frequency
            word_freq = Counter(token.text.lower() for token in doc if not token.is_stop)
            sentence_scores = {}
            
            for sentence in sentences:
                score = sum(word_freq[word.lower()] for word in word_tokenize(sentence)
                           if word.lower() in word_freq)
                sentence_scores[sentence] = score
            
            # Get top sentences
            summary_sentences = sorted(sentence_scores.items(),
                                    key=lambda x: x[1], reverse=True)[:3]
            return ' '.join(sent for sent, score in summary_sentences)
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        # Encode texts
        embeddings1 = self.sentence_transformer.encode([text1])[0]
        embeddings2 = self.sentence_transformer.encode([text2])[0]
        
        # Calculate cosine similarity
        similarity = np.dot(embeddings1, embeddings2) / (
            np.linalg.norm(embeddings1) * np.linalg.norm(embeddings2)
        )
        
        return float(similarity) 