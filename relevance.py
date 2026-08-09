import math
import re

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def extract_concepts(text: str):
    """
    Lightweight CPU-friendly concept extraction.
    Removes common stop words and returns a set of lowercase keywords.
    """
    stopwords = {
        'the', 'is', 'in', 'at', 'of', 'on', 'and', 'a', 'to', 'for', 'with',
        'what', 'how', 'does', 'do', 'are', 'can', 'it', 'by', 'an', 'this', 'that',
        'which', 'who', 'whom', 'whose', 'where', 'when', 'why', 'whether', 'if',
        'or', 'as', 'be', 'from', 'but', 'not', 'have', 'has', 'had', 'will'
    }
    # Simple regex to get words (3 letters or more)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    concepts = set([w for w in words if w not in stopwords])
    return concepts

def validate_relevance(query: str, doc_text: str, cross_encoder_logit: float):
    """
    Calculates a relevance score between 0 and 1.
    Combines calibrated cross-encoder logit with a concept match ratio.
    """
    # 1. Calibrate Cross-Encoder output to probability (0 to 1)
    ce_prob = sigmoid(cross_encoder_logit)
    
    # 2. Concept Match Ratio
    query_concepts = extract_concepts(query)
    doc_concepts = extract_concepts(doc_text)
    
    matched_concepts = query_concepts.intersection(doc_concepts)
    missing_concepts = query_concepts - doc_concepts
    
    concept_match_ratio = 0.0
    if len(query_concepts) > 0:
        concept_match_ratio = len(matched_concepts) / len(query_concepts)
        
    # 3. Combine scores
    # Cross-encoder is very accurate, so we give it 80% weight.
    # Keyword overlap gives 20% weight to ensure explicit query terms are present.
    relevance_score = (0.8 * ce_prob) + (0.2 * concept_match_ratio)
    
    return {
        "relevance_score": round(relevance_score, 4),
        "matched_concepts": list(matched_concepts),
        "missing_concepts": list(missing_concepts)
    }
