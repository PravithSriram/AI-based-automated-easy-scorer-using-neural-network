import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from spellchecker import SpellChecker
import textstat

# Download punkt tokenizer if not already present
nltk.download('punkt', quiet=True)

def clean_words(words):
    """Keep tokens that contain at least one alphabetic character and lowercase them."""
    cleaned = [w.lower() for w in words if any(c.isalpha() for c in w)]
    return cleaned

def score_essay(text):
    # Tokenize sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    words = clean_words(words)

    total_words = len(words)
    num_sentences = max(1, len(sentences))
    unique_words = len(set(words))
    vocab_ratio = (unique_words / total_words) if total_words else 0
    avg_sentence_len = (total_words / num_sentences) if num_sentences else 0

    # spelling errors
    spell = SpellChecker()
    misspelled = list(spell.unknown(words))
    spelling_err_rate = (len(misspelled) / total_words) if total_words else 0

    # heuristic scores (0-10)
    grammar_score = max(0.0, (1 - spelling_err_rate)) * 10
    ideal = 15.0
    structure_score = max(0.0, 1 - abs(avg_sentence_len - ideal) / ideal) * 10
    vocab_score = min(vocab_ratio * 20, 10)
    flesch = textstat.flesch_reading_ease(text) if total_words else 0
    readability_score = min(max(flesch / 10.0, 0.0), 10.0)

    overall = (grammar_score * 0.4 +
               structure_score * 0.3 +
               vocab_score * 0.2 +
               readability_score * 0.1)

    return {
        'grammar': round(grammar_score, 2),
        'structure': round(structure_score, 2),
        'vocabulary': round(vocab_score, 2),
        'readability': round(readability_score, 2),
        'overall': round(overall, 2),
        'misspelled': misspelled,
        'stats': {
            'words': total_words,
            'sentences': len(sentences),
            'unique_words': unique_words,
            'avg_sentence_len': round(avg_sentence_len, 2)
        }
    }
