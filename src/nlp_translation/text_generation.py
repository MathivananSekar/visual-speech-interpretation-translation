import spacy

# Load the spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    nlp = None

def tokenize_sentence(sentence):
    """
    Tokenizes a sentence into words using spaCy.

    Args:
        sentence (str): The sentence to tokenize.

    Returns:
        list: A list of tokenized words.
    """
    if nlp is None:
        raise ValueError("spaCy model is not loaded.")
    
    doc = nlp(sentence)
    return [token.text for token in doc]
