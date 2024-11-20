from keybert import KeyBERT

# Initialize KeyBERT model
keybert_model = KeyBERT()


# Function to extract keywords with KeyBERT
def extract_keywords(text, top_n=5, diversity=0.5):
    """
    Extracts keywords from a text using KeyBERT.

    Args:
        text (str): The input text.
        top_n (int): Number of top keywords to extract.
        diversity (float): Controls the diversity of keywords (0 = low diversity, 1 = high diversity).

    Returns:
        list: List of tuples containing keywords and their scores.
    """
    return keybert_model.extract_keywords(text, top_n=top_n, diversity=diversity)


# Function to create prompts using the kw_score strategy
def create_kw_score_prompt(text, top_n=5, diversity=0.5):
    """
    Creates a prompt using the kw_score strategy.

    Args:
        text (str): The input text.
        top_n (int): Number of keywords to extract.
        diversity (float): Controls the diversity of keywords (0 = low diversity, 1 = high diversity).

    Returns:
        str: The generated prompt.
    """
    keywords = extract_keywords(text, top_n, diversity)
    keyword_prompt = " ".join([f"{kw[0]}:{kw[1]:.2f}" for kw in keywords])
    return f"{keyword_prompt} {text}"


# Function to create prompts using the kw_sep strategy
def create_kw_sep_prompt(text, top_n=5, diversity=0.5):
    """
    Creates a prompt using the kw_sep strategy.

    Args:
        text (str): The input text.
        top_n (int): Number of keywords to extract.
        diversity (float): Controls the diversity of keywords (0 = low diversity, 1 = high diversity).

    Returns:
        str: The generated prompt.
    """
    keywords = extract_keywords(text, top_n, diversity)
    keyword_prompt = " </s> ".join([kw[0] for kw in keywords]) + " </s>"
    return f"{keyword_prompt} {text}"
