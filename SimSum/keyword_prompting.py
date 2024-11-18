from keybert import KeyBERT
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Initialize KeyBERT model
keybert_model = KeyBERT()

# Function to extract keywords with KeyBERT
def extract_keywords(text, top_n=5):
    """
    Extracts keywords from a text using KeyBERT.
    
    Args:
        text (str): The input text.
        top_n (int): Number of top keywords to extract.
        
    Returns:
        list: List of tuples containing keywords and their scores.
    """
    return keybert_model.extract_keywords(text, top_n=top_n)

# Function to create prompts using the kw_score strategy
def create_kw_score_prompt(text, top_n=5):
    """
    Creates a prompt using the kw_score strategy.
    
    Args:
        text (str): The input text.
        top_n (int): Number of keywords to extract.
        
    Returns:
        str: The generated prompt.
    """
    keywords = extract_keywords(text, top_n)
    keyword_prompt = " ".join([f"{kw[0]}:{kw[1]:.2f}" for kw in keywords])
    return f"{keyword_prompt} {text}"

# Function to create prompts using the kw_sep strategy
def create_kw_sep_prompt(text, top_n=5):
    """
    Creates a prompt using the kw_sep strategy.
    
    Args:
        text (str): The input text.
        top_n (int): Number of keywords to extract.
        
    Returns:
        str: The generated prompt.
    """
    keywords = extract_keywords(text, top_n)
    keyword_prompt = " </s> ".join([kw[0] for kw in keywords]) + " </s>"
    return f"{keyword_prompt} {text}"

# Example Usage
if __name__ == "__main__":
    input_text = (
        "Machine learning is a subset of artificial intelligence that focuses on the use of "
        "data and algorithms to imitate the way humans learn, gradually improving its accuracy."
    )
    top_n_keywords = 5  # Number of keywords to extract
    
    # Generate prompts
    kw_score_prompt = create_kw_score_prompt(input_text, top_n=top_n_keywords)
    kw_sep_prompt = create_kw_sep_prompt(input_text, top_n=top_n_keywords)
    
    print("kw_score Prompt:")
    print(kw_score_prompt)
    print("\nkw_sep Prompt:")
    print(kw_sep_prompt)
