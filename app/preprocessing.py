import re

def clean_text(text: str) -> str:
    """
    Remove unnecessary whitespace and page artifacts.
    """

    # Replace multiple spaces with one
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing spaces
    text = text.strip()

    return text