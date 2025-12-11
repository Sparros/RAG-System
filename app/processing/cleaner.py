def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - Normalize whitespace
    - Remove excessive newlines
    - Strip leading/trailing spaces
    """
    if not text:
        return ""

    # Replace line breaks with spaces
    text = text.replace("\r", " ").replace("\n", " ")

    # Collapse multiple spaces
    while "  " in text:
        text = text.replace("  ", " ")

    return text.strip()
