def load_raw_text_data(file_path: str) -> str:
    """Load raw text from a dataset file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
    