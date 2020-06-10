import re

def normalized_text(comments):
    """
    Series: series containing text you would like normalized.
    Normalized meaning raw text is converted into lower-case w/ punctuation and stopwords removed, as well as tokenized"""
    comments = comments.lower()
    comments = re.sub(r'[^\w\s]+', ' ', comments) #Remove punctuation
    comments= re.sub(r'\s\s+', ' ', comments) #Remove excess white spaces between text
    comments.strip() #strip leading/trailing whitespace
    return comments