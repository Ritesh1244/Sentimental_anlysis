

import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize the Porter Stemmer
port_stem = PorterStemmer()

def clean_text(text):
    """Cleans text by removing non-alphabetic characters and converting to lowercase"""
    text = re.sub('[^a-zA-Z]', ' ', text)
    return text.lower()

def remove_stopwords(text):
    """Removes English stopwords"""
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in text.split() if word not in stop_words])

def stem_text(text):
    """Applies stemming to words"""
    return ' '.join([port_stem.stem(word) for word in text.split()])

def preprocess_text(text):
    """Runs full preprocessing pipeline on input text"""
    text = clean_text(text)
    text = remove_stopwords(text)
    return stem_text(text)
