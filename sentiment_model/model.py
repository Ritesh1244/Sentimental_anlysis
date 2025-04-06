
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer



class SentimentModel:
    def __init__(self, model_path, vectorizer_path):
        self.model = self.load_model(model_path)
        self.vectorizer = self.load_vectorizer(vectorizer_path)

    def load_model(self, model_path):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model

    def load_vectorizer(self, vectorizer_path):
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        return vectorizer

    def predict(self, text):
        text_vectorized = self.vectorizer.transform([text])  # Transform input text
        prediction = self.model.predict(text_vectorized)
        return prediction[0]
