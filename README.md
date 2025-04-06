# Streamlit Sentiment Analysis

This project implements a sentiment analysis application using Streamlit. Users can input text and receive predictions on whether the sentiment is positive or negative based on a pre-trained model.

## Project Structure

```
streamlit-sentiment-analysis
├── app.py                # Main entry point for the Streamlit application
├── sentiment_model       # Package containing model and preprocessing code
│   ├── __init__.py      # Initializes the sentiment_model package
│   ├── model.py         # Loads the trained model and makes predictions
│   └── preprocess.py     # Contains functions for text preprocessing
├── requirements.txt      # Lists project dependencies
└── README.md             # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd streamlit-sentiment-analysis
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`.

3. Input your text in the provided text box and click the "Predict" button to see the sentiment prediction.

## Dependencies

This project requires the following Python packages:
- Streamlit
- scikit-learn
- pandas
- numpy
- nltk

Make sure to install all dependencies listed in `requirements.txt` before running the application.#   S e n t i m e n t a l _ a n l y s i s  
 