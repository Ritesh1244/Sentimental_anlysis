import streamlit as st  
import os  
from openai import OpenAI  
from dotenv import load_dotenv  
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline  

# Load environment variables from .env file  
load_dotenv()  

# Load OpenAI API key  
api_key = os.getenv("OPENAI_API_KEY")  
client = OpenAI(api_key=api_key)  

# Load intent model and tokenizer  
intent_model = AutoModelForSequenceClassification.from_pretrained("yeniguno/bert-uncased-intent-classification")  
intent_tokenizer = AutoTokenizer.from_pretrained("yeniguno/bert-uncased-intent-classification")  
intent_pipe = pipeline("text-classification", model=intent_model, tokenizer=intent_tokenizer)  

# === Streamlit UI ===  
st.title("Smart Intent & Sentiment Analyzer")  
st.write("Enter a message to detect its intent and sentiment:")

# Input box  
user_input = st.text_area("Input Text", "", height=100)  

# Button to process input  
if st.button("Analyze"):  
    if user_input.strip() == "":  
        st.warning("⚠️ Please enter some text to analyze.")  
    else:  
        try:  
            # --- Intent Detection ---  
            intent_result = intent_pipe(user_input)  
            intent_label = intent_result[0]["label"]  

            # --- Sentiment Analysis ---  
            sentiment_response = client.chat.completions.create(  
                model="gpt-3.5-turbo",  
                messages=[  
                    {  
                        "role": "system",  
                        "content": "You are a sentiment analysis assistant. Respond with 'Positive', 'Negative', or 'Neutral' and give a brief reason."  
                    },  
                    {  
                        "role": "user",  
                        "content": user_input  
                    }  
                ]  
            )  
            sentiment_result = sentiment_response.choices[0].message.content.strip()  

            # --- Output Results ---  
            st.subheader("🔎 Predicted Intent:")  
            st.info(intent_label)  

            st.subheader("✅ Predicted Sentiment:")  
            st.success(sentiment_result)  

        except Exception as e:  
            st.error(f"❌ Error: {e}")
