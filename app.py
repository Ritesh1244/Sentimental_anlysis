import streamlit as st  
import os  
from openai import OpenAI  
from dotenv import load_dotenv  

# Load environment variables from .env file  
load_dotenv()  

# Use your OpenAI API key from environment variable  
api_key = os.getenv("OPENAI_API_KEY")  
client = OpenAI(api_key=api_key)  

# === Streamlit UI ===  
st.title("Sentiment Analysis App")  
st.write("Enter text to analyze its sentiment:")  

# Input text area (larger than text_input)  
user_input = st.text_area("Input Text", "", height=100)  

# Button to analyze sentiment  
if st.button("Analyze"):  
    if user_input.strip() == "":  
        st.warning("⚠️ Please enter some text to analyze.")  
    else:  
        try:  
            # Make GPT call  
            response = client.chat.completions.create(  
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

            # Get result  
            result = response.choices[0].message.content.strip()  
            st.subheader("Predicted Sentiment:")  
            st.success(result)  

        except Exception as e:  
            st.error(f"❌ Error: {e}")  
