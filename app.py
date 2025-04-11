import streamlit as st
import openai

# Use the secret key from Streamlit secrets
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# === Streamlit UI ===
st.title("Sentiment Analysis App")
st.write("Enter text to analyze its sentiment:")

# Input text area
user_input = st.text_area("Input Text", "", height=100)

# Button to analyze sentiment
if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        try:
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

            result = response.choices[0].message.content.strip()
            st.subheader("Predicted Sentiment:")
            st.success(result)

        except Exception as e:
            st.error(f"❌ Error: {e}")
