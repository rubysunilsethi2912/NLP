import re
import string
import joblib
import streamlit as st

# Load the saved model and vectorizer
model = joblib.load("trained_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


# Define the text cleaning function
def clean_text(text):
  text = str(text).lower()
  text = re.sub(r"http\S+|www\S+|https\S+", "", text)
  text = re.sub(r"@\w+", "", text)
  text = re.sub(r"#", "", text)
  text = text.translate(str.maketrans("", "", string.punctuation))
  text = re.sub(r"\d+", "", text)
  text = re.sub(r"\s+", " ", text).strip()
  return text


st.title("🚨 Disaster Tweet Classification App")
st.write("Enter a tweet below to check if it's a disaster or not.")

user_input = st.text_area("Enter Tweet Text here:", "")

if st.button("Classify Tweet"):
  if user_input.strip() == "":
    st.warning("Please enter some text to classify.")
  else:
    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    if prediction == 1:
      st.error(
          "⚠️ **Disaster Tweet Detected!** This looks like a real emergency"
          " report."
      )
    else:
      st.success(
          "✅ **Non-Disaster Tweet.** This does not appear to be an emergency."
      )