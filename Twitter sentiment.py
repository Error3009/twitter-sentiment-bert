import streamlit as st
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Page Configuration
st.set_page_config(page_title="Twitter Sentiment Analyzer", page_icon="🐦")
st.title("🐦 Twitter Sentiment Analysis with BERT")
st.write("Enter any tweet below to analyze whether its sentiment is **Positive**, **Negative**, or **Neutral**.")

# 2. Text Preprocessing Function (removes @user handles and URLs)
def clean_tweet(text):
    text = re.sub(r'http\S+', '', text)  # Remove links
    text = re.sub(r'@\w+', '', text)     # Remove @mentions
    return text.strip()

# 3. Load Pre-trained BERT Model (Cached so it loads fast)
@st.cache_resource
def load_bert_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

with st.spinner("Loading BERT model weights... Please wait."):
    tokenizer, model = load_bert_model()

# 4. User Input Interface
tweet_input = st.text_area("Enter Tweet Text:", "Just tried the new app update, absolute game changer! Loving it.")

if st.button("Analyze Sentiment"):
    if tweet_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Preprocess input
        cleaned_text = clean_tweet(tweet_input)
        
        # Tokenize and predict
        inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1)[0]
        
        # Output labels mapping
        labels = ['Negative', 'Neutral', 'Positive']
        prediction = torch.argmax(scores).item()
        confidence = scores[prediction].item()

        # Display Results
        st.subheader("Result")
        if labels[prediction] == 'Positive':
            st.success(f"**Sentiment:** {labels[prediction]} 😊 (Confidence: {confidence:.2%})")
        elif labels[prediction] == 'Negative':
            st.error(f"**Sentiment:** {labels[prediction]} 😡 (Confidence: {confidence:.2%})")
        else:
            st.info(f"**Sentiment:** {labels[prediction]} 😐 (Confidence: {confidence:.2%})")

        # Display confidence breakdown
        st.write("---")
        st.write("**Full Confidence Breakdown:**")
        for label, score in zip(labels, scores):
            st.write(f"- **{label}:** {score.item():.2%}")