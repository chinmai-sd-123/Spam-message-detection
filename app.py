import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

# Load model & vectorizer
tfidf = joblib.load('trained_models/tfidf_vectorizer.pkl')
model = joblib.load('trained_models/spam_classifier.pkl')

# Suspicious phrases (RULES)
suspicious_phrases = [
    "not a scam",
    "this is not a scam",
    "trust me",
    "official message",
    "you have been chosen"
]

# Character normalization map
char_map = {
    '0': 'o',
    '1': 'i',
    '3': 'e',
    '4': 'a',
    '5': 's',
    '7': 't',
    '@': 'a'
}

def normalize_text(text):
    for char, replacement in char_map.items():
        text = text.replace(char, replacement)
    return text

# 🔹 RULE-ONLY NORMALIZATION (NO STEMMING)
def normalize_for_rules(text):
    text = normalize_text(text)
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    return text

# 🔹 ML CLEANING (WITH STEMMING)
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_for_ml(text):
    text = normalize_text(text)
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Streamlit UI
st.set_page_config(page_title='Spam Message Detection')
st.title('Spam Message Detection App')
st.write('Enter a message below to check whether it is **spam** or **not spam**.')

user_input = st.text_area('Enter your message here')

if st.button('Predict'):
    if user_input.strip() == '':
        st.warning('Please enter a message.')
    else:
        # RULE-BASED CHECK (FIRST)
        rule_text = normalize_for_rules(user_input)

        for phrase in suspicious_phrases:
            if phrase in rule_text:
                st.error('This message is **SPAM** (rule-based detection)')
                st.write('Reason: Suspicious reassurance phrase detected')
                st.stop()

        #  ML PREDICTION
        clean_text = clean_for_ml(user_input)
        vectorized = tfidf.transform([clean_text])

        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0][prediction]

        if prediction == 1:
            st.error(f'This message is **SPAM** (confidence: {probability*100:.2f}%)')
        else:
            st.success(f'This message is **NOT SPAM** (confidence: {probability*100:.2f}%)')
