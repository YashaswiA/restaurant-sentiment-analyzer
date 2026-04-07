import streamlit as st
import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')

st.set_page_config(page_title="Restaurant Sentiment Analyzer", page_icon="🍽️", layout="centered")

# Load models
@st.cache_resource
def load_models():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('cv.pkl', 'rb') as f:
        cv = pickle.load(f)
    return model, cv

model, cv = load_models()

def predict_sentiment(review):
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    if 'not' in all_stopwords:
        all_stopwords.remove('not')
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    X = cv.transform([review]).toarray()
    prediction = model.predict(X)
    return "Positive" if prediction[0] == 1 else "Negative"

st.title("🍽️ Restaurant Review Sentiment Analyzer")
st.markdown("Analyze customer reviews to determine whether they are **Positive** or **Negative**.")

tab1, tab2 = st.tabs(["Single Review Analysis", "Batch Analysis via CSV"])

with tab1:
    st.subheader("Analyze a Single Review")
    user_input = st.text_area("Enter a restaurant review:", placeholder="The food was extremely delicious, but the service was a bit slow...")
    
    if st.button("Analyze Review"):
        if user_input.strip() == "":
            st.warning("Please enter a review first.")
        else:
            sentiment = predict_sentiment(user_input)
            if sentiment == "Positive":
                st.success(f"**Sentiment: {sentiment}** 😊")
            else:
                st.error(f"**Sentiment: {sentiment}** 😞")
                
with tab2:
    st.subheader("Batch Review Analysis")
    st.write("Upload a CSV file containing a column named **'Review'**. The app will predict the sentiment for each review and provide a summary.")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv", "tsv"])
    
    if uploaded_file is not None:
        try:
            # Try parsing as CSV, if fails try TSV or vice versa based on extension
            if uploaded_file.name.endswith('.tsv'):
                df = pd.read_csv(uploaded_file, delimiter='\t', quoting=3)
            else:
                df = pd.read_csv(uploaded_file)
                
            if 'Review' not in df.columns:
                st.error("The uploaded file must contain a column named 'Review'.")
            else:
                st.write("Analyzing reviews...")
                
                # Make predictions
                df['Sentiment Prediction'] = df['Review'].apply(predict_sentiment)
                
                # Counts
                pos_count = len(df[df['Sentiment Prediction'] == "Positive"])
                neg_count = len(df[df['Sentiment Prediction'] == "Negative"])
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Reviews", len(df))
                col2.metric("Positive Reviews 😊", pos_count)
                col3.metric("Negative Reviews 😞", neg_count)
                
                st.markdown("### Processed Data")
                st.dataframe(df)
                
                # Make it downloadable
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Analyzed Data",
                    csv_data,
                    "analyzed_reviews.csv",
                    "text/csv",
                    key='download-csv'
                )
        except Exception as e:
            st.error(f"Error processing file: {e}")
