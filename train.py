import pandas as pd
import numpy as np
import re
import nltk
import pickle
import os

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB

# Download stopwords if not already present
nltk.download('stopwords')

# Read dataset
# Using absolute path based on user's location
dataset_path = r"C:\Users\admin\Downloads\Restaurant_Reviews.tsv"
dataset = pd.read_csv(dataset_path, delimiter='\t', quoting=3)

corpus = []
for i in range(0, dataset.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'].iloc[i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    # According to original notebook, don't remove 'not'
    if 'not' in all_stopwords:
        all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

# Create Bag of Words model
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Train Naive Bayes model on entire dataset
classifier = GaussianNB()
classifier.fit(X, y)

# Ensure output directory exists (current dir)
output_dir = os.path.dirname(os.path.abspath(__file__))

# Save the models
with open(os.path.join(output_dir, 'cv.pkl'), 'wb') as f:
    pickle.dump(cv, f)

with open(os.path.join(output_dir, 'model.pkl'), 'wb') as f:
    pickle.dump(classifier, f)

print("Training completed. Pickled models saved.")
