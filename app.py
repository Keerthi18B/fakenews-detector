import pandas as pd
import numpy as np
import requests
from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)

# Fetch latest news
def fetch_latest_news():
    url = "https://newsapi.org/v2/top-headlines?country=us&apiKey=d7475d0c527945dd908e1c8df585f586"
    response = requests.get(url).json()
    articles = response.get("articles", [])
    return [article["title"] + " " + article["description"] for article in articles if article["title"] and article["description"]]

# Load Dataset
df_fake = pd.read_csv("Fake.csv")
df_real = pd.read_csv("True.csv")

df_fake['label'] = 0  # 0 for fake news
df_real['label'] = 1  # 1 for real news

min_samples = min(len(df_fake), len(df_real))
df_fake_balanced = df_fake.sample(n=min_samples, random_state=42)
df_real_balanced = df_real.sample(n=min_samples, random_state=42)

# Create balanced dataset
df = pd.concat([df_fake_balanced, df_real_balanced]).sample(frac=1, random_state=42).reset_index(drop=True)

df = pd.concat([df_fake, df_real])
df = df[['text', 'label']]
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the dataset

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Convert text data into numerical vectors
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9, min_df=5, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Function to predict news
def predict_news(text):
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]
    return "Real News" if prediction == 1 else "Fake News"

# Compare Input with Latest News
def check_with_latest_news(text):
    latest_news = fetch_latest_news()
    latest_news_tfidf = vectorizer.transform(latest_news)
    text_tfidf = vectorizer.transform([text])
    similarities = (latest_news_tfidf * text_tfidf.T).toarray()
    
    max_similarity = np.max(similarities) if similarities.size > 0 else 0

    if max_similarity > 0.2:  # Lowered from 0.3 to 0.2
        return "Matches real news"
    elif max_similarity > 0.1:
        return "Possibly Real"
    else:
        return "Potentially Fake"


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    prediction = predict_news(text)
    match_result = check_with_latest_news(text)
    return jsonify({"result": prediction, "match": match_result})

if __name__ == '__main__':
    app.run(debug=True)
