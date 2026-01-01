import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import warnings
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
nltk.download('stopwords')

# Load and merge data
df1 = pd.read_csv("dreaddit-train.csv")
df3 = pd.read_csv("dreaddit-test.csv")
df = pd.concat([df1, df3], ignore_index=True)

# Sentiment detection
def detect_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Text cleaning
stemmer = SnowballStemmer("english")
stopwords_set = set(stopwords.words("english"))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split() if word not in stopwords_set]
    text = [stemmer.stem(word) for word in text]
    return " ".join(text)

# Apply cleaning and sentiment
df2 = df.copy()
df2["text"] = df2["text"].apply(clean)
df2["sentiment"] = df2["text"].apply(detect_sentiment)
df2["label"] = df2["label"].map({0: "No Stress", 1: "Stress"})
df2 = df2[["text", "label", "sentiment"]]
print(df2["label"].value_counts())

# Word cloud
def generate_wc(data, bgcolor):
    plt.figure(figsize=(20, 20))
    mask = np.array(Image.open(r"C:\Users\akann\OneDrive\Desktop\stressed\stress-954814_960_720.png"))
    wc = WordCloud(background_color=bgcolor, stopwords=STOPWORDS, mask=mask)
    wc.generate(' '.join(data))
    plt.imshow(wc)
    plt.axis("off")
    plt.show()

print("Generating word cloud...")
generate_wc(df2["text"], 'white')

# Label distribution
import seaborn as sns
sns.countplot(x=df2["label"])
plt.show()

# Feature extraction and model training
X = df2["text"]
y = df2["label"]

vectorizer = TfidfVectorizer(stop_words="english")
X_vect = vectorizer.fit_transform(X)

from scipy.sparse import hstack
sentiment_values = (df2["sentiment"] * 100).values.reshape(-1, 1)
X_combined = hstack([X_vect, sentiment_values])

X_train, X_test, y_train, y_test = train_test_split(X_combined, y, random_state=42)
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)

import numpy as np

feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]

# Add sentiment to the feature list
feature_names = np.append(feature_names, "sentiment")

print("\nTop features for 'Stress':")
top_stress = np.argsort(coefficients)[-20:]
for i in reversed(top_stress):
    print(f"{feature_names[i]}: {coefficients[i]:.4f}")

print("\nTop features for 'No Stress':")
top_nostress = np.argsort(coefficients)[:20]
for i in top_nostress:
    print(f"{feature_names[i]}: {coefficients[i]:.4f}")

# Evaluation
y_pred = model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))

# BERT-based sentiment analysis using Hugging Face
from transformers import pipeline

# Load sentiment analysis pipeline FIRST
sentiment_model = pipeline("sentiment-analysis")


def predict_with_bert(text):
    result = sentiment_model(text)[0]
    return "No Stress" if result['label'] == 'POSITIVE' else "Stress"

print("BERT → Feeling happy and relaxed:", predict_with_bert("Feeling happy and relaxed"))
print("BERT → I need help and feel overwhelmed:", predict_with_bert("I need help and feel overwhelmed"))

# Load sentiment analysis pipeline
sentiment_model = pipeline("sentiment-analysis")

# Test it
print("\nBERT Sentiment Predictions:")
print("Feeling happy and relaxed →", sentiment_model("Feeling happy and relaxed")[0])
print("I need help and feel overwhelmed →", sentiment_model("I need help and feel overwhelmed")[0])

# Sentiment-only model for comparison
from sklearn.linear_model import LogisticRegression

X_sentiment = df2["sentiment"].values.reshape(-1, 1)
model_sentiment = LogisticRegression()
model_sentiment.fit(X_sentiment, y)

def predict_sentiment_only(text):
    polarity = detect_sentiment(clean(text))
    return model_sentiment.predict([[polarity]])[0]

print("\nSentiment-only predictions:")
print("Sentiment-only → Feeling happy and relaxed:", predict_sentiment_only("Feeling happy and relaxed"))
print("Sentiment-only → I need help and feel overwhelmed:", predict_sentiment_only("I need help and feel overwhelmed"))


# Test with user input
def predict_text(text):
    cleaned = clean(text)
    sentiment = detect_sentiment(cleaned)
    vect_input = vectorizer.transform([cleaned])
    combined_input = hstack([vect_input, np.array([[sentiment]])])
    prediction = model.predict(combined_input)[0]
    return prediction
examples = [
    "I feel calm and peaceful today",
    "Everything is going well",
    "I'm happy and content",
    "Life feels balanced and joyful"
]

for text in examples:
    df2.loc[len(df2)] = [clean(text), "No Stress", detect_sentiment(text)]

# Examples
print("Full model → Feeling happy and relaxed:", predict_text("Feeling happy and relaxed"))
print("Full model → I feel calm and peaceful today:", predict_text("I feel calm and peaceful today"))
print("Full model → I need help and feel overwhelmed:", predict_text("I need help and feel overwhelmed"))

# BERT-based sentiment analysis using Hugging Face
from transformers import pipeline

# Load the BERT sentiment model
sentiment_model = pipeline("sentiment-analysis")

# Define prediction function
def predict_with_bert(text):
    result = sentiment_model(text)[0]
    return "No Stress" if result['label'] == 'POSITIVE' else "Stress"

# Test predictions
print("BERT → Feeling happy and relaxed:", predict_with_bert("Feeling happy and relaxed"))
print("BERT → I need help and feel overwhelmed:", predict_with_bert("I need help and feel overwhelmed"))
print("BERT → I feel calm and peaceful today:", predict_with_bert("I feel calm and peaceful today"))
print("BERT → Life feels balanced and joyful:", predict_with_bert("Life feels balanced and joyful"))
