from transformers import pipeline

sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

def predict_with_bert(text):
    result = sentiment_model(text)[0]
    return "No Stress" if result['label'] == 'POSITIVE' else "Stress"