from flask import Flask, render_template, request, jsonify
from model.bert_model import predict_with_bert

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['text']
    prediction = predict_with_bert(user_input)
    return jsonify({'result': prediction})

if __name__ == '__main__':
    app.run(debug=True) 