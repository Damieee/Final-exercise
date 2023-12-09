from flask import Flask, request, jsonify
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

app = Flask(__name__)

# Home Route 
@app.route('/', methods=['GET'])
def index():
    return jsonify({'Message': "Welcome to our sentiment_model"})

# Create a route that manages user requests and performs sentiment prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']

    # Analyze sentiment
    sentiment = analyzer.polarity_scores(text)

    # Interpret the compound score
    compound_score = sentiment['compound']

    if compound_score >= 0.5:
        sentiment_label = 'This is a positive statement'
    elif 0.0 <= compound_score < 0.5:
        sentiment_label = 'This is a neutral statement'
    else:
        sentiment_label = 'This is a negative statement'

    # Return simplified sentiment label
    return jsonify({'sentiment': sentiment_label})

if __name__ == '__main__':
    app.run(debug=True)
