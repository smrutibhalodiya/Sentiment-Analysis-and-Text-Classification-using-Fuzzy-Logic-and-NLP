# app.py
from flask import Flask, request, jsonify, render_template
import nltk
import os
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import skfuzzy as fuzz
import numpy as np

# Ensure NLTK resources are downloaded
def ensure_nltk_resources():
    nltk_data_path = nltk.data.path
    if not os.path.exists(os.path.join(nltk_data_path[0], 'tokenizers/punkt')):
        nltk.download('punkt')
    if not os.path.exists(os.path.join(nltk_data_path[0], 'corpora/stopwords')):
        nltk.download('stopwords')

class FuzzySentimentAnalyzer:
    def __init__(self):
        ensure_nltk_resources()
        
        # Initialize stopwords
        self.stop_words = set(stopwords.words('english'))
        
        # Words that negate sentiment
        self.negations = ['not', 'no', "don't", 'none', 'neither', 'nor', 'without']
        
        # Words that intensify sentiment
        self.intensifiers = {
            'very': 1.5,
            'extremely': 2.0,
            'somewhat': 0.5,
            'slightly': 0.3,
            'really': 1.7,
            'absolutely': 2.0,
            'barely': 0.2,
            'hardly': 0.2,
            'quite': 1.3,
            'exceptionally': 1.8,
            'particularly': 1.4
        }
        
        # Sentiment lexicon with fuzzy membership values [negative, neutral, positive]
        self.sentiment_lexicon = {

            #  Positive Sentiment
            'good': [0.0, 0.3, 0.7],
            'nice': [0.0, 0.2, 0.8],
            'excellent': [0.0, 0.1, 0.9],
            'great': [0.0, 0.2, 0.8],
            'wonderful': [0.0, 0.1, 0.9],
            'amazing': [0.0, 0.1, 0.9],
            'fantastic': [0.0, 0.0, 1.0],
            'outstanding': [0.0, 0.1, 0.9],
            'perfect': [0.0, 0.0, 1.0],
            'awesome': [0.0, 0.0, 1.0],
            'love': [0.0, 0.1, 0.9],
            'like': [0.0, 0.3, 0.7],
            'enjoy': [0.0, 0.2, 0.8],
            'enjoyed': [0.0, 0.2, 0.8],
            'enjoying': [0.0, 0.2, 0.8],
            'recommend': [0.0, 0.3, 0.7],
            'recommended': [0.0, 0.3, 0.7],
            'worth': [0.0, 0.3, 0.7],
            'excited': [0.0, 0.2, 0.8],
            'exciting': [0.0, 0.2, 0.8],
            'convenient': [0.0, 0.4, 0.6],
            'happy': [0.0, 0.2, 0.8],
            'fast': [0.0, 0.4, 0.6],
            'reasonable': [0.0, 0.6, 0.4],
            'spacious': [0.0, 0.3, 0.7],
            'clean': [0.0, 0.4, 0.6],
            'friendly': [0.0, 0.2, 0.8],
            'polite': [0.0, 0.3, 0.7],
            'helpful': [0.0, 0.3, 0.7],
            'efficient': [0.0, 0.3, 0.7],

            # Neutral Sentiment
            'fine': [0.1, 0.8, 0.1],
            'okay': [0.1, 0.8, 0.1],
            'ok': [0.1, 0.8, 0.1],
            'average': [0.2, 0.7, 0.1],
            'mediocre': [0.5, 0.5, 0.0],
            'interesting': [0.0, 0.3, 0.7],

            # Negative Sentiment
            'bad': [0.7, 0.3, 0.0],
            'poor': [0.8, 0.2, 0.0],
            'terrible': [0.9, 0.1, 0.0],
            'horrible': [1.0, 0.0, 0.0],
            'awful': [0.9, 0.1, 0.0],
            'disappointing': [0.8, 0.2, 0.0],
            'disappointed': [0.8, 0.2, 0.0],
            'hate': [0.9, 0.1, 0.0],
            'dislike': [0.7, 0.3, 0.0],
            'waste': [0.8, 0.2, 0.0],
            'wasted': [0.8, 0.2, 0.0],
            'boring': [0.6, 0.4, 0.0],
            'inconvenient': [0.6, 0.4, 0.0],
            'sad': [0.8, 0.2, 0.0],
            'slow': [0.7, 0.3, 0.0],
            'unreasonable': [0.7, 0.3, 0.0],
            'cramped': [0.7, 0.3, 0.0],
            'dirty': [0.8, 0.2, 0.0],
            'unfriendly': [0.8, 0.2, 0.0],
            'rude': [0.9, 0.1, 0.0],
            'unhelpful': [0.7, 0.3, 0.0],
            'inefficient': [0.7, 0.3, 0.0],

            # Ambiguous / Mixed Feelings
            'interesting': [0.0, 0.3, 0.7],  # Could be positive or neutral
            'challenging': [0.4, 0.4, 0.2],  # Might be a good thing or not
            'unexpected': [0.4, 0.3, 0.3],   # Depends on context
            'bittersweet': [0.4, 0.1, 0.5],  # Mixed feeling
            'surprising': [0.3, 0.4, 0.3],   # Could be good or bad
            'unusual': [0.3, 0.5, 0.2],      # Neutral, leaning toward negative
        }

        
        # Set up fuzzy universe variables
        self.sentiment_range = np.arange(0, 1.01, 0.01)
        
        # Define fuzzy membership functions
        self.negative_mf = fuzz.trimf(self.sentiment_range, [0, 0, 0.6])
        self.neutral_mf = fuzz.trimf(self.sentiment_range, [0, 0.5, 1])
        self.positive_mf = fuzz.trimf(self.sentiment_range, [0.4, 1, 1])
    
    def preprocess_text(self, text):
        """Tokenize text and remove irrelevant stopwords"""
        tokens = word_tokenize(text.lower())
        # Keep negations and intensifiers, remove other stopwords
        return [token for token in tokens if token not in self.stop_words or 
                token in self.negations or token in self.intensifiers]
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of the text and return fuzzy results"""
        tokens = self.preprocess_text(text)
        
        if not tokens:
            return {
                'fuzzy_label': 'neutral',
                'crisp_score': 0.5,
                'membership': {
                    'negative': 0.0,
                    'neutral': 1.0,
                    'positive': 0.0
                }
            }
        
        # Analyze sentiments with negation and intensifier handling
        sentiments = []
        negation_active = False
        current_intensifier = 1.0
        
        for i in range(len(tokens)):
            
            token = tokens[i]

            # Check for negation
            if token in self.negations:
                print(f"Negation found: {token}")
                negation_active = True
                continue
            
            # Check for intensifiers
            if token in self.intensifiers:
                current_intensifier = self.intensifiers[token]
                print(f"Intensifier found: {token} with multiplier: {current_intensifier}")
                continue

            # Check if word is in sentiment lexicon
            if token in self.sentiment_lexicon:
                scores = self.sentiment_lexicon[token]
                
                # Apply negation if active (swap positive and negative scores)
                if negation_active:
                    scores = [scores[2], scores[1], scores[0]]  # Swap neg/pos
                    negation_active = False  # Reset negation
                
                # Apply intensifier to the dominant sentiment
                max_score_index = np.argmax(scores)
                intensified_scores = scores.copy()
                print(f"Applying intensifier: {current_intensifier} to token: {token} -> {scores}")

                # best value ko bada and chota kar rahe on basis of current_intensifier
                # Increase the dominant sentiment and decrease others
                modified_sentiment = [0, 0, 0]
                for j in range(3):
                    if j == max_score_index:
                        # Increase the dominant sentiment
                        modified_sentiment[j] = min(1.0, scores[j] * current_intensifier)
                    else:
                        # Decrease others proportionally
                        modified_sentiment[j] = max(0.0, scores[j] / current_intensifier)
                
                # Normalize to ensure sum is 1
                total = sum(modified_sentiment)
                if total > 0:
                    scores = [v / total for v in modified_sentiment]
                
                sentiments.append(scores)
                current_intensifier = 1.0  # Reset intensifier
        
        # Calculate average sentiment
        if sentiments:
            # avg_sentiment = [
            #     sum(s[0] for s in sentiments) / len(sentiments),
            #     sum(s[1] for s in sentiments) / len(sentiments),
            #     sum(s[2] for s in sentiments) / len(sentiments)
            # ]

            # or 

            avg_sentiment = np.mean(sentiments, axis=0)
            print(f"Average sentiment: {avg_sentiment}")
        else:
            avg_sentiment = [0.2, 0.6, 0.2]  # Default sentiment if no words match
        
        # Calculate crisp score from fuzzy values (weighted average)
        crisp_score = 0.0 * avg_sentiment[0] + 0.5 * avg_sentiment[1] + 1.0 * avg_sentiment[2]
        
        # Determine membership in each fuzzy set
        negative_membership = fuzz.interp_membership(self.sentiment_range, self.negative_mf, crisp_score)
        neutral_membership = fuzz.interp_membership(self.sentiment_range, self.neutral_mf, crisp_score)
        positive_membership = fuzz.interp_membership(self.sentiment_range, self.positive_mf, crisp_score)
        
        # Normalize the values
        total_membership = negative_membership + neutral_membership + positive_membership
        
        if total_membership == 0:
           negative_membership = neutral_membership = positive_membership = 1 / 3
        else:
           negative_membership /= total_membership
           neutral_membership /= total_membership
           positive_membership /= total_membership
        
        # Determine fuzzy label
        memberships = [negative_membership, neutral_membership, positive_membership]
        print(f"Memberships: {memberships}")
        fuzzy_categories = ['very negative', 'negative', 'slightly negative', 'neutral', 
                           'slightly positive', 'positive', 'very positive']
        
        # Map crisp score to fuzzy category
        if crisp_score < 0.1:
            fuzzy_label = fuzzy_categories[0]  # very negative
        elif crisp_score < 0.30:
            fuzzy_label = fuzzy_categories[1]  # negative
        elif crisp_score < 0.45:
            fuzzy_label = fuzzy_categories[2]  # slightly negative
        elif crisp_score < 0.55:
            fuzzy_label = fuzzy_categories[3]  # neutral
        elif crisp_score < 0.70:
            fuzzy_label = fuzzy_categories[4]  # slightly positive
        elif crisp_score < 0.90:
            fuzzy_label = fuzzy_categories[5]  # positive
        else:
            fuzzy_label = fuzzy_categories[6]  # very positive
        
        return {
            'fuzzy_label': fuzzy_label,
            'crisp_score': round(crisp_score, 2),
            'membership': {
                'negative': round(memberships[0], 2),
                'neutral': round(memberships[1], 2),
                'positive': round(memberships[2], 2)
            }
        }

# Initialize Flask app
app = Flask(__name__)

# Initialize sentiment analyzer
analyzer = FuzzySentimentAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    result = analyzer.analyze_sentiment(text)
    return jsonify(result)

@app.route('/examples')
def examples():
    test_sentences = [
        "The movie was excellent, I really enjoyed it.",
        "The service was terrible and the food was cold.",
        "It was an okay experience, nothing special.",
        "The movie was not bad actually.",
        "I'm disappointed with this product.",
        "The room was clean, but the service was very slow.",
        "This book is somewhat interesting but also quite challenging.",
        "The experience was absolutely fantastic!",
        "It was okay, but I like the ambiance.",
        "The weather is a bit challenging today, but still beautiful."
        "While the hotel room was spacious and clean, the staff were rude and the location was far from ideal.",
        "I've never been so disappointed with a product in my entire life, everything about it is horrible."
    ]
    
    results = []
    for sentence in test_sentences:
        result = analyzer.analyze_sentiment(sentence)
        results.append({'text': sentence, 'analysis': result})
    
    return jsonify(results)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)