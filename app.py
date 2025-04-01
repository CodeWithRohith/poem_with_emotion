from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model components
try:
    model = BertForSequenceClassification.from_pretrained("deployment/model")
    tokenizer = BertTokenizer.from_pretrained("deployment/tokenizer")
    with open("deployment/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    print("✅ All models loaded successfully!")
except Exception as e:
    print(f"❌ Loading failed: {str(e)}")
    raise

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get and validate input
        data = request.get_json()
        if not data or 'poem' not in data:
            return jsonify({"error": "Missing 'poem' parameter"}), 400
        
        poem = data['poem'].strip()
        if len(poem) < 5:  # Minimum length check
            return jsonify({"error": "Poem too short"}), 400

        # Tokenize and predict
        inputs = tokenizer(poem, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Format results
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]
        predicted_id = np.argmax(probs)
        
        return jsonify({
            "status": "success",
            "prediction": {
                "emotion": label_encoder.inverse_transform([predicted_id])[0],
                "confidence": round(float(probs[predicted_id]), 4)
            },
            "probabilities": {
                label: round(float(prob), 4) 
                for label, prob in zip(label_encoder.classes_, probs)
            }
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)