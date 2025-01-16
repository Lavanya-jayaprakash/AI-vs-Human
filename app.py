from flask import Flask, render_template, request, jsonify
import torch
from transformers import BertTokenizer
from model import BERTClassifier  # Import BERTClassifier class from your model.py file

app = Flask(__name__)

# Load the trained model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTClassifier(bert_model_name='bert-base-uncased', num_classes=2)
model.load_state_dict(torch.load("bert_model.pth"))
model.to(device)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Home route to serve the HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction endpoint to process user input and predict text
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    
    # Prepare the text for prediction
    encoding = tokenizer(text, return_tensors='pt', max_length=256, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
    
    result = "AI-generated" if preds.item() == 1 else "Human-written"
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)
