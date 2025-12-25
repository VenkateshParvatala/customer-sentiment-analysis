from flask import Flask, request, jsonify
import joblib
import re
import nltk
import threading
import webbrowser
from nltk.corpus import stopwords

# Download stopwords (only first time)
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Clean text function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# Load trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = Flask(__name__)

# ---------------- API ROUTE ----------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    prediction = model.predict(X)[0]

    return jsonify({
        "input": text,
        "cleaned": cleaned,
        "sentiment": prediction
    })


# ---------------- UI ROUTE ----------------
@app.route("/", methods=["GET", "POST"])
def ui():
    sentiment = None
    color = "black"

    if request.method == "POST":
        user_text = request.form.get("user_text", "")
        cleaned = clean_text(user_text)
        X = vectorizer.transform([cleaned])
        sentiment = model.predict(X)[0]

        if sentiment == "positive":
            color = "green"
        elif sentiment == "negative":
            color = "red"
        else:
            color = "orange"

    return f"""
    <html>
    <head>
        <title>Customer Sentiment Analyzer</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f4f6f8;
                padding: 40px;
            }}
            .container {{
                background: white;
                padding: 30px;
                width: 500px;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            textarea {{
                width: 100%;
                height: 120px;
                font-size: 14px;
                padding: 10px;
            }}
            button {{
                margin-top: 10px;
                padding: 10px 20px;
                font-size: 15px;
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }}
            button:hover {{
                background-color: #0056b3;
            }}
            .result {{
                margin-top: 20px;
                font-size: 20px;
                font-weight: bold;
                color: {color};
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Customer Sentiment Analyzer</h2>
            <form method="POST">
                <textarea name="user_text" placeholder="Enter text here"></textarea><br>
                <button type="submit">Analyze</button>
            </form>
            <div class="result">
                {sentiment if sentiment else ""}
            </div>
        </div>
    </body>
    </html>
    """

# ---------------- RUN APP ----------------
def open_browser():
    webbrowser.open("http://127.0.0.1:5000/")

if __name__ == "__main__":
    threading.Timer(0.5, open_browser).start()
    app.run(debug=True)