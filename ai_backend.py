from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import random
import os
import json

app = Flask(__name__)
CORS(app)

OLLAMA_API_URL = "http://localhost:11434/api/generate"  
FORENSIC_API_URL = "http://127.0.0.1:5003/analyze"
UPLOAD_FOLDER = "uploads"
REPORTS_FOLDER = "reports"


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        user_message = data.get("message", "").lower().strip()

        if not user_message:
            return jsonify({"error": "No message provided"}), 400


        if user_message == "analyze":
            files = os.listdir(UPLOAD_FOLDER)
            if not files:
                return jsonify({"response": "No uploaded file found. Please upload a file first."})

            latest_file = sorted(files, key=lambda x: os.path.getmtime(os.path.join(UPLOAD_FOLDER, x)), reverse=True)[0]
            forensic_response = requests.post(FORENSIC_API_URL, json={"file_name": latest_file})
            forensic_data = forensic_response.json()

            if "error" in forensic_data:
                return jsonify({"response": forensic_data["error"]})


        
        payload = {"model": "llama3.2:1b-instruct-q4_0", "prompt": user_message, "stream": False}
        response = requests.post(OLLAMA_API_URL, json=payload)
        response_data = response.json()
        ai_message = response_data.get("response", "").strip()

        short_response = ". ".join(ai_message.split(". ")[:10]).strip() + "."
        return jsonify({"response": ai_message})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5004, debug=True)
