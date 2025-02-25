from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

DETAILS_FILE = "details.txt"

@app.route("/save-details", methods=["POST"])
def save_details():
    try:
        data = request.json
        name = data.get("name", "").strip()
        email = data.get("email", "").strip()
        dob = data.get("dob", "").strip()

        if not name or not email or not dob:
            return jsonify({"error": "Invalid input"}), 400

        with open(DETAILS_FILE, "a") as file:
            file.write(f"Name: {name}, Email: {email}, DOB: {dob}\n")

        return jsonify({"message": "Details saved successfully!"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5002, debug=True)
