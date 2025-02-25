from flask import Flask, jsonify
from flask_cors import CORS
import os
import json

app = Flask(__name__)
CORS(app)

REPORTS_FOLDER = "reports"


CINEMATIC_TEMPLATE = (
    "file name - '{file_name}', "
    "file type - '{file_type}"
    "AI probability -  {ai_report}, "
    "edit status -  {edit_status},  "
    "possibe edits found - '{possible_edits} "
)

@app.route("/report_summary", methods=["GET"])
def report_summary():
    try:
        report_files = [f for f in os.listdir(REPORTS_FOLDER) if f.endswith(".json")]
        if not report_files:
            return jsonify({"response": "No forensic reports found."})

        latest_report = sorted(report_files, key=lambda x: os.path.getmtime(os.path.join(REPORTS_FOLDER, x)), reverse=True)[0]
        report_path = os.path.join(REPORTS_FOLDER, latest_report)

        with open(report_path, "r") as report_file:
            report_data = json.load(report_file)

        file_name = report_data.get("file_name", "Unknown File")
        ai_report = report_data.get("ai_report", 0)
        edit_status = "unaltered" if not report_data.get("edited", False) else "digitally modified"
        summary = report_data.get("summary", "No summary available.")
        possible_edits = report_data.get("possible_edits")

        formatted_response = (
    "FORBIDDEN FRAME ANALYSIS\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    f"File: {file_name}\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    f"AI Probability: {ai_report}\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    f"Edit Status: {edit_status}\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    f"Summary: {summary}\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    f"Anomalies Detected: {', '.join(possible_edits) if possible_edits else 'None'}\n"
)

        return jsonify({"response": formatted_response})

    

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5005, debug=True)
