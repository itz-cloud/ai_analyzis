# AI Analysis Tool

## Introduction
This is an AI analysis tool that helps analyze images, videos, PDFs, text, or Word documents to detect whether AI-generated content is present. It also includes a locally running AI that answers user questions in real time.

## Features
- AI-powered analysis of images, videos, PDFs, and text
- Local AI assistant using LLaMA 3.2
- Interactive chatbox
- File upload functionality
- Secure backend with Flask and Python

## Download & Setup Instructions

### Step 1: Download the Project
Download the repository as a ZIP file:
[Download Here](https://github.com/itz-cloud/ai_analyzis/archive/refs/heads/main.zip)

Unzip the file in Windows.

### Step 2: Open Terminal & Set Up Virtual Environment
Navigate to the extracted folder and open the terminal. Then, run the following commands:

#### Create a Virtual Environment (venv)
```sh
python -m venv venv
```

#### Activate the Virtual Environment
For Windows:
```sh
venv\Scripts\activate
```
For Linux/Mac:
```sh
source venv/bin/activate
```

#### Install Dependencies
```sh
pip install -r requirements.txt
```

### Step 3: Start the Backend Services
Open **five** terminal windows and run the following commands **one by one** in each terminal:

```sh
python save_details.py
python ai_backend.py
python upload_backend.py
python forensic_backend.py
python report_backend.py
```

Make sure each backend is running successfully.

### Step 4: Install WSL & Ollama
In a **new terminal**, install WSL and Ollama:

#### Install WSL (for Windows users only)
```sh
wsl --install
```

#### Install Ollama
```sh
curl -fsSL https://ollama.com/install.sh | sh
```

#### Download and Run Local AI Model
```sh
ollama run llama3.2:1b-instruct-q4_0
```

### Step 5: Open the Web Application
1. Click on `index.html` to open it in a browser.
2. Click **Get Started** and enter your details to log in.
3. Ask any question in the chatbox.
4. Click the **+ button** to upload files.
5. Type `analyze` to start analyzing the uploaded file and get a report.

## Notes
- Ensure all five backend services and Ollama AI are running properly before testing.
- If you encounter any issues, check that your virtual environment is activated and all dependencies are installed correctly.

Enjoy using the AI Analysis Tool!

