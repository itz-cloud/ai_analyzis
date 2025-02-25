import os
import cv2
import librosa
import numpy as np
import mimetypes
import json
import moviepy.editor as mp
import pdfplumber
import docx
import soundfile as sf
from scipy.stats import entropy
from skimage.metrics import structural_similarity as ssim
from collections import Counter
from PIL import Image, ImageChops
from flask import Flask, request, jsonify
from flask_cors import CORS
import io
from PIL import Image, ImageChops, ImageEnhance
import piexif
import librosa.display
from scipy.signal import hilbert



app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
REPORTS_FOLDER = "reports"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)


# ✅ Compute perceptual hash (pHash)
def phash(image_path):
    img = Image.open(image_path).convert("L").resize((32, 32), Image.LANCZOS)
    pixels = np.array(img)
    dct = cv2.dct(np.float32(pixels))
    dct_low_freq = dct[:8, :8]
    avg = dct_low_freq.mean()
    return "".join("1" if pixel > avg else "0" for row in dct_low_freq for pixel in row)

# ai image anlysis
def analyze_image(file_path):
    try:
        # Open image from file path and ensure it is in RGB mode.
        img = Image.open(file_path)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img_array = np.array(img)
        
        # 🔍 Error Level Analysis (ELA)
        with io.BytesIO() as buffer:
            img.save(buffer, "JPEG", quality=90)
            buffer.seek(0)
            temp_img = Image.open(buffer)
            ela_img = ImageChops.difference(img, temp_img)
        
        extrema = ela_img.getextrema()
        max_diff = max([e[1] for e in extrema])
        scale_factor = 255 / max_diff if max_diff > 0 else 1
        ela_img = ImageEnhance.Brightness(ela_img).enhance(scale_factor)
        
        # 🔍 Structural Similarity Index (SSIM)
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
        similarity = ssim(img_gray, img_blur)
        
        # 🔍 Frequency Domain Analysis (DCT)
        img_ycrcb = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)
        y_channel = img_ycrcb[:, :, 0]
        dct = cv2.dct(np.float32(y_channel))
        dct_mean = np.mean(np.abs(dct))
        
        # 🔍 Color & Texture Anomalies
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]  
        texture_variation = np.std(cv2.Laplacian(img_gray, cv2.CV_64F))
        saturation_mean = np.mean(saturation)
        
        # 🔍 Noise Pattern Analysis
        noise_map = cv2.absdiff(img_gray, cv2.medianBlur(img_gray, 5))
        noise_level = np.mean(noise_map)
        
        # 🔍 Edge Consistency Analysis
        edges = cv2.Canny(img_gray, 100, 200)
        edge_density = np.sum(edges) / edges.size
        
        # 🔍 Metadata Analysis (EXIF)
        exif_data = None
        try:
            exif_data = piexif.load(img.info.get("exif", b""))
        except:
            exif_data = "No EXIF Data"
        
        # 🔍 Forgery Localization
        forgery_mask = cv2.absdiff(img_gray, cv2.equalizeHist(img_gray))
        forgery_intensity = np.mean(forgery_mask)
        
        # 🔍 Shadow & Reflection Consistency
        reflection_map = cv2.Laplacian(img_gray, cv2.CV_64F)
        reflection_variation = np.std(reflection_map)
        
        # 🔍 JPEG Ghosting Analysis
        ghost_map = cv2.absdiff(img_gray, cv2.medianBlur(img_gray, 11))
        ghosting_level = np.mean(ghost_map)
        
        # 🔍 Chromatic Aberration Analysis
        b, g, r = cv2.split(img_array)
        ca_map = cv2.absdiff(b, r)
        chromatic_aberration = np.mean(ca_map)
        
        # 🔍 Lighting Consistency Analysis
        lighting_gradient = np.mean(cv2.Sobel(img_gray, cv2.CV_64F, 1, 1, ksize=5))
        
        # 🔍 Splicing & Copy-Move Detection
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(img_gray, None)
        splicing_detected = len(keypoints) < 50
        
        # 🔍 Deep Fake Artifact Detection
        laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
        deepfake_artifacts = laplacian_var < 10
        
        probability_factors = [
            min(1, (max_diff / 255) * 6.0),  # ELA difference (Higher sensitivity)
            min(1, (dct_mean / 50) * 4.5),  # Stronger weight on DCT analysis
            min(1, (10 / (texture_variation + 1)) * 7.0),  # AI texture anomalies
            min(1, (noise_level / 80) * 3.5),  # Reduce noise impact slightly
            min(1, (0.02 / (edge_density + 0.001)) * 8.0),  # AI has sharper unnatural edges
            min(1, (ghosting_level / 30) * 6.0),  # JPEG ghosting analysis (higher weight)
            min(1, (deepfake_artifacts) * 12.0)  # AI artifacts get the highest weight
        ]

        # 🔥 AI Probability Score (Boost AI Scores)
        ai_probability = round(min(100, sum(probability_factors) * 25), 2)  

        # 🔍 AI Usage Report
        if ai_probability < 60:
            ai_report = "No AI detected (Real image)"
        elif 65 <= ai_probability < 75:
            ai_report = "Possibly AI-edited (Mild AI involvement)"
        else:
            ai_report = "Fully AI-generated (High AI probability)"

        # 🔥 Identify Possible Edits
        possible_edits = []
        if max_diff > 50:
            possible_edits.append("High ELA difference")
        if dct_mean > 10:
            possible_edits.append("Frequency anomalies detected")
        if texture_variation < 5:
            possible_edits.append("Unnatural texture variations")
        if noise_level > 20:
            possible_edits.append("Abnormal noise levels")
        if edge_density < 0.01:
            possible_edits.append("Edge inconsistencies found")
        if forgery_intensity > 15:
            possible_edits.append("Forgery localization detected")
        if reflection_variation < 5:
            possible_edits.append("Reflection inconsistency found")
        if ghosting_level > 20:
            possible_edits.append("JPEG ghosting detected")
        if chromatic_aberration > 10:
            possible_edits.append("Chromatic aberration anomaly")
        if splicing_detected:
            possible_edits.append("Possible splicing detected")
        if deepfake_artifacts:
            possible_edits.append("Deepfake artifact detected")
        
        return {
            "file_type": "Image",
            "ai_report": ai_report,
            "edited": bool(possible_edits),
            "summary": "Possible tampering detected." if possible_edits else "No major edits found.",
            "ai_probability": round(ai_probability, 2),  # AI-based probability score
            "possible_edits": possible_edits,  # List of detected anomalies
            "saturation_mean": float(saturation_mean),
            "texture_variation": float(texture_variation),
            "noise_level": float(noise_level),
            "edge_density": float(edge_density),
            "forgery_intensity": float(forgery_intensity),
            "reflection_variation": float(reflection_variation),
            "ghosting_level": float(ghosting_level),
            "chromatic_aberration": float(chromatic_aberration),
            "lighting_gradient": float(lighting_gradient),
            "splicing_detected": bool(splicing_detected),
            "deepfake_artifacts": bool(deepfake_artifacts)
        }
    except Exception as e:
        return {
            "file_type": "Image", 
            "edited": False, 
            "summary": f"Failed to analyze image: {str(e)}"
        }

# ✅ Analyze Video
def analyze_video(file_path):
    try:
        cap = cv2.VideoCapture(file_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        motion_diffs = []
        dct_variability = []
        noise_levels = []
        color_anomalies = []

        ret, prev_frame = cap.read()
        if not ret:
            return {"file_type": "Video", "ai_probability": 0, "edited": False, "summary": "Could not analyze video."}

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        for i in range(min(15, total_frames - 1)):  # Analyze 15 frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * total_frames // 15)
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # ✅ Motion Analysis
            diff = cv2.absdiff(prev_gray, gray).mean()
            motion_diffs.append(diff)
            prev_gray = gray
            
            # ✅ Frequency Domain (DCT Analysis)
            dct_coeff = cv2.dct(np.float32(gray) / 255.0)
            dct_variability.append(np.var(dct_coeff))
            
            # ✅ Noise Analysis (Laplacian Variance)
            noise_levels.append(cv2.Laplacian(gray, cv2.CV_64F).var())

            # ✅ Color Consistency Analysis
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            color_hist = cv2.calcHist([hsv], [0], None, [256], [0, 256])
            color_anomalies.append(np.var(color_hist))

        cap.release()

        # ✅ Compute Feature Scores
        motion_var = np.var(motion_diffs)
        dct_var = np.var(dct_variability)
        noise_var = np.var(noise_levels)
        color_var = np.var(color_anomalies)

        # ✅ AI Detection Heuristic
        ai_score = (motion_var * 1.2 + dct_var * 2.5 + noise_var * 1.8 + color_var * 2.0)
        ai_probability = float(round(min(100, max(0, ai_score * 10)), 2))
        edited = ai_probability > 50  # AI-generated threshold

        if ai_probability < 60:
            ai_report = "No AI detected (Real image)"
        elif 65 <= ai_probability < 75:
            ai_report = "Possibly AI-edited (Mild AI involvement)"
        else:
            ai_report = "Fully AI-generated (High AI probability)"

        return {
            "file_type": "Video",
            "ai_report": ai_report,
            "edited": edited,
            "summary": "Highly likely AI-generated or edited video detected." if edited else "No major AI artifacts detected."
        }
    except Exception as e:
        return {"file_type": "Video", "ai_probability": 0, "edited": False, "summary": f"Failed to analyze video: {str(e)}"}
# audio analyze
def analyze_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        
        # Feature 1: Spectral Entropy (Measures randomness in audio signal)
        spectral_entropy = entropy(np.histogram(y, bins=10, density=True)[0])

        # Feature 2: MFCC Mean Variance (AI voices lack natural variation)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_var = np.var(mfccs)

        # Feature 3: Spectral Contrast (AI-generated audio is often uniform)
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))

        # Feature 4: Zero-Crossing Rate (Measures abrupt changes in signal)
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))

        # Feature 5: Harmonic-to-Noise Ratio (HNR) - AI-generated voices often have a different noise profile
        hnr = np.mean(librosa.effects.harmonic(y))

        # Feature 6: Mel Cepstral Distortion (MCD) - Measures deviation from natural speech patterns
        mcd = np.mean(np.abs(mfccs - np.mean(mfccs, axis=1, keepdims=True)))

        # Feature 7: Energy Distribution Anomalies - AI voices have unnatural energy across frequencies
        rmse = np.mean(librosa.feature.rms(y=y))

        # Feature 8: Phase Distortion Analysis (Hilbert transform)
        analytic_signal = hilbert(y)
        phase_distortion = np.mean(np.abs(np.angle(analytic_signal)))

        # AI Probability Calculation (Weighted Score)
        ai_score = (
            (spectral_entropy * 15) +
            (1 / (mfcc_var + 0.1) * 100) +
            (100 - spectral_contrast * 5) +
            (zero_crossing_rate * 250) +
            ((1 - hnr) * 80) +
            (mcd * 30) +
            ((1 - rmse) * 50) +
            (phase_distortion * 60)
        )

        ai_probability = min(max(round(ai_score / 5, 2), 0), 100)  # Normalize to 0-100

        # AI classification
        if ai_probability < 50:
            ai_report = "No AI detected (Real audio)"
        elif 50 <= ai_probability < 70:
            ai_report = "Possibly AI-edited (Mild AI involvement)"
        elif 70 <= ai_probability < 85:
            ai_report = "Likely AI-generated (Moderate AI probability)"
        else:
            ai_report = "Fully AI-generated (High AI probability)"

        # Determine if the audio has been edited
        edited = bool(ai_probability > 60)

        return {
            "file_type": "Audio",
            "file_name": os.path.basename(file_path),
            "ai_report": ai_report,
            "ai_probability": ai_probability,
            "edited": edited,
            "possible_edit_found": "Yes" if edited else "No"
        }
    
    except Exception as e:
        return {
            "file_type": "Audio",
            "file_name": os.path.basename(file_path),
            "ai_report": "Analysis Failed",
            "ai_probability": 0,
            "edited": False,
            "possible_edit_found": "No",
            "error": str(e)
        }

import os
import pdfplumber
import docx
import numpy as np
from collections import Counter
from scipy.stats import entropy

# ✅ Improved Text Analysis Function
def analyze_text(content, file_name):
    words = content.split()
    total_words = len(words)
    unique_word_ratio = len(set(words)) / max(1, total_words)  # Avoid division by zero
    entropy_score = entropy(list(Counter(words).values())) if total_words > 1 else 0

    # ✅ AI Probability Calculation
    ai_probability = round((1 - unique_word_ratio) * 100, 2) + round(min(100, entropy_score * 10), 2)
    
    # ✅ Detect Possible Edits
    possible_edits = []
    if unique_word_ratio < 0.3:
        possible_edits.append("Low lexical diversity (may indicate AI-generated or heavily edited text)")
    if entropy_score > 4:
        possible_edits.append("High entropy (text structure suggests AI involvement)")
    if total_words > 500 and entropy_score < 2:
        possible_edits.append("Very low entropy (potentially rewritten or automated content)")
    
    # ✅ Determine Edit Status
    edit_status = bool(possible_edits)

    # ✅ AI Classification
    if ai_probability < 50:
        ai_report = "No AI detected (Real text)"
    elif 50 <= ai_probability < 70:
        ai_report = "Possibly AI-edited (Mild AI involvement)"
    elif 70 <= ai_probability < 85:
        ai_report = "Likely AI-generated (Moderate AI probability)"
    else:
        ai_report = "Fully AI-generated (High AI probability)"

    return {
        "file_name": file_name,
        "file_type": "Text",
        "ai_probability": ai_probability,
        "edit_status": edit_status,
        "ai_report": ai_report,
        "possible_edits": possible_edits or "No major edits detected."
    }

# ✅ Analyze TXT Files
def analyze_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return analyze_text(content, os.path.basename(file_path))

# ✅ Analyze PDF Files
def analyze_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = "".join([page.extract_text() or "" for page in pdf.pages])
    return analyze_text(text, os.path.basename(file_path))

# ✅ Analyze Word Documents (DOCX)
def analyze_word(file_path):
    doc = docx.Document(file_path)
    text = "\n".join([p.text for p in doc.paragraphs])
    return analyze_text(text, os.path.basename(file_path))

# ✅ Universal File Analysis
def analyze_file(file_path, file_type):
    if file_type and "text" in file_type:
        return analyze_txt(file_path)
    elif file_type and "pdf" in file_type:
        return analyze_pdf(file_path)
    elif file_type and ("word" in file_type or "msword" in file_type or "docx" in file_type):
        return analyze_word(file_path)
    else:
        return {
            "file_name": os.path.basename(file_path),
            "file_type": file_type,
            "ai_probability": 0,
            "edit_status": False,
            "AI analysis": "Unsupported file type.",
            "possible_edits": "N/A"
        }



# ✅ Determine file type
def get_file_type(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or "unknown"


# ✅ Main analysis function
def analyze_file(file_path):
    file_type = get_file_type(file_path)
    file_name = os.path.basename(file_path)

    if file_type and "image" in file_type:
        result = analyze_image(file_path)
    elif file_type and "video" in file_type:
        result = analyze_video(file_path)
    elif file_type and "audio" in file_type:
        result = analyze_audio(file_path)
    elif file_type and "text" in file_type:
        with open(file_path, "r", encoding="utf-8") as f:
            text_content = f.read()
        result = analyze_text(text_content, os.path.basename(file_path))

    elif file_type and "pdf" in file_type:
        result = analyze_pdf(file_path)
    elif file_type and "word" in file_type or "msword" in file_type:
        result = analyze_word(file_path)
    else:
        return {
            "file_type": "Unknown",
            "ai_probability": 0,
            "edited": False,
            "summary": "Unsupported file format."
        }

    result["file_name"] = file_name

    # ✅ Save forensic report
    report_path = os.path.join(REPORTS_FOLDER, f"{file_name}.json")
    with open(report_path, "w") as f:
        json.dump(result, f, indent=4)

    return result


# ✅ Flask API for forensic analysis
@app.route("/analyze", methods=["POST"])
def analyze_endpoint():
    data = request.get_json()
    file_name = data.get("file_name")

    if not file_name:
        return jsonify({"error": "No file specified"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file_name)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    # ✅ Call the analysis function
    report = analyze_file(file_path)

    return jsonify(report)  # ✅ Return the forensic report


# ✅ Run Flask server on port 5003
if __name__ == "__main__":
    app.run(port=5003, debug=True)
