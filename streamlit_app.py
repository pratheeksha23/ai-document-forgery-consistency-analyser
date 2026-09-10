import streamlit as st
import cv2
import numpy as np
import easyocr
import re
import xml.etree.ElementTree as ET
from PIL import Image
from pyzbar.pyzbar import decode
from deepface import DeepFace
import mediapipe as mp
import tempfile
import os

st.set_page_config(page_title="VerifAI | Identity Forensics", page_icon="🛡️", layout="wide")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp {
    background: radial-gradient(circle at 15% 10%, #16192a 0%, #0b0d16 55%, #05060a 100%);
    color: #e6e8f0;
}
#MainMenu, footer, header {visibility: hidden;}
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00e0a0, #4ea1ff, #b96bff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.hero-sub {
    color: #9aa0b4;
    font-size: 0.95rem;
    margin-top: -4px;
}
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 18px 20px;
    backdrop-filter: blur(8px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    margin-bottom: 16px;
}
.pill-real {
    background: rgba(0,224,160,0.12);
    border: 1px solid #00e0a0;
    color: #00e0a0;
    padding: 10px 18px;
    border-radius: 12px;
    font-weight: 700;
    font-size: 1.1rem;
    text-align: center;
}
.pill-fake {
    background: rgba(255,77,109,0.12);
    border: 1px solid #ff4d6d;
    color: #ff4d6d;
    padding: 10px 18px;
    border-radius: 12px;
    font-weight: 700;
    font-size: 1.1rem;
    text-align: center;
}
.mono-log {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    background: #0d0f18;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 12px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False, model_storage_directory="./models", download_enabled=True)

reader = load_ocr()
mp_face_mesh = mp.solutions.face_mesh

d_table = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
]
p_table = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
]

def validate_verhoeff(num_str):
    clean = re.sub(r'\D', '', str(num_str))
    if len(clean) != 12:
        return False
    c = 0
    for i, item in enumerate(reversed(clean)):
        c = d_table[c][p_table[i % 8][int(item)]]
    return c == 0

def check_tampering(img_np):
    _, enc = cv2.imencode('.jpg', img_np, [cv2.IMWRITE_JPEG_QUALITY, 90])
    resaved = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    ela = cv2.absdiff(img_np, resaved)
    ela_gray = cv2.cvtColor(ela, cv2.COLOR_RGB2GRAY)
    is_tampered = bool(np.mean(ela_gray) > 13.0 and np.max(ela_gray) > 80)
    return is_tampered, ela_gray

def check_localized_tampering(img_np, ocr_boxes):
    _, enc = cv2.imencode('.jpg', img_np, [cv2.IMWRITE_JPEG_QUALITY, 90])
    resaved = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    ela_full = cv2.cvtColor(cv2.absdiff(img_np, resaved), cv2.COLOR_RGB2GRAY)

    region_stats = []
    for bbox, text, conf in ocr_boxes:
        pts = np.array(bbox, np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        if w < 10 or h < 10:
            continue
        crop = ela_full[y:y + h, x:x + w]
        if crop.size == 0:
            continue
        region_stats.append((text, float(np.mean(crop))))

    flagged = []
    if len(region_stats) >= 3:
        means = [m for _, m in region_stats]
        median_m = float(np.median(means))
        mad = float(np.median([abs(m - median_m) for m in means])) + 1e-6
        for text, m in region_stats:
            z_score = abs(m - median_m) / (mad * 1.4826)
            if z_score > 3.2 and m > (median_m + 5.0):
                flagged.append((text, m, z_score))
    return flagged

def parse_qr_data(qr_objs):
    if not qr_objs:
        return None
    for q in qr_objs:
        try:
            raw_bytes = q.data
            decoded_text = raw_bytes.decode('utf-8', errors='ignore')
            if "<PrintLetterBarcodeData" in decoded_text or "xml" in decoded_text:
                root = ET.fromstring(decoded_text)
                return {
                    "type": "aadhaar_xml",
                    "uid": root.attrib.get("uid", ""),
                    "dob": root.attrib.get("dob", "") or root.attrib.get("yob", ""),
                    "name": root.attrib.get("name", "")
                }
            if len(decoded_text) > 0:
                return {"type": "standard_qr", "raw": decoded_text}
        except Exception:
            continue
    return None

def mask_pii(img_np, boxes):
    masked = img_np.copy()
    pan_regex = r'[A-Z]{5}[0-9]{4}[A-Z]{1}'
    for (bbox, text, _) in boxes:
        clean_digits = re.sub(r'\D', '', text)
        clean_text = text.replace(" ", "").upper()
        if len(clean_digits) >= 8 or re.search(r'\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b', text) or re.search(pan_regex, clean_text):
            pts = np.array(bbox, np.int32)
            cv2.fillPoly(masked, [pts], (0, 0, 0))
    return masked

def extract_dates(text):
    dates = []
    patterns = [
        r'\b(0[1-9]|[12][0-9]|3[01])[-/.](0[1-9]|1[012])[-/.](19|20)\d\d\b',
        r'\b(19|20)\d\d[-/.](0[1-9]|1[012])[-/.](0[1-9]|[12][0-9]|3[01])\b',
        r'\b(19|20)\d\d\b'
    ]
    for p in patterns:
        matches = re.findall(p, text)
        if matches:
            for m in re.finditer(p, text):
                dates.append(m.group(0))
    return dates

def detect_doc_type(raw_text, clean_alnum, qr_data=None):
    pan_matches = re.findall(r'[A-Z]{5}[0-9]{4}[A-Z]{1}', clean_alnum)
    pan_keywords = ["INCOME TAX", "PERMANENT ACCOUNT", "INCOMETAX", "FATHER'S NAME", "GOVT. OF INDIA"]
    has_pan_kw = any(k in raw_text for k in pan_keywords)

    if pan_matches or (has_pan_kw and not any(k in raw_text for k in ["AADHAAR", "UIDAI"])):
        return "PAN Card"

    if qr_data and isinstance(qr_data, dict) and qr_data.get("type") == "aadhaar_xml":
        return "Aadhaar Card"

    aadhaar_keywords = ["AADHAAR", "UNIQUE IDENTIFICATION", "MERA AADHAAR", "UIDAI", "GOVERNMENT OF INDIA"]
    has_aadhaar_kw = any(k in raw_text for k in aadhaar_keywords)

    all_digits = re.findall(r'\b\d{4}\s?\d{4}\s?\d{4}\b', raw_text)
    if not all_digits:
        digits_only = re.sub(r'\D', '', raw_text)
        for i in range(max(len(digits_only) - 11, 0)):
            seq = digits_only[i:i + 12]
            if len(seq) == 12 and seq[0] not in '01' and validate_verhoeff(seq):
                all_digits.append(seq)
                break

    if has_aadhaar_kw or all_digits:
        return "Aadhaar Card"

    if any(k in raw_text for k in ["DRIVING LICENCE", "DRIVING LICENSE", "UNION OF INDIA DRIVING"]):
        return "Driving Licence"
    if any(k in raw_text for k in ["ELECTION COMMISSION", "ELECTORAL", "VOTER"]):
        return "Voter ID"

    return "Unrecognized"

def validate_pan(raw_text, clean_alnum):
    reasons, valid = [], True
    pan_hits = re.findall(r'[A-Z]{5}[0-9]{4}[A-Z]{1}', clean_alnum)
    if pan_hits:
        pan_val = pan_hits[0]
        if pan_val[3] not in "PCHFATBLJG":
            valid = False
            reasons.append(f"Invalid 4th character holder-category code: '{pan_val[3]}'")
    else:
        valid = False
        reasons.append("Valid 10-character alphanumeric PAN format not detected")

    if not any(k in raw_text for k in ["INCOME", "TAX", "GOVT", "INDIA", "ACCOUNT"]):
        valid = False
        reasons.append("Missing Income Tax Department validation headers")
    return valid, reasons

def validate_aadhaar(raw_text, qr_data, ocr_dob, ocr_boxes):
    reasons, valid = [], True
    all_digits = re.sub(r'\D', '', raw_text)
    extracted_uid = ""

    for i in range(max(len(all_digits) - 11, 0)):
        c = all_digits[i:i + 12]
        if len(c) == 12 and c[0] not in '01' and validate_verhoeff(c):
            extracted_uid = c
            break

    if not extracted_uid:
        valid = False
        reasons.append("Aadhaar checksum integrity verification failed (Verhoeff validation)")

    if not any(k in raw_text for k in ["GOVERNMENT OF INDIA", "UNIQUE IDENTIFICATION", "AADHAAR", "MERA AADHAAR", "UIDAI"]):
        valid = False
        reasons.append("Missing UIDAI / Government of India authentication header")

    if qr_data and isinstance(qr_data, dict):
        qr_dob = str(qr_data.get("dob", "")).strip()
        if qr_dob:
            clean_qr = re.sub(r'\D', '', qr_dob)
            clean_ocr = re.sub(r'\D', '', str(ocr_dob)) if ocr_dob != "NOT_FOUND" else ""
            
            dob_matches = False
            if clean_ocr and clean_qr:
                if clean_qr in clean_ocr or clean_ocr in clean_qr:
                    dob_matches = True
                elif len(clean_qr) >= 4 and len(clean_ocr) >= 4:
                    if clean_qr[-4:] == clean_ocr[-4:]:
                        dob_matches = True

            if not dob_matches:
                valid = False
                reasons.append(f"DOB Tampering Flagged: Physical Card ({ocr_dob}) conflicts with Secure QR Record ({qr_dob})")

    return valid, reasons, extracted_uid

def extract_face(img_np):
    try:
        faces = DeepFace.extract_faces(img_np, detector_backend='opencv', enforce_detection=False)
        if faces and len(faces) > 0:
            fa = faces[0]['facial_area']
            x, y, w, h = fa['x'], fa['y'], fa['w'], fa['h']
            H, W, _ = img_np.shape
            pad = int(min(w, h) * 0.1)
            x0, y0 = max(x - pad, 0), max(y - pad, 0)
            x1, y1 = min(x + w + pad, W), min(y + h + pad, H)
            face = img_np[y0:y1, x0:x1]
            if face.size > 0:
                return face
    except Exception:
        pass
    return None

def analyze_biometrics(doc_face_rgb, selfie_rgb, model_name="Facenet512"):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f1, \
         tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f2:
        cv2.imwrite(f1.name, cv2.cvtColor(doc_face_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f2.name, cv2.cvtColor(selfie_rgb, cv2.COLOR_RGB2BGR))
        p1, p2 = f1.name, f2.name

    deepface_verified, distance, threshold = False, 1.0, 0.40
    try:
        res = DeepFace.verify(
            img1_path=p1,
            img2_path=p2,
            model_name=model_name,
            detector_backend='opencv',
            distance_metric='cosine',
            enforce_detection=False
        )
        deepface_verified = res.get("verified", False)
        distance = float(res.get("distance", 1.0))
        threshold = float(res.get("threshold", 0.40))
    except Exception:
        pass
    finally:
        for p in (p1, p2):
            if os.path.exists(p):
                os.remove(p)

    return {
        "deepface_verified": deepface_verified,
        "distance": distance,
        "threshold": threshold,
        "confidence": "MATCHED - HIGH CONFIDENCE" if deepface_verified else "MISMATCH"
    }

def compute_trust_score(metadata_valid, is_tampered, biometric_result):
    if not metadata_valid:
        return 15
    if is_tampered:
        return 25
    score = 50
    if biometric_result is not None:
        if biometric_result["deepface_verified"]:
            score += 50
        else:
            return 20
    else:
        score += 40
    return min(score, 100)

def render_trust_gauge(score, verdict):
    color = "#00e0a0" if verdict == "REAL" else "#ff4d6d"
    html = f"""
    <div style="display:flex;flex-direction:column;align-items:center;margin-top:8px;">
      <div style="width:160px;height:160px;border-radius:50%;
        background:conic-gradient({color} {score * 3.6}deg, #1e2230 0deg);
        display:flex;align-items:center;justify-content:center;
        box-shadow:0 0 25px {color}55;">
        <div style="width:124px;height:124px;border-radius:50%;background:#0d0f18;
             display:flex;flex-direction:column;align-items:center;justify-content:center;">
          <span style="font-size:2.0rem;font-weight:700;color:{color};">{score}%</span>
          <span style="font-size:0.65rem;color:#9aa0b4;letter-spacing:1px;">TRUST SCORE</span>
        </div>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

st.markdown('<div class="hero-title">🛡️ VerifAI — Identity Forensics Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Automated OCR cross-verification · ELA tamper detection · Algorithmic checksums · Biometric face match</div>', unsafe_allow_html=True)
st.write("")

col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Step 1: ID Document")
    doc_mode = st.radio("Source", ["Upload File", "Capture via Camera"], horizontal=True, key="doc_mode")
    doc_input = st.file_uploader("Upload ID Card", type=["jpg", "jpeg", "png"]) if doc_mode == "Upload File" else st.camera_input("Capture ID")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Step 2: Biometric Selfie (Optional)")
    selfie_mode = st.radio("Source", ["Upload File", "Capture via Camera"], horizontal=True, key="selfie_mode")
    selfie_input = st.file_uploader("Upload Selfie", type=["jpg", "jpeg", "png"]) if selfie_mode == "Upload File" else st.camera_input("Capture Selfie")
    st.markdown('</div>', unsafe_allow_html=True)

run = st.button("🔍 Run Forensic Verification Analysis", type="primary", use_container_width=True)

if run:
    if not doc_input:
        st.error("Please supply a document image.")
    else:
        doc_img = Image.open(doc_input).convert("RGB")
        max_size = 1280
        if max(doc_img.size) > max_size:
            doc_img.thumbnail((max_size, max_size))
        doc_np = np.array(doc_img)

        with st.spinner("Executing document forensics and biometric inference..."):
            ocr_boxes = reader.readtext(doc_np, paragraph=False)
            raw_text = " ".join([b[1] for b in ocr_boxes]).upper()
            clean_alnum = re.sub(r'[^A-Z0-9]', '', raw_text)

            masked_preview = mask_pii(doc_np, ocr_boxes)
            qr_objects = decode(doc_img)
            qr_data = parse_qr_data(qr_objects)

            is_tampered, ela_map = check_tampering(doc_np)
            localized_flags = check_localized_tampering(doc_np, ocr_boxes)
            if localized_flags:
                is_tampered = True

            dates_found = extract_dates(raw_text)
            ocr_dob = dates_found[0] if dates_found else "NOT_FOUND"

            doc_type = detect_doc_type(raw_text, clean_alnum, qr_data)
            reasons = []
            metadata_valid = True

            if doc_type == "PAN Card":
                metadata_valid, reasons = validate_pan(raw_text, clean_alnum)
            elif doc_type == "Aadhaar Card":
                metadata_valid, reasons, _ = validate_aadhaar(raw_text, qr_data, ocr_dob, ocr_boxes)
            else:
                metadata_valid = False
                reasons.append("Document could not be recognized as a valid PAN or Aadhaar card")

            if is_tampered:
                metadata_valid = False
                reasons.append("Digital manipulation/splicing detected via Error Level Analysis")

            face_crop = extract_face(doc_np)
            biometric_result = None

            if selfie_input is not None:
                if face_crop is None:
                    reasons.append("Facial portrait could not be cropped from ID for biometric match")
                    metadata_valid = False
                else:
                    selfie_img = Image.open(selfie_input).convert("RGB")
                    selfie_np = np.array(selfie_img)
                    biometric_result = analyze_biometrics(face_crop, selfie_np, model_name="Facenet512")
                    if not biometric_result["deepface_verified"]:
                        metadata_valid = False
                        reasons.append("Biometric mismatch: ID portrait does not match selfie")

            final_verdict = "REAL" if metadata_valid else "FAKE"
            trust_score = compute_trust_score(metadata_valid, is_tampered, biometric_result)

        st.divider()
        c_left, c_right = st.columns([1.2, 1])

        with c_left:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Forensic Audit Results")
            st.write(f"**Identified Document:** {doc_type}")
            if final_verdict == "REAL":
                st.markdown(f'<div class="pill-real">✅ VERDICT: REAL — {trust_score}% Trust Score</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="pill-fake">🚫 VERDICT: FAKE — {trust_score}% Trust Score</div>', unsafe_allow_html=True)

            st.write("")
            audit_data = {
                "Document Type": doc_type,
                "Authentication Verdict": final_verdict,
                "Trust Score": f"{trust_score}%",
                "Extracted OCR DOB": ocr_dob,
                "Decoded QR DOB": qr_data.get("dob", "No QR decoded") if isinstance(qr_data, dict) else "None",
                "Tampering (ELA)": "DETECTED" if is_tampered else "CLEAN",
                "Integrity Violations": reasons if reasons else ["None - All forensic tests passed"]
            }
            if biometric_result is not None:
                audit_data["Biometric Status"] = biometric_result["confidence"]
                audit_data["Face Match Distance"] = f"{biometric_result['distance']:.4f} (Threshold: {biometric_result['threshold']:.4f})"

            st.markdown('<div class="mono-log">', unsafe_allow_html=True)
            st.json(audit_data)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c_right:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Trust Assessment")
            render_trust_gauge(trust_score, final_verdict)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Visual Audits")
            st.image(masked_preview, caption="Redacted Document (Sensitive PII Masked)", use_container_width=True)
            if face_crop is not None:
                st.image(face_crop, caption="Cropped ID Portrait", width=160)
            st.markdown('</div>', unsafe_allow_html=True)
