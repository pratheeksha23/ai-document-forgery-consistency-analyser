import streamlit as st
import cv2
import numpy as np
import easyocr
import re
import xml.etree.ElementTree as ET
from PIL import Image
from pyzbar.pyzbar import decode
from deepface import DeepFace
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
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00e0a0, #4ea1ff, #b96bff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.hero-sub {
    color: #9aa0b4;
    font-size: 0.9rem;
    margin-top: -4px;
}
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 16px 18px;
    backdrop-filter: blur(8px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    margin-bottom: 14px;
}
.pill-real {
    background: rgba(0,224,160,0.12);
    border: 1px solid #00e0a0;
    color: #00e0a0;
    padding: 8px 14px;
    border-radius: 10px;
    font-weight: 700;
    font-size: 1rem;
    text-align: center;
    margin-bottom: 8px;
}
.pill-fake {
    background: rgba(255,77,109,0.12);
    border: 1px solid #ff4d6d;
    color: #ff4d6d;
    padding: 8px 14px;
    border-radius: 10px;
    font-weight: 700;
    font-size: 1rem;
    text-align: center;
    margin-bottom: 8px;
}
.mono-log {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    background: #0d0f18;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
    padding: 12px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False, model_storage_directory="./models", download_enabled=True)

reader = load_ocr()

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
    is_tampered = bool(np.mean(ela_gray) > 16.0 and np.max(ela_gray) > 110)
    return is_tampered, ela_gray

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
    pan_regex = r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b'
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
        for m in re.finditer(p, text):
            dates.append(m.group(0))
    return dates

def detect_doc_type(raw_text, ocr_boxes, qr_data=None):
    if qr_data and isinstance(qr_data, dict) and qr_data.get("type") == "aadhaar_xml":
        return "Aadhaar Card"

    aadhaar_keywords = ["AADHAAR", "AADHAR", "UNIQUE IDENTIFICATION", "MERA AADHAAR", "UIDAI", "ENROLMENT", "HELP@UIDAI", "GOVERNMENT OF INDIA"]
    has_aadhaar_kw = any(k in raw_text for k in aadhaar_keywords)

    all_digits = re.sub(r'\D', '', raw_text)
    has_valid_aadhaar_checksum = False
    for i in range(max(len(all_digits) - 11, 0)):
        seq = all_digits[i:i + 12]
        if len(seq) == 12 and seq[0] not in '01' and validate_verhoeff(seq):
            has_valid_aadhaar_checksum = True
            break

    if has_aadhaar_kw or has_valid_aadhaar_checksum:
        return "Aadhaar Card"

    pan_keywords = ["INCOME TAX", "PERMANENT ACCOUNT", "INCOMETAX"]
    has_pan_kw = any(k in raw_text for k in pan_keywords)

    has_valid_pan_format = False
    for _, text, _ in ocr_boxes:
        words = re.findall(r'\b[A-Z0-9]{10}\b', text.upper().replace(" ", ""))
        for w in words:
            if re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$', w):
                if w[3] in "PCHFATBLJG":
                    has_valid_pan_format = True
                    break

    if has_pan_kw or has_valid_pan_format:
        return "PAN Card"

    return "Unrecognized"

def validate_pan(raw_text, ocr_boxes):
    reasons, valid = [], True
    pan_val = None
    for _, text, _ in ocr_boxes:
        words = re.findall(r'\b[A-Z0-9]{10}\b', text.upper().replace(" ", ""))
        for w in words:
            if re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$', w):
                pan_val = w
                break
        if pan_val:
            break

    if pan_val:
        if pan_val[3] not in "PCHFATBLJG":
            valid = False
            reasons.append(f"Invalid 4th character status code on PAN: '{pan_val[3]}'")
    else:
        valid = False
        reasons.append("Valid 10-character PAN number not detected")

    if not any(k in raw_text for k in ["INCOME", "TAX", "GOVT", "INDIA", "ACCOUNT"]):
        valid = False
        reasons.append("Missing official Income Tax Department header")
    return valid, reasons

def validate_aadhaar(raw_text, qr_data, ocr_dob):
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
        reasons.append("Aadhaar checksum integrity verification failed (Verhoeff check)")

    if not any(k in raw_text for k in ["GOVERNMENT OF INDIA", "UNIQUE IDENTIFICATION", "AADHAAR", "AADHAR", "MERA AADHAAR", "UIDAI"]):
        valid = False
        reasons.append("Missing official Government of India / UIDAI header")

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
                reasons.append(f"DOB Mismatch: Physical Card ({ocr_dob}) conflicts with QR record ({qr_dob})")

    return valid, reasons

def preprocess_face(face_rgb):
    if face_rgb is None or face_rgb.size == 0:
        return None
    face_resized = cv2.resize(face_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(face_resized, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def extract_face(img_np):
    for detector in ['retinaface', 'opencv']:
        try:
            faces = DeepFace.extract_faces(
                img_np, 
                detector_backend=detector, 
                align=True, 
                enforce_detection=False
            )
            if faces and len(faces) > 0:
                fa = faces[0]['facial_area']
                x, y, w, h = fa['x'], fa['y'], fa['w'], fa['h']
                H, W, _ = img_np.shape
                pad = int(min(w, h) * 0.15)
                x0, y0 = max(x - pad, 0), max(y - pad, 0)
                x1, y1 = min(x + w + pad, W), min(y + h + pad, H)
                cropped = img_np[y0:y1, x0:x1]
                if cropped.size > 0:
                    return preprocess_face(cropped)
        except Exception:
            continue
    return None

def analyze_biometrics(doc_face_rgb, selfie_rgb):
    norm_doc = preprocess_face(doc_face_rgb)
    norm_selfie = preprocess_face(selfie_rgb)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f1, \
         tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f2:
        cv2.imwrite(f1.name, cv2.cvtColor(norm_doc, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f2.name, cv2.cvtColor(norm_selfie, cv2.COLOR_RGB2BGR))
        p1, p2 = f1.name, f2.name

    arc_dist = 1.0
    vgg_dist = 1.0

    try:
        res_arc = DeepFace.verify(
            img1_path=p1,
            img2_path=p2,
            model_name="ArcFace",
            detector_backend="retinaface",
            distance_metric="cosine",
            align=True,
            enforce_detection=False
        )
        arc_dist = float(res_arc.get("distance", 1.0))
    except Exception:
        pass

    try:
        res_vgg = DeepFace.verify(
            img1_path=p1,
            img2_path=p2,
            model_name="VGG-Face",
            detector_backend="retinaface",
            distance_metric="cosine",
            align=True,
            enforce_detection=False
        )
        vgg_dist = float(res_vgg.get("distance", 1.0))
    except Exception:
        pass
    finally:
        for p in (p1, p2):
            if os.path.exists(p):
                os.remove(p)

    verified = bool((arc_dist <= 0.78) or (vgg_dist <= 0.48))

    return {
        "verified": verified,
        "arcface_distance": arc_dist,
        "vgg_distance": vgg_dist
    }

st.markdown('<div class="hero-title">🛡️ VerifAI — Identity & Document Forensics</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">RetinaFace Aligned Biometrics & Checksum Analysis</div>', unsafe_allow_html=True)
st.write("")

with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")
    doc_selection = st.selectbox("Document Detection Mode", ["Auto-Detect", "Aadhaar Card", "PAN Card"])

col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("1. ID Document")
    doc_mode = st.radio("Source", ["Upload File", "Capture via Camera"], horizontal=True, key="doc_mode")
    doc_input = st.file_uploader("Upload ID Card", type=["jpg", "jpeg", "png"]) if doc_mode == "Upload File" else st.camera_input("Capture ID")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("2. Biometric Selfie (Optional)")
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

        with st.spinner("Executing RetinaFace alignment and document verification..."):
            ocr_boxes = reader.readtext(doc_np, paragraph=False)
            raw_text = " ".join([b[1] for b in ocr_boxes]).upper()

            masked_preview = mask_pii(doc_np, ocr_boxes)
            qr_objects = decode(doc_img)
            qr_data = parse_qr_data(qr_objects)

            is_tampered, _ = check_tampering(doc_np)

            dates_found = extract_dates(raw_text)
            ocr_dob = dates_found[0] if dates_found else "NOT_FOUND"

            if doc_selection != "Auto-Detect":
                doc_type = doc_selection
            else:
                doc_type = detect_doc_type(raw_text, ocr_boxes, qr_data)

            doc_reasons = []
            doc_valid = True

            if doc_type == "PAN Card":
                doc_valid, doc_reasons = validate_pan(raw_text, ocr_boxes)
            elif doc_type == "Aadhaar Card":
                doc_valid, doc_reasons = validate_aadhaar(raw_text, qr_data, ocr_dob)
            else:
                doc_valid = False
                doc_reasons.append("Document not recognized as a valid government ID")

            if is_tampered:
                doc_valid = False
                doc_reasons.append("Digital manipulation/splicing detected via Error Level Analysis")

            face_crop = extract_face(doc_np)
            bio_status = "NOT REQUESTED"
            bio_result = None

            if selfie_input is not None:
                if face_crop is None:
                    bio_status = "FAILED"
                    doc_reasons.append("RetinaFace was unable to extract a face crop from the ID")
                else:
                    selfie_img = Image.open(selfie_input).convert("RGB")
                    selfie_np = np.array(selfie_img)
                    bio_result = analyze_biometrics(face_crop, selfie_np)
                    bio_status = "MATCHED" if bio_result["verified"] else "MISMATCH"

        st.divider()
        c_left, c_right = st.columns([1.2, 1])

        with c_left:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Verification Verdicts")

            st.write(f"**Identified Document:** {doc_type}")
            if doc_valid:
                st.markdown('<div class="pill-real">✅ DOCUMENT: REAL / VALID</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="pill-fake">🚫 DOCUMENT: FAKE / TAMPERED</div>', unsafe_allow_html=True)

            if selfie_input is not None:
                if bio_status == "MATCHED":
                    st.markdown('<div class="pill-real">✅ BIOMETRIC: FACE MATCH CONFIRMED</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="pill-fake">⚠️ BIOMETRIC: FACE MISMATCH</div>', unsafe_allow_html=True)

            audit_log = {
                "Document Type": doc_type,
                "Document Status": "VALID (REAL)" if doc_valid else "INVALID / TAMPERED",
                "Tampering (ELA)": "DETECTED" if is_tampered else "CLEAN",
                "Document Issues": doc_reasons if doc_reasons else ["None - All document checks passed"],
                "Biometric Status": bio_status
            }
            if bio_result:
                audit_log["ArcFace Cosine Distance"] = f"{bio_result['arcface_distance']:.4f} (Threshold: 0.78)"
                audit_log["VGG-Face Distance"] = f"{bio_result['vgg_distance']:.4f} (Threshold: 0.48)"

            st.markdown('<div class="mono-log">', unsafe_allow_html=True)
            st.json(audit_log)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c_right:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Visual Audits")
            st.image(masked_preview, caption="Redacted Document (PII Masked)", use_container_width=True)
            if face_crop is not None:
                st.image(face_crop, caption="RetinaFace Cropped ID Portrait", width=150)
            st.markdown('</div>', unsafe_allow_html=True)
