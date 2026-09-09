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

st.set_page_config(page_title="AI Document Forgery & Consistency Analyser", layout="wide")

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
    clean = re.sub(r'\D', '', num_str)
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
    return bool(np.mean(ela_gray) > 14.0 and np.max(ela_gray) > 85)

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

def extract_face(img_np):
    try:
        faces = DeepFace.extract_faces(img_np, detector_backend='opencv', enforce_detection=False)
        if faces and len(faces) > 0:
            facial_area = faces[0]['facial_area']
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            return img_np[y:y+h, x:x+w]
    except Exception:
        pass
    return None

def match_faces(img1_np, img2_np):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f1, tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f2:
        cv2.imwrite(f1.name, cv2.cvtColor(img1_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f2.name, cv2.cvtColor(img2_np, cv2.COLOR_RGB2BGR))
        p1, p2 = f1.name, f2.name
    try:
        res = DeepFace.verify(img1_path=p1, img2_path=p2, model_name='VGG-Face', detector_backend='opencv', enforce_detection=False)
        return res.get('verified', False), res.get('distance', 1.0)
    except Exception:
        return False, 1.0
    finally:
        if os.path.exists(p1):
            os.remove(p1)
        if os.path.exists(p2):
            os.remove(p2)

st.title("AI Document Forgery & Consistency Analyser")
st.caption("Ephemeral Execution | DeepFace & EasyOCR Engine")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Document Input")
    doc_mode = st.radio("Select source:", ["Upload File", "Capture via Camera"], horizontal=True, key="doc_mode")
    doc_input = st.file_uploader("Upload ID Card", type=["jpg", "jpeg", "png"]) if doc_mode == "Upload File" else st.camera_input("Capture ID")

with col2:
    st.subheader("2. Biometric Selfie (Optional)")
    selfie_mode = st.radio("Select source:", ["Upload File", "Capture via Camera"], horizontal=True, key="selfie_mode")
    selfie_input = st.file_uploader("Upload Selfie", type=["jpg", "jpeg", "png"]) if selfie_mode == "Upload File" else st.camera_input("Capture Selfie")

if st.button("Run Verification Analysis", type="primary"):
    if not doc_input:
        st.error("Please supply a document image.")
    else:
        doc_img = Image.open(doc_input).convert("RGB")
        max_size = 1280
        if max(doc_img.size) > max_size:
            doc_img.thumbnail((max_size, max_size))

        doc_np = np.array(doc_img)

        with st.spinner("Processing document forensic checks..."):
            ocr_boxes = reader.readtext(doc_np, paragraph=False)
            raw_text = " ".join([b[1] for b in ocr_boxes]).upper()
            clean_alnum = re.sub(r'[^A-Z0-9]', '', raw_text)

            masked_preview = mask_pii(doc_np, ocr_boxes)
            qr_objects = decode(doc_img)
            qr_data = parse_qr_data(qr_objects)
            is_tampered = check_tampering(doc_np)

            dob_match = re.search(r'\b(0[1-9]|[12][0-9]|3[01])[-/.](0[1-9]|1[012])[-/.](19|20)\d\d\b', raw_text)
            ocr_dob = dob_match.group(0) if dob_match else "NOT_FOUND"

            is_pan = any(k in raw_text for k in ["INCOME TAX", "PERMANENT ACCOUNT", "FATHER", "GOVT. OF INDIA", "SIGNATURE"]) or re.search(r'[A-Z]{5}[0-9]{4}[A-Z]{1}', clean_alnum)
            is_aadhaar = any(k in raw_text for k in ["AADHAAR", "UNIQUE IDENTIFICATION", "MERA AADHAAR"]) or re.search(r'\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b', raw_text)

            metadata_match = True
            reasons = []
            doc_type = "Unrecognized"

            if is_pan and not is_aadhaar:
                doc_type = "PAN Card"
                pan_hits = re.findall(r'[A-Z]{5}[0-9]{4}[A-Z]{1}', clean_alnum)
                if pan_hits:
                    pan_val = pan_hits[0]
                    if pan_val[3] not in "CPHFATBLJG":
                        metadata_match = False
                        reasons.append("Invalid 4th character status code on PAN")
                else:
                    metadata_match = False
                    reasons.append("Valid 10-character PAN number not detected")

                if not any(k in raw_text for k in ["INCOME TAX", "GOVT. OF INDIA", "PERMANENT ACCOUNT"]):
                    metadata_match = False
                    reasons.append("Missing Income Tax Department header")

            else:
                doc_type = "Aadhaar Card"
                all_digits = re.sub(r'\D', '', raw_text)
                candidates = [all_digits[i:i+12] for i in range(len(all_digits) - 11)]
                extracted_uid = ""
                for c in candidates:
                    if c[0] not in '01' and validate_verhoeff(c):
                        extracted_uid = c
                        break

                if not extracted_uid:
                    metadata_match = False
                    reasons.append("Aadhaar checksum validation failed (Verhoeff check)")

                if not any(k in raw_text for k in ["GOVERNMENT OF INDIA", "UNIQUE IDENTIFICATION", "AADHAAR", "MERA AADHAAR"]):
                    metadata_match = False
                    reasons.append("Missing official Government of India / UIDAI header")

                if qr_objects and qr_data and "dob" in qr_data:
                    clean_ocr_dob = re.sub(r'\D', '', ocr_dob)
                    clean_qr_dob = re.sub(r'\D', '', qr_data["dob"])
                    if clean_ocr_dob and clean_qr_dob and (clean_qr_dob not in clean_ocr_dob and clean_ocr_dob not in clean_qr_dob):
                        metadata_match = False
                        reasons.append(f"DOB mismatch: Card ({ocr_dob}) vs QR ({qr_data['dob']})")

            if is_tampered:
                metadata_match = False
                reasons.append("Digital manipulation/splicing detected (ELA)")

            face_crop = extract_face(doc_np)

            biometric_verified = None
            face_distance = None
            if selfie_input is not None and face_crop is not None:
                selfie_img = Image.open(selfie_input).convert("RGB")
                selfie_np = np.array(selfie_img)
                biometric_verified, face_distance = match_faces(face_crop, selfie_np)
                if not biometric_verified:
                    metadata_match = False
                    reasons.append("Biometric face match failed between document photo and selfie")

            final_verdict = "REAL" if metadata_match else "FAKE"
            trust_score = 95 if final_verdict == "REAL" else 20

            st.divider()
            c_left, c_right = st.columns([1.2, 1])

            with c_left:
                st.subheader("Verification Breakdown")
                st.write(f"**Document Type:** {doc_type}")
                if final_verdict == "REAL":
                    st.success(f"### Verdict: {final_verdict} (Trust Score: {trust_score}%)")
                else:
                    st.error(f"### Verdict: {final_verdict} (Trust Score: {trust_score}%)")

                audit_data = {
                    "Document Detected": doc_type,
                    "Verdict": final_verdict,
                    "Trust Score": f"{trust_score}%",
                    "Printed DOB": ocr_dob,
                    "Physical/Digital Tampering": "DETECTED" if is_tampered else "CLEAN",
                    "Issues Flagged": reasons if reasons else ["None - All integrity checks passed"]
                }
                if biometric_verified is not None:
                    audit_data["Biometric Match"] = "VERIFIED" if biometric_verified else "MISMATCH"
                    audit_data["Face Distance"] = f"{face_distance:.4f}"

                st.json(audit_data)

            with c_right:
                st.subheader("Visual Audits")
                st.image(masked_preview, caption="Redacted Document (PII Masked)", use_container_width=True)
                if face_crop is not None:
                    st.image(face_crop, caption="Detected ID Photo Crop", width=160)
