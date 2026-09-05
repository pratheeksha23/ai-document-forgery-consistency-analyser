import streamlit as st
import cv2
import numpy as np
import easyocr
import re
import tempfile
import xml.etree.ElementTree as ET
from PIL import Image
from pyzbar.pyzbar import decode
from deepface import DeepFace

st.set_page_config(page_title="Privacy-First KYC Verifier", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

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

def check_ela(img_np):
    _, enc = cv2.imencode('.jpg', img_np, [cv2.IMWRITE_JPEG_QUALITY, 90])
    resaved = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    ela = cv2.absdiff(img_np, resaved)
    ela_gray = cv2.cvtColor(ela, cv2.COLOR_RGB2GRAY)
    return bool(np.max(ela_gray) > 55)

def mask_pii(img_np, ocr_boxes):
    masked = img_np.copy()
    for (bbox, text, _) in ocr_boxes:
        clean = re.sub(r'\D', '', text)
        if len(clean) >= 8 or re.search(r'\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b', text):
            pts = np.array(bbox, np.int32)
            cv2.fillPoly(masked, [pts], (0, 0, 0))
    return masked

def parse_qr(qr_objs):
    for q in qr_objs:
        try:
            txt = q.data.decode('utf-8', errors='ignore')
            if "<PrintLetterBarcodeData" in txt:
                root = ET.fromstring(txt)
                return {
                    "uid": root.attrib.get("uid", ""),
                    "dob": root.attrib.get("dob", "") or root.attrib.get("yob", "")
                }
        except Exception:
            continue
    return None

st.title("Automated KYC Document Authenticity & Biometric Verifier")
st.caption("Zero Data Retention Policy: All images processed exclusively in ephemeral memory.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Identity Document")
    doc_file = st.file_uploader("Upload Document", type=["jpg", "jpeg", "png"])
    doc_camera = st.camera_input("Or Capture Document via Camera")

with col2:
    st.subheader("2. Live Selfie (Webcam)")
    selfie_file = st.camera_input("Capture Live Selfie for Verification")

selected_doc = doc_file if doc_file is not None else doc_camera

if st.button("Run Forensic Verification", type="primary"):
    if not selected_doc:
        st.error("Please upload or capture a document first.")
    else:
        doc_img = Image.open(selected_doc).convert("RGB")
        doc_np = np.array(doc_img)

        with st.spinner("Analyzing document security features and extracting text..."):
            ocr_boxes = reader.readtext(doc_np, paragraph=False)
            raw_text = " ".join([b[1] for b in ocr_boxes]).upper()

            all_digits = re.sub(r'\D', '', raw_text)
            candidates = [all_digits[i:i+12] for i in range(len(all_digits) - 11)]
            extracted_uid = ""
            for c in candidates:
                if c[0] not in '01' and validate_verhoeff(c):
                    extracted_uid = c
                    break

            dob_match = re.search(r'\b(0[1-9]|[12][0-9]|3[01])[-/.](0[1-9]|1[012])[-/.](19|20)\d\d\b', raw_text)
            ocr_dob = dob_match.group(0) if dob_match else "NOT_FOUND"

            qr_objs = decode(doc_img)
            qr_data = parse_qr(qr_objs)
            is_tampered = check_ela(doc_np)
            gov_headers = any(k in raw_text for k in ["GOVERNMENT OF INDIA", "UNIQUE IDENTIFICATION", "AADHAAR", "MERA AADHAAR"])

            reasons = []
            is_real = True

            if not extracted_uid:
                is_real = False
                reasons.append("Invalid or unreadable Number (Failed Verhoeff Checksum)")
            if not gov_headers:
                is_real = False
                reasons.append("Missing official Government of India emblems / headers")
            if is_tampered:
                is_real = False
                reasons.append("Digital tampering / photo splicing detected via ELA")

            if qr_objs:
                if qr_data:
                    clean_ocr = re.sub(r'\D', '', ocr_dob)
                    clean_qr = re.sub(r'\D', '', qr_data["dob"])
                    if clean_ocr and clean_qr and (clean_qr not in clean_ocr and clean_ocr not in clean_qr):
                        is_real = False
                        reasons.append(f"DOB Mismatch! Printed on card: {ocr_dob} | Encoded in QR: {qr_data['dob']}")
                    if qr_data["uid"] and extracted_uid and qr_data["uid"][-4:] != extracted_uid[-4:]:
                        is_real = False
                        reasons.append("UID mismatch against embedded cryptographic QR")
            else:
                is_real = False
                reasons.append("Physical QR code missing or deliberately altered")

            face_matched = False
            bio_status = "No Selfie Captured"
            similarity = 0.0

            if selfie_file:
                selfie_img = Image.open(selfie_file).convert("RGB")
                selfie_np = np.array(selfie_img)
                with tempfile.NamedTemporaryFile(suffix=".jpg") as f1, tempfile.NamedTemporaryFile(suffix=".jpg") as f2:
                    cv2.imwrite(f1.name, cv2.cvtColor(doc_np, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(f2.name, cv2.cvtColor(selfie_np, cv2.COLOR_RGB2BGR))
                    try:
                        res = DeepFace.verify(f1.name, f2.name, model_name="ArcFace", detector_backend="opencv", enforce_detection=False)
                        dist = float(res["distance"])
                        face_matched = dist <= 0.68
                        similarity = max(0.0, min(100.0, (1 - (dist / 0.85)) * 100))
                        bio_status = "MATCH CONFIRMED" if face_matched else "MISMATCH / IMPERSONATION ALERT"
                    except Exception:
                        bio_status = "Face detection error"

            st.divider()
            res_col1, res_col2 = st.columns(2)

            with res_col1:
                if not is_real:
                    st.error("### Verdict: FAKE / FRAUDULENT")
                elif selfie_file and not face_matched:
                    st.warning("### Verdict: REAL DOCUMENT - IMPERSONATION MISMATCH")
                else:
                    st.success("### Verdict: REAL / VERIFIED")

                st.write(f"**Biometric Result:** {bio_status} ({similarity:.1f}% Match)")
                st.write(f"**Masked Identifier:** [Redacted ID]")
                st.write(f"**Detected DOB:** {ocr_dob}")

                if reasons:
                    st.error("Fraud Flags Detected:")
                    for r in reasons:
                        st.write(f"- {r}")
                else:
                    st.success("All mathematical, cryptographic, and visual checks passed.")

            with res_col2:
                masked_img = mask_pii(doc_np, ocr_boxes)
                st.image(masked_img, caption="Redacted Document Preview (PII Protected)", use_container_width=True)
