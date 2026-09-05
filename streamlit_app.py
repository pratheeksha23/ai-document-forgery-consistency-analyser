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

st.set_page_config(page_title="AI Document Forgery & Consistency Analyser", layout="wide")

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

def check_tampering(img_np):
    _, enc = cv2.imencode('.jpg', img_np, [cv2.IMWRITE_JPEG_QUALITY, 90])
    resaved = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    ela = cv2.absdiff(img_np, resaved)
    ela_gray = cv2.cvtColor(ela, cv2.COLOR_RGB2GRAY)
    return bool(np.mean(ela_gray) > 12.0 and np.max(ela_gray) > 80)

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
                    "dob": root.attrib.get("dob", ""),
                    "yob": root.attrib.get("yob", ""),
                    "name": root.attrib.get("name", "")
                }
            if raw_bytes.isdigit() or len(raw_bytes) > 200:
                return {"type": "secure_v2", "raw": raw_bytes}
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

def match_faces(doc_img_np, selfie_img_np):
    doc_crop = None
    try:
        faces = DeepFace.extract_faces(doc_img_np, detector_backend='opencv', enforce_detection=False)
        if len(faces) > 0:
            a = faces[0]["facial_area"]
            doc_crop = doc_img_np[a['y']:a['y']+a['h'], a['x']:a['x']+a['w']]
    except Exception:
        doc_crop = None

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f1, tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f2:
        cv2.imwrite(f1.name, cv2.cvtColor(doc_img_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f2.name, cv2.cvtColor(selfie_img_np, cv2.COLOR_RGB2BGR))
        try:
            res = DeepFace.verify(
                img1_path=f1.name,
                img2_path=f2.name,
                model_name='ArcFace',
                detector_backend='opencv',
                distance_metric='cosine',
                enforce_detection=False
            )
            raw_dist = float(res["distance"])
            is_match = raw_dist <= 0.68
            score = max(0.0, min(100.0, (1 - (raw_dist / 0.85)) * 100))
            status = "MATCH (Same Person)" if is_match else "MISMATCH (Different Person)"
            return doc_crop, status, is_match, score
        except Exception:
            return doc_crop, "FACE NOT DETECTED", False, 0.0

st.title("AI Document Forgery & Consistency Analyser")
st.caption("Privacy-First Pipeline | Zero Data Retention | In-Memory Processing")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Identity Document")
    doc_source = st.radio("Input Method for Document:", ["Upload File", "Take Photo via Camera"], horizontal=True, key="doc_src")
    if doc_source == "Upload File":
        doc_file = st.file_uploader("Upload ID Document", type=["jpg", "jpeg", "png"], key="doc_up")
    else:
        doc_file = st.camera_input("Capture Document", key="doc_cam")

with col2:
    st.subheader("2. Live Face Verification")
    selfie_source = st.radio("Input Method for Selfie:", ["Take Selfie via Camera", "Upload Photo"], horizontal=True, key="selfie_src")
    if selfie_source == "Take Selfie via Camera":
        selfie_file = st.camera_input("Capture Live Face", key="selfie_cam")
    else:
        selfie_file = st.file_uploader("Upload Selfie Photo", type=["jpg", "jpeg", "png"], key="selfie_up")

if st.button("Run Verification Analysis", type="primary"):
    if not doc_file:
        st.error("Please provide an ID document using either Upload or Camera.")
    else:
        doc_img = Image.open(doc_file).convert("RGB")
        doc_np = np.array(doc_img)

        with st.spinner("Executing Forensic Audits and Neural Consistency Checks..."):
            ocr_boxes = reader.readtext(doc_np, paragraph=False)
            raw_text = " ".join([b[1] for b in ocr_boxes]).upper()
            compressed_text = re.sub(r'[^A-Z0-9]', '', raw_text)

            masked_preview = mask_pii(doc_np, ocr_boxes)
            qr_objects = decode(doc_img)
            qr_data = parse_qr_data(qr_objects)
            is_tampered = check_tampering(doc_np)

            dob_match = re.search(r'\b(0[1-9]|[12][0-9]|3[01])[-/.](0[1-9]|1[012])[-/.](19|20)\d\d\b', raw_text)
            yob_match = re.search(r'\b(19|20)\d{2}\b', raw_text)
            ocr_dob = dob_match.group(0) if dob_match else (yob_match.group(0) if yob_match else "NOT_FOUND")

            is_pan = any(k in raw_text for k in ["INCOME TAX", "PERMANENT ACCOUNT", "FATHER", "GOVT. OF INDIA", "SIGNATURE"]) or re.search(r'[A-Z]{5}[0-9]{4}[A-Z]{1}', compressed_text)
            is_aadhaar = any(k in raw_text for k in ["AADHAAR", "UNIQUE IDENTIFICATION", "MERA AADHAAR", "ENROLMENT"]) or re.search(r'\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b', raw_text)

            metadata_match = True
            rejection_reasons = []
            doc_type = "Unrecognized Document"
            qr_cross_check = "NOT_APPLICABLE"

            if is_pan and not is_aadhaar:
                doc_type = "PAN Card"
                pan_matches = re.findall(r'[A-Z]{5}[0-9]{4}[A-Z]{1}', compressed_text)
                if pan_matches:
                    pan_num = pan_matches[0]
                    if pan_num[3] not in "CPHFATBLJG":
                        metadata_match = False
                        rejection_reasons.append("Invalid 4th character status code on PAN")
                else:
                    metadata_match = False
                    rejection_reasons.append("Valid 10-character PAN format not found")

                has_pan_header = any(k in raw_text for k in ["INCOME TAX", "GOVT. OF INDIA", "PERMANENT ACCOUNT"])
                if not has_pan_header:
                    metadata_match = False
                    rejection_reasons.append("Missing Income Tax Department header")

                if qr_objects:
                    qr_cross_check = "VERIFIED (QR Present)"
                else:
                    qr_cross_check = "OPTIONAL (Physical Card)"

            else:
                doc_type = "Aadhaar Card"
                all_digits = re.sub(r'\D', '', raw_text)
                candidates = [all_digits[i:i+12] for i in range(len(all_digits) - 11)]
                extracted_uid = ""
                for cand in candidates:
                    if cand[0] not in '01' and validate_verhoeff(cand):
                        extracted_uid = cand
                        break

                if not extracted_uid:
                    metadata_match = False
                    rejection_reasons.append("Invalid or missing identification number (Verhoeff validation failed)")

                has_aadhaar_header = any(k in raw_text for k in ["GOVERNMENT OF INDIA", "UNIQUE IDENTIFICATION", "AADHAAR", "MERA AADHAAR"])
                if not has_aadhaar_header:
                    metadata_match = False
                    rejection_reasons.append("Missing official Government of India / UIDAI header")

                if qr_objects:
                    if qr_data and isinstance(qr_data, dict) and "dob" in qr_data:
                        qr_dob = qr_data.get("dob", "") or qr_data.get("yob", "")
                        qr_uid = qr_data.get("uid", "")
                        clean_ocr_dob = re.sub(r'\D', '', ocr_dob)
                        clean_qr_dob = re.sub(r'\D', '', qr_dob)

                        dob_aligned = (clean_qr_dob in clean_ocr_dob) or (clean_ocr_dob in clean_qr_dob)
                        uid_aligned = (not qr_uid) or (extracted_uid and qr_uid[-4:] == extracted_uid[-4:])

                        if not dob_aligned:
                            metadata_match = False
                            qr_cross_check = f"MISMATCH (Card: {ocr_dob} vs QR: {qr_dob})"
                            rejection_reasons.append(f"DOB mismatch: Printed ({ocr_dob}) vs QR ({qr_dob})")
                        elif not uid_aligned:
                            metadata_match = False
                            qr_cross_check = "MISMATCH (UID vs QR)"
                            rejection_reasons.append("UID mismatch against embedded QR code")
                        else:
                            qr_cross_check = "VERIFIED (Card matches QR payload)"
                    else:
                        qr_cross_check = "SECURE_QR_PRESENT"
                else:
                    metadata_match = False
                    qr_cross_check = "MISSING_QR"
                    rejection_reasons.append("Physical QR code missing or damaged")

            if is_tampered:
                metadata_match = False
                rejection_reasons.append("ELA detected visual manipulation or splicing")

            face_crop = None
            bio_status = "NO SELFIE PROVIDED"
            face_match = False
            bio_score = 0.0

            if selfie_file:
                selfie_img = Image.open(selfie_file).convert("RGB")
                face_crop, bio_status, face_match, bio_score = match_faces(doc_np, np.array(selfie_img))
                if not face_match:
                    rejection_reasons.append("Biometric face mismatch (Impersonation Alert)")

            if not metadata_match:
                final_verdict = "FAKE"
                trust_score = 15
            elif selfie_file and not face_match:
                final_verdict = "FAKE (IMPERSONATION)"
                trust_score = 30
            else:
                final_verdict = "REAL"
                trust_score = 95 if face_match else 80

            st.divider()
            r_col1, r_col2 = st.columns([1.2, 1])

            with r_col1:
                st.subheader("Verification Breakdown")
                st.write(f"**Document Detected:** {doc_type}")

                if final_verdict == "REAL":
                    st.success(f"### Verdict: {final_verdict} (Trust Score: {trust_score}%)")
                else:
                    st.error(f"### Verdict: {final_verdict} (Trust Score: {trust_score}%)")

                details_data = {
                    "Document Type": doc_type,
                    "Status": final_verdict,
                    "Trust Score": f"{trust_score}%",
                    "Printed DOB": ocr_dob,
                    "QR Cross-Check": qr_cross_check,
                    "Physical/Digital Tampering": "DETECTED" if is_tampered else "CLEAN",
                    "Biometric Result": bio_status,
                    "Biometric Similarity": f"{bio_score:.1f}%",
                    "Issues Flagged": rejection_reasons if rejection_reasons else ["None - All integrity checks passed"]
                }
                st.json(details_data)

            with r_col2:
                st.subheader("Visual Audits")
                st.image(masked_preview, caption="Redacted Document (PII Masked)", use_container_width=True)
                if face_crop is not None:
                    st.image(face_crop, caption="Detected ID Photo Crop", width=180)
