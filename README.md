# ai-document-forgery-consistency-analyser
https://github.com/user-attachments/assets/10e58ea8-f25d-4eed-bab6-c218fdfc4057
https://github.com/user-attachments/assets/40bc82cd-6cfa-4aaa-81ee-7da1446d5dc9

An identity verification and document forensics web application built with Streamlit, EasyOCR, OpenCV, and pyzbar. VerifAI cross-references visual card text against digitally encoded security elements, detects digital image manipulation, and executes lightweight biometric facial matching within resource-constrained environments.

---

## Features

- **Multi-Document Classification Engine:** Weighted, score-based document detection for Indian national identity documents:
  - PAN Card
  - Aadhaar Card
  - Driving Licence
  - Voter ID (EPIC)
- **Cryptographic & Data Cross-Verification:**
  - Automated QR decoding via `pyzbar` (XML and raw payload parsing).
  - Cross-checks visual OCR text (e.g., printed DOB/Name) against digitally signed QR records to detect text alterations.
  - Verhoeff checksum validation for Aadhaar numbers.
  - Alphanumeric structure and category-code validation for PAN identifiers.
- **Forgery & Tamper Detection:**
  - Full-frame Error Level Analysis (ELA) to detect resaving and compression anomalies.
  - Localized noise analysis around extracted text regions to flag font splicing and manual edits.
- **Lightweight Biometric Face Matching:**
  - Histogram equalization (CLAHE) to handle scanner glare and contrast disparities.
  - Dual-metric template matching and histogram correlation (`cv2.HISTCMP_CORREL` + `cv2.TM_CCOEFF_NORMED`) optimized for CPU-only, low-memory environments (avoids OOM crashes on free hosting tiers).
- **Automated PII Redaction:**
  - Automatically masks sensitive account numbers and personal data before displaying audit previews.

---

## System Architecture
ID Document / Live Capture
│
├──► Preprocessing & Contrast Balancing (CLAHE)
│           │
│           ├──► EasyOCR Engine ──► Regex & Verhoeff Validations
│           ├──► pyzbar ──────────► Cryptographic QR Extraction
│           └──► ELA Module ──────► Digital Splicing Detection
│
└──► Facial Crop Detection
│
Live Selfie ─────────┴──► Histogram Correlation & Template Normalization
│
Audit Verdict & Trust Score


---

## Tech Stack

- **Framework:** Streamlit
- **Optical Character Recognition:** EasyOCR
- **Computer Vision:** OpenCV (headless), NumPy, Pillow
- **Barcode & QR Decoding:** pyzbar
- **Security & Integrity:** Verhoeff Algorithm, Error Level Analysis (ELA)

