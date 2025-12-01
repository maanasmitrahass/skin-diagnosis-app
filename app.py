# app.py
"""
Royal Skin Diagnosis - Streamlit app (single-file).

Features:
- Welcome -> Register / Login -> Diagnose
- Local user store fallback (session_state) and optional Firebase persistence
- AI diagnosis via Gemini (if GENAI_API_KEY), or Hugging Face inference (if HF settings),
  otherwise a safe dummy predictor
- Chat Q/A using Gemini if available, otherwise an automatic summary
- No torch import to avoid heavy dependencies on Streamlit Cloud
- PBKDF2 password hashing fallback (no bcrypt required)
"""

import os
import io
import json
import time
import base64
import hashlib
import binascii
from datetime import datetime

import streamlit as st
from PIL import Image
import requests

# --- Page config: must be the first Streamlit call after imports ---
st.set_page_config(page_title="Royal Skin Diagnosis", layout="centered", initial_sidebar_state="expanded")

# -------------------------
# Configuration / Secrets
# -------------------------
# Get keys from Streamlit secrets or environment variables
GENAI_API_KEY = st.secrets.get("GENAI_API_KEY") if "GENAI_API_KEY" in st.secrets else os.environ.get("GENAI_API_KEY")
HF_API_TOKEN = st.secrets.get("HF_API_TOKEN") if "HF_API_TOKEN" in st.secrets else os.environ.get("HF_API_TOKEN")
HF_MODEL = st.secrets.get("HF_MODEL") if "HF_MODEL" in st.secrets else os.environ.get("HF_MODEL")

FIREBASE_SERVICE_ACCOUNT_JSON = st.secrets.get("FIREBASE_SERVICE_ACCOUNT_JSON") if "FIREBASE_SERVICE_ACCOUNT_JSON" in st.secrets else None
FIREBASE_STORAGE_BUCKET = st.secrets.get("FIREBASE_STORAGE_BUCKET") if "FIREBASE_STORAGE_BUCKET" in st.secrets else os.environ.get("FIREBASE_STORAGE_BUCKET")

# Try to import google generative ai library only if key present (optional)
USE_GENAI = False
if GENAI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GENAI_API_KEY)
        USE_GENAI = True
    except Exception:
        USE_GENAI = False

# -------------------------
# Disease metadata (expandable)
# -------------------------
DISEASE_INFO = {
    "Melanoma": {"severity":"high", "description":"Potential skin cancer. Requires urgent dermatologist evaluation.", "precautions":"Avoid sun; see dermatologist.", "medications":"Depends on staging—surgery/oncology."},
    "Basal Cell Carcinoma": {"severity":"high", "description":"Common skin cancer; treatable when diagnosed early.", "precautions":"See dermatologist; sun protection.", "medications":"Surgical excision, topical therapies."},
    "Psoriasis": {"severity":"low", "description":"Chronic autoimmune skin disease with flaky plaques.", "precautions":"Moisturize; avoid triggers.", "medications":"Topical steroids, vitamin D analogs."},
    "Nevus": {"severity":"low", "description":"Benign mole.", "precautions":"Monitor for changes.", "medications":"Usually none."},
    "Actinic Keratosis": {"severity":"medium", "description":"Precancerous sun-damage lesion.", "precautions":"Sun protection; dermatology review.", "medications":"Topical 5-FU, cryotherapy."},
}
DISEASE_CLASSES = list(DISEASE_INFO.keys())

# -------------------------
# Firebase optional init
# -------------------------
FIREBASE_AVAILABLE = False
db = None
bucket = None
if FIREBASE_SERVICE_ACCOUNT_JSON and FIREBASE_STORAGE_BUCKET:
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore, storage as fb_storage
        cred_dict = json.loads(FIREBASE_SERVICE_ACCOUNT_JSON)
        cred = credentials.Certificate(cred_dict)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {"storageBucket": FIREBASE_STORAGE_BUCKET})
        db = firestore.client()
        bucket = fb_storage.bucket()
        FIREBASE_AVAILABLE = True
    except Exception as e:
        FIREBASE_AVAILABLE = False
        # don't crash — we'll fallback to local storage
        st.warning("Firebase init failed (continuing with local storage).")

# -------------------------
# Utility: hashing (PBKDF2) - secure fallback
# -------------------------
def hash_password(pw: str) -> str:
    """Hash password using PBKDF2-HMAC-SHA256. Returns a string 'pbkdf2$salthex$keyhex'."""
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac('sha256', pw.encode('utf-8'), salt, 200000)
    return "pbkdf2$" + binascii.hexlify(salt).decode() + "$" + binascii.hexlify(key).decode()

def check_password(pw: str, stored: str) -> bool:
    """Check PBKDF2 password hash."""
    try:
        tag, salt_hex, key_hex = stored.split("$")
        salt = binascii.unhexlify(salt_hex)
        key = binascii.unhexlify(key_hex)
        new_key = hashlib.pbkdf2_hmac('sha256', pw.encode('utf-8'), salt, 200000)
        return binascii.hexlify(new_key) == binascii.hexlify(key)
    except Exception:
        return False

# -------------------------
# Local fallback stores (session_state)
# -------------------------
if "local_users" not in st.session_state:
    st.session_state["local_users"] = {}  # email -> user dict
if "local_records" not in st.session_state:
    st.session_state["local_records"] = []  # list of records
if "user" not in st.session_state:
    st.session_state["user"] = None
if "page" not in st.session_state:
    st.session_state["page"] = "welcome"
if "current_diagnosis" not in st.session_state:
    st.session_state["current_diagnosis"] = None
if "last_avatar_bytes" not in st.session_state:
    st.session_state["last_avatar_bytes"] = None

def local_store_user(user_doc: dict):
    u = st.session_state.get("local_users", {})
    u[user_doc["email"]] = user_doc
    st.session_state["local_users"] = u

def local_get_user(email: str):
    return st.session_state.get("local_users", {}).get(email)

# -------------------------
# Helpers: Firebase upload & save
# -------------------------
def upload_image_to_storage(path: str, image_bytes: bytes, content_type="image/jpeg"):
    """Upload bytes to Firebase Storage; returns public URL or None."""
    if not FIREBASE_AVAILABLE or not bucket:
        return None
    try:
        blob = bucket.blob(path)
        blob.upload_from_string(image_bytes, content_type=content_type)
        try:
            blob.make_public()
            return blob.public_url
        except Exception:
            return f"gs://{blob.bucket.name}/{blob.name}"
    except Exception as e:
        st.error(f"Upload error: {e}")
        return None

def save_record_cloud(record: dict):
    """Save diagnosis record to Firestore. Returns True/False."""
    if not FIREBASE_AVAILABLE or not db:
        return False
    try:
        db.collection("diagnoses").add(record)
        return True
    except Exception as e:
        st.error(f"Cloud save failed: {e}")
        return False

# -------------------------
# Diagnosis backends
# -------------------------
def dummy_predict(image_bytes=None):
    """Safe dummy predictor used when no model/inference API is available."""
    # choose a benign default class
    return "Nevus", 0.42

def hf_predict(image_bytes: bytes):
    """Use HuggingFace image inference API (if HF_API_TOKEN & HF_MODEL are set).
    This is a basic implementation that posts image bytes to HF inference endpoint.
    Adapt the parsing depending on your HF model's output format.
    """
    if not HF_API_TOKEN or not HF_MODEL:
        return dummy_predict(image_bytes)
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    api_url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    try:
        # send as file
        response = requests.post(api_url, headers=headers, files={"file": image_bytes}, timeout=60)
        response.raise_for_status()
        data = response.json()
        # Parsing strategy: try to read a top label text field or first item
        # Many HF image-classification models return [{"label":"...","score":...}, ...]
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            label = data[0].get("label") or data[0].get("class") or str(data[0])
            score = data[0].get("score", 0.0)
            # map to known classes if possible
            return label, float(score)
        # otherwise fallback to string
        return str(data), 0.0
    except Exception as e:
        st.warning(f"Hugging Face inference failed: {e}")
        return dummy_predict(image_bytes)

def genai_vision_predict(image_bytes: bytes):
    """Use Gemini Vision via google.generativeai if available.
    The exact API for sending images can change — this is a simple prompt-image flow.
    """
    if not USE_GENAI:
        return dummy_predict(image_bytes)
    try:
        # For vision + text, use content generation with multimodal input
        # Create a "prompt" describing task and include image bytes in the request structure
        model = genai.GenerativeModel("models/gemini-2.5-pro")
        prompt = "You are an expert dermatologist. Look at the image and provide a single-line disease label only."
        # The new SDK specifics vary — attempt a simple call that works with many APIs
        response = model.generate_content([prompt, {"mime_type":"image/jpeg", "data": image_bytes}])
        text = ""
        if hasattr(response, "text"):
            text = response.text.strip()
        elif isinstance(response, dict):
            text = response.get("content", "")
        else:
            text = str(response)
        # Try to extract a label and confidences (best-effort)
        # We'll assume the first line contains the label
        label = text.splitlines()[0] if text else "Unknown"
        return label, 0.0
    except Exception as e:
        st.warning(f"Gemini vision inference failed: {e}")
        return dummy_predict(image_bytes)

def diagnose_image(image_bytes: bytes):
    """Master routing: try Gemini -> HF -> dummy."""
    # Try genai (highest preference)
    if USE_GENAI:
        try:
            return genai_vision_predict(image_bytes)
        except Exception:
            pass
    # Try Hugging Face if configured
    if HF_API_TOKEN and HF_MODEL:
        try:
            return hf_predict(image_bytes)
        except Exception:
            pass
    # Fallback
    return dummy_predict(image_bytes)

# -------------------------
# Chat / QA wrapper
# -------------------------
def genai_answer_question(disease_label: str, question: str):
    """Ask Gemini (if available) for a helpful answer based on disease context."""
    if not USE_GENAI:
        # fallback: automatic summary from DISEASE_INFO
        info = DISEASE_INFO.get(disease_label, {})
        desc = info.get("description", "No description available.")
        prec = info.get("precautions", "Follow general precautions and consult a doctor.")
        return f"Summary for {disease_label}:\n{desc}\nPrecautions: {prec}\nIf symptoms are severe, consult a dermatologist."
    try:
        model = genai.GenerativeModel("models/gemini-2.5-pro")
        prompt = f"""You are a dermatologist. Disease: {disease_label}\nDisease info: {DISEASE_INFO.get(disease_label, {})}\nPatient question: {question}\nAnswer concisely and include when to see a doctor."""
        response = model.generate_content(prompt)
        if hasattr(response, "text"):
            return response.text
        return str(response)
    except Exception as e:
        return f"AI error: {e}"

# -------------------------
# UI: CSS and styling
# -------------------------
st.markdown(
    """
    <style>
    :root { --bg:#12021a; --card:#1e0930; --muted: #e6ddff; --accent1:#9b4dff; --accent2:#6e2ee6; }
    .stApp { background: linear-gradient(180deg,var(--bg), #1a0530); color:var(--muted); }
    .royal-card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border:1px solid rgba(155,77,255,0.08); border-radius:12px; padding:12px; margin-bottom:10px;}
    h1,h2,h3 { color:#f7ecff !important; }
    .btn-primary { background: linear-gradient(90deg,var(--accent1),var(--accent2)); color:white; padding:8px 12px; border-radius:8px;}
    .muted { color: rgba(230,220,255,0.8); }
    input, textarea, select { color: black !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Page flow: Welcome -> Register -> Login -> App
# -------------------------
def go_home():
    st.session_state["page"] = "welcome"
    st.experimental_rerun()

def go_login():
    st.session_state["page"] = "login"
    st.experimental_rerun()

def go_register():
    st.session_state["page"] = "register"
    st.experimental_rerun()

# --- Welcome page ---
def page_welcome():
    st.markdown('<div class="royal-card">', unsafe_allow_html=True)
    st.title("👑 Royal Skin Diagnosis")
    st.write("Educational tool to analyze skin images. Not a medical diagnosis.")
    st.write("Create an account or login to continue. Data can be saved to your account.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Login"):
            go_login()
    with c2:
        if st.button("Register"):
            go_register()
    st.markdown("</div>", unsafe_allow_html=True)

# --- Register page ---
def page_register():
    st.header("Create account")
    cols = st.columns(2)
    with cols[0]:
        name = st.text_input("Full name", key="reg_name")
        email = st.text_input("Email", key="reg_email")
        phone = st.text_input("Phone (optional)", key="reg_phone")
    with cols[1]:
        pw = st.text_input("Password", type="password", key="reg_pw")
        pw2 = st.text_input("Confirm password", type="password", key="reg_pw2")
        avatar = st.file_uploader("Profile picture (optional)", type=["jpg","jpeg","png"], key="reg_avatar")
    if st.button("Register account"):
        if not name or not email or not pw:
            st.error("Fill name, email and password.")
        elif pw != pw2:
            st.error("Passwords do not match.")
        else:
            # Save locally or to firebase
            hashed = hash_password(pw)
            user_doc = {"name": name, "email": email, "phone": phone, "password_hash": hashed, "created_at": datetime.utcnow().isoformat()}
            if avatar:
                b = avatar.read()
                st.session_state["last_avatar_bytes"] = b
                if FIREBASE_AVAILABLE:
                    path = f"profiles/{email.replace('@','_at_')}_{int(time.time())}.jpg"
                    url = upload_image_to_storage(path, b, content_type="image/jpeg")
                    if url:
                        user_doc["avatar_url"] = url
            if FIREBASE_AVAILABLE and db:
                try:
                    db.collection("users").add(user_doc)
                    st.success("Account created (cloud). Please login.")
                except Exception as e:
                    st.error(f"Cloud save failed: {e}")
            else:
                local_store_user(user_doc)
                st.success("Local account created. (Cloud not configured.)")

# --- Login page ---
def page_login():
    st.header("Login")
    em = st.text_input("Email", key="login_email")
    pw = st.text_input("Password", type="password", key="login_pw")
    if st.button("Login"):
        user_doc = None
        # try cloud first
        if FIREBASE_AVAILABLE and db:
            try:
                res = db.collection("users").where("email", "==", em).limit(1).get()
                if res and len(res) > 0:
                    user_doc = res[0].to_dict()
            except Exception:
                user_doc = None
        # fallback local
        if not user_doc:
            user_doc = local_get_user(em)
        if not user_doc:
            st.error("No such user. Please register.")
        else:
            if check_password(pw, user_doc["password_hash"]):
                st.session_state["user"] = {"email": user_doc["email"], "name": user_doc.get("name"), "phone": user_doc.get("phone"), "avatar": user_doc.get("avatar_url") if FIREBASE_AVAILABLE else st.session_state.get("last_avatar_bytes")}
                st.success("Logged in.")
                st.session_state["page"] = "home"
                st.experimental_rerun()
            else:
                st.error("Incorrect password.")

# --- Home page after login ---
def page_home():
    st.markdown('<div class="royal-card">', unsafe_allow_html=True)
    st.title("Home")
    st.write("Use the sidebar to navigate to Diagnose, Profile, or Sign out.")
    st.markdown("</div>", unsafe_allow_html=True)

# --- Profile page ---
def page_profile():
    if not st.session_state.get("user"):
        st.info("Please login.")
        return
    user = st.session_state["user"]
    st.header("Profile")
    col1, col2 = st.columns([1,3])
    with col1:
        if isinstance(user.get("avatar"), (bytes, bytearray)):
            st.image(user["avatar"], width=140)
        elif user.get("avatar"):
            st.image(user["avatar"], width=140)
        else:
            st.image("https://via.placeholder.com/140x140.png?text=No+Avatar", width=140)
    with col2:
        st.write(f"**Name:** {user.get('name')}")
        st.write(f"**Email:** {user.get('email')}")
        st.write(f"**Phone:** {user.get('phone')}")
        st.write("Cloud: " + ("Connected" if FIREBASE_AVAILABLE else "Not configured (local mode)"))
    if st.button("View saved records"):
        if FIREBASE_AVAILABLE and db:
            try:
                qry = db.collection("diagnoses").where("user_email", "==", user["email"]).limit(50).stream()
                docs = [d.to_dict() for d in qry]
                st.write(docs)
            except Exception as e:
                st.error(f"Cloud read failed: {e}")
        else:
            st.write(st.session_state.get("local_records", []))

# --- Diagnose page ---
def page_diagnose():
    st.header("🔬 Diagnose Skin Condition")
    uploaded = st.file_uploader("Upload skin image (jpg/png)", type=["jpg","jpeg","png"])
    if uploaded:
        try:
            img = Image.open(uploaded)
            st.image(img, caption="Uploaded image", use_column_width=True)
        except Exception:
            st.error("Could not open image.")
            return
        b = io.BytesIO()
        img.save(b, format="JPEG")
        img_bytes = b.getvalue()

        # diagnoze
        if st.button("Analyze image"):
            with st.spinner("Analyzing..."):
                label, conf = diagnose_image(img_bytes)
                st.success(f"Predicted: **{label}** (confidence {conf:.2f})")
                st.session_state["current_diagnosis"] = label
                st.session_state["current_confidence"] = conf
                # show info if exists
                info = DISEASE_INFO.get(label, {})
                st.markdown(f"**Description:** {info.get('description','No description available.')}")
                st.markdown(f"**Precautions:** {info.get('precautions','Follow general precautions.')}")
                st.markdown(f"**Treatment / Medications:** {info.get('medications','Consult a dermatologist.')}")
                # show practo link for severe or high confidence
                severity = info.get("severity", "low")
                if severity == "high" or conf > 0.85:
                    st.warning("This looks potentially serious. Please consult a dermatologist promptly.")
                    st.markdown("[Book a dermatologist on Practo](https://www.practo.com/dermatologist)")

                # Save record option
                if st.button("Save this record"):
                    if not st.session_state.get("user"):
                        st.error("Log in to save records.")
                    else:
                        rec = {
                            "user_email": st.session_state["user"]["email"],
                            "disease": label,
                            "confidence": float(conf),
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        if FIREBASE_AVAILABLE and bucket:
                            path = f"diagnoses/{st.session_state['user']['email'].replace('@','_at_')}/{int(time.time())}.jpg"
                            url = upload_image_to_storage(path, img_bytes, content_type="image/jpeg")
                            if url:
                                rec["image_url"] = url
                        else:
                            rec["image_b64"] = base64.b64encode(img_bytes).decode("utf-8")
                        # Save either to cloud or local
                        if FIREBASE_AVAILABLE:
                            ok = save_record_cloud(rec)
                            if ok:
                                st.success("Saved to cloud.")
                            else:
                                st.error("Cloud save failed; saved locally.")
                                st.session_state["local_records"].append(rec)
                        else:
                            st.session_state["local_records"].append(rec)
                            st.success("Saved locally.")
    else:
        st.info("Upload an image to diagnose.")

    # Chat / Ask about diagnosis
    st.markdown("---")
    st.subheader("Ask about the diagnosis")
    if not st.session_state.get("current_diagnosis"):
        st.info("Diagnose an image first so AI has context.")
    question = st.text_input("Your question about the diagnosis:", key="qa_input")
    if st.button("Ask AI"):
        if not st.session_state.get("current_diagnosis"):
            st.warning("Diagnose an image first.")
        else:
            ans = genai_answer_question(st.session_state["current_diagnosis"], question)
            st.markdown(ans)

# -------------------------
# Sidebar & routing
# -------------------------
def render_sidebar():
    st.sidebar.title("Navigation")
    if st.session_state.get("user"):
        st.sidebar.write(f"Signed in as: {st.session_state['user'].get('name')}")
        choice = st.sidebar.radio("Go to", ["Home", "Diagnose", "Profile", "Sign out"])
        if choice == "Home":
            st.session_state["page"] = "home"
        elif choice == "Diagnose":
            st.session_state["page"] = "app"
        elif choice == "Profile":
            st.session_state["page"] = "profile"
        elif choice == "Sign out":
            st.session_state["user"] = None
            st.session_state["page"] = "welcome"
            st.experimental_rerun()
    else:
        choice = st.sidebar.radio("Go to", ["Home", "Login", "Register"])
        if choice == "Home":
            st.session_state["page"] = "home"
        elif choice == "Login":
            st.session_state["page"] = "login"
        elif choice == "Register":
            st.session_state["page"] = "register"

# -------------------------
# Main renderer
# -------------------------
def main():
    render_sidebar()
    page = st.session_state.get("page", "welcome")
    if page == "welcome":
        page_welcome()
    elif page == "register":
        page_register()
    elif page == "login":
        page_login()
    elif page == "home":
        page_home()
    elif page == "profile":
        page_profile()
    elif page == "app":
        page_diagnose()
    else:
        page_welcome()

# Footer
def footer():
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div style="text-align:center;color:rgba(230,220,255,0.7)">Built with ❤️ — educational tool only. Not a medical diagnosis.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    footer()
