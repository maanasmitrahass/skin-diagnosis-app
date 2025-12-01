# app.py
import os
import io
import time
import base64
from datetime import datetime

import streamlit as st

# ---- IMPORTANT: this MUST be the first Streamlit call in the script ----
st.set_page_config(page_title="Skin Diagnosis Assistant", layout="centered")
# ------------------------------------------------------------------------

from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

# Optional cloud (Firebase)
try:
    import firebase_admin
    from firebase_admin import credentials, firestore, storage as fb_storage
    FIREBASE_AVAILABLE = True
except Exception:
    FIREBASE_AVAILABLE = False

    FIREBASE_AVAILABLE = False

# Password hashing
try:
    import bcrypt
except Exception:
    bcrypt = None

# -------------------------
# Config / Secrets (instructions below)
# -------------------------
# Put sensitive secrets into Streamlit secrets or environment variables.
# Required keys (recommended to store in streamlit secrets):
#   FIREBASE_SERVICE_ACCOUNT_JSON  -> JSON string of service account (or set path in FIREBASE_SERVICE_ACCOUNT_PATH)
#   FIREBASE_STORAGE_BUCKET       -> "your-project.appspot.com"
#   GENAI_API_KEY                 -> Gemini key (optional)
#
# The app will attempt to initialize Firebase only if FIREBASE_SERVICE_ACCOUNT_JSON (or PATH) and FIREBASE_STORAGE_BUCKET are present.
GENAI_API_KEY = None
if "GENAI_API_KEY" in st.secrets:
    GENAI_API_KEY = st.secrets["GENAI_API_KEY"]
else:
    GENAI_API_KEY = os.environ.get("GENAI_API_KEY")

# -------------------------
# Disease info & classes (kept from your original app)
# -------------------------
DISEASE_INFO = {
    "Melanoma": { "description": "A dangerous form of skin cancer requiring prompt attention.", "cure": "Surgical excision by a dermatologist or oncologist.", "precautions": "Regular skin checks, avoid tanning beds, use sunscreen (SPF 30+).", "medications": "Immunotherapy (pembrolizumab), targeted therapies.", "department": "Dermatology" },
    "Nevus": { "description": "Common mole, typically benign.", "cure": "None needed unless suspicious; may be removed surgically.", "precautions": "Monitor for changes in color, size, or shape.", "medications": "None typically required.", "department": "Dermatology" },
    "Basal Cell Carcinoma": { "description": "The most common type of skin cancer, slow-growing and highly treatable.", "cure": "Surgical excision, topical treatments, or cryotherapy.", "precautions": "Reduce sun exposure and use protective sunscreen.", "medications": "Imiquimod or fluorouracil cream as prescribed.", "department": "Dermatology" },
    "Actinic Keratosis": { "description": "Sun-induced precancerous lesion.", "cure": "Cryotherapy, topical treatments.", "precautions": "Minimize sun exposure, use hats and protective clothing.", "medications": "5-fluorouracil cream, diclofenac gel.", "department": "Dermatology" },
    "Benign Keratosis": { "description": "Non-cancerous skin growth, often waxy or wart-like.", "cure": "Removal by cryotherapy or laser if bothersome.", "precautions": "Monitor for unusual changes in appearance.", "medications": "None needed.", "department": "Dermatology" },
    "Dermatofibroma": { "description": "Firm, benign nodule under the skin.", "cure": "Surgical removal if painful or bothersome.", "precautions": "Observe for rapid growth or changes.", "medications": "None required.", "department": "Dermatology" },
    "Vascular Lesion": { "description": "Abnormal cluster of blood vessels in the skin.", "cure": "Laser therapy or minor surgical removal.", "precautions": "Monitor for bleeding or size increase.", "medications": "Laser therapy as recommended.", "department": "Dermatology" },
    "Psoriasis": { "description": "A chronic skin condition causing red, scaly patches. Often found on elbows, knees, and scalp.", "cure": "No cure, but symptoms are manageable.", "precautions": "Moisturize skin, avoid triggers such as stress and skin injury.", "medications": "Topical steroids, vitamin D analogues, light therapy, oral immunosuppressants (as prescribed).", "department": "Dermatology" }
}
DISEASE_CLASSES = list(DISEASE_INFO.keys())

# -------------------------
# Device & model loader
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model(checkpoint_bytes=None):
    # Recreate same architecture as original
    try:
        model = models.resnet50(weights=None)
    except Exception:
        model = models.resnet50()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(DISEASE_CLASSES))
    if checkpoint_bytes:
        try:
            import io
            buf = io.BytesIO(checkpoint_bytes)
            state_dict = torch.load(buf, map_location="cpu")
            model.load_state_dict(state_dict)
            st.info("Loaded model checkpoint.")
        except Exception as e:
            st.error(f"Checkpoint load failed: {e}")
    model.to(DEVICE)
    model.eval()
    return model

def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    if image.mode == "RGBA":
        image = image.convert("RGB")
    return transform(image).unsqueeze(0).to(DEVICE)

def predict_disease(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, idx = torch.max(probs, dim=1)
        predicted = DISEASE_CLASSES[idx.item()]
        return predicted, float(confidence.item())

# -------------------------
# Firebase initialization helpers
# -------------------------
@st.cache_resource
def init_firebase():
    """
    Initialize Firebase Admin SDK using a service account JSON.
    Place the service account JSON string into st.secrets["FIREBASE_SERVICE_ACCOUNT_JSON"]
    or set an env var FIREBASE_SERVICE_ACCOUNT_PATH pointing to the JSON file.
    Also set st.secrets["FIREBASE_STORAGE_BUCKET"] to your bucket name.
    """
    if not FIREBASE_AVAILABLE:
        return None, None

    # read service account JSON from secrets or path
    service_account_json = st.secrets.get("FIREBASE_SERVICE_ACCOUNT_JSON") if "FIREBASE_SERVICE_ACCOUNT_JSON" in st.secrets else None
    service_account_path = os.environ.get("FIREBASE_SERVICE_ACCOUNT_PATH")
    bucket_name = st.secrets.get("FIREBASE_STORAGE_BUCKET") if "FIREBASE_STORAGE_BUCKET" in st.secrets else os.environ.get("FIREBASE_STORAGE_BUCKET")

    if not (service_account_json or service_account_path) or not bucket_name:
        st.info("Firebase not configured (missing service account or storage bucket). Running in LOCAL mode.")
        return None, None

    try:
        if service_account_json:
            import json
            cred_dict = json.loads(service_account_json)
            cred = credentials.Certificate(cred_dict)
        else:
            cred = credentials.Certificate(service_account_path)

        # Avoid re-initializing app if running hot-reload
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {"storageBucket": bucket_name})
        db = firestore.client()
        bucket = fb_storage.bucket()
        return db, bucket
    except Exception as e:
        st.error(f"Failed to initialize Firebase: {e}")
        return None, None

db, bucket = init_firebase()

# -------------------------
# Cloud helpers (only active if db and bucket available)
# -------------------------
def make_blob_public_and_get_url(blob):
    try:
        blob.make_public()
        return blob.public_url
    except Exception:
        try:
            # fallback: return gs:// path if public not allowed
            return f"gs://{blob.bucket.name}/{blob.name}"
        except Exception:
            return None

def upload_image_to_storage(path: str, image_bytes: bytes, content_type="image/jpeg"):
    if not bucket:
        return None
    try:
        blob = bucket.blob(path)
        blob.upload_from_string(image_bytes, content_type=content_type)
        url = make_blob_public_and_get_url(blob)
        return url
    except Exception as e:
        st.error(f"Upload failed: {e}")
        return None

def save_user_to_firestore(user_doc: dict):
    if not db:
        return False
    try:
        users = db.collection("users")
        # use auto-id
        doc_ref = users.document()
        doc_ref.set(user_doc)
        return True
    except Exception as e:
        st.error(f"Failed to save user: {e}")
        return False

def find_user_by_email(email: str):
    if not db:
        return None
    users = db.collection("users")
    q = users.where("email", "==", email).limit(1).get()
    if q and len(q) > 0:
        return q[0]
    return None

def save_diagnosis_record(record: dict):
    if not db:
        return False
    try:
        db.collection("diagnoses").add(record)
        return True
    except Exception as e:
        st.error(f"Failed to save diagnosis record: {e}")
        return False

# -------------------------
# Simple auth helpers (local/db)
# -------------------------
def hash_password(clear_password: str) -> str:
    if bcrypt is None:
        raise RuntimeError("bcrypt is not installed. Please install bcrypt.")
    hashed = bcrypt.hashpw(clear_password.encode("utf-8"), bcrypt.gensalt())
    return hashed.decode('utf-8')

def check_password(clear_password: str, hashed_password: str) -> bool:
    if bcrypt is None:
        raise RuntimeError("bcrypt is not installed. Please install bcrypt.")
    try:
        return bcrypt.checkpw(clear_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False

def local_store_user(local_users, user_doc):
    # local_users is a dict in session_state for demo fallback
    local_users[user_doc["email"]] = user_doc
    st.session_state["local_users"] = local_users

def local_get_user(email):
    return st.session_state.get("local_users", {}).get(email)

# -------------------------
# Gemini wrapper (keeps your original pattern)
# -------------------------
def get_bot_answer(context: str, user_question: str) -> str:
    # If you want to use Gemini, configure GENAI_API_KEY in secrets/env and install google.generativeai
    if not GENAI_API_KEY:
        return "Chat is disabled because API key is not configured."
    try:
        import google.generativeai as genai
        genai.configure(api_key=GENAI_API_KEY)
        model = genai.GenerativeModel("models/gemini-2.5-pro")
        prompt = f"You are a medical assistant. Here is diagnosis info: {context}\nPatient question: {user_question}\nGive a concise, evidence-based, non-legal advisory answer and recommend seeing a dermatologist if appropriate."
        response = model.generate_content(prompt)
        if hasattr(response, "text"):
            return response.text
        return str(response)
    except Exception as e:
        return f"AI assistant failed to respond: {e}"

# -------------------------
# UI / Royal CSS
# -------------------------

st.markdown(
    """
    <style>
    :root {
      --royal-bg: linear-gradient(180deg, #0f0724 0%, #211035 100%);
      --card-bg: rgba(255,255,255,0.05);
      --accent1: #9b4dff;
      --accent2: #6e2ee6;
      --muted: #dcd6f7;
    }
    html, body, [class*="stApp"] {
      background: var(--royal-bg);
      color: var(--muted);
      font-family: "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }
    .royal-card {
      background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02));
      border: 1px solid rgba(155,77,255,0.12);
      border-radius: 14px;
      padding: 20px;
      box-shadow: 0 8px 30px rgba(0,0,0,0.5);
    }
    .brand {
      font-size: 24px;
      font-weight: 700;
      color: var(--accent1);
      letter-spacing: 0.6px;
    }
    .muted {
      color: rgba(220,214,247,0.85);
      font-size: 13px;
    }
    .primary-btn {
      background: linear-gradient(90deg, var(--accent1), var(--accent2));
      padding: 10px 18px;
      border-radius: 10px;
      color: white;
      border: none;
      font-weight: 600;
      box-shadow: 0 8px 24px rgba(107,46,230,0.18);
    }
    input[type="file"] { color: white; }
    .small { font-size: 13px; color: rgba(220,220,255,0.85); }
    </style>
    """, unsafe_allow_html=True
)

st.markdown('<div class="royal-card">', unsafe_allow_html=True)
st.markdown('<span class="brand">👑 Royal Skin Diagnosis</span>  <span class="muted">— secure user portal & cloud save</span>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Sidebar for Login / Register
st.sidebar.header("Account")
menu = st.sidebar.selectbox("Choose", ["Home", "Register", "Login", "Profile", "Sign out"])

# optional checkpoint upload in sidebar (same idea as your original)
checkpoint_file = st.sidebar.file_uploader("Optional: upload model checkpoint (.pt/.pth)", type=["pt","pth"])
checkpoint_bytes = None
if checkpoint_file:
    checkpoint_bytes = checkpoint_file.read()
model = load_model(checkpoint_bytes)

# Initialize local fallback store
if "local_users" not in st.session_state:
    st.session_state["local_users"] = {}

if "user" not in st.session_state:
    st.session_state["user"] = None

# -------------------------
# Registration
# -------------------------
if menu == "Register":
    st.subheader("Create an account")
    cols = st.columns(2)
    with cols[0]:
        name = st.text_input("Full name")
        email = st.text_input("Email")
        phone = st.text_input("Phone (optional)")
    with cols[1]:
        password = st.text_input("Password", type="password")
        password2 = st.text_input("Confirm password", type="password")
        avatar = st.file_uploader("Profile picture (optional)", type=["jpg","jpeg","png"])
    if st.button("Register", key="register_btn"):
        if not name or not email or not password:
            st.error("Please fill name, email and password.")
        elif password != password2:
            st.error("Passwords do not match.")
        else:
            # Check cloud first
            if db:
                # check if user exists
                existing = find_user_by_email(email)
                if existing:
                    st.error("An account with this email already exists.")
                else:
                    hashed = hash_password(password)
                    user_doc = {
                        "name": name,
                        "email": email,
                        "phone": phone,
                        "password_hash": hashed,
                        "created_at": datetime.utcnow(),
                    }
                    if avatar:
                        img_bytes = avatar.read()
                        path = f"profiles/{email.replace('@','_at_')}_{int(time.time())}.jpg"
                        url = upload_image_to_storage(path, img_bytes, content_type=avatar.type or "image/jpeg")
                        if url:
                            user_doc["avatar_url"] = url
                    ok = save_user_to_firestore(user_doc)
                    if ok:
                        st.success("Account created. Please log in from sidebar -> Login.")
            else:
                # Local fallback
                if local_get_user(email):
                    st.error("Account already exists locally.")
                else:
                    hashed = hash_password(password)
                    user_doc = {
                        "name": name, "email": email, "phone": phone,
                        "password_hash": hashed, "created_at": datetime.utcnow()
                    }
                    if avatar:
                        st.session_state["last_avatar_bytes"] = avatar.read()
                    local_store_user(st.session_state["local_users"], user_doc)
                    st.success("Local account created. (Cloud not configured.)")

# -------------------------
# Login
# -------------------------
elif menu == "Login":
    st.subheader("Login to your account")
    lemail = st.text_input("Email", key="login_email")
    lpass = st.text_input("Password", type="password", key="login_pass")
    if st.button("Login", key="login_btn"):
        user_doc = None
        user_ref = None
        if db:
            found = find_user_by_email(lemail)
            if not found:
                st.error("No account with that email.")
            else:
                user_ref = found
                data = found.to_dict()
                user_doc = data
        else:
            user_doc = local_get_user(lemail)
        if user_doc:
            if check_password(lpass, user_doc["password_hash"]):
                # set session
                st.session_state["user"] = {
                    "email": user_doc["email"],
                    "name": user_doc.get("name"),
                    "phone": user_doc.get("phone"),
                    "avatar": user_doc.get("avatar_url") if db else st.session_state.get("last_avatar_bytes")
                }
                st.success(f"Welcome back, {st.session_state['user']['name']}!")
            else:
                st.error("Incorrect password.")

# -------------------------
# Profile
# -------------------------
elif menu == "Profile":
    if not st.session_state.get("user"):
        st.info("Please login first.")
    else:
        user = st.session_state["user"]
        st.subheader("Your profile")
        cols = st.columns([1,3])
        with cols[0]:
            if isinstance(user.get("avatar"), (bytes, bytearray)):
                st.image(user["avatar"], width=140)
            elif user.get("avatar"):
                st.image(user["avatar"], width=140)
            else:
                st.image("https://via.placeholder.com/140x140.png?text=No+Avatar", width=140)
        with cols[1]:
            st.markdown(f"**Name:** {user.get('name')}")
            st.markdown(f"**Email:** {user.get('email')}")
            st.markdown(f"**Phone:** {user.get('phone')}")
            if db:
                st.markdown("**Cloud:** Connected")
            else:
                st.markdown("**Cloud:** Not configured (local mode)")
        st.markdown("---")
        if st.button("Go to Diagnosis page"):
            st.session_state["current_page"] = "diagnose"

# -------------------------
# Sign out
# -------------------------
elif menu == "Sign out":
    st.session_state["user"] = None
    st.success("Signed out.")

# -------------------------
# Home / Diagnosis / Chat main area
# -------------------------
# main container
st.markdown("</div>", unsafe_allow_html=True)  # close header card
st.markdown("")

main_col1, main_col2 = st.columns([2, 3])

with main_col1:
    st.markdown('<div class="royal-card">', unsafe_allow_html=True)
    st.header("Features")
    st.markdown("- Diagnose from image (uses your uploaded model or default ResNet50)")
    st.markdown("- Save diagnosis records to cloud (if Firebase configured)")
    st.markdown("- Register / Login with secure hashed passwords")
    st.markdown("- Consultation chat (Gemini) if API key configured")
    st.markdown("</div>", unsafe_allow_html=True)

with main_col2:
    st.markdown('<div class="royal-card">', unsafe_allow_html=True)
    st.header("🔬 Diagnose Skin Disease")
    uploaded = st.file_uploader("Upload skin image (jpg, png)", type=["jpg","jpeg","png"])
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded", use_column_width=True)
        tensor = preprocess_image(image)
        with st.spinner("Running model..."):
            predicted, confidence = predict_disease(model, tensor)
        st.success(f"Identified: **{predicted}** (confidence {confidence:.2f})")
        info = DISEASE_INFO.get(predicted, {})
        st.markdown(f"**Description:** {info.get('description','')}")
        st.markdown(f"**Precautions:** {info.get('precautions','')}")
        # Save record option
        if st.button("Save this record to your account"):
            if not st.session_state.get("user"):
                st.error("Please login to save records.")
            else:
                user_email = st.session_state["user"]["email"]
                ts = int(time.time())
                filename = f"{user_email.replace('@','_at_')}_{ts}.jpg"
                # image bytes
                buf = io.BytesIO()
                image.save(buf, format="JPEG")
                img_bytes = buf.getvalue()
                # upload if cloud configured
                record = {
                    "user_email": user_email,
                    "disease": predicted,
                    "confidence": float(confidence),
                    "timestamp": datetime.utcnow()
                }
                if bucket:
                    path = f"diagnoses/{user_email.replace('@','_at_')}/{filename}"
                    url = upload_image_to_storage(path, img_bytes, content_type="image/jpeg")
                    if url:
                        record["image_url"] = url
                else:
                    # store small base64 in record if no cloud (for local demo)
                    record["image_b64"] = base64.b64encode(img_bytes).decode("utf-8")
                ok = save_diagnosis_record(record)
                if ok:
                    st.success("Diagnosis saved.")
                else:
                    st.error("Failed to save diagnosis (cloud may be unconfigured).")
    st.markdown("</div>", unsafe_allow_html=True)

# Chat assistant (uses Gemini if key available)
st.markdown("")
st.markdown('<div class="royal-card">', unsafe_allow_html=True)
st.subheader("💬 Ask about the diagnosis")
if "current_diagnosis" in st.session_state:
    st.markdown(f"Context disease: **{st.session_state['current_diagnosis']}**")
user_q = st.text_input("Ask a question to the AI doctor:", key="user_question_input")
if st.button("Ask AI"):
    if not st.session_state.get("current_diagnosis"):
        st.warning("It helps to diagnose an image first so AI has context.")
    else:
        diag = st.session_state.get("current_diagnosis")
        diag_context = DISEASE_INFO.get(diag, {})
        diagnosis_text = (
            f"Disease: {diag}\n"
            f"Description: {diag_context.get('description','')}\n"
            f"Cure: {diag_context.get('cure','')}\n"
            f"Precautions: {diag_context.get('precautions','')}\n"
            f"Medications: {diag_context.get('medications','')}\n"
        )
        with st.spinner("AI is thinking..."):
            answer = get_bot_answer(diagnosis_text, st.session_state.get("user_question_input",""))
        st.markdown(f"**AI Doctor:** {answer}")
st.markdown("</div>", unsafe_allow_html=True)

# Footer with small tips
st.markdown('<div style="margin-top:18px;text-align:center;color:rgba(220,214,247,0.6)">Built with ❤️ — remember: this is an educational tool, not a medical diagnosis.</div>', unsafe_allow_html=True)
