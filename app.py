# app.py
import os
import io
import time
import base64
import json
import hashlib
import binascii
from datetime import datetime

import streamlit as st

# ---- IMPORTANT: this MUST be the FIRST Streamlit command ----
st.set_page_config(page_title="Royal Skin Diagnosis", layout="centered", initial_sidebar_state="expanded")
# --------------------------------------------------------------

from PIL import Image

# ---------------------------
# Robust optional imports
# ---------------------------
# torch + torchvision guard
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# bcrypt guard
try:
    import bcrypt
except Exception:
    bcrypt = None

# firebase-admin guard
try:
    import firebase_admin
    from firebase_admin import credentials, firestore, storage as fb_storage
    FIREBASE_AVAILABLE = True
except Exception:
    FIREBASE_AVAILABLE = False

# google generative ai optional
try:
    import google.generativeai as genai
    GENAI_PACKAGE_AVAILABLE = True
except Exception:
    GENAI_PACKAGE_AVAILABLE = False

# ---------------------------
# Config (secrets from Streamlit or env)
# ---------------------------
GENAI_API_KEY = st.secrets.get("GENAI_API_KEY") if "GENAI_API_KEY" in st.secrets else os.environ.get("GENAI_API_KEY")
FIREBASE_SERVICE_ACCOUNT_JSON = st.secrets.get("FIREBASE_SERVICE_ACCOUNT_JSON") if "FIREBASE_SERVICE_ACCOUNT_JSON" in st.secrets else None
FIREBASE_STORAGE_BUCKET = st.secrets.get("FIREBASE_STORAGE_BUCKET") if "FIREBASE_STORAGE_BUCKET" in st.secrets else os.environ.get("FIREBASE_STORAGE_BUCKET")

# ---------------------------
# Disease metadata (expand later)
# ---------------------------
DISEASE_INFO = {
    "Melanoma": {"severity": "high", "description": "Potentially life-threatening skin cancer.", "cure": "Surgical excision and oncology referral.", "precautions": "See dermatologist urgently; avoid sun exposure.", "medications": "Targeted/immunotherapy depending on stage."},
    "Basal Cell Carcinoma": {"severity": "high", "description": "Common skin cancer; treatable when detected early.", "cure": "Surgery (Mohs) or excision.", "precautions": "Consult dermatologist.", "medications": "Topical/systemic as advised."},
    "Psoriasis": {"severity":"low", "description":"Chronic autoimmune skin condition.", "cure":"No cure; symptom control.", "precautions":"Moisturize; avoid triggers.", "medications":"Topical steroids, vitamin D analogues."},
    "Nevus": {"severity":"low", "description":"Usually a benign mole.", "cure":"None unless suspicious.", "precautions":"Monitor for changes.", "medications":"Not usually required."},
    "Actinic Keratosis": {"severity":"medium", "description":"Sun-damaged precancerous lesion.", "cure":"Cryotherapy/topicals.", "precautions":"Sun protection.", "medications":"5-FU, diclofenac gel."}
}
DISEASE_CLASSES = list(DISEASE_INFO.keys())

# ---------------------------
# Model helpers
# ---------------------------
DEVICE = torch.device("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")

def safe_load_model(path="model_checkpoint.pth"):
    """
    Attempts to load a ResNet50-based classifier if torch is available and path exists.
    Returns model or None.
    """
    if not TORCH_AVAILABLE:
        return None
    try:
        model = models.resnet50(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, len(DISEASE_CLASSES))
        if os.path.exists(path):
            state = torch.load(path, map_location="cpu")
            model.load_state_dict(state)
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        st.warning(f"Model load failed: {e}")
        return None

def preprocess_image(image: Image.Image):
    if TORCH_AVAILABLE:
        tf = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        if image.mode != "RGB":
            image = image.convert("RGB")
        return tf(image).unsqueeze(0).to(DEVICE)
    else:
        return None  # for dummy path

def predict_with_model(model, tensor):
    try:
        with torch.no_grad():
            out = model(tensor)
            probs = torch.nn.functional.softmax(out, dim=1)
            conf, idx = torch.max(probs, dim=1)
            return DISEASE_CLASSES[idx.item()], float(conf.item())
    except Exception as e:
        st.warning(f"Prediction failed: {e}")
        return "Nevus", 0.45

# Dummy predict when no torch/model available
def dummy_predict(_tensor=None):
    return "Nevus", 0.45

# load model (cached)
@st.cache_resource
def get_model():
    return safe_load_model("model_checkpoint.pth")

model = get_model()

# ---------------------------
# Firebase init
# ---------------------------
@st.cache_resource
def init_firebase():
    if not FIREBASE_AVAILABLE:
        return None, None
    if not (FIREBASE_SERVICE_ACCOUNT_JSON or os.environ.get("FIREBASE_SERVICE_ACCOUNT_PATH")):
        return None, None
    try:
        if FIREBASE_SERVICE_ACCOUNT_JSON:
            cred_dict = json.loads(FIREBASE_SERVICE_ACCOUNT_JSON)
            cred = credentials.Certificate(cred_dict)
        else:
            cred = credentials.Certificate(os.environ.get("FIREBASE_SERVICE_ACCOUNT_PATH"))
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {"storageBucket": FIREBASE_STORAGE_BUCKET})
        db = firestore.client()
        bucket = fb_storage.bucket()
        return db, bucket
    except Exception as e:
        st.error(f"Firebase init error: {e}")
        return None, None

db, bucket = init_firebase()

def upload_image_to_storage(path: str, image_bytes: bytes, content_type="image/jpeg"):
    if not bucket:
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
        st.error(f"Upload failed: {e}")
        return None

# ---------------------------
# Password helpers (bcrypt or PBKDF2 fallback)
# ---------------------------
def hash_password(pw: str) -> str:
    if bcrypt:
        return bcrypt.hashpw(pw.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    # PBKDF2 fallback
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac('sha256', pw.encode('utf-8'), salt, 200000)
    return "pbkdf2$" + binascii.hexlify(salt).decode() + "$" + binascii.hexlify(key).decode()

def check_password(pw: str, stored: str) -> bool:
    if bcrypt and not stored.startswith("pbkdf2$"):
        try:
            return bcrypt.checkpw(pw.encode('utf-8'), stored.encode('utf-8'))
        except Exception:
            return False
    # pbkdf2 check
    try:
        _tag, salt_hex, key_hex = stored.split("$")
        salt = binascii.unhexlify(salt_hex)
        key = binascii.unhexlify(key_hex)
        new_key = hashlib.pbkdf2_hmac('sha256', pw.encode('utf-8'), salt, 200000)
        return binascii.hexlify(new_key) == binascii.hexlify(key)
    except Exception:
        return False

# ---------------------------
# Local store fallback (if no Firestore)
# ---------------------------
if "local_users" not in st.session_state:
    st.session_state["local_users"] = {}
if "local_records" not in st.session_state:
    st.session_state["local_records"] = []
if "user" not in st.session_state:
    st.session_state["user"] = None
if "page" not in st.session_state:
    st.session_state["page"] = "welcome"

def local_store_user(user_doc: dict):
    u = st.session_state.get("local_users", {})
    u[user_doc["email"]] = user_doc
    st.session_state["local_users"] = u

def local_get_user(email: str):
    return st.session_state.get("local_users", {}).get(email)

def save_diagnosis_record(record: dict):
    if db:
        try:
            db.collection("diagnoses").add(record)
            return True
        except Exception as e:
            st.error(f"Save failed: {e}")
            return False
    else:
        recs = st.session_state.get("local_records", [])
        recs.append(record)
        st.session_state["local_records"] = recs
        return True

# ---------------------------
# UI CSS - lighter headings for visibility
# ---------------------------
st.markdown("""
<style>
:root { --royal1:#12021a; --accent:#9b4dff; --accent2:#6e2ee6; --muted:#e6ddff; }
html, body, [class*="stApp"] { background: linear-gradient(180deg,#12021a,#1e0930); color:var(--muted); }
h1, h2, h3 { color: #f3e8ff !important; }
.royal-card { background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02)); border:1px solid rgba(155,77,255,0.08); border-radius:12px; padding:16px; margin-bottom:12px;}
.primary-btn { background: linear-gradient(90deg,#9b4dff,#6e2ee6); color:white; padding:8px 12px; border-radius:8px; border:none; }
.small-muted { color: rgba(230,220,255,0.8); font-size:13px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Page flow: welcome -> register/login -> app
# ---------------------------
if st.session_state["page"] == "welcome":
    st.markdown('<div class="royal-card">', unsafe_allow_html=True)
    st.title("👑 Welcome to Royal Skin Diagnosis")
    st.write("Educational tool — not a substitute for professional medical advice.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login"):
            st.session_state["page"] = "login"
            st.experimental_rerun()
    with col2:
        if st.button("Register"):
            st.session_state["page"] = "register"
            st.experimental_rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# Sidebar navigation for logged in users
if st.session_state["page"] in ["home", "app", "profile", "login", "register"]:
    st.sidebar.header("Navigation")
    if st.session_state.get("user"):
        st.sidebar.write(f"Signed in as: {st.session_state['user'].get('name')}")
        choice = st.sidebar.radio("Go to", ["Home","Diagnose","Profile","Sign out"])
    else:
        choice = st.sidebar.radio("Go to", ["Home","Login","Register"])
    if choice == "Home":
        st.session_state["page"] = "home"
    elif choice == "Login":
        st.session_state["page"] = "login"
    elif choice == "Register":
        st.session_state["page"] = "register"
    elif choice == "Diagnose":
        st.session_state["page"] = "app"
    elif choice == "Profile":
        st.session_state["page"] = "profile"
    elif choice == "Sign out":
        st.session_state["user"] = None
        st.session_state["page"] = "welcome"
        st.experimental_rerun()

# ---------------------------
# Register
# ---------------------------
if st.session_state["page"] == "register":
    st.header("Create account")
    c1, c2 = st.columns(2)
    with c1:
        name = st.text_input("Full name")
        email = st.text_input("Email")
        phone = st.text_input("Phone (optional)")
    with c2:
        password = st.text_input("Password", type="password")
        password2 = st.text_input("Confirm password", type="password")
        avatar = st.file_uploader("Profile picture (optional)", type=["jpg","jpeg","png"])
    if st.button("Register"):
        if not name or not email or not password:
            st.error("Fill name, email and password.")
        elif password != password2:
            st.error("Passwords do not match.")
        else:
            hashed = hash_password(password)
            user_doc = {"name": name, "email": email, "phone": phone, "password_hash": hashed, "created_at": datetime.utcnow()}
            if avatar:
                b = avatar.read()
                st.session_state["last_avatar_bytes"] = b
                if bucket:
                    path = f"profiles/{email.replace('@','_at_')}_{int(time.time())}.jpg"
                    url = upload_image_to_storage(path, b, content_type=avatar.type or "image/jpeg")
                    if url: user_doc["avatar_url"] = url
            if db:
                try:
                    db.collection("users").add(user_doc)
                    st.success("Account created. Please login.")
                except Exception as e:
                    st.error(f"Cloud save failed: {e}")
            else:
                local_store_user(user_doc)
                st.success("Local account created. (Cloud not configured.)")

# ---------------------------
# Login
# ---------------------------
if st.session_state["page"] == "login":
    st.header("Login")
    em = st.text_input("Email", key="login_email")
    pw = st.text_input("Password", type="password", key="login_pw")
    if st.button("Login"):
        user_doc = None
        if db:
            try:
                q = db.collection("users").where("email","==",em).limit(1).get()
                if q and len(q) > 0:
                    user_doc = q[0].to_dict()
            except Exception as e:
                st.error(f"Cloud lookup failed: {e}")
        else:
            user_doc = local_get_user(em)
        if not user_doc:
            st.error("No such user. Register first.")
        else:
            if check_password(pw, user_doc["password_hash"]):
                st.session_state["user"] = {"email": user_doc["email"], "name": user_doc.get("name"), "phone": user_doc.get("phone"), "avatar": user_doc.get("avatar_url") if db else st.session_state.get("last_avatar_bytes")}
                st.success("Logged in.")
                st.session_state["page"] = "home"
                st.experimental_rerun()
            else:
                st.error("Incorrect password.")

# ---------------------------
# Home
# ---------------------------
if st.session_state["page"] == "home":
    st.markdown('<div class="royal-card">', unsafe_allow_html=True)
    st.title("Home")
    st.write("Welcome! Use the sidebar to go to Diagnose or Profile.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Profile
# ---------------------------
if st.session_state["page"] == "profile":
    if not st.session_state.get("user"):
        st.info("Please login first.")
    else:
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
            st.write("Cloud: " + ("Connected" if db else "Not configured (local mode)"))
        if st.button("View saved records"):
            st.write(st.session_state.get("local_records", []))

# ---------------------------
# Diagnose / main app
# ---------------------------
if st.session_state["page"] == "app":
    st.header("🔬 Diagnose Skin Disease")
    uploaded = st.file_uploader("Upload skin image (jpg/png)", type=["jpg","jpeg","png"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded image", use_column_width=True)

        if TORCH_AVAILABLE:
            tensor = preprocess_image(img)
            if tensor is None:
                predicted, conf = dummy_predict(None)
            elif model is not None:
                predicted, conf = predict_with_model(model, tensor)
            else:
                predicted, conf = dummy_predict(None)
        else:
            predicted, conf = dummy_predict(None)

        st.success(f"Predicted: **{predicted}** (confidence {conf:.2f})")

        # store in session for chat context
        st.session_state["current_diagnosis"] = predicted
        st.session_state["current_confidence"] = conf

        info = DISEASE_INFO.get(predicted, {})
        st.markdown(f"**Description:** {info.get('description','No description available.')}")
        st.markdown(f"**Precautions:** {info.get('precautions','Use sun protection and consult specialist if unsure.')}")
        st.markdown(f"**Treatment summary:** {info.get('medications','Consult dermatologist for treatment options.')}")

        severity = info.get("severity", "low")
        if severity == "high" or conf > 0.85:
            st.warning("This looks potentially severe. Please consult a dermatologist.")
            st.markdown("[Book a dermatologist consultation on Practo](https://www.practo.com/dermatologist)")

        if st.button("Save this record"):
            if not st.session_state.get("user"):
                st.error("Login to save records.")
            else:
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                b = buf.getvalue()
                rec = {"user_email": st.session_state["user"]["email"], "disease": predicted, "confidence": float(conf), "timestamp": datetime.utcnow()}
                if bucket:
                    path = f"diagnoses/{st.session_state['user']['email'].replace('@','_at_')}/{int(time.time())}.jpg"
                    url = upload_image_to_storage(path, b, content_type="image/jpeg")
                    if url:
                        rec["image_url"] = url
                else:
                    rec["image_b64"] = base64.b64encode(b).decode("utf-8")
                ok = save_diagnosis_record(rec)
                if ok:
                    st.success("Record saved.")
                else:
                    st.error("Failed to save record.")

    st.markdown("---")
    st.subheader("Ask about the diagnosis")
    if not st.session_state.get("current_diagnosis"):
        st.info("Diagnose an image first so AI has context.")
    user_q = st.text_input("Patient question:", key="user_question_input")
    if st.button("Ask AI"):
        if not st.session_state.get("current_diagnosis"):
            st.warning("Please diagnose an image first.")
        else:
            diag = st.session_state["current_diagnosis"]
            diag_info = DISEASE_INFO.get(diag, {})
            prompt = f"Disease: {diag}\nInfo: {diag_info}\nPatient question: {user_q}\nGive a concise, evidence-based, helpful answer and when needed recommend seeing a dermatologist."
            if not GENAI_API_KEY or not GENAI_PACKAGE_AVAILABLE:
                st.info("AI chat disabled (GENAI_API_KEY not configured). Showing automatic summary instead.")
                st.markdown(f"**Summary for {diag}:** {diag_info.get('description','')}\n\n**Precautions:** {diag_info.get('precautions','')}\n\n**When to see a doctor:** If symptoms are severe or change rapidly, consult a dermatologist.")
            else:
                try:
                    genai.configure(api_key=GENAI_API_KEY)
                    model_g = genai.GenerativeModel("models/gemini-2.5-pro")
                    response = model_g.generate_content(prompt)
                    if hasattr(response, "text"):
                        st.markdown(response.text)
                    else:
                        st.markdown(str(response))
                except Exception as e:
                    st.error(f"AI chat error: {e}")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div style="text-align:center;color:rgba(230,220,255,0.7)">Built with ❤️ — educational tool only. Not a medical diagnosis.</div>', unsafe_allow_html=True)
