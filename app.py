# app.py — safer / hardened version of your Royal Skin Diagnosis app
# Overwrite your current app.py with this file.

import os
import io
import json
import time
import base64
import hashlib
import binascii
from datetime import datetime
from typing import Optional, Tuple

import streamlit as st
from PIL import Image
import requests

# -------------------------
# Page config (first Streamlit call)
# -------------------------
st.set_page_config(page_title="Royal Skin Diagnosis", layout="centered", initial_sidebar_state="expanded")

# -------------------------
# Read secrets / env
# -------------------------
GENAI_API_KEY = st.secrets.get("GENAI_API_KEY") if "GENAI_API_KEY" in st.secrets else os.environ.get("GENAI_API_KEY")
HF_API_TOKEN = st.secrets.get("HF_API_TOKEN") if "HF_API_TOKEN" in st.secrets else os.environ.get("HF_API_TOKEN")
HF_MODEL = st.secrets.get("HF_MODEL") if "HF_MODEL" in st.secrets else os.environ.get("HF_MODEL")

FIREBASE_SERVICE_ACCOUNT_JSON = st.secrets.get("FIREBASE_SERVICE_ACCOUNT_JSON") if "FIREBASE_SERVICE_ACCOUNT_JSON" in st.secrets else None
FIREBASE_STORAGE_BUCKET = st.secrets.get("FIREBASE_STORAGE_BUCKET") if "FIREBASE_STORAGE_BUCKET" in st.secrets else os.environ.get("FIREBASE_STORAGE_BUCKET")

# -------------------------
# Optional Google Generative AI (Gemini)
# -------------------------
USE_GENAI = False
genai = None
if GENAI_API_KEY:
    try:
        import google.generativeai as genai_mod
        genai_mod.configure(api_key=GENAI_API_KEY)
        genai = genai_mod
        USE_GENAI = True
    except Exception:
        # leave USE_GENAI False if import/config fails
        USE_GENAI = False
        genai = None

# -------------------------
# Optional Firebase init (lazy)
# -------------------------
FIREBASE_AVAILABLE = False
db = None
bucket = None

def try_init_firebase():
    """
    Lazy init for firebase. Return (db, bucket) or (None, None).
    Do NOT call Streamlit UI functions in this function at import time.
    """
    global FIREBASE_AVAILABLE, db, bucket
    if not (FIREBASE_SERVICE_ACCOUNT_JSON and FIREBASE_STORAGE_BUCKET):
        return None, None
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore, storage as fb_storage
        cred_dict = json.loads(FIREBASE_SERVICE_ACCOUNT_JSON)
        cred = credentials.Certificate(cred_dict)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {"storageBucket": FIREBASE_STORAGE_BUCKET})
        db_local = firestore.client()
        bucket_local = fb_storage.bucket()
        FIREBASE_AVAILABLE = True
        db = db_local
        bucket = bucket_local
        return db_local, bucket_local
    except Exception:
        # initialization failed, fallback to local only
        FIREBASE_AVAILABLE = False
        db = None
        bucket = None
        return None, None

# do not call UI from here; call later to surface any message
_firebase = try_init_firebase()
if _firebase:
    db, bucket = _firebase

# -------------------------
# Disease metadata
# -------------------------
DISEASE_INFO = {
    "Melanoma": {"severity":"high", "description":"Potentially life-threatening skin cancer.", "precautions":"Avoid sun; see dermatologist urgently.", "medications":"Surgery/oncology-managed therapies."},
    "Basal Cell Carcinoma": {"severity":"high", "description":"Common skin cancer; treatable when detected early.", "precautions":"Dermatologist consult; sun protection.", "medications":"Surgical excision, topical options."},
    "Psoriasis": {"severity":"low", "description":"Chronic autoimmune skin condition causing scaly plaques.", "precautions":"Moisturize; avoid triggers.", "medications":"Topical steroids, vitamin D analogues."},
    "Nevus": {"severity":"low", "description":"Benign mole in most cases.", "precautions":"Monitor for changes in size/colour.", "medications":"Not needed unless suspicious."},
    "Actinic Keratosis": {"severity":"medium", "description":"Sun-damaged precancerous lesion.", "precautions":"Sun protection; dermatology review.", "medications":"Cryotherapy, topical 5-FU or diclofenac."}
}

# -------------------------
# Hashing helpers (PBKDF2 fallback)
# -------------------------
def hash_password(pw: str) -> str:
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac('sha256', pw.encode('utf-8'), salt, 200000)
    return "pbkdf2$" + binascii.hexlify(salt).decode() + "$" + binascii.hexlify(key).decode()

def check_password(pw: str, stored: str) -> bool:
    try:
        tag, salt_hex, key_hex = stored.split("$")
        salt = binascii.unhexlify(salt_hex)
        key = binascii.unhexlify(key_hex)
        new_key = hashlib.pbkdf2_hmac('sha256', pw.encode('utf-8'), salt, 200000)
        # compare hex strings explicitly (safer and clear)
        return binascii.hexlify(new_key).decode() == binascii.hexlify(key).decode()
    except Exception:
        return False

# -------------------------
# Local session fallback stores
# -------------------------
if "local_users" not in st.session_state:
    st.session_state["local_users"] = {}
if "local_records" not in st.session_state:
    st.session_state["local_records"] = []
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

def local_get_user(email: str) -> Optional[dict]:
    return st.session_state.get("local_users", {}).get(email)

# -------------------------
# Firebase helpers (defensive)
# -------------------------
def upload_image_to_storage(path: str, image_bytes: bytes, content_type="image/jpeg") -> Optional[str]:
    if not bucket:
        return None
    try:
        blob = bucket.blob(path)
        # upload from bytes
        blob.upload_from_string(image_bytes, content_type=content_type)
        try:
            blob.make_public()
            return blob.public_url
        except Exception:
            return f"gs://{blob.bucket.name}/{blob.name}"
    except Exception as e:
        # do not crash app
        st.error(f"Upload failed: {e}")
        return None

def save_record_cloud(record: dict) -> bool:
    if not db:
        return False
    try:
        db.collection("diagnoses").add(record)
        return True
    except Exception as e:
        st.error(f"Cloud save failed: {e}")
        return False

# -------------------------
# Diagnosis backends (fixed HF request usage)
# -------------------------
def dummy_predict(image_bytes: bytes) -> Tuple[str, float]:
    return "Nevus", 0.42

def hf_predict(image_bytes: bytes) -> Tuple[str, float]:
    if not HF_API_TOKEN or not HF_MODEL:
        return dummy_predict(image_bytes)
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    api_url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    try:
        # files expects tuple like ("file", ("filename", data, "image/jpeg"))
        files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
        response = requests.post(api_url, headers=headers, files=files, timeout=60)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            label = data[0].get("label") or data[0].get("class") or str(data[0])
            score = data[0].get("score", 0.0)
            return label, float(score)
        # sometimes HF returns dict with label keys
        if isinstance(data, dict):
            # try common patterns
            if "error" in data:
                st.warning(f"HF error: {data.get('error')}")
                return dummy_predict(image_bytes)
            # fallback: return stringified body
            return str(data), 0.0
        return str(data), 0.0
    except Exception as e:
        # do not crash app; warn user
        st.warning(f"Hugging Face inference failed: {e}")
        return dummy_predict(image_bytes)

def genai_vision_predict(image_bytes: bytes) -> Tuple[str, float]:
    if not USE_GENAI or genai is None:
        return dummy_predict(image_bytes)
    try:
        # Build a short prompt and send single image content if SDK supports mixed inputs.
        model = genai.GenerativeModel("models/gemini-2.5-pro")
        prompt = "You are an experienced dermatologist. Look at the image and provide a single-line disease label only."
        # Use safest API shape: if SDK supports images, it may accept dict entries — catch exceptions
        response = model.generate_content([prompt, {"mime_type":"image/jpeg", "data": image_bytes}])
        text = ""
        if hasattr(response, "text") and response.text:
            text = response.text.strip()
        elif isinstance(response, dict):
            text = response.get("content", "")
        else:
            text = str(response)
        label = text.splitlines()[0].strip() if text else "Unknown"
        return label, 0.0
    except Exception as e:
        st.warning(f"Gemini vision inference failed: {e}")
        return dummy_predict(image_bytes)

def diagnose_image(image_bytes: bytes) -> Tuple[str, float]:
    # priority: Gemini vision -> HF model -> dummy
    if USE_GENAI and genai is not None:
        try:
            return genai_vision_predict(image_bytes)
        except Exception:
            pass
    if HF_API_TOKEN and HF_MODEL:
        try:
            return hf_predict(image_bytes)
        except Exception:
            pass
    return dummy_predict(image_bytes)

# -------------------------
# Chat / QA (descriptive)
# -------------------------
def genai_answer_question(disease_label: str, question: str) -> str:
    if not USE_GENAI or genai is None:
        info = DISEASE_INFO.get(disease_label, {})
        return (f"Summary for {disease_label}:\n\n"
                f"{info.get('description','No description available.')}\n\n"
                f"Precautions: {info.get('precautions','Follow general precautions.')}\n\n"
                f"If symptoms are severe or rapidly changing, consult a dermatologist.")
    try:
        model = genai.GenerativeModel("models/gemini-2.5-pro")
        system_prompt = (
            "You are a senior dermatologist and teacher. Answer in clear, structured sections. "
            "Be evidence-based, cautious, and include: (1) short summary, (2) likely causes, "
            "(3) recommended first-line home care, (4) medical treatments used, (5) red flags when to seek a doctor, "
            "(6) common medication classes, and (7) suggested next steps."
        )
        user_prompt = (
            f"Disease: {disease_label}\nPatient question: {question}\n\n"
            "Please provide a detailed, structured reply in numbered sections as described above."
        )
        response = model.generate_content([system_prompt, user_prompt], temperature=0.2, max_output_tokens=800)
        if hasattr(response, "text") and response.text:
            return response.text
        return str(response)
    except Exception as e:
        return f"AI error: {e}\n\n(Showing fallback summary)\n\n" + (DISEASE_INFO.get(disease_label, {}).get("description",""))

# -------------------------
# UI CSS
# -------------------------
st.markdown("""
<style>
:root { --bg:#12021a; --card:#1e0930; --muted:#e6ddff; --accent1:#9b4dff; --accent2:#6e2ee6; }
.stApp { background: linear-gradient(180deg,var(--bg), #1a0530); color:var(--muted); }
.royal-card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border:1px solid rgba(155,77,255,0.08); border-radius:12px; padding:12px; margin-bottom:10px;}
h1,h2,h3 { color:#f7ecff !important; }
.small-muted { color: rgba(230,220,255,0.8); }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Page flow helpers
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

# -------------------------
# Pages
# -------------------------
def page_welcome():
    # Surface firebase/init status here (safe)
    if FIREBASE_SERVICE_ACCOUNT_JSON or FIREBASE_STORAGE_BUCKET:
        if not (db and bucket):
            st.info("Firebase keys provided but connection failed (cloud features disabled).")
    st.markdown('<div class="royal-card">', unsafe_allow_html=True)
    st.title("👑 Royal Skin Diagnosis")
    st.write("Educational tool — not a medical diagnosis.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Login"):
            go_login()
    with c2:
        if st.button("Register"):
            go_register()
    st.markdown("</div>", unsafe_allow_html=True)

def page_register():
    st.header("Create account")
    c1, c2 = st.columns(2)
    with c1:
        name = st.text_input("Full name", key="reg_name")
        email = st.text_input("Email", key="reg_email")
        phone = st.text_input("Phone (optional)", key="reg_phone")
    with c2:
        pw = st.text_input("Password", type="password", key="reg_pw")
        pw2 = st.text_input("Confirm password", type="password", key="reg_pw2")
        avatar = st.file_uploader("Profile picture (optional)", type=["jpg","jpeg","png"], key="reg_avatar")
    if st.button("Register account"):
        if not name or not email or not pw:
            st.error("Fill name, email and password.")
        elif pw != pw2:
            st.error("Passwords do not match.")
        else:
            hashed = hash_password(pw)
            user_doc = {"name": name, "email": email, "phone": phone, "password_hash": hashed, "created_at": datetime.utcnow().isoformat()}
            if avatar:
                try:
                    b = avatar.read()
                    st.session_state["last_avatar_bytes"] = b
                    if FIREBASE_AVAILABLE and bucket:
                        path = f"profiles/{email.replace('@','_at_')}_{int(time.time())}.jpg"
                        url = upload_image_to_storage(path, b, content_type="image/jpeg")
                        if url:
                            user_doc["avatar_url"] = url
                except Exception:
                    # continue with local avatar fallback
                    pass
            if FIREBASE_AVAILABLE and db:
                try:
                    db.collection("users").add(user_doc)
                    st.success("Account created (cloud). Please login.")
                except Exception as e:
                    st.error(f"Cloud save failed: {e}")
                    local_store_user(user_doc)
                    st.success("Saved locally instead.")
            else:
                local_store_user(user_doc)
                st.success("Local account created. (Cloud not configured.)")

def page_login():
    st.header("Login")
    em = st.text_input("Email", key="login_email")
    pw = st.text_input("Password", type="password", key="login_pw")
    if st.button("Login"):
        user_doc = None
        if FIREBASE_AVAILABLE and db:
            try:
                q = db.collection("users").where("email","==",em).limit(1).get()
                if q and len(q) > 0:
                    user_doc = q[0].to_dict()
            except Exception:
                user_doc = None
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

def page_home():
    st.markdown('<div class="royal-card">', unsafe_allow_html=True)
    st.title("Home")
    st.write("Use the sidebar to go to Diagnose or Profile.")
    st.markdown("</div>", unsafe_allow_html=True)

def page_profile():
    if not st.session_state.get("user"):
        st.info("Please login.")
        return
    user = st.session_state["user"]
    st.header("Profile")
    col1, col2 = st.columns([1,3])
    with col1:
        avatar = user.get("avatar")
        if isinstance(avatar, (bytes, bytearray)):
            from io import BytesIO
            try:
                st.image(BytesIO(avatar), width=140)
            except Exception:
                st.image("https://via.placeholder.com/140x140.png?text=Avatar", width=140)
        elif isinstance(avatar, str) and (avatar.startswith("http") or avatar.startswith("gs://")):
            try:
                st.image(avatar, width=140)
            except Exception:
                st.image("https://via.placeholder.com/140x140.png?text=Avatar", width=140)
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
                docs = db.collection("diagnoses").where("user_email","==", user["email"]).order_by("timestamp", direction="DESCENDING").limit(100).stream()
                recs = [d.to_dict() for d in docs]
                if not recs:
                    st.info("No cloud records found for this user.")
                else:
                    for r in recs:
                        with st.expander(f"{r.get('disease')} — {r.get('timestamp','')}"):
                            st.write(f"**Disease:** {r.get('disease')}")
                            st.write(f"**Confidence:** {r.get('confidence')}")
                            if r.get("image_url"):
                                try:
                                    st.image(r.get("image_url"), use_column_width=True)
                                except Exception:
                                    st.write("Image not available.")
                            elif r.get("image_b64"):
                                st.image(base64.b64decode(r.get("image_b64")), use_column_width=True)
                            st.json(r)
            except Exception as e:
                st.error(f"Failed to read cloud records: {e}")
                # fallback to local records display
                recs_local = st.session_state.get("local_records", [])
                if not recs_local:
                    st.info("No local records.")
                else:
                    for r in recs_local[::-1]:
                        with st.expander(f"{r.get('disease')} — {r.get('timestamp','')}"):
                            st.write(f"**Disease:** {r.get('disease')}")
                            st.write(f"**Confidence:** {r.get('confidence')}")
                            if r.get("image_b64"):
                                st.image(base64.b64decode(r.get("image_b64")), use_column_width=True)
                            st.json(r)
        else:
            recs = st.session_state.get("local_records", [])
            if not recs:
                st.info("No local records found. Diagnose & save a record first.")
            else:
                for r in recs[::-1]:
                    with st.expander(f"{r.get('disease')} — {r.get('timestamp','')}"):
                        st.write(f"**Disease:** {r.get('disease')}")
                        st.write(f"**Confidence:** {r.get('confidence')}")
                        if r.get("image_b64"):
                            st.image(base64.b64decode(r.get("image_b64")), use_column_width=True)
                        st.json(r)

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

        if st.button("Analyze image"):
            with st.spinner("Analyzing..."):
                label, conf = diagnose_image(img_bytes)
                st.success(f"Predicted: **{label}** (confidence {conf:.2f})")
                st.session_state["current_diagnosis"] = label
                st.session_state["current_confidence"] = conf
                info = DISEASE_INFO.get(label, {})
                st.markdown(f"**Description:** {info.get('description','No description available.')}")
                st.markdown(f"**Precautions:** {info.get('precautions','Follow general precautions.')}")
                st.markdown(f"**Treatment / Medications:** {info.get('medications','Consult a dermatologist.')}")
                severity = info.get("severity","low")
                if severity == "high" or conf > 0.85:
                    st.warning("This looks potentially serious. Please consult a dermatologist promptly.")
                    st.markdown("[Book a dermatologist on Practo](https://www.practo.com/dermatologist)")

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
                            try:
                                path = f"diagnoses/{st.session_state['user']['email'].replace('@','_at_')}/{int(time.time())}.jpg"
                                url = upload_image_to_storage(path, img_bytes, content_type="image/jpeg")
                                if url:
                                    rec["image_url"] = url
                            except Exception:
                                pass
                        else:
                            rec["image_b64"] = base64.b64encode(img_bytes).decode("utf-8")

                        saved_cloud = False
                        if FIREBASE_AVAILABLE and db:
                            try:
                                db.collection("diagnoses").add(rec)
                                saved_cloud = True
                                st.success("Saved to cloud.")
                            except Exception as e:
                                st.error(f"Cloud save failed: {e}\nSaved locally instead.")
                        if not saved_cloud:
                            st.session_state["local_records"].append(rec)
                            st.success("Saved locally.")
    else:
        st.info("Upload an image to diagnose.")

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
            st.session_state["page"] = "welcome"
        elif choice == "Login":
            st.session_state["page"] = "login"
        elif choice == "Register":
            st.session_state["page"] = "register"

# -------------------------
# Main
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

def footer():
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div style="text-align:center;color:rgba(230,220,255,0.7)">Built with ❤️ — educational tool only. Not a medical diagnosis.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    footer()
