# app.py
"""
Royal Skin Diagnosis — Streamlit app with Supabase persistence.
- Reads SUPABASE_URL, SUPABASE_KEY, SUPABASE_BUCKET from Streamlit secrets
- Optional GENAI (Gemini) and HF (Hugging Face) integration for AI answers
- Local fallback storage if cloud isn't configured
- PBKDF2 password hashing
"""

import os
import io
import time
import base64
import hashlib
import binascii
from datetime import datetime

import streamlit as st
from PIL import Image
import requests

# -------------------------
# Page config (must be first Streamlit call)
# -------------------------
st.set_page_config(page_title="Royal Skin Diagnosis", layout="centered", initial_sidebar_state="expanded")

# -------------------------
# Read secrets
# -------------------------
SUPABASE_URL = st.secrets.get("SUPABASE_URL") if "SUPABASE_URL" in st.secrets else os.environ.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY") if "SUPABASE_KEY" in st.secrets else os.environ.get("SUPABASE_KEY")
SUPABASE_BUCKET = st.secrets.get("SUPABASE_BUCKET") if "SUPABASE_BUCKET" in st.secrets else os.environ.get("SUPABASE_BUCKET", "records")

GENAI_API_KEY = st.secrets.get("GENAI_API_KEY") if "GENAI_API_KEY" in st.secrets else os.environ.get("GENAI_API_KEY")
HF_API_TOKEN = st.secrets.get("HF_API_TOKEN") if "HF_API_TOKEN" in st.secrets else os.environ.get("HF_API_TOKEN")
HF_MODEL = st.secrets.get("HF_MODEL") if "HF_MODEL" in st.secrets else os.environ.get("HF_MODEL")

# -------------------------
# Try to init Gemini (optional)
# -------------------------
USE_GENAI = False
if GENAI_API_KEY:
    try:
        import google.generativeai as genai  # optional
        genai.configure(api_key=GENAI_API_KEY)
        USE_GENAI = True
    except Exception:
        USE_GENAI = False

# -------------------------
# Supabase init (cached)
# -------------------------
SUPABASE_AVAILABLE = False
supabase = None

@st.cache_resource(show_spinner=False)
def init_supabase_client():
    global SUPABASE_AVAILABLE
    if not (SUPABASE_URL and SUPABASE_KEY):
        return None
    try:
        from supabase import create_client
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        SUPABASE_AVAILABLE = True
        return client
    except Exception as e:
        st.warning("Supabase client init failed — running in local mode.")
        return None

supabase = init_supabase_client()

# -------------------------
# Fallback disease info
# -------------------------
DISEASE_INFO = {
    "Melanoma": {"severity":"high", "description":"Potentially life-threatening skin cancer.", "precautions":"Avoid sun; see dermatologist urgently.", "medications":"Surgery/oncology-managed therapies."},
    "Psoriasis": {"severity":"low", "description":"Chronic autoimmune scaly plaques.", "precautions":"Moisturize; avoid triggers.", "medications":"Topical steroids, vitamin D analogues."},
    "Nevus": {"severity":"low", "description":"Common benign mole.", "precautions":"Monitor for changes.", "medications":"Usually none."},
    "Actinic Keratosis": {"severity":"medium", "description":"Sun-damaged precancerous lesion.", "precautions":"Sun protection; dermatology review.", "medications":"Cryotherapy or topical therapies."}
}

# -------------------------
# Password hashing (PBKDF2)
# -------------------------
def hash_password(pw: str) -> str:
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac('sha256', pw.encode('utf-8'), salt, 200_000)
    return "pbkdf2$" + binascii.hexlify(salt).decode() + "$" + binascii.hexlify(key).decode()

def check_password(pw: str, stored: str) -> bool:
    try:
        _, salt_hex, key_hex = stored.split("$")
        salt = binascii.unhexlify(salt_hex)
        key = binascii.unhexlify(key_hex)
        new_key = hashlib.pbkdf2_hmac('sha256', pw.encode('utf-8'), salt, 200_000)
        return binascii.hexlify(new_key) == key
    except Exception:
        return False

# -------------------------
# Local fallback session stores
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

def local_get_user(email: str):
    return st.session_state.get("local_users", {}).get(email)

# -------------------------
# Supabase helpers
# -------------------------
def upload_image_to_supabase(path: str, image_bytes: bytes, content_type="image/jpeg"):
    if not supabase:
        return None
    try:
        bucket = supabase.storage().from_(SUPABASE_BUCKET)
        bucket.upload(path, io.BytesIO(image_bytes), content_type=content_type)
        public_url = f"{SUPABASE_URL.rstrip('/')}/storage/v1/object/public/{SUPABASE_BUCKET}/{path}"
        return public_url
    except Exception as e:
        st.warning(f"Supabase upload failed: {e}")
        return None

def save_record_supabase(record: dict):
    if not supabase:
        return False
    try:
        res = supabase.table("records").insert(record).execute()
        if hasattr(res, "error") and res.error:
            st.warning(f"Supabase insert error: {res.error}")
            return False
        return True
    except Exception as e:
        st.warning(f"Supabase save failed: {e}")
        return False

def save_user_supabase(user_doc: dict):
    if not supabase:
        return False
    try:
        res = supabase.table("users").insert(user_doc).execute()
        if hasattr(res, "error") and res.error:
            st.warning(f"Supabase user insert error: {res.error}")
            return False
        return True
    except Exception as e:
        st.warning(f"Supabase user save failed: {e}")
        return False

def get_user_supabase(email: str):
    if not supabase:
        return None
    try:
        res = supabase.table("users").select("*").eq("email", email).limit(1).execute()
        data = getattr(res, "data", None) or (res.json() if hasattr(res, "json") else None)
        if data and isinstance(data, list) and len(data) > 0:
            return data[0]
        return None
    except Exception as e:
        st.warning(f"Supabase read user failed: {e}")
        return None

def get_records_supabase(email: str):
    if not supabase:
        return []
    try:
        res = supabase.table("records").select("*").eq("user_email", email).order("timestamp", desc=True).execute()
        data = getattr(res, "data", None) or (res.json() if hasattr(res, "json") else None)
        return data or []
    except Exception as e:
        st.warning(f"Supabase read records failed: {e}")
        return []

# -------------------------
# Predictors (dummy / HF / optional GenAI)
# -------------------------
def dummy_predict(image_bytes=None):
    return "Nevus", 0.42

def hf_predict(image_bytes: bytes):
    if not HF_API_TOKEN or not HF_MODEL:
        return dummy_predict(image_bytes)
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    api_url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    try:
        response = requests.post(api_url, headers=headers, files={"file": ("image.jpg", image_bytes)}, timeout=60)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            label = data[0].get("label") or data[0].get("class") or str(data[0])
            score = data[0].get("score", 0.0)
            return label, float(score)
        return str(data), 0.0
    except Exception as e:
        st.warning(f"Hugging Face inference failed: {e}")
        return dummy_predict(image_bytes)

def genai_vision_predict(image_bytes: bytes):
    if not USE_GENAI:
        return dummy_predict(image_bytes)
    try:
        model = genai.GenerativeModel("models/gemini-2.5-pro")
        prompt = "You are an experienced dermatologist. Look at the image and provide a single-line disease label only."
        response = model.generate_content([prompt, {"mime_type":"image/jpeg", "data": image_bytes}])
        if hasattr(response, "text") and response.text:
            text = response.text.strip()
        elif isinstance(response, dict):
            text = response.get("content", "")
        else:
            text = str(response)
        label = text.splitlines()[0] if text else "Unknown"
        return label, 0.0
    except Exception as e:
        st.warning(f"GenAI vision failed: {e}")
        return dummy_predict(image_bytes)

def diagnose_image(image_bytes: bytes):
    if USE_GENAI:
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
# AI answer
# -------------------------
def genai_answer_question(disease_label: str, question: str):
    if not USE_GENAI:
        info = DISEASE_INFO.get(disease_label, {})
        return (f"**Summary for {disease_label}:**\n\n"
                f"{info.get('description','No description available.')}\n\n"
                f"**Precautions:** {info.get('precautions','Follow general precautions.')}\n\n"
                f"**Treatment / Medications:** {info.get('medications','Consult a dermatologist.')}\n\n"
                f"If symptoms are severe or rapidly changing, consult a dermatologist.")
    try:
        model = genai.GenerativeModel("models/gemini-2.5-pro")
        system_prompt = (
            "You are a senior dermatologist and teacher. Answer in clear, structured sections. "
            "Be evidence-based, cautious, and include: (1) short summary, (2) likely causes, "
            "(3) recommended first-line home care, (4) medical treatments used, (5) red flags when to seek a doctor, "
            "(6) common medication classes, and (7) suggested next steps."
        )
        user_prompt = f"Disease: {disease_label}\nUser question: {question}\nPlease answer in numbered sections."
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
.big-btn { background: linear-gradient(90deg,#ffecf8, #f0e6ff); border-radius:10px; padding:8px; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Pages and flow
# -------------------------
def go(page_name):
    st.session_state["page"] = page_name
    st.experimental_rerun()

def page_welcome():
    st.markdown('<div class="royal-card">', unsafe_allow_html=True)
    st.title("👑 Royal Skin Diagnosis")
    st.write("Educational tool — not a medical diagnosis.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Login"):
            go("login")
    with c2:
        if st.button("Register"):
            go("register")
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
                b = avatar.read()
                st.session_state["last_avatar_bytes"] = b
                if SUPABASE_AVAILABLE and supabase:
                    try:
                        path = f"profiles/{email.replace('@','_at_')}_{int(time.time())}.jpg"
                        url = upload_image_to_supabase(path, b, content_type="image/jpeg")
                        if url:
                            user_doc["avatar_url"] = url
                    except Exception:
                        pass
            if SUPABASE_AVAILABLE and supabase:
                ok = save_user_supabase(user_doc)
                if ok:
                    st.success("Account created in cloud. Please log in.")
                else:
                    st.error("Cloud save failed; saved locally.")
                    local_store_user(user_doc)
            else:
                local_store_user(user_doc)
                st.success("Local account created. (Cloud not configured.)")

def page_login():
    st.header("Login")
    em = st.text_input("Email", key="login_email")
    pw = st.text_input("Password", type="password", key="login_pw")
    if st.button("Login"):
        user_doc = None
        if SUPABASE_AVAILABLE and supabase:
            user_doc = get_user_supabase(em)
        if not user_doc:
            user_doc = local_get_user(em)
        if not user_doc:
            st.error("No such user. Please register.")
        else:
            if check_password(pw, user_doc["password_hash"]):
                st.session_state["user"] = {"email": user_doc["email"], "name": user_doc.get("name"), "phone": user_doc.get("phone"), "avatar": user_doc.get("avatar_url") if SUPABASE_AVAILABLE else st.session_state.get("last_avatar_bytes")}
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
            st.image(avatar, width=140)
        elif isinstance(avatar, str) and avatar.startswith("http"):
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
        st.write("Cloud: " + ("Connected" if SUPABASE_AVAILABLE else "Not configured (local mode)"))

    if st.button("View saved records"):
        if SUPABASE_AVAILABLE and supabase:
            recs = get_records_supabase(user["email"])
            if not recs:
                st.info("No cloud records found for this user.")
            else:
                for r in recs:
                    with st.expander(f"{r.get('diagnosis')} — {r.get('timestamp','')}"):
                        st.write(f"**Diagnosis:** {r.get('diagnosis')}")
                        st.write(f"**Confidence:** {r.get('confidence')}")
                        if r.get("image_url"):
                            try:
                                st.image(r.get("image_url"), use_column_width=True)
                            except Exception:
                                st.write("Image not available.")
                        elif r.get("image_b64"):
                            st.image(base64.b64decode(r.get("image_b64")), use_column_width=True)
                        st.json(r)
        else:
            recs = st.session_state.get("local_records", [])
            if not recs:
                st.info("No local records found.")
            else:
                for r in recs[::-1]:
                    with st.expander(f"{r.get('diagnosis')} — {r.get('timestamp','')}"):
                        st.write(f"**Diagnosis:** {r.get('diagnosis')}")
                        st.write(f"**Confidence:** {r.get('confidence')}")
                        if r.get("image_b64"):
                            st.image(base64.b64decode(r.get("image_b64")), use_column_width=True)
                        st.json(r)

def page_diagnose():
    st.header("🔬 Diagnose Skin Condition")
    uploaded = st.file_uploader("Upload skin image (jpg/png)", type=["jpg","jpeg","png"])
    if uploaded:
        try:
            img = Image.open(uploaded).convert("RGB")
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
                if severity == "high" or (isinstance(conf, float) and conf > 0.85):
                    st.warning("This looks potentially serious. Please consult a dermatologist promptly.")
                    st.markdown("[Book a dermatologist on Practo](https://www.practo.com/dermatologist)")

                if st.button("Save this record"):
                    if not st.session_state.get("user"):
                        st.error("Log in to save records.")
                    else:
                        rec = {
                            "user_email": st.session_state["user"]["email"],
                            "diagnosis": label,
                            "confidence": float(conf),
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        saved_cloud = False
                        if SUPABASE_AVAILABLE and supabase:
                            try:
                                path = f"records/{st.session_state['user']['email'].replace('@','_at_')}/{int(time.time())}.jpg"
                                url = upload_image_to_supabase(path, img_bytes, content_type="image/jpeg")
                                if url:
                                    rec["image_url"] = url
                            except Exception:
                                pass

                            ok = save_record_supabase(rec)
                            if ok:
                                saved_cloud = True
                                st.success("Saved to cloud.")
                            else:
                                st.error("Cloud save failed; saved locally.")
                        if not saved_cloud:
                            rec["image_b64"] = base64.b64encode(img_bytes).decode("utf-8")
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
