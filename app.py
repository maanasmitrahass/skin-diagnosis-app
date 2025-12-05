# app.py — Local-only Skin Diagnosis Center (complete)
# NO SUPABASE — no cloud configuration needed.

import os
import io
import base64
import hashlib
import binascii
import random
from datetime import datetime
from typing import Tuple, Dict

import streamlit as st
from PIL import Image

# -----------------------
# Page config - must be first Streamlit call
# -----------------------
st.set_page_config(page_title="Skin Disease Diagnosis Center", layout="wide")

# -----------------------
# Small performance tweak (watchdog polling in some envs)
# -----------------------
os.environ["WATCHDOG_USE_POLLING"] = "1"

# -----------------------
# Session-state boot
# -----------------------
if "local_users" not in st.session_state:
    st.session_state["local_users"] = {}  # email -> user dict

if "local_records" not in st.session_state:
    st.session_state["local_records"] = []  # list of records

if "user" not in st.session_state:
    st.session_state["user"] = None  # logged-in user (dict)

if "page" not in st.session_state:
    st.session_state["page"] = "welcome"

if "current_diagnosis" not in st.session_state:
    st.session_state["current_diagnosis"] = None  # tuple(label, confidence, meta)

# -----------------------
# Password hashing helpers
# -----------------------
def hash_password(pw: str) -> str:
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"), salt, 200_000)
    return "pbkdf2$" + binascii.hexlify(salt).decode() + "$" + binascii.hexlify(key).decode()

def check_password(pw: str, stored: str) -> bool:
    try:
        _, salt_hex, key_hex = stored.split("$")
        salt = binascii.unhexlify(salt_hex)
        key = binascii.unhexlify(key_hex)
        new_key = hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"), salt, 200_000)
        return binascii.hexlify(new_key).decode() == binascii.hexlify(key).decode()
    except Exception:
        return False

# -----------------------
# Dummy disease knowledge base (expand later)
# -----------------------
DISEASE_INFO: Dict[str, Dict[str, str]] = {
    "Melanoma": {
        "description": "Melanoma is a serious form of skin cancer that can spread to other organs.",
        "precautions": "Seek immediate evaluation by a dermatologist if lesion is changing, bleeding, or painful.",
        "medications": "Medical/surgical treatment only. See specialist; not for OTC self-treatment.",
        "severity": "high"
    },
    "Psoriasis": {
        "description": "Chronic inflammatory condition with red, scaly patches.",
        "precautions": "Avoid triggers, keep skin moisturized, consult dermatologist for topical therapy.",
        "medications": "Topical steroids, vitamin D analogues; oral/biologic for severe cases.",
        "severity": "medium"
    },
    "Eczema": {
        "description": "Also called atopic dermatitis — itchy, inflamed skin.",
        "precautions": "Use gentle skincare, moisturize frequently, avoid strong soaps.",
        "medications": "Topical emollients and steroids for flares; see physician if severe.",
        "severity": "medium"
    },
    "Nevus (mole)": {
        "description": "A benign mole. Usually harmless but monitor for changes.",
        "precautions": "Watch for asymmetry, color change, increasing size — see dermatologist if changes noted.",
        "medications": "None usually. Surgical excision only if clinically indicated.",
        "severity": "low"
    },
    "Unknown": {
        "description": "The image did not match known patterns in the local database.",
        "precautions": "Please consult a dermatologist for an accurate diagnosis.",
        "medications": "N/A",
        "severity": "unknown"
    }
}

# -----------------------
# Fake "model" - lightweight deterministic-ish function
# NOTE: This is a placeholder. For real detection you'd plug an ML model.
# -----------------------
def fake_diagnose_image(image_bytes: bytes) -> Tuple[str, float]:
    """
    Return a (label, confidence) pair.
    Deterministic-ish: use image size parity + random seed for variety.
    """
    # Use image length and some randomness to choose label but stable per run
    seed = len(image_bytes) % 9973
    rng = random.Random(seed)
    labels = list(DISEASE_INFO.keys())
    # bias toward nevus if image small, else pick others
    size_factor = len(image_bytes) % 1000
    if size_factor < 200:
        label = "Nevus (mole)"
        confidence = 0.6 + rng.random() * 0.3
    else:
        label = rng.choice(labels)
        confidence = 0.4 + rng.random() * 0.5
    if label not in DISEASE_INFO:
        label = "Unknown"
    return label, float(confidence)

# -----------------------
# UI styling (royal purple theme + light text)
# -----------------------
st.markdown(
    """
    <style>
    :root {
      --bg: #0f0416;
      --panel: rgba(255,255,255,0.03);
      --accent: #d88cff;
      --muted: #cfc2e6;
      --light: #f3eefc;
    }
    .stApp {
      background: linear-gradient(180deg, #0f0416 0%, #1b072a 100%);
      color: var(--light);
    }
    .big-title {
      font-size:48px !important;
      color: var(--light) !important;
      font-weight:700;
    }
    .subtitle {
      color: var(--muted) !important;
      font-size:16px;
      margin-bottom:18px;
    }
    .card {
      background: var(--panel);
      padding: 22px;
      border-radius: 12px;
      border: 1px solid rgba(216,140,255,0.06);
      margin-bottom: 18px;
    }
    .btn-primary {
      background-color: #ffd8ff;
      color: #3c0b3f;
      border-radius: 8px;
      padding: 8px 14px;
      font-weight:600;
    }
    .muted-text { color: var(--muted) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# Navigation helpers
# -----------------------
def go(to: str):
    st.session_state["page"] = to
    st.experimental_rerun()

# -----------------------
# Page components
# -----------------------
def welcome_page():
    left, right = st.columns([1, 2])
    with left:
        st.write("")  # spacer
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="big-title">Skin Disease Diagnosis Center</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Upload an image, get an educational diagnosis and suggested precautions. Not a substitute for medical advice.</div>', unsafe_allow_html=True)
        st.write("")  # spacer
        if st.button("Login", key="welcome_login"):
            go("login")
        st.write(" ")
        st.markdown("</div>", unsafe_allow_html=True)


def register_page():
    st.header("Create an account")
    with st.form("register_form"):
        name = st.text_input("Full name")
        email = st.text_input("Email")
        pw = st.text_input("Password", type="password")
        pw2 = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Register")
    if submitted:
        if not name or not email or not pw:
            st.error("Please fill all fields.")
            return
        if pw != pw2:
            st.error("Passwords do not match.")
            return
        if email in st.session_state["local_users"]:
            st.error("User already exists. Please login.")
            return
        st.session_state["local_users"][email] = {
            "name": name,
            "email": email,
            "password_hash": hash_password(pw)
        }
        st.success("Account created — please login.")
        go("login")


def login_page():
    st.header("Login")
    with st.form("login_form"):
        email = st.text_input("Email")
        pw = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
    if submitted:
        user = st.session_state["local_users"].get(email)
        if not user:
            st.error("No such user. Please register.")
            return
        if check_password(pw, user["password_hash"]):
            st.session_state["user"] = user
            st.success("Login successful.")
            go("home")
        else:
            st.error("Incorrect password.")
    st.write("Don't have an account? ")
    if st.button("Create account"):
        go("register")


def home_page():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Welcome back, " + (st.session_state["user"]["name"] if st.session_state["user"] else "guest"))
    st.write("Use the left menu to Diagnose or view Profile.")
    st.markdown("</div>", unsafe_allow_html=True)


def diagnose_page():
    st.header("Diagnose Skin Condition")
    uploaded = st.file_uploader("Upload a clear photo of the affected skin area (jpg, png)", type=["jpg", "jpeg", "png"])
    if uploaded:
        try:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, use_column_width=True)
            # get raw bytes
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            bytes_data = buf.getvalue()
        except Exception as e:
            st.error("Cannot read image: " + str(e))
            return

        if st.button("Analyze"):
            label, conf = fake_diagnose_image(bytes_data)
            st.session_state["current_diagnosis"] = (label, conf, bytes_data)
            st.success(f"Diagnosis: {label}  —  Confidence: {conf:.2f}")
            meta = DISEASE_INFO.get(label, DISEASE_INFO["Unknown"])
            st.markdown("**Description:** " + meta.get("description", ""))
            st.markdown("**Precautions:** " + meta.get("precautions", ""))
            st.markdown("**Medications / Notes:** " + meta.get("medications", ""))

            if meta.get("severity") == "high":
                st.warning("This looks potentially serious. Consider consulting a medical professional immediately.")
                st.markdown("[Book a consultation on Practo](https://www.practo.com)")

        # Save record button
        if st.session_state.get("current_diagnosis"):
            if st.button("Save this record"):
                label, conf, bytes_data = st.session_state["current_diagnosis"]
                rec = {
                    "diagnosis": label,
                    "confidence": float(conf),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "image_b64": base64.b64encode(bytes_data).decode("utf-8"),
                    "user_email": st.session_state["user"]["email"] if st.session_state["user"] else None
                }
                st.session_state["local_records"].append(rec)
                st.success("Record saved locally in this session.")


def profile_page():
    st.header("Profile")
    user = st.session_state.get("user")
    if user:
        st.markdown(f"**Name:** {user['name']}")
        st.markdown(f"**Email:** {user['email']}")
    else:
        st.info("You are not logged in.")

    st.subheader("Saved Records (local session)")
    records = [r for r in st.session_state["local_records"] if (user is None or r.get("user_email") == user.get("email"))]
    if not records:
        st.info("No saved records in this session.")
    else:
        for i, r in enumerate(reversed(records)):
            with st.expander(f"{r['diagnosis']} — {r['timestamp']}"):
                st.write(f"**Confidence:** {r['confidence']:.2f}")
                imgdata = base64.b64decode(r["image_b64"])
                st.image(imgdata, use_column_width=True)

    # Export CSV button
    if records:
        import csv
        csv_buf = io.StringIO()
        writer = csv.writer(csv_buf)
        writer.writerow(["diagnosis", "confidence", "timestamp", "user_email"])
        for r in records:
            writer.writerow([r["diagnosis"], r["confidence"], r["timestamp"], r.get("user_email","")])
        csv_bytes = csv_buf.getvalue().encode()
        st.download_button("Download records CSV", data=csv_bytes, file_name="records.csv", mime="text/csv")


def chatbot_area():
    st.subheader("Ask about the diagnosis")
    prompt = st.text_input("Ask a question to the assistant about the current diagnosis (e.g., 'What is the cure?')", key="chat_input")
    if st.button("Ask"):
        if not st.session_state.get("current_diagnosis"):
            st.info("Please diagnose an image first so the assistant has context.")
        else:
            label, conf, _ = st.session_state["current_diagnosis"]
            meta = DISEASE_INFO.get(label, DISEASE_INFO["Unknown"])
            lower = prompt.lower()
            answer = ""
            if "cure" in lower or "treatment" in lower or "medicat" in lower:
                answer = f"Suggested approach: {meta.get('medications','Not available')}. " \
                         "This is educational — for actual prescription or surgery, consult a dermatologist."
            elif "precaution" in lower or "prevent" in lower:
                answer = f"Precautions: {meta.get('precautions','Follow up with a physician')}"
            elif "what" in lower or "describe" in lower or "info" in lower:
                answer = f"{meta.get('description','No additional info')}"
            else:
                # fallback
                answer = f"I suggest: {meta.get('description','Consult a dermatologist')} — for severe cases use Practo: https://www.practo.com"
            st.write(answer)

# -----------------------
# Sidebar (routing & auth)
# -----------------------
def render_sidebar():
    st.sidebar.title("Navigation")
    if st.session_state["user"]:
        choice = st.sidebar.radio("Menu", ["Home", "Diagnose", "Profile", "Logout"])
        if choice == "Home":
            go("home")
        elif choice == "Diagnose":
            go("diagnose")
        elif choice == "Profile":
            go("profile")
        elif choice == "Logout":
            st.session_state["user"] = None
            st.success("Logged out.")
            go("welcome")
    else:
        choice = st.sidebar.radio("Menu", ["Welcome", "Login", "Register"])
        if choice == "Welcome":
            go("welcome")
        elif choice == "Login":
            go("login")
        elif choice == "Register":
            go("register")

# -----------------------
# Router / main
# -----------------------
def main():
    render_sidebar()
    page = st.session_state.get("page", "welcome")
    if page == "welcome":
        welcome_page()
    elif page == "login":
        login_page()
    elif page == "register":
        register_page()
    elif page == "home":
        home_page()
    elif page == "diagnose":
        diagnose_page()
    elif page == "profile":
        profile_page()

    # chatbot block always visible near bottom of main area
    st.markdown("---")
    chatbot_area()

if __name__ == "__main__":
    main()
