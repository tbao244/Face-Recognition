import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pickle
import time
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Face Recognition",
    page_icon="📸",
    layout="centered" 
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* Dark industrial theme */
.stApp { background-color: #0d0f14; color: #e8eaf0; }

/* Title */
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem; font-weight: 700;
    color: #ffffff; letter-spacing: -0.5px;
    margin-bottom: 0.2rem;
}
.hero-subtitle {
    font-size: 0.9rem; color: #6b7280;
    font-weight: 300; margin-bottom: 1.5rem;
}

/* Metric card */
.metric-card {
    background: #13161e; border: 1px solid #1e2230;
    border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.75rem;
}
.metric-label {
    font-size: 0.72rem; text-transform: uppercase;
    letter-spacing: 1px; color: #6b7280;
    font-family: 'Space Mono', monospace; margin-bottom: 4px;
}
.metric-value {
    font-size: 1.4rem; font-weight: 600;
    color: #f0f2f8; font-family: 'Space Mono', monospace;
}
.metric-value.highlight { color: #10b981; }
.metric-value.warn      { color: #f59e0b; }

/* Prediction box */
.pred-box {
    background: #13161e; border: 1px solid #1e2230;
    border-left: 3px solid #10b981; border-radius: 8px;
    padding: 0.9rem 1.1rem; margin-top: 0.5rem;
}
.pred-name { font-family: 'Space Mono', monospace; font-size: 1.1rem; font-weight: 700; color: #f0f2f8; }
.pred-conf { font-size: 0.82rem; color: #6b7280; margin-top: 2px; }
.pred-bar-bg { background: #1e2230; border-radius: 4px; height: 5px; margin-top: 8px; overflow: hidden; }
.pred-bar-fg { height: 5px; border-radius: 4px; background: linear-gradient(90deg, #10b981, #34d399); }

/* Video frame */
div[data-testid="stImage"] img { border-radius: 10px; border: 1px solid #1e2230; }

/* Checkbox */
.stCheckbox label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important; color: #c8ccd8 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d0f14; }
::-webkit-scrollbar-thumb { background: #1e2230; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# --- LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model('face_model.h5')
    with open('class_names.pkl', 'rb') as f:
        class_indices = pickle.load(f)
    class_names = {v: k for k, v in class_indices.items()}
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    img_size = model.input_shape[1]
    return model, class_names, face_cascade, img_size

resources_ok = False
try:
    model, class_names, face_cascade, IMG_SIZE = load_resources()
    resources_ok = True
except Exception:
    pass

# --- HEADER ---
st.markdown('<div class="hero-title">Face Recognition</div>', unsafe_allow_html=True)
if not resources_ok:
    st.error("❌ Could not load `face_model.h5` or `class_names.pkl`. Make sure both files are in the same folder as `app.py`.")
    st.stop()

# --- VIDEO ---
col_video, col_stats = st.columns([3, 1], gap="large")

with col_video:
    run_webcam = st.checkbox("▶  Activate Webcam", value=False)
    FRAME_WINDOW = st.empty()

    if not run_webcam:
        FRAME_WINDOW.markdown("""
        <div style="background:#13161e;border:1px solid #1e2230;border-radius:10px;
                    height:380px;display:flex;align-items:center;justify-content:center;
                    flex-direction:column;gap:12px;">
            <span style="font-size:3rem;">📷</span>
            <span style="font-family:'Space Mono',monospace;font-size:0.8rem;color:#3a3f50;
                         letter-spacing:1px;text-transform:uppercase;">Camera offline</span>
        </div>""", unsafe_allow_html=True)

with col_stats:
    st.markdown("#### Stats")
    fps_placeholder   = st.empty()
    faces_placeholder = st.empty()
    result_placeholder = st.empty()

    fps_placeholder.markdown("""
    <div class="metric-card">
        <div class="metric-label">FPS</div>
        <div class="metric-value">—</div>
    </div>""", unsafe_allow_html=True)

    faces_placeholder.markdown("""
    <div class="metric-card">
        <div class="metric-label">Faces</div>
        <div class="metric-value">—</div>
    </div>""", unsafe_allow_html=True)

    result_placeholder.markdown("""
    <div class="pred-box" style="border-left-color:#3a3f50;">
        <div class="pred-name" style="color:#3a3f50;">Waiting...</div>
        <div class="pred-conf">No detection yet</div>
        <div class="pred-bar-bg"><div class="pred-bar-fg" style="width:0%"></div></div>
    </div>""", unsafe_allow_html=True)

# --- WEBCAM LOOP ---
if run_webcam:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        st.error("❌ Cannot open webcam. Make sure no other app is using it.")
        st.stop()
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠ Lost webcam feed.")
                break

            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=4, minSize=(60, 60)
            )

            best_name = None
            best_conf = 0.0
            n_faces   = len(faces) if isinstance(faces, np.ndarray) else 0

            for (x, y, w, h) in (faces if n_faces > 0 else []):
                
                padding = 20
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(frame_rgb.shape[1], x + w + padding)
                y2 = min(frame_rgb.shape[0], y + h + padding)
                face_crop = frame_rgb[y1:y2, x1:x2]

                face_resize = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
                face_input = preprocess_input(face_resize.astype("float32"))
                face_input = np.expand_dims(face_input, axis=0)

                predictions = model.predict(face_input, verbose=0)
                class_idx   = int(np.argmax(predictions))
                confidence  = float(np.max(predictions))

                if confidence >= 0.35: 
                    name  = class_names.get(class_idx, "Unknown")
                    color = (16, 185, 129)
                else:
                    name  = "Unknown"
                    color = (239, 68, 68)

                if confidence > best_conf:
                    best_conf = confidence
                    best_name = name

                cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), color, 2)
                label = f"{name}  {confidence*100:.1f}%"
                font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                (tw, lh), _ = cv2.getTextSize(label, font, fs, th)
                pad = 6
                cv2.rectangle(frame_rgb, (x, y - lh - pad*2), (x + tw + pad*2, y), color, -1)
                cv2.putText(frame_rgb, label, (x + pad, y - pad), font, fs, (255, 255, 255), th)

            FRAME_WINDOW.image(frame_rgb, channels="RGB", use_container_width=True)

            fps_color = "highlight" if fps >= 20 else "warn" if fps >= 10 else ""
            fps_placeholder.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">FPS</div>
                <div class="metric-value {fps_color}">{fps:.1f}</div>
            </div>""", unsafe_allow_html=True)

            faces_placeholder.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Faces</div>
                <div class="metric-value {'highlight' if n_faces > 0 else ''}">{n_faces}</div>
            </div>""", unsafe_allow_html=True)

            if best_name and n_faces > 0:
                bar_color  = "#10b981" if best_name != "Unknown" else "#ef4444"
                border_col = "#10b981" if best_name != "Unknown" else "#ef4444"
                result_placeholder.markdown(f"""
                <div class="pred-box" style="border-left-color:{border_col};">
                    <div class="pred-name">{best_name}</div>
                    <div class="pred-conf">Confidence: {best_conf*100:.1f}%</div>
                    <div class="pred-bar-bg">
                        <div class="pred-bar-fg" style="width:{int(best_conf*100)}%;background:{bar_color};"></div>
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                result_placeholder.markdown("""
                <div class="pred-box" style="border-left-color:#3a3f50;">
                    <div class="pred-name" style="color:#3a3f50;">No face</div>
                    <div class="pred-conf">Point camera at a face</div>
                    <div class="pred-bar-bg"><div class="pred-bar-fg" style="width:0%"></div></div>
                </div>""", unsafe_allow_html=True)

    finally:
        cap.release()
