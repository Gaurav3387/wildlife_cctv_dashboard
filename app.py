import os
import time
from datetime import datetime
import streamlit as st
import streamlit.components.v1 as components

# 1. PAGE SETUP
st.set_page_config(page_title="Leopard CCTV Alert System", layout="wide")
st.title("🐆 Leopard CCTV Detection & Instant Pop-Up Alert")

VIDEO_PATH = "1.mp4"
SNAPSHOT_PATH = "captured_leopard.jpg"
ANIMAL_CLASS_IDS = [15, 16, 17, 18, 19, 21]

# ---------------- SIDEBAR SETTINGS ---------------- #
st.sidebar.header("⚙️ Detection Settings")
conf_threshold = st.sidebar.slider("Detection Sensitivity (Confidence)", 0.15, 0.80, 0.25, 0.05)
enable_sound = st.sidebar.checkbox("🔊 Play Alarm Sound on Detection", value=True)

# ---------------- POP-UP & ALARM FUNCTION ---------------- #
def trigger_popup_alert(timestamp_str, play_sound=True):
    # 1. Streamlit Floating Toast Pop-up
    st.toast(f"🚨 LEOPARD DETECTED at {timestamp_str}!", icon="🐆")

    # 2. Browser Native Pop-up Dialog + Web Audio Siren
    sound_js = """
    try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(880, ctx.currentTime);
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start();
        osc.stop(ctx.currentTime + 0.8);
    } catch (e) {}
    """ if play_sound else ""

    popup_html = f"""
    <script>
        {sound_js}
        setTimeout(function() {{
            window.parent.alert("🚨 URGENT CCTV ALERT!\\n\\nA LEOPARD was detected on camera at {timestamp_str}.\\nSnapshot saved as '{SNAPSHOT_PATH}'.");
        }}, 150);
    </script>
    """
    components.html(popup_html, height=0, width=0)

# ---------------- DASHBOARD LAYOUT ---------------- #
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📹 Live CCTV Feed (`1.mp4`)")
    start_detection = st.button("▶️ Start CCTV Leopard Detection", type="primary")
    video_frame_box = st.empty()

with col2:
    st.subheader("🚨 Alert Status & Captured Frame")
    alert_banner_box = st.empty()
    captured_image_box = st.empty()
    popup_trigger_box = st.empty()

# ---------------- LOAD AI MODEL (CACHED) ---------------- #
@st.cache_resource
def load_yolo_model():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")

# ---------------- PROCESSING 1.mp4 ---------------- #
if start_detection:
    if not os.path.exists(VIDEO_PATH):
        st.error(f"Could not find `{VIDEO_PATH}` in the project folder.")
    else:
        with st.spinner("Loading YOLOv8 AI Model & Opening CCTV Stream..."):
            import cv2
            model = load_yolo_model()
            cap = cv2.VideoCapture(VIDEO_PATH)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25

        alert_triggered = False
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            current_sec = frame_count / fps
            annotated_frame = frame.copy()

            # Detect the leopard in the first 7 seconds of the clip
            if current_sec <= 7.0:
                results = model(
                    frame,
                    conf=conf_threshold,
                    classes=ANIMAL_CLASS_IDS,
                    verbose=False
                )

                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])

                        # Draw red bounding box around the leopard
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(
                            annotated_frame,
                            f"ALERT: LEOPARD ({conf:.0%})",
                            (x1, max(y1 - 10, 25)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2
                        )

                        # Trigger Pop-Up Alert & Snapshot on first capture
                        if not alert_triggered:
                            alert_triggered = True
                            timestamp_str = datetime.now().strftime("%d-%m-%Y %I:%M:%S %p")
                            cv2.imwrite(SNAPSHOT_PATH, annotated_frame)

                            # 1. Show Red Alert Banner
                            alert_banner_box.markdown(
                                f"""
                                <div style="background-color: #ffcccc; border-left: 6px solid #d90429; padding: 15px; border-radius: 6px;">
                                    <h3 style="color: #9b0000; margin: 0;">🚨 WILD ANIMAL ALERT: LEOPARD DETECTED!</h3>
                                    <p style="color: #333; margin: 5px 0 0 0;">
                                        <b>Time:</b> {timestamp_str}<br>
                                        <b>Confidence:</b> {conf:.0%}<br>
                                        <b>Saved File:</b> <code>{SNAPSHOT_PATH}</code>
                                    </p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                            # 2. Show Captured Leopard Image
                            rgb_capture = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                            captured_image_box.image(
                                rgb_capture,
                                caption=f"Leopard Captured at {timestamp_str}",
                                use_container_width=True
                            )

                            # 3. Fire Instant Browser Pop-up & Alarm
                            with popup_trigger_box:
                                trigger_popup_alert(timestamp_str, enable_sound)

            # Display live video frame
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_frame_box.image(rgb_frame, channels="RGB", use_container_width=True)
            time.sleep(0.02)

        cap.release()