import cv2
import time
import numpy as np

from face_detector import FaceDetector
from emotion_model import EmotionModel
import overlay_utils as ou


def main():
    import traceback

    cap = None
    detector = None
    emotion = None

    try:
        print("Starting webcam…")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Could not open webcam.")
            return

        detector = FaceDetector()
        emotion = EmotionModel(mode="image")
        emotion.dl_backend = "onnx"

        try:
            emotion.load()
        except Exception as e:
            print("Warning: Could not load HuggingFace model. Running fallback.", e)

        fps_smooth = 30
        last_time = time.time()
        scan_y = 0
        display_bbox = None

        label_decay = emotion.recent_decay
        bbox_lerp = emotion.bbox_lerp

        use_dl = False
        dl_backend = emotion.dl_backend

        print("Running real-time emotion analysis…")

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Error reading frame.")
                break

            h, w = frame.shape[:2]
            bbox, lms = detector.detect(frame)

            # --------------------------------------------------------------------
            # BBOX SMOOTHING
            # --------------------------------------------------------------------
            if bbox:
                if display_bbox is None:
                    display_bbox = bbox
                else:
                    x0, y0, w0, h0 = display_bbox
                    x1, y1, w1, h1 = bbox
                    lerp = bbox_lerp
                    display_bbox = (
                        int(x0 + (x1 - x0) * lerp),
                        int(y0 + (y1 - y0) * lerp),
                        int(w0 + (w1 - w0) * lerp),
                        int(h0 + (h1 - h0) * lerp),
                    )
            else:
                display_bbox = None

            # --------------------------------------------------------------------
            # SAFE EMOTION PREDICTION LOGIC
            # --------------------------------------------------------------------
            label, conf, persona, alpha = ("neutral", 0.0, "Calm Sentinel", 1.0)
            top3 = [("neutral", 100.0), ("neutral", 0.0), ("neutral", 0.0)]

            result = None

            if bbox and use_dl:
                try:
                    result = emotion.predict_dl(frame, bbox)
                except Exception:
                    result = None
                    print("DL failed, fallback to landmark model.")

            if result is None and lms:
                try:
                    result = emotion.predict(lms, (h, w))
                except Exception:
                    result = None

            # Handle 4-output or 5-output versions safely
            if result:
                if len(result) == 5:
                    label, conf, persona, alpha, top3 = result
                else:
                    label, conf, persona, alpha = result
                    top3 = [(label, conf * 100), ("neutral", 0), ("neutral", 0)]

            # --------------------------------------------------------------------
            # HUD DRAWING
            # --------------------------------------------------------------------
            if display_bbox:
                ou.draw_rounded_rect(frame, display_bbox, ou.NEON_CYAN, thickness=2, radius=22, glow=True)
                ou.draw_emotion_label(frame, label, conf, persona, display_bbox, alpha)

                # ⭐ NEW HOLOGRAM EMOTION PANEL ⭐
                draw_hologram_panel(frame, display_bbox, top3)

            # Animated scanline
            scan_y = (scan_y + int((time.time() - last_time) * 180)) % h
            ou.draw_scanline(frame, scan_y, ou.NEON_CYAN, 2)

            # Title
            ou.draw_glitch_text(frame, "AI EMOTION NEURAL-HUD", pos=(40, 42))

            # FPS
            now = time.time()
            fps = 1.0 / max(1e-6, (now - last_time))
            last_time = now
            fps_smooth = fps_smooth * 0.85 + fps * 0.15
            ou.draw_fps(frame, fps_smooth)

            cv2.imshow("AI Face Persona", frame)
            key = cv2.waitKey(1) & 0xFF

            # Interactive controls
            if key == ord("+"):
                label_decay = min(0.98, label_decay + 0.03)
                emotion.recent_decay = label_decay
            elif key == ord("-"):
                label_decay = max(0.50, label_decay - 0.03)
                emotion.recent_decay = label_decay
            elif key == ord("]"):
                bbox_lerp = min(0.9, bbox_lerp + 0.05)
                emotion.bbox_lerp = bbox_lerp
            elif key == ord("["):
                bbox_lerp = max(0.02, bbox_lerp - 0.05)
                emotion.bbox_lerp = bbox_lerp
            elif key == ord("d"):
                use_dl = not use_dl
            elif key == ord("m"):
                dl_backend = "onnx" if dl_backend == "deepface" else "deepface"
                emotion.dl_backend = dl_backend
            elif key == ord("s"):
                ou.save_screenshot(frame)

            if key == 27:
                break

    except Exception:
        traceback.print_exc()
        print("Fatal error.")

    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()


# ============================================================================================
#  CYBERPUNK HOLOGRAM – TOP-3 EMOTION PANEL
# ============================================================================================
def draw_hologram_panel(img, bbox, top3):
    x, y, w, h = bbox

    px = x + w + 20
    py = y + 10

    panel_width = 180
    bar_width = 120
    line_height = 28

    overlay = img.copy()

    # Panel background
    cv2.rectangle(
        overlay,
        (px - 10, py - 25),
        (px - 10 + panel_width, py - 25 + line_height * 4),
        (10, 15, 30),
        -1,
    )

    img[:] = cv2.addWeighted(img, 1.0, overlay, 0.25, 0)

    # Title
    cv2.putText(img, "EMOTION MATRIX", (px, py - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, ou.NEON_ACCENT, 2, cv2.LINE_AA)

    # Loop through top3 emotions
    for i, (name, pct) in enumerate(top3):
        y_pos = py + i * line_height

        # Text
        cv2.putText(img, f"{name.capitalize():<10} {pct:.1f}%",
                    (px, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    ou.NEON_CYAN,
                    1,
                    cv2.LINE_AA)

        # Glow color mapping
        glow = (min(255, int(2.3 * pct)), 80, 255)

        # Progress bar (animated)
        bar_len = int(bar_width * (pct / 100))

        cv2.rectangle(img, (px, y_pos + 6), (px + bar_width, y_pos + 16),
                      (40, 40, 40), -1)
        cv2.rectangle(img, (px, y_pos + 6), (px + bar_len, y_pos + 16),
                      glow, -1)


if __name__ == "__main__":
    main()