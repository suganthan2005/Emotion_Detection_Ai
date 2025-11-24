import time
from typing import List, Tuple, Optional
import numpy as np
import threading
from collections import deque, defaultdict
import os
import cv2

pipeline = None
DeepFace = None
onnxruntime = None

try:
    import onnxruntime as _ort
    onnxruntime = _ort
except Exception:
    onnxruntime = None


class EmotionModel:
    """EmotionModel wraps a HF text-classification pipeline for go-emotions.

    It converts simple numeric facial metrics into a textual description (e.g. "smiling, eyes open")
    and asks the text classifier to predict. This is a pragmatic adapter so we can use the
    requested model for visual emotion via heuristics.

    NOTE: predict(...) and predict_dl(...) now return an extra `top3` list:
      (label, conf, persona, alpha, top3)
    where top3 is List[Tuple[label, percent_float]]
    """

    PERSONA_MAP = {
        'joy': 'AI Dreamer',
        'surprise': 'Curious Synth',
        'anger': 'Chrome Rebel',
        'sadness': 'Neon Loner',
        'confused': 'Quantum Puzzler',
        'happy': 'Sunset Coder',
        'excited': 'Pulse Rider',
        'fear': 'Circuit Warden',
        'disgust': 'Acid Critic',
        'neutral': 'Calm Sentinel'
    }

    def __init__(self, model_name: str = "joeddav/distilbert-base-uncased-go-emotions", mode: str = "image"):
        """Create EmotionModel.

        Parameters
        - model_name: name of HF model to use if mode=='text'
        - mode: 'text' to use HF text classifier (original), 'image' to use landmark-based
                heuristic classifier, or 'hybrid' to prefer image heuristics and fallback
                to text when available.
        """
        self.model_name = model_name
        self.mode = mode
        self.classifier = None
        self._load_lock = threading.Lock()
        self.last_label = None
        self.display_label = None
        self.display_alpha = 1.0
        self.last_change_time = time.time()
        # smoothing history for labels: recent predicted (label, conf)
        self.recent = deque(maxlen=8)
        # decay factor for recency weighting (0.0..1.0). Smaller -> more weight to recent
        self.recent_decay = 0.85
        # bbox smoothing lerp default (can be tuned from main)
        self.bbox_lerp = 0.22
        # DL backend selection ('deepface' or 'onnx')
        self.dl_backend = 'deepface'
        # cached ONNX session
        self._onnx_sess = None
        # URL to a small ONNX FER+ model (official ONNX models repo). Will be downloaded on demand.
        # Use raw.githubusercontent.com to avoid GitHub redirect issues
        self._onnx_model_url = 'https://raw.githubusercontent.com/onnx/models/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx'

    def load(self):
        # Lazy import transformers.pipeline to avoid heavy imports at module import time
        global pipeline
        try:
            if pipeline is None:
                from transformers import pipeline as _pipeline
                pipeline = _pipeline
        except Exception:
            raise RuntimeError("transformers library not available. Install requirements.txt")

        with self._load_lock:
            if self.classifier is None:
                # return_all_scores so we can pick top label and confidence
                self.classifier = pipeline("text-classification", model=self.model_name, return_all_scores=True)

    def _landmarks_to_text(self, landmarks: List[Tuple[int,int]], image_shape: Tuple[int,int]) -> str:
        """Simple heuristic to describe face as text for the text classifier.

        Uses mouth corners, eye openness and eyebrow slope heuristics to produce a short sentence.
        """
        if not landmarks:
            return "neutral face"

        h, w = image_shape
        # indices from MediaPipe face mesh for lips and eyes (approximate)
        left_mouth = landmarks[61] if len(landmarks) > 61 else landmarks[-1]
        right_mouth = landmarks[291] if len(landmarks) > 291 else landmarks[-1]
        top_lip = landmarks[13] if len(landmarks) > 13 else landmarks[-1]
        bottom_lip = landmarks[14] if len(landmarks) > 14 else landmarks[-1]

        left_eye_top = landmarks[159] if len(landmarks) > 159 else landmarks[-1]
        left_eye_bottom = landmarks[145] if len(landmarks) > 145 else landmarks[-1]
        right_eye_top = landmarks[386] if len(landmarks) > 386 else landmarks[-1]
        right_eye_bottom = landmarks[374] if len(landmarks) > 374 else landmarks[-1]

        mouth_width = np.hypot(right_mouth[0]-left_mouth[0], right_mouth[1]-left_mouth[1])
        mouth_open = np.hypot(top_lip[0]-bottom_lip[0], top_lip[1]-bottom_lip[1])

        eye_left_h = abs(left_eye_top[1] - left_eye_bottom[1])
        eye_right_h = abs(right_eye_top[1] - right_eye_bottom[1])
        eye_avg = (eye_left_h + eye_right_h) / 2.0

        # normalize by face height approx
        face_h = h
        smile_score = mouth_width / (face_h * 0.3 + 1e-6)
        open_score = mouth_open / (face_h * 0.05 + 1e-6)
        eye_open_score = eye_avg / (face_h * 0.03 + 1e-6)

        parts = []
        if smile_score > 0.28 and open_score < 0.06:
            parts.append("smiling")
        elif open_score > 0.06:
            parts.append("mouth open")
        else:
            parts.append("neutral mouth")

        if eye_open_score > 1.2:
            parts.append("eyes wide")
        elif eye_open_score < 0.6:
            parts.append("eyes squint")
        else:
            parts.append("eyes normal")

        desc = ", ".join(parts)
        return desc

    def _predict_from_landmarks(self, landmarks: List[Tuple[int,int]], image_shape: Tuple[int,int]):
        """A stronger image-based heuristic classifier that computes facial features
        (eye aspect ratio, mouth aspect, smile curvature, eyebrow slope) and maps
        them to one of the target emotion labels with a confidence score.

        This is intentionally lightweight and deterministic (no heavy models).
        """
        h, w = image_shape
        if not landmarks:
            return 'neutral', 0.0

        def dist(a, b):
            return np.hypot(a[0]-b[0], a[1]-b[1])

        # pick approximate landmark indices (MediaPipe face mesh)
        # mouth corners
        lm_left = landmarks[61] if len(landmarks) > 61 else landmarks[-1]
        lm_right = landmarks[291] if len(landmarks) > 291 else landmarks[-1]
        top_lip = landmarks[13] if len(landmarks) > 13 else landmarks[-1]
        bottom_lip = landmarks[14] if len(landmarks) > 14 else landmarks[-1]

        # eyes
        left_eye_top = landmarks[159] if len(landmarks) > 159 else landmarks[-1]
        left_eye_bottom = landmarks[145] if len(landmarks) > 145 else landmarks[-1]
        right_eye_top = landmarks[386] if len(landmarks) > 386 else landmarks[-1]
        right_eye_bottom = landmarks[374] if len(landmarks) > 374 else landmarks[-1]

        mouth_w = dist(lm_left, lm_right)
        mouth_h = dist(top_lip, bottom_lip)
        ear = (abs(left_eye_top[1]-left_eye_bottom[1]) + abs(right_eye_top[1]-right_eye_bottom[1]))/2.0

        # normalize by face size (approx as image height)
        norm_w = mouth_w / (h + 1e-6)
        norm_h = mouth_h / (h + 1e-6)
        norm_ear = ear / (h + 1e-6)

        # compute heuristic scores
        smile_score = norm_w - 0.8 * norm_h
        open_mouth_score = norm_h * 3.0
        surprise_score = open_mouth_score + (norm_ear * 2.0)
        squint_score = 1.0 - norm_ear

        # initial label
        label = 'neutral'
        conf = 0.0

        if smile_score > 0.035 and norm_ear < 0.02:
            label = 'happy'
            conf = min(0.95, 0.45 + smile_score*8)
        elif surprise_score > 0.12:
            label = 'surprise'
            conf = min(0.95, 0.3 + (surprise_score-0.12)*3)
        elif abs(norm_ear - 0.03) > 0.02 and abs(norm_w - 0.02) < 0.02:
            # eyebrow asymmetry / slight mouth neutral -> confused
            label = 'confused'
            conf = min(0.85, 0.3 + abs(norm_ear - 0.03)*10)
        elif squint_score > 0.04 and norm_h < 0.02:
            label = 'anger'
            conf = min(0.9, 0.25 + squint_score*6)
        elif norm_h > 0.08 and smile_score > 0.02:
            label = 'excited'
            conf = min(0.95, 0.35 + (norm_h-0.08)*5)
        elif norm_h > 0.06 and smile_score < 0.01:
            label = 'sadness'
            conf = min(0.85, 0.2 + (norm_h-0.06)*4)
        else:
            label = 'neutral'
            conf = 0.5

        return label, float(max(0.0, min(1.0, conf)))

    def _predict_from_deepface(self, frame, bbox: Tuple[int,int,int,int]):
        """Use DeepFace to analyze the face region and return (label, confidence, top3).

        Returns:
            mapped_label, top_conf (0..1), top3_list where top3_list is [(label, percent), ...]
        """
        # import DeepFace lazily to avoid heavy imports/crashes on module import
        global DeepFace
        if DeepFace is None:
            try:
                from deepface import DeepFace as _DeepFace
                DeepFace = _DeepFace
            except Exception:
                raise RuntimeError('DeepFace not available')

        x,y,w,h = bbox
        # safe crop
        H,W = frame.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(W, x+w)
        y2 = min(H, y+h)
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return 'neutral', 0.0, [('neutral', 100.0)]

        try:
            result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
            if isinstance(result, list) and result:
                result = result[0]
            emotions = result.get('emotion', {}) or {}
            if not emotions:
                return 'neutral', 0.0, [('neutral', 100.0)]

            # Normalize DeepFace scores to 0..1
            total = sum(emotions.values()) or 1.0
            # map DeepFace labels to canonical labels
            label_map = {
                'happy': 'happy',
                'sad': 'sadness',
                'angry': 'anger',
                'surprise': 'surprise',
                'neutral': 'neutral',
                'disgust': 'disgust',
                'fear': 'fear',
                'contempt': 'neutral'
            }
            mapped_scores = {}
            for k,v in emotions.items():
                mk = label_map.get(k.lower(), 'neutral')
                mapped_scores[mk] = mapped_scores.get(mk, 0.0) + float(v)

            # build top3
            items = sorted(mapped_scores.items(), key=lambda kv: kv[1], reverse=True)
            top3 = []
            total_mapped = sum(v for _,v in items) or 1.0
            for lab, sc in items[:3]:
                pct = float(sc) / float(total_mapped) * 100.0
                top3.append((lab, round(pct, 1)))

            # top label and confidence
            top_label = items[0][0]
            top_conf = float(items[0][1]) / total_mapped
            return top_label, top_conf, top3
        except Exception:
            return 'neutral', 0.0, [('neutral', 100.0)]

    def _predict_from_onnx(self, frame, bbox: Tuple[int,int,int,int]):
        """Predict using a local ONNX model file placed in assets/emotion_model.onnx.

        Returns:
            mapped_label, confidence, top3_list
        """
        # Ensure onnxruntime is available
        if onnxruntime is None:
            raise RuntimeError('onnxruntime not available')

        model_path = 'assets/emotion_model.onnx'
        # download model on demand if missing
        if not os.path.exists(model_path):
            try:
                self._ensure_onnx_model(model_path)
            except Exception as e:
                raise RuntimeError(f'Failed to acquire ONNX model: {e}')

        x,y,w,h = bbox
        H,W = frame.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(W, x+w)
        y2 = min(H, y+h)
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return 'neutral', 0.0, [('neutral', 100.0)]

        # Many FER+ ONNX models expect grayscale 64x64 input. Prepare accordingly.
        try:
            img = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        except Exception:
            img = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (64,64)).astype('float32')
        # normalize to 0..1 and expand to (1,1,64,64)
        img = (img / 255.0).astype('float32')
        img = img[None, None, :, :]

        # cache onnx session for speed
        if self._onnx_sess is None:
            self._onnx_sess = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        sess = self._onnx_sess
        input_name = sess.get_inputs()[0].name
        out = sess.run(None, {input_name: img})
        probs = np.asarray(out[0]).ravel()
        # softmax (some models already output probabilities)
        try:
            exp = np.exp(probs - np.max(probs))
            probs = exp / (exp.sum() + 1e-9)
        except Exception:
            pass

        # FER+ original mapping
        fer_labels = ['neutral','happiness','surprise','sadness','anger','disgust','fear','contempt']
        mapping = {
            'neutral': 'neutral',
            'happiness': 'happy',
            'surprise': 'surprise',
            'sadness': 'sadness',
            'anger': 'anger',
            'disgust': 'disgust',
            'fear': 'confused',
            'contempt': 'neutral'
        }

        # build mapped list of (label, prob)
        mapped_probs = {}
        for i, p in enumerate(probs):
            fer = fer_labels[i] if i < len(fer_labels) else 'neutral'
            lab = mapping.get(fer, 'neutral')
            mapped_probs[lab] = mapped_probs.get(lab, 0.0) + float(p)

        items = sorted(mapped_probs.items(), key=lambda kv: kv[1], reverse=True)
        total = sum(v for _, v in items) or 1.0
        top3 = []
        for lab, sc in items[:3]:
            pct = float(sc) / total * 100.0
            top3.append((lab, round(pct, 1)))

        top_label = items[0][0]
        top_conf = float(items[0][1]) / total
        return top_label, top_conf, top3

    def _ensure_onnx_model(self, model_path: str):
        """Download a small ONNX FER+ model into assets/ if it's not present.

        This downloads the official ONNX models repo's FER+ model. If requests is
        available it will be used; otherwise urllib is used.
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        url = self._onnx_model_url
        print(f'Downloading ONNX emotion model from {url} ...')
        # try requests first
        try:
            import requests
            r = requests.get(url, stream=True, timeout=30)
            r.raise_for_status()
            with open(model_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print('Downloaded ONNX model to', model_path)
            return
        except Exception:
            pass

        # fallback to urllib
        try:
            from urllib.request import urlopen
            with urlopen(url, timeout=30) as src, open(model_path, 'wb') as dst:
                dst.write(src.read())
            print('Downloaded ONNX model to', model_path)
            return
        except Exception as e:
            if os.path.exists(model_path):
                return
            raise RuntimeError(f'Could not download ONNX model: {e}')

    def predict(self, landmarks: List[Tuple[int,int]], image_shape: Tuple[int,int]):
        """Predict using configured mode.

        Returns:
            best_label, conf_smoothed, persona, display_alpha, top3
        where top3 is a list of (label, percent_float)
        """

        # Decide which prediction mode to use: image heuristics, text model, deepface DL, or hybrid
        label = 'neutral'
        conf = 0.0
        # text-mode will produce scores list; capture top3 if available
        text_top3 = None
        if self.mode == 'image':
            label, conf = self._predict_from_landmarks(landmarks, image_shape)
        elif self.mode == 'text':
            # ensure classifier loaded
            if self.classifier is None:
                try:
                    self.load()
                except Exception:
                    label, conf = 'neutral', 0.0
                else:
                    desc = self._landmarks_to_text(landmarks, image_shape)
                    try:
                        scores_list = self.classifier(desc)
                        scores = scores_list[0] if isinstance(scores_list, list) and scores_list else scores_list
                        # scores is list of dicts: [{'label':..., 'score':...}, ...]
                        items = sorted(scores, key=lambda x: x['score'], reverse=True)
                        top = items[0]
                        label = top['label']
                        conf = float(top['score'])
                        # build top3
                        text_top3 = [(it['label'], round(float(it['score']) * 100.0, 1)) for it in items[:3]]
                    except Exception:
                        label, conf = 'neutral', 0.0
            else:
                desc = self._landmarks_to_text(landmarks, image_shape)
                try:
                    scores_list = self.classifier(desc)
                    scores = scores_list[0] if isinstance(scores_list, list) and scores_list else scores_list
                    items = sorted(scores, key=lambda x: x['score'], reverse=True)
                    top = items[0]
                    label = top['label']
                    conf = float(top['score'])
                    text_top3 = [(it['label'], round(float(it['score']) * 100.0, 1)) for it in items[:3]]
                except Exception:
                    label, conf = 'neutral', 0.0
        elif self.mode == 'hybrid':  # hybrid: prefer image heuristics, fallback to text
            label, conf = self._predict_from_landmarks(landmarks, image_shape)
            # if low confidence, try text classifier
            if conf < 0.35:
                try:
                    if self.classifier is None:
                        self.load()
                    desc = self._landmarks_to_text(landmarks, image_shape)
                    scores_list = self.classifier(desc)
                    scores = scores_list[0] if isinstance(scores_list, list) and scores_list else scores_list
                    items = sorted(scores, key=lambda x: x['score'], reverse=True)
                    tlabel = items[0]['label']
                    tconf = float(items[0]['score'])
                    text_top3 = [(it['label'], round(float(it['score']) * 100.0, 1)) for it in items[:3]]
                    # prefer text if clearly stronger
                    if tconf > conf + 0.15:
                        label, conf = tlabel, tconf
                except Exception:
                    pass
        elif self.mode == 'dl':
            # fallback to heuristics here; DL predictions are done through predict_dl
            if landmarks:
                label, conf = self._predict_from_landmarks(landmarks, image_shape)
            else:
                label, conf = 'neutral', 0.0
        else:
            # unknown mode - fallback to landmarks
            label, conf = self._predict_from_landmarks(landmarks, image_shape)

        # Append to recent history and compute smoothed label
        self.recent.append((label, conf))
        agg = defaultdict(float)
        weight = 1.0
        total_w = 0.0
        # weighted by recency (more recent -> higher weight)
        for lab, c in reversed(self.recent):
            agg[lab] += c * weight
            total_w += weight
            weight *= self.recent_decay

        # pick label with highest aggregated score
        if agg:
            best_label = max(agg.items(), key=lambda kv: kv[1])[0]
        else:
            best_label = label

        # compute averaged confidence for best label
        conf_smoothed = agg[best_label] / max(1e-6, sum(agg.values())) if sum(agg.values()) > 0 else conf

        # Build top3 from aggregated scores (if text_top3 not available)
        items = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
        if items and sum(v for _, v in items) > 0:
            total_agg = sum(v for _, v in items)
            top3 = [(lab, round(float(v) / float(total_agg) * 100.0, 1)) for lab, v in items[:3]]
        elif text_top3 is not None:
            top3 = text_top3
        else:
            # fallback: show best_label plus neutral + unknown
            top3 = [(best_label, round(conf_smoothed * 100.0, 1)), ('neutral', round((1.0 - conf_smoothed) * 100.0, 1))]

        persona = self.PERSONA_MAP.get(best_label.lower(), self.PERSONA_MAP.get('neutral'))

        # smooth fade animation state update --- trigger change time
        if best_label != self.last_label:
            self.last_change_time = time.time()
            self.last_label = best_label
            # reset alpha to 0 to start fade-in
            self.display_alpha = 0.0
            self.display_label = best_label

        # progress alpha towards 1.0
        dt = time.time() - self.last_change_time
        # 0.25s fade-in
        self.display_alpha = min(1.0, dt / 0.25)

        return best_label, float(max(0.0, min(1.0, conf_smoothed))), persona, self.display_alpha, top3

    def predict_dl(self, frame, bbox: Tuple[int,int,int,int]):
        """Convenience method to call the DL predictor directly using frame+bbox.

        Returns (label, conf, persona, alpha, top3)
        """
        try:
            if self.dl_backend == 'onnx':
                lab, conf, top3 = self._predict_from_onnx(frame, bbox)
            else:
                lab, conf, top3 = self._predict_from_deepface(frame, bbox)
        except Exception:
            lab, conf, top3 = 'neutral', 0.0, [('neutral', 100.0)]

        persona = self.PERSONA_MAP.get(lab.lower(), self.PERSONA_MAP.get('neutral'))
        if lab != self.last_label:
            self.last_change_time = time.time()
            self.last_label = lab
            self.display_alpha = 0.0
        dt = time.time() - self.last_change_time
        self.display_alpha = min(1.0, dt / 0.25)
        return lab, conf, persona, self.display_alpha, top3


if __name__ == "__main__":
    # basic test
    em = EmotionModel()
    try:
        em.load()
        print('Loaded model (may download first time).')
    except Exception as e:
        print('Could not load transformer model:', e)