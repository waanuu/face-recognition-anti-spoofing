import cv2
import numpy as np
import onnxruntime
import time
from collections import deque
import faiss
import pickle
import os
import uuid


# =============================
# PATH
# =============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "antispoof_mobilenetv3.onnx")
SCRFD_PATH = os.path.join(MODEL_DIR, "det_500m.onnx")
ARCFACE_PATH = os.path.join(MODEL_DIR, "w600k_mbf.onnx")
FAISS_INDEX_PATH = os.path.join(MODEL_DIR, "face_gallery.index")
FAISS_NAMES_PATH = os.path.join(MODEL_DIR, "face_names.pkl")

RECOG_THRESHOLD = 0.22
ANTISPOOF_THRESHOLD = 0.60

REGISTER_SAMPLES = 20
DUPLICATE_THRESHOLD = 0.45


# =============================
# CHECK FILES
# =============================

for path in [
    MODEL_PATH,
    SCRFD_PATH,
    ARCFACE_PATH,
    FAISS_INDEX_PATH,
    FAISS_NAMES_PATH,
]:
    print(path, "=>", os.path.exists(path))
antispoof_session = onnxruntime.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

antispoof_input = antispoof_session.get_inputs()[0].name
antispoof_output = antispoof_session.get_outputs()[0].name
# =============================
# LOAD ARCFACE / MOBILEFACENET
# =============================

arcface_session = onnxruntime.InferenceSession(
    ARCFACE_PATH,
    providers=["CPUExecutionProvider"]
)

arcface_input = arcface_session.get_inputs()[0].name
arcface_output = arcface_session.get_outputs()[0].name


# =============================
# LOAD FAISS
# =============================

index = faiss.read_index(FAISS_INDEX_PATH)

with open(FAISS_NAMES_PATH, "rb") as f:
    gallery_names = pickle.load(f)

print("FAISS loaded:", index.ntotal)


# =============================
# UTILS
# =============================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def nms(boxes, scores, iou_threshold=0.4):
    if len(boxes) == 0:
        return []

    boxes = boxes.astype(np.float32)
    scores = scores.astype(np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h
        union = areas[i] + areas[order[1:]] - inter

        iou = inter / np.maximum(union, 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


# =============================
# FACE ALIGN 5-POINT
# =============================

ARC_FACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def estimate_norm(src, dst=ARC_FACE_DST):
    src = src.astype(np.float32)

    M, _ = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.LMEDS
    )

    return M


def align_face(img_bgr, kps):
    M = estimate_norm(kps)

    if M is None:
        return None

    aligned = cv2.warpAffine(
        img_bgr,
        M,
        (112, 112),
        borderValue=0
    )

    return aligned


# =============================
# FACE EMBEDDING
# =============================

def get_embedding(face_bgr):
    img = cv2.resize(face_bgr, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)
    img = (img - 127.5) / 127.5

    img = np.transpose(img, (2, 0, 1))[None].astype(np.float32)

    emb = arcface_session.run(
        [arcface_output],
        {arcface_input: img}
    )[0]

    emb = emb.astype(np.float32)

    norm = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    emb = emb / norm

    return emb


# =============================
# RECOGNITION
# =============================

def recognize(face_bgr):
    emb = get_embedding(face_bgr)

    scores, idxs = index.search(emb, 1)

    score = float(scores[0][0])
    idx = int(idxs[0][0])

    if idx < 0 or score < RECOG_THRESHOLD:
        return "Unknown", score

    name = gallery_names[idx]

    return name, score


# =============================
# REGISTER NEW PERSON
# =============================

def save_faiss():
    faiss.write_index(index, FAISS_INDEX_PATH)

    with open(FAISS_NAMES_PATH, "wb") as f:
        pickle.dump(gallery_names, f)

    print("Saved FAISS index and names")


def generate_random_id():
    while True:
        new_id = "ID_" + uuid.uuid4().hex[:8].upper()

        if new_id not in gallery_names:
            return new_id


def add_new_person(embs):
    global index, gallery_names

    if len(embs) == 0:
        print("No embeddings collected")
        return False, None, 0.0

    mean_emb = np.mean(
        np.vstack(embs),
        axis=0,
        keepdims=True
    ).astype(np.float32)

    norm = np.linalg.norm(mean_emb, axis=1, keepdims=True) + 1e-12
    mean_emb = mean_emb / norm

    scores, idxs = index.search(mean_emb, 1)

    best_score = float(scores[0][0])
    best_idx = int(idxs[0][0])

    if best_idx >= 0 and best_score >= DUPLICATE_THRESHOLD:
        existed_name = gallery_names[best_idx]

        print("Face already exists!")
        print(f"Matched: {existed_name} | score={best_score:.4f}")

        return False, existed_name, best_score

    new_id = generate_random_id()

    index.add(mean_emb)
    gallery_names.append(new_id)

    save_faiss()

    print(f"Added new person: {new_id}")
    print("Total identities:", index.ntotal)

    return True, new_id, best_score


# =============================
# SCRFD WITH LANDMARK
# =============================

class SCRFD:
    def __init__(self, model_path):
        self.session = onnxruntime.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.input_size = (640, 640)
        self.strides = [8, 16, 32]
        self.num_anchors = 2

        print("SCRFD outputs:", self.output_names)

    def detect(self, img, conf=0.4, iou_threshold=0.4):
        h0, w0 = img.shape[:2]

        size = max(h0, w0)

        padded = np.zeros((size, size, 3), dtype=np.uint8)
        padded[:h0, :w0] = img

        resized = cv2.resize(padded, self.input_size)

        blob = resized.astype(np.float32)
        blob = (blob - 127.5) / 128.0
        blob = blob.transpose(2, 0, 1)[None].astype(np.float32)

        outputs = self.session.run(
            self.output_names,
            {self.input_name: blob}
        )

        boxes_all = []
        scores_all = []
        kps_all = []

        input_h, input_w = self.input_size

        for i, stride in enumerate(self.strides):
            scores = outputs[i].reshape(-1)

            if scores.min() < 0 or scores.max() > 1:
                scores = sigmoid(scores)

            boxes = (outputs[i + 3] * stride).reshape(-1, 4)

            kps = (outputs[i + 6] * stride).reshape(-1, 10)

            height = input_h // stride
            width = input_w // stride

            grid = np.stack(
                np.meshgrid(np.arange(width), np.arange(height)),
                axis=-1
            ).reshape(-1, 2)

            grid = np.repeat(grid, self.num_anchors, axis=0)
            grid = (grid + 0.5) * stride

            mask = scores > conf

            if mask.sum() == 0:
                continue

            boxes = boxes[mask]
            kps = kps[mask]
            grid_masked = grid[mask]
            scores_masked = scores[mask]

            x1 = grid_masked[:, 0] - boxes[:, 0]
            y1 = grid_masked[:, 1] - boxes[:, 1]
            x2 = grid_masked[:, 0] + boxes[:, 2]
            y2 = grid_masked[:, 1] + boxes[:, 3]

            boxes_decoded = np.stack([x1, y1, x2, y2], axis=1)

            kps_decoded = kps.reshape(-1, 5, 2)
            kps_decoded[:, :, 0] = grid_masked[:, 0:1] + kps_decoded[:, :, 0]
            kps_decoded[:, :, 1] = grid_masked[:, 1:2] + kps_decoded[:, :, 1]

            boxes_all.append(boxes_decoded)
            scores_all.append(scores_masked)
            kps_all.append(kps_decoded)

        if len(boxes_all) == 0:
            return [], [], []

        boxes = np.vstack(boxes_all)
        scores = np.concatenate(scores_all)
        kpss = np.vstack(kps_all)

        scale = size / self.input_size[0]

        boxes *= scale
        kpss *= scale

        boxes[:, 0] = np.clip(boxes[:, 0], 0, w0)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h0)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w0)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h0)

        kpss[:, :, 0] = np.clip(kpss[:, :, 0], 0, w0)
        kpss[:, :, 1] = np.clip(kpss[:, :, 1], 0, h0)

        keep = nms(boxes, scores, iou_threshold=iou_threshold)

        boxes = boxes[keep]
        scores = scores[keep]
        kpss = kpss[keep]

        order = scores.argsort()[::-1]

        boxes = boxes[order]
        scores = scores[order]
        kpss = kpss[order]

        return boxes.astype(int), kpss.astype(np.float32), scores


detector = SCRFD(SCRFD_PATH)


# =============================
# Webcam
# =============================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev = time.time()

score_buffer = deque(maxlen=10)

last_name = "Unknown"
last_recog_score = 0.0

frame_count = 0

missing_count = 0
MAX_MISSING = 5

register_mode = False
register_embs = []
register_message = ""
register_message_timer = 0


def reset_state():
    global last_name, last_recog_score

    score_buffer.clear()
    last_name = "Unknown"
    last_recog_score = 0.0

fps_buffer = deque(maxlen=10)
while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    boxes, kpss, det_scores = detector.detect(frame)

    if len(boxes) == 0:
        missing_count += 1

        if missing_count >= MAX_MISSING:
            reset_state()

    else:
        missing_count = 0

        areas = [
            (b[2] - b[0]) * (b[3] - b[1])
            for b in boxes
        ]

        best_idx = int(np.argmax(areas))

        box = boxes[best_idx]
        kps = kpss[best_idx]

        x1, y1, x2, y2 = box

        w = x2 - x1
        h = y2 - y1

        pad_left = int(w * 0.10)
        pad_right = int(w * 0.10)

        pad_top = int(h * 0.15)

        pad_bottom = int(h * -0.12)

        H, W = frame.shape[:2]

        x1 = max(0, x1 - pad_left)
        y1 = max(0, y1 - pad_top)
        x2 = min(W, x2 + pad_right)
        y2 = min(H, y2 + pad_bottom)

        face_bgr = frame[y1:y2, x1:x2]

        aligned_bgr = align_face(frame, kps)

        if aligned_bgr is None and face_bgr.size != 0:
            aligned_bgr = cv2.resize(face_bgr, (112, 112))

        if face_bgr.size != 0:
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            face_input = cv2.resize(face_rgb, (224, 224))
            face_input = face_input.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            face_input = (face_input - mean) / std
            face_input = np.transpose(face_input, (2, 0, 1))
            face_input = np.expand_dims(face_input, axis=0).astype(np.float32)

            out = antispoof_session.run(
                [antispoof_output],
                {antispoof_input: face_input}
            )[0]

            exp = np.exp(out)
            prob = exp / np.sum(exp, axis=1, keepdims=True)

            real_score = float(prob[0][1])

            score_buffer.append(real_score)
            smooth_score = sum(score_buffer) / len(score_buffer)

            if smooth_score > ANTISPOOF_THRESHOLD:

                if register_mode:
                    if aligned_bgr is not None:
                        emb = get_embedding(aligned_bgr)
                        register_embs.append(emb[0])

                        color = (255, 255, 0)
                        text = f"REGISTERING: {len(register_embs)}/{REGISTER_SAMPLES}"

                        if len(register_embs) >= REGISTER_SAMPLES:
                            saved, info, dup_score = add_new_person(register_embs)

                            if saved:
                                register_message = f"SAVED {info}"
                            else:
                                register_message = f"ALREADY EXISTS: {info} {dup_score:.2f}"

                            register_message_timer = 60
                            register_mode = False
                            register_embs.clear()

                    else:
                        color = (0, 255, 255)
                        text = "ALIGN FAILED"

                else:
                    if aligned_bgr is not None and frame_count % 3 == 0:
                        last_name, last_recog_score = recognize(aligned_bgr)

                    if last_name == "Unknown":
                        color = (0, 255, 255)
                        text = f"UNKNOWN | {last_recog_score:.2f} | REAL {smooth_score:.2f}"
                    else:
                        color = (0, 255, 0)
                        text = f"{last_name} | {last_recog_score:.2f} | REAL {smooth_score:.2f}"

            else:
                color = (0, 0, 255)
                text = f"FAKE | REAL_SCORE {smooth_score:.2f}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            for px, py in kps.astype(int):
                cv2.circle(frame, (px, py), 2, (255, 255, 0), -1)

            cv2.putText(
                frame,
                text,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_DUPLEX,
                0.65,
                color,
                2
            )

    if register_message_timer > 0:
        register_message_timer -= 1

        cv2.putText(
            frame,
            register_message,
            (20, 90),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (255, 255, 0),
            2
        )

    now = time.time()
    fps = 1 / max(now - prev, 1e-6)
    prev = now
    fps_buffer.append(fps)
    smooth_fps = sum(fps_buffer) / len(fps_buffer)

    cv2.putText(
        frame,
        f"FPS: {int(smooth_fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_DUPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Press S: register | ESC: exit",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_DUPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow("Face Anti Spoofing + Recognition", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

    elif key == ord("s"):
        register_mode = True
        register_embs.clear()
        register_message = "START REGISTERING"
        register_message_timer = 30
        print("Start registering new face...")

cap.release()
cv2.destroyAllWindows()