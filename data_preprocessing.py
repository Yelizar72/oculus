import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import mediapipe as mp
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Setup FaceMesh
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

EYE_BOX = [33, 133, 362, 263]  # Eye corners indices

# Eye ROI extraction
def extract_eye_roi(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    res = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        raise ValueError("No landmarks detected")
    lm = res.multi_face_landmarks[0].landmark
    xs = [lm[i].x * w for i in EYE_BOX]
    ys = [lm[i].y * h for i in EYE_BOX]
    x1, x2 = int(min(xs)), int(max(xs))
    y1, y2 = int(min(ys)), int(max(ys))
    dx, dy = int(0.10 * (x2 - x1)), int(0.10 * (y2 - y1))
    x1, y1 = max(0, x1 - dx), max(0, y1 - dy)
    x2, y2 = min(w, x2 + dx), min(h, y2 + dy)
    return img[y1:y2, x1:x2]

# CSV loader
def load_split(split_dir: str):
    csv_path = os.path.join(split_dir, "_classes.csv")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    filepaths = [os.path.join(split_dir, f) for f in df["filename"]]
    labels    = df[["ESOTROPIA", "EXOTROPIA", "HYPERTROPIA", "HYPOTROPIA", "NORMAL"]].values.astype("float32")
    return np.array(filepaths), labels

# Binary class-weights
def compute_binary_class_weights(labels_onehot):  # 0 - normal, 1 - strabismus
    normal_mask = labels_onehot[:, 4] == 1
    n_normal    = normal_mask.sum()
    n_total     = len(labels_onehot)
    n_strab     = n_total - n_normal
    w_normal    = n_total / (2.0 * n_normal)
    w_strab     = n_total / (2.0 * n_strab)
    return {0: float(w_normal), 1: float(w_strab)}

# Image loader
def _cv_read(path):
    img = cv2.imread(path.decode("utf-8"))
    if img is None:
        raise ValueError(f"Cannot read {path}")
    try:
        img = extract_eye_roi(img)
    except ValueError:
        # fallback: center square crop
        h, w = img.shape[:2]
        side = min(h, w)
        img = img[(h-side)//2:(h+side)//2, (w-side)//2:(w+side)//2]
    img = cv2.resize(img, (299, 299))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img.astype("float32"))
    return img

# Augmentations
def _augment(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    angle = tf.random.uniform([], -10, 10) * np.pi/180
    img = tfa.image.rotate(img, angle, fill_mode="reflect")
    return img

# Dataset builder
def make_dataset(paths, labels_onehot, mode="binary",
                 batch=32, aug=True, shuffle=True, oversample=False):    # modes: binary, multitask, type
    if oversample and mode == "binary":
        normal_idx = np.where(labels_onehot[:, 4] == 1)[0]
        strab_idx  = np.where(labels_onehot[:, 4] == 0)[0]
        reps       = int(np.ceil(len(strab_idx) / len(normal_idx)))
        extra_idx  = np.random.choice(normal_idx,
                                      size=reps * len(normal_idx) - len(normal_idx),
                                      replace=True)
        new_idx    = np.concatenate([strab_idx, normal_idx, extra_idx])
        np.random.shuffle(new_idx)
        paths      = paths[new_idx]
        labels_onehot = labels_onehot[new_idx]

    # label format conversion
    if mode == "binary":
        y = (1 - labels_onehot[:, 4]).astype("float32")
    elif mode == "type":
        y = labels_onehot[:, :4].astype("float32")
    elif mode == "multitask":
        bin_ = (1 - labels_onehot[:, 4]).astype("float32")
        y = np.concatenate([bin_[:, None], labels_onehot[:, :4]], axis=1)
    else:
        raise ValueError("mode must be binary | type | multitask")

    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=42)

    def _py_load(path, lab):
        img = tf.py_function(_cv_read, [path], tf.float32)
        img.set_shape((299, 299, 3))
        return img, lab

    ds = ds.map(_py_load, num_parallel_calls=tf.data.AUTOTUNE)
    if aug:
        ds = ds.map(lambda x, l: (_augment(x), l),
                    num_parallel_calls=tf.data.AUTOTUNE)

    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)

# Loader for all three splits
def build_all_datasets(root_dir, mode="binary", batch=32, augment_train=True, oversample=False):
    train_p, train_l = load_split(os.path.join(root_dir, "train"))
    val_p,   val_l   = load_split(os.path.join(root_dir, "val"))
    test_p,  test_l  = load_split(os.path.join(root_dir, "test"))

    train_ds = make_dataset(train_p, train_l,
                            mode=mode, batch=batch,
                            aug=augment_train, shuffle=True,
                            oversample=oversample)

    val_ds   = make_dataset(val_p, val_l,
                            mode=mode, batch=batch,
                            aug=False, shuffle=False)

    test_ds  = make_dataset(test_p, test_l,
                            mode=mode, batch=batch,
                            aug=False, shuffle=False)

    return train_ds, val_ds, test_ds

# Main for testing
if __name__ == "__main__":
    DATA_ROOT = "Downloads/Strab1.v2i.multiclass/train" 
    train_ds, val_ds, test_ds = build_all_datasets(DATA_ROOT,
                                                   mode="multitask",
                                                   batch=32,
                                                   augment_train=True,
                                                   oversample=False)
    for imgs, lbls in train_ds.take(1):
        print("Batch:", imgs.shape, "Labels:", lbls.shape)
