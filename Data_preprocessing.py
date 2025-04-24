import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import tensorflow_addons as tfa
from tensorflow.keras.applications.inception_v3 import preprocess_input

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

def extract_eye_roi(image: np.ndarray):
    h, w, _ = image.shape
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        raise ValueError("No face/eyes detected")

    lm = results.multi_face_landmarks[0].landmark
    eye_indices = [33, 133, 362, 263]
    xs = [lm[i].x * w for i in eye_indices]
    ys = [lm[i].y * h for i in eye_indices]
    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))
    dx, dy = int(0.1 * (x_max - x_min)), int(0.1 * (y_max - y_min))
    x1, y1 = max(0, x_min - dx), max(0, y_min - dy)
    x2, y2 = min(w, x_max + dx), min(h, y_max + dy)
    return image[y1:y2, x1:x2]

def load_and_preprocess(path: str, label):
    img = cv2.imread(path.decode('utf-8'))
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    try:
        roi = extract_eye_roi(img)
    except ValueError:
        h, w, _ = img.shape
        side = min(h, w)
        roi = img[(h-side)//2:(h+side)//2, (w-side)//2:(w+side)//2]
    roi = cv2.resize(roi, (299, 299))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = preprocess_input(roi.astype(np.float32))
    return roi, label

def tf_load_and_preprocess(path, label):
    img, lbl = tf.py_function(func=load_and_preprocess, inp=[path, label], Tout=(tf.float32, label.dtype))
    img.set_shape((299, 299, 3))
    lbl.set_shape(label.shape)
    return img, lbl

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    angle = tf.random.uniform((), -10, 10) * np.pi / 180
    image = tfa.image.rotate(image, angle, fill_mode='reflect')
    return image, label

def build_dataset(image_paths, labels, batch_size=32, shuffle=True, augment_data=True):
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(image_paths))
    ds = ds.map(tf_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if augment_data:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

if __name__ == "__main__":
    data_dir = "images.jpg"
    filenames = []
    labels = []
    for fname in os.listdir(data_dir):
        if fname.lower().endswith(('.jpg','.png')):
            filenames.append(os.path.join(data_dir, fname))
            labels.append(1 if "strabismus" in fname else 0)

    image_paths = np.array(filenames, dtype=str)
    labels = np.array(labels, dtype=np.int32)

    ds = build_dataset(image_paths, labels, batch_size=16)

    for batch_imgs, batch_lbls in ds.take(1):
        print(batch_imgs.shape, batch_lbls.shape)
