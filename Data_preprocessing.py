import os, glob, cv2, numpy as np, pandas as pd, tensorflow as tf, mediapipe as mp
import tensorflow_addons as tfa
from tensorflow.keras.applications.inception_v3 import preprocess_input

# 1.  eye ROI helper ─ MediaPipe
_mp_face  = mp.solutions.face_mesh
_face_mesh = _mp_face.FaceMesh(static_image_mode=True,
                               max_num_faces=1,
                               refine_landmarks=True,
                               min_detection_confidence=0.5)

_EYE_IDXS = [33, 133, 362, 263]        # inner / outer corners of both eyes

def extract_eye_roi(img: np.ndarray) -> np.ndarray:
    """Crop the eye rectangle with 10 % padding or raise ValueError."""
    h, w = img.shape[:2]
    res  = _face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        raise ValueError("no landmarks")
    lm = res.multi_face_landmarks[0].landmark
    xs = [lm[i].x * w for i in _EYE_IDXS]
    ys = [lm[i].y * h for i in _EYE_IDXS]
    x1, x2 = int(min(xs)), int(max(xs))
    y1, y2 = int(min(ys)), int(max(ys))
    dx, dy = int(0.10 * (x2 - x1)), int(0.10 * (y2 - y1))
    x1, y1 = max(0, x1 - dx), max(0, y1 - dy)
    x2, y2 = min(w, x2 + dx), min(h, y2 + dy)
    return img[y1:y2, x1:x2]

# 2.  one-time CSV parser
def load_split(split_dir: str):
    """
    Args   : split_dir  e.g.   /data/images.jpg/train
    Returns: filepaths, labels_onehot (N,5)
             column order = [ESOTROPIA, EXOTROPIA, HYPERTROPIA, HYPOTROPIA, NORMAL]
    """
    csv_path = os.path.join(split_dir, "_classes.csv")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()       
    filepaths = [os.path.join(split_dir, f) for f in df["filename"]]
    labels    = df[["ESOTROPIA","EXOTROPIA","HYPERTROPIA","HYPOTROPIA","NORMAL"]].values.astype("float32")
    return np.array(filepaths), labels

# 3.  tf.data preprocessing 
def _cv_read(path):
    img = cv2.imread(path.decode("utf-8"))
    if img is None:
        raise ValueError(f"cannot read {path}")
    try:
        img = extract_eye_roi(img)
    except ValueError:                         # fallback = centred square
        h, w = img.shape[:2]
        side = min(h, w)
        img   = img[(h-side)//2:(h+side)//2, (w-side)//2:(w+side)//2]
    img = cv2.resize(img, (299, 299))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img.astype("float32"))
    return img

def _augment(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.1)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    angle = tf.random.uniform([], -10, 10) * np.pi/180
    img = tfa.image.rotate(img, angle, fill_mode="reflect")
    return img

def make_dataset(paths, labels_onehot,
                 mode="binary",            # "binary", "type", or "multitask"
                 batch=32, aug=True, shuffle=True):

    # convert numeric labels 
    if mode == "binary":
        y = (1 - labels_onehot[:,4]).astype("float32")      # 1=strabismus, 0=normal
    elif mode == "type":
        y = labels_onehot[:,:4].astype("float32")           
    elif mode == "multitask":
        bin_ = (1 - labels_onehot[:,4]).astype("float32")   
        y    = np.concatenate([bin_[:,None], labels_onehot[:,:4]], axis=1)  
    else:
        raise ValueError("mode must be binary | type | multitask")

    # dataset construction
    ds = tf.data.Dataset.from_tensor_slices((paths, y))

    if shuffle:
        ds = ds.shuffle(len(paths), seed=42)

    def _py_load(path, lab):
        img = tf.py_function(_cv_read, [path], tf.float32)
        img.set_shape((299,299,3))
        return img, lab

    ds = ds.map(_py_load,  num_parallel_calls=tf.data.AUTOTUNE)

    if aug:
        ds = ds.map(lambda x,l: (_augment(x), l),
                    num_parallel_calls=tf.data.AUTOTUNE)

    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)

# 4.  convenience loader for all three splits
def build_all_datasets(root_dir, mode="binary", batch=32, augment_train=True):
    train_p, train_l = load_split(os.path.join(root_dir, "train"))
    val_p,   val_l   = load_split(os.path.join(root_dir, "val"))
    test_p,  test_l  = load_split(os.path.join(root_dir, "test"))

    train_ds = make_dataset(train_p, train_l, mode, batch, aug=augment_train, shuffle=True)
    val_ds   = make_dataset(val_p,   val_l,   mode, batch, aug=False, shuffle=False)
    test_ds  = make_dataset(test_p,  test_l,  mode, batch, aug=False, shuffle=False)
    return train_ds, val_ds, test_ds
    
# 5.  sanity check
if __name__ == "__main__":
    DATA_ROOT = "Strab1.v2i.multiclass.zip" 
    train_ds, val_ds, test_ds = build_all_datasets(DATA_ROOT, mode="multitask")

    for imgs, lbl in train_ds.take(1):
        print("batch", imgs.shape, "labels", lbl.shape)
