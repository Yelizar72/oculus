import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import tensorflow_addons as tfa
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S
import numpy as np

# import data pipeline
from processing import build_all_datasets, load_split, compute_binary_class_weights

# Hyperparameters 
ROOT_DIR        = "/path/to/images.jpg" 
BATCH_SIZE      = 16
EPOCHS          = 20
LEARNING_RATE   = 3e-4
INPUT_SHAPE     = (299, 299, 3)
LOSS_TYPE_GAMMA = 2.0
LOSS_TYPE_ALPHA = 0.25
LOSS_TYPE_WEIGHT= 1.5  

# Prepare Data
train_ds, val_ds, test_ds = build_all_datasets(
    root_dir=ROOT_DIR,
    mode="multitask",
    batch=BATCH_SIZE,
    augment_train=True,
    oversample=True
)

train_paths, train_labels = load_split(os.path.join(ROOT_DIR, "train"))
class_weights = compute_binary_class_weights(train_labels)

# Build Model
base = EfficientNetV2S(
    input_shape=INPUT_SHAPE,
    include_top=False,
    weights="imagenet"
)
x = layers.GlobalAveragePooling2D()(base.output)
x = layers.Dropout(0.3)(x)

binary_head = layers.Dense(1, activation="sigmoid", name="binary_head")(x)
type_head   = layers.Dense(4, activation="softmax", name="type_head")(x)

model = models.Model(inputs=base.input, outputs=[binary_head, type_head])

# Custom Multi-Task Loss
def multitask_loss(y_true, y_pred):
    y_bin_true  = y_true[:, 0]
    y_type_true = y_true[:, 1:]
    p_bin, p_type = y_pred

    fl = tfa.losses.SigmoidFocalCrossEntropy(
        alpha=LOSS_TYPE_ALPHA, gamma=LOSS_TYPE_GAMMA
    )
    L_bin = fl(y_bin_true, p_bin)

    mask = tf.expand_dims(y_bin_true, axis=-1) 
    ce = tf.keras.losses.CategoricalCrossentropy()
    L_type = ce(y_type_true, p_type) * mask

    return tf.reduce_mean(L_bin + LOSS_TYPE_WEIGHT * L_type)

# Compile
model.compile(
    optimizer=optimizers.Adam(LEARNING_RATE),
    loss=multitask_loss,
    metrics={
        "binary_head": [tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="auc")],
        "type_head":   [tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
    }
)

# Callbacks 
ckpt = callbacks.ModelCheckpoint(
    "best_model_v2.h5",
    monitor="val_binary_head_recall",
    save_best_only=True,
    mode="max"
)
es = callbacks.EarlyStopping(
    monitor="val_binary_head_recall",
    patience=5,
    restore_best_weights=True,
    mode="max"
)
rlr = callbacks.ReduceLROnPlateau(
    monitor="val_binary_head_auc",
    factor=0.5,
    patience=3,
    mode="max"
)

# Training
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[ckpt, es, rlr],
    class_weight={"binary_head": class_weights}
)

# F1 scoring
y_true_bin, y_pred_bin = [], []
y_true_type, y_pred_type = [], []

for x_batch, y_batch in test_ds:
    p_bin, p_type = model.predict(x_batch)
    y_true_bin.append(y_batch[:, 0].numpy())
    y_pred_bin.append((p_bin > 0.5).astype(int).reshape(-1))
    y_true_type.append(np.argmax(y_batch[:, 1:].numpy(), axis=1))
    y_pred_type.append(np.argmax(p_type, axis=1))

y_true_bin  = np.concatenate(y_true_bin)
y_pred_bin  = np.concatenate(y_pred_bin)
y_true_type = np.concatenate(y_true_type)
y_pred_type = np.concatenate(y_pred_type)

bin_f1 = f1_score(y_true_bin, y_pred_bin)
print(f"\nBinary F1 score (strabismus vs normal): {bin_f1:.4f}")

print("\nStrabismus sub-type classification report:")
print(classification_report(
    y_true_type,
    y_pred_type,
    target_names=["ESOTROPIA","EXOTROPIA","HYPERTROPIA","HYPOTROPIA"]
))

print("\nKeras evaluate metrics:")
print(model.evaluate(test_ds, return_dict=True))
