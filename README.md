# Oculus project
Oculus is a web-based platform that provides personalized eye exercises based on a user's current eye health status. By analyzing user input and eye condition, the site recommends targeted routines to support and improve visual well-being.

This repository focuses on of the core components of the Oculus platform:
Strabismus Detection – Using uploaded eye photos to automatically detect and if any classify it into one of four types:

+ Esotropia (eye turns inward)
+ Exotropia (eye turns outward)
+ Hypertropia (eye turns upward)
+ Hypotropia (eye turns downward)

These functionalities serve as the foundation for personalized diagnosis and recommendation within the Oculus system.

The dataset was extracted from the Roboflow website: https://universe.roboflow.com/strabismusdet/strab1/dataset/2
## Contents
### Data Analysis and Feature Engineering
<details>
<summary><strong>📁 Dataset Structure</strong></summary>

**Main folders and files:**

- `images.jpg/` — Main dataset directory  
  - `train/` — Training set  
    - Image files (`.jpg`, `.png`)  
    - `_classes.csv` — One-hot encoded class labels  
  - `val/` — Validation set  
    - Image files  
    - `_classes.csv`  
  - `test/` — Final evaluation set  
    - Image files  
    - `_classes.csv`  

</details>

#### Overall Class Distribution Analysis
The dataset is close to balanced across the four strabismus sub-types (≈ 17-23 % each), while the NORMAL class is slightly smaller (≈ 19 %). Every record carries exactly one active label, so there is no multi-labelling noise, and the bar-chart confirms the visual balance.

#### Binary Imbalance Check
When collapsed to a two-class task, strabismus outnumbers normal ≈ 4 : 1 in every split. Mitigate this by applying class-weights or focal loss in the binary gate, or by oversampling NORMAL images during training.

#### Image Size & Format Check
All files share the same resolution (640 × 320, aspect 2 : 1) and are stored as .jpg. No duplicate images were detected by MD5 hashes. The uniform size simplifies resizing to 299 × 299 without large interpolation artefacts.

#### Basic Brightness Evaluation
A random sample (200 images) shows a mean pixel value of 127 ± 26 (0–255 BGR), approximating a normal distribution with no extreme exposures. Standard preprocess normalisation is sufficient; CLAHE is unnecessary.

#### Colour-Bias Check
Scleral hue is tightly clustered (≈ 50-62°) across all classes, signalling no colour bias in the “white of the eye.” Iris hue shows a tentative shift: ESOTROPIA/EXOTROPIA/NORMAL ≈ 60°, HYPER/HYPO ≈ 110-123°, but the sample size for iris masks is tiny (15-17 measurements max). Should add hue-jitter during training and monitor Grad-CAM to ensure the network learns geometry rather than subtle colour cues.

#### Overall Conclusions and Training Recommendations
+ Binary gate – weight ratio ≈ 4 : 1
+ Augmentation – horizontal flip, ±10° rotation, brightness/contrast jitter preserve clinical validity.
+ Cross-validation – stratified K-fold on the 5-column one-hot matrix gives robust estimates for a 700-image dataset


### Data Preprocessing
#### Image Loading
Each image is loaded from disk using OpenCV’s cv2.imread() function inside the _cv_read() helper. This function reads an image file from a specified dataset split and returns it as a NumPy array. If the image fails to load a ValueError is immediately raised to maintain data integrity and prevent silent failures during preprocessing.

#### Eye Region Extraction (ROI Detection)
After loading, the script attempts to extract the eye region from the face using MediaPipe’s FaceMesh model, initialized in the module as _face_mesh. It detects facial landmarks and specifically uses four key indices: 33, 133, 362, 263, which correspond to the inner and outer corners of both eyes. The landmark coordinates are scaled according to the image size to define a bounding box. To ensure better visual context, a 10% margin is added around the detected eye region before cropping. This step ensures the model focuses consistently on the eyes, which are the key area for detecting strabismus.

#### Fallback Cropping
If MediaPipe fails to detect facial landmarks (due to reasons like poor image quality, occlusion, or unusual head poses), the code applies a fallback strategy. Instead of discarding the image, it performs a centered square crop, selecting the largest possible square from the middle of the image. This guarantees that all images are preserved in the dataset while keeping the crop centered on the face region, which still captures valuable diagnostic features.

#### Image Preprocessing
Once the eye region is extracted, the image undergoes standard preprocessing steps to make it compatible with deep learning models. The cropped region is resized to a fixed size of 299x299 pixels, converted from OpenCV’s default BGR format to RGB, and normalized using the preprocess_input() function from TensorFlow’s Keras API (InceptionV3 settings). This normalization scales pixel values and adjusts color distributions to match the expectations of the pretrained backbone network, improving convergence speed and model performance.

#### Data Augmentation
To enhance model generalization and reduce overfitting, optional real-time data augmentation is applied using TensorFlow and TensorFlow Addons libraries. The augmentations include random horizontal flipping (simulating left and right eye variations), random brightness adjustments, random contrast changes, and small random rotations within ±10 degrees. These augmentations increase the variability of the training set, helping the model become robust against common image variations like lighting differences, slight head tilts, and mirrored images.

#### Dataset Construction
After preprocessing, image tensors and their corresponding labels (depending on the mode: binary classification, 4-type classification, or multitask) are combined into a tf.data.Dataset. The dataset pipeline supports optional shuffling (for randomized batches), batching (mini-batches of a configurable size, e.g., 32), real-time augmentation, and prefetching (loading future batches in the background). This design ensures efficient and scalable data loading for training and evaluation.

### Computer Vision Model

#### Defining Hyperparameters
Sets the key training hyperparameters:
+ BATCH_SIZE, EPOCHS, LEARNING_RATE: Training configuration.
+ INPUT_SHAPE: The input image shape expected by the model.
+ LOSS_ALPHA, LOSS_GAMMA: Focal loss parameters to handle class imbalance.
+ LOSS_TYPE_WEIGHT: Weight assigned to the 4-class head loss relative to the binary head.

#### Data Preparation
Calls build_all_datasets() to load train, val, and test datasets as TensorFlow pipelines.
Augmentation and oversampling (for NORMAL class) are applied automatically.
Also computes binary class weights (NORMAL vs STRABISMUS) from training data using compute_binary_class_weights(), to balance classes during loss calculation.

#### Model Construction
The model architecture:
- EfficientNetV2S is used as a feature extractor backbone (include_top=False) with pretrained ImageNet weights.
- A GlobalAveragePooling2D and Dropout layer are added for regularization.
- Two outputs (heads) are built:  
  **binary_head**: A sigmoid output predicting strabismus presence (binary classification).  
  **type_head**: A softmax output predicting the 4 strabismus sub-types (multi-class classification).  
  Both heads share the same base encoder.

#### Custom Multi-Task Loss Function
Defines multitask_loss():
+ Binary head uses Sigmoid Focal Loss to address class imbalance (rare NORMAL samples).
+ Type head uses Categorical Cross-Entropy Loss, masked by strabismus presence (only contribute when positive strabismus).
+ Both losses are combined, with the type loss scaled by LOSS_TYPE_WEIGHT.

This ensures the model prioritizes detecting strabismus, while still learning to classify subtypes when applicable.

#### Model Compilation
The model is compiled with:
+ Adam optimizer
+ The custom multitask loss function
Metrics:
+ Recall and AUC for binary output (important due to dataset skew).
+ Categorical Accuracy for sub-type classification.

#### Setting Up Callbacks
Three callbacks are configured:
+ ModelCheckpoint: Save the best model based on val_binary_head_recall.
+ EarlyStopping: Stop training early if recall doesn't improve for 5 epochs.
+ ReduceLROnPlateau: Reduce learning rate if val_binary_head_auc plateaus.

These callbacks help optimize convergence and avoid overfitting.

#### Manual Evaluation: F1 Score and Classification Report
Predictions:
+ Collects model predictions for binary and multi-class outputs across the test dataset.
F1 Score (binary):
+ Calculates the F1-score for strabismus vs normal classification using f1_score().
Classification Report (sub-types):
+ Generates a detailed precision, recall, and F1-score report for each of the 4 subtypes (classification_report()).
Keras Evaluation:
+ Additionally prints standard evaluation metrics using model.evaluate() for completeness.

This two-part evaluation (manual F1 + model metrics) provides a much deeper understanding of model performance, especially given the class imbalance.
