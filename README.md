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
![output (5)](https://github.com/user-attachments/assets/768a748d-ca78-46de-bff8-3e073152fc40)



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
