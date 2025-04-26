# Oculus project
Oculus is a web-based platform that provides personalized eye exercises based on a user's current eye health status. By analyzing user input and eye condition, the site recommends targeted routines to support and improve visual well-being.

This repository focuses on of the core components of the Oculus platform:
Strabismus Detection – Using uploaded eye photos to automatically detect and classify strabismus into one of four types:

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


### Data Preprocessing
#### Image Loading
Each image is loaded from disk using OpenCV’s cv2.imread() function. This function reads an image file and returns it as a NumPy array, enabling further manipulation in Python. The image must be in .jpg or .png format and located within a specified folder (e.g., images.jpg/). If the image fails to load (e.g., due to corruption or an incorrect path), an error is raised to ensure data integrity.
#### Eye Region Extraction (ROI Detection)
To focus the model on relevant visual features, the script extracts the eye region from the face using MediaPipe's FaceMesh model. This step identifies facial landmarks, and specifically uses landmark indices 33, 133, 362, 263 corresponding to the inner and outer corners of both eyes. These landmarks are scaled according to the image size to calculate bounding box coordinates. A 10% padding is added for spatial context before cropping the image to the eye region. This ensures consistent focus on the diagnostic area (the eyes), which is essential for accurate strabismus detection.
#### Fallback Cropping
If the MediaPipe landmark detection fails—due to factors like poor image resolution, occlusion, or lighting conditions—a fallback strategy is applied. Instead of discarding the sample, the code performs a square center crop, taking the central region of the image based on the smaller dimension (height or width). This guarantees that the dataset remains complete and balanced while still focusing on the most informative part of the image when no landmarks are available.
#### Image Preprocessing
Once the eye region is extracted or a fallback crop is taken, the image undergoes several preprocessing steps to meet the input requirements of the deep learning model (InceptionV3). The cropped image is resized to 299x299 pixels, converted from OpenCV's default BGR format to RGB, and then normalized using the preprocess_input() function provided by TensorFlow’s Keras API. This function scales pixel intensities and adjusts color distributions in a way that aligns with the model’s pretraining, thereby improving learning performance and convergence.
#### Data Augmentation
To improve model generalization and robustness, data augmentation is optionally applied using TensorFlow functions and tensorflow_addons. This includes random horizontal flipping (to simulate left/right eye variations), slight brightness and contrast jittering (to account for lighting differences), and small-angle rotations (±10°) to simulate head tilts or misalignments. These transformations introduce diversity to the training data, helping the model learn to distinguish genuine strabismus from natural variations in eye appearance.
#### Dataset Construction
After preprocessing, all image paths and their associated binary labels (1 for strabismus, 0 for normal) are used to construct a tf.data.Dataset. The dataset pipeline applies all transformation functions (load_and_preprocess, and optionally augment), shuffles the data if specified, batches it into mini-batches (e.g., of size 16 or 32), and uses prefetching to load future batches while the current one is being processed. This efficient pipeline structure allows the dataset to be streamed directly into a neural network for training or evaluation, with minimal I/O bottlenecks and optimal memory usage.
### Computer Vision Model
