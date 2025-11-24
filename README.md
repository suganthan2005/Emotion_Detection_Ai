# Emotion Detection AI

Emotion Detection AI is a Python-based project for recognizing human emotions from facial images (or webcam frames) using machine learning and computer vision techniques. It is intended as a learning / demo project and can be extended into a full real‑time application.
by Suganthan
## Features

- Detects faces in images or video frames using OpenCV.
- Classifies basic emotions (e.g., happy, sad, angry, surprised, neutral, etc.) using a trained model.
- Simple, modular Python code that is easy to extend.
- Can be adapted for real‑time emotion detection from a webcam feed.

## Requirements

- Python 3.8+
- Recommended packages:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `opencv-python`
  - `tensorflow` or `torch` (depending on your implementation)
  - `scikit-learn`
  - `jupyter`

Install dependencies:

pip install -r requirements.txt

text

If you do not have a `requirements.txt` yet, create one and add the libraries you are using in the notebook.

## Getting Started

1. **Clone the repository**

git clone https://github.com/suganthan2005/Emotion_Detection_Ai.git
cd Emotion_Detection_Ai

text

2. **Create and activate a virtual environment (optional but recommended)**

python -m venv .venv
source .venv/bin/activate # On Linux/macOS
.venv\Scripts\activate # On Windows

text

3. **Install dependencies**

pip install -r requirements.txt

text

4. **Open the Jupyter notebook**

jupyter notebook Emotion_detection.ipynb

text

Then run all cells to:
- Load the dataset
- Train or load the model
- Evaluate emotion detection on sample images

## Usage

Depending on how you implement the pipeline, typical use cases might include:

- **Training the model**: Run the training section in the notebook to train a model on your dataset (e.g., FER2013 or a custom dataset).
- **Testing on images**: Use images in a test folder and modify the relevant cells to point to those images.
- **Real‑time webcam detection** (if implemented):
- Ensure a webcam is connected.
- Run the code cell that captures frames from the webcam and overlays predicted emotions on each detected face.

Update this section to show the exact commands or notebook cells that users should run.

## Dataset

This project can be used with public facial emotion datasets such as FER2013 or your own labeled dataset.

- Place the dataset in the `data/` directory or configure the path inside the notebook.
- Ensure the folder structure and labels match the expectations of your data loading code.

Add details here about:
- Which dataset you actually used.
- How to download or prepare it.
- How the data directory should be structured.

## Model

Describe your model architecture and training setup here, for example:

- Convolutional Neural Network (CNN)-based classifier.
- Input: grayscale or RGB face images, size \(48 \times 48\) or similar.
- Output: probability distribution over emotion classes.

Mention:
- Loss function and optimizer.
- Number of epochs and batch size.
- Any data augmentation or preprocessing used.

## Results

Include a brief summary once you have results:

- Accuracy, F1‑score, or other metrics on the validation/test set.
- Example predictions or screenshots of the model in action.

You can also add plots (confusion matrix, training curves) exported from the notebook.

## Keyboard controls & runtime tuning

When the application window is focused, the following keys are available for control and live tuning:

- `S` – Save a screenshot to the `screenshots/` directory with a timestamped filename. On Windows, a short shutter sound is played. [web:10]
- `+` / `=` – Increase label smoothing to make the displayed emotion label less jittery. [web:10]
- `-` – Decrease label smoothing so the label reacts more quickly to changes. [web:10]
- `]` / `[` – Increase / decrease bounding-box lerp, which changes how smoothly the HUD bounding box follows the detected face. [web:10]
- `d` – Toggle deep-learning (DL) inference on or off; DL inference is OFF by default. [web:10]
- `m` – Switch the DL backend between `onnx` and `deepface` if both backends are available. [web:13]
- `ESC` – Exit the application. [web:10]


## Future Work

- Improve accuracy with deeper architectures or transfer learning.
- Add real‑time webcam emotion detection with a simple GUI.
- Export model for use in mobile or web applications.
- Extend to multi‑modal emotion recognition (text, audio, etc.).

## Contributing

Contributions, issues, and feature requests are welcome.

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit and push your changes.
4. Open a pull request describing your modifications.

## License

MIT License

## Acknowledgements

- OpenCV and deep learning frameworks used in this project.
- Public emotion recognition datasets and prior open‑source implementations that inspired this work.
