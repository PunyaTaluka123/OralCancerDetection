# Oral Cancer Detection

A binary image classifier that takes a photo of the mouth (lips or tongue) and estimates whether it shows oral cancer. It's a transfer-learned MobileNetV2, exported to an 8-bit-quantised TensorFlow Lite model (2.5 MB) small enough to run on a phone.

It's a prototype screening aid and a learning project, not a medical device.

---

## Results

Measured on 29 Oct 2026 by running the shipped `models/optimized_model.tflite` over both image sets. A prediction counts as cancer when P(non-cancer) < 0.5.

| Evaluation set | Images | Accuracy | Sensitivity (cancer) | Specificity |
| --- | --- | --- | --- | --- |
| Development set (500 cancer / 450 non-cancer, Kaggle) | 950 | 94.6% | 93.4% | 96.0% |
| [Oral Cancer (Lips and Tongue) images](https://www.kaggle.com/shivam17299/oral-cancer-lips-and-tongue-images), a separate Kaggle set (87 / 44) | 131 | 80.2% | 88.5% | 63.6% |

**How to read these numbers.** The model was trained on the development set and the train/validation split wasn't saved, so 94.6% is closer to training accuracy than to a test score. The 131-image set comes from a different source, so it's the better estimate of how the model generalises. On it, recall stays high but specificity drops: the model over-calls cancer on healthy mouths it hasn't seen. For a screening tool that's the safer direction to fail, but it is clearly where the work is.

## Model

| | |
| --- | --- |
| Backbone | MobileNetV2, ImageNet weights, frozen |
| Head | GlobalAveragePooling → Dense(128, ReLU) → Dropout(0.5) → Dense(1, sigmoid) |
| Input | 224 × 224 RGB, pixels scaled to [0, 1] (`x / 255`) |
| Output | **P(non-cancer)**. Keras' `flow_from_directory` orders classes alphabetically, so `CANCER` = 0 and `NON CANCER` = 1. |
| Loss / optimiser | Binary cross-entropy, Adam (final LR 1e-5) |
| Export | `TFLiteConverter` with `Optimize.DEFAULT` (dynamic-range quantisation): 11 MB → 2.5 MB |

Why MobileNetV2: about 1,000 images is far too few to train a CNN from scratch, so a pretrained backbone does the feature extraction and only a small head is learned. MobileNetV2 suits the on-device goal: it's small, fast on CPU, and its depthwise-separable architecture quantises well.

## Project history

1. **Baseline**: a quick MobileNet-based model with a 2-way softmax (`models/baseline/`, Teachable Machine export format), evaluated on the 131-image set (`results.csv`).
2. **Grad-CAM**: started on explainability for the baseline (`scripts/gradcam.py`), to show which region of the mouth drove a prediction. Still a work in progress, see below.
3. **Custom model**: retrained in Keras on the larger 950-image set with the architecture above (`models/best_oral_cancer_model.keras`).
4. **Mobile export**: quantised to TFLite for an on-device app (`scripts/convert_to_tflite.py`). The plan then moved from a mobile app to a web front end, which wasn't built.

## Repository layout

```
models/
  best_oral_cancer_model.keras   final Keras model (MobileNetV2 + head)
  optimized_model.tflite         quantised TFLite export, the one to use
  baseline/
    keras_model.h5, labels.txt   baseline model (0 = No Cancer, 1 = Cancer)
    results.csv                  baseline's per-image predictions on the 131-image set
scripts/
  convert_to_tflite.py           Keras → quantised TFLite
  gradcam.py                     Grad-CAM heatmap overlay (work in progress)
```

Datasets aren't committed. Put images under `data/` (git-ignored) to run anything.

## Usage

```bash
pip install ai-edge-litert pillow numpy     # or full tensorflow (tf.lite.Interpreter has the same API)
```

```python
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

interpreter = Interpreter(model_path="models/optimized_model.tflite")
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

img = Image.open("mouth.jpg").convert("RGB").resize((224, 224))
x = np.asarray(img, dtype=np.float32)[None] / 255.0

interpreter.set_tensor(inp["index"], x)
interpreter.invoke()
p_non_cancer = float(interpreter.get_tensor(out["index"])[0][0])
print("cancer" if p_non_cancer < 0.5 else "non-cancer", f"(P(cancer) = {1 - p_non_cancer:.2f})")
```

## Known issues and next steps

- **No saved split or training script.** The notebook that produced `best_oral_cancer_model.keras` isn't in the repo, so the training run can't be reproduced from here. Next: a scripted training run with a fixed, saved split and a held-out test set.
- **Specificity on unseen data (64%).** More varied healthy-mouth images, colour and lighting augmentation, and calibrating the decision threshold on a validation set rather than using 0.5.
- **Grad-CAM is unfinished.** `gradcam.py` takes the gradient with respect to the argmax *index* rather than the class score, and writes into an immutable tensor. The fix is `tape.gradient(predictions[:, class_idx], conv_output)` plus TF ops for the channel weighting.
- **Baseline label encoding.** `models/baseline/results.csv` stores true labels as cancer = 0 (folder order) while the baseline's `labels.txt` uses cancer = 1. Read with one consistent encoding, the baseline agrees with the ground truth on 93/131 images (71%).

## Tech stack

Python · TensorFlow / Keras · MobileNetV2 · TensorFlow Lite · OpenCV · NumPy
