# ⚡ NeuralVision Studio | KerasLab

An end-to-end, high-performance Deep Learning Computer Vision studio and image classification workbench built with **TensorFlow / Keras**, **OpenCV**, **Plotly**, and **Streamlit**.

Designed and maintained by **Somesh Pandey** ([spandey5368@gmail.com](mailto:spandey5368@gmail.com)).

---

## 🌟 Key Highlights & Capabilities

- **Multi-Architecture Neural Suite**:
  - **Custom CNN**: 3-block hierarchical convolutional network ($32 \rightarrow 64 \rightarrow 128$ filters, $3\times3$ kernels, ReLU, MaxPooling2D).
  - **VGG16 (Transfer Learning)**: Deep 16-layer feature representation pre-trained on ImageNet with frozen feature extraction and `GlobalAveragePooling2D`.
  - **ResNet50 (Transfer Learning)**: 50-layer deep Residual Network with identity skip connections that overcome vanishing gradients.
- **Adaptive Regularization & Overfitting Defense**:
  - Configurable **Dropout** ($p \in [0.0, 0.7]$) preventing neuron co-adaptation.
  - **Early Stopping** callback monitoring validation loss with configurable patience and automatic restoration of the best model weights.
- **Real-Time Live Telemetry (`StreamlitCallback`)**:
  - Live progress tracking with asynchronous Plotly loss and accuracy curves updated at the end of every epoch.
- **Diagnostic Evaluation & Confusion Matrix**:
  - Interactive Confusion Matrix Heatmap.
  - Quantitative diagnostics: Overall Accuracy, Precision, Recall (Sensitivity), and F1-Score.
- **Explainable AI (XAI) & Feature Map Visualizer**:
  - Peeks inside the neural network's receptive fields by rendering the activations of the first convolutional layer (`Conv2D`), showing detected edges, gradients, and textures in real-time.
- **Zero-Setup Procedural Dataset Synthesizer**:
  - Built-in geometric demo generator using OpenCV for cats and dogs with randomized noise. Runs out-of-the-box without needing to download external multi-gigabyte datasets!

---

## 📁 Repository Structure

```
KerasLab/
├── app.py                      # Main Streamlit web application & UI dashboard
├── pyfiles/
│   ├── constants.py            # Global hyperparameter constants, image dimensions & class maps
│   ├── preprocessing.py        # Image decoding, tensor normalization, and color conversion
│   ├── model.py                # Model builders: Custom CNN, VGG16, and ResNet50
│   ├── data_loader.py          # Dataset ingestion, validation checks, and synthetic data generator
│   ├── callbacks.py            # Custom Keras callback for real-time Streamlit & Plotly updates
│   └── visualization.py        # Interactive loss/acc curves, Confusion Matrix, and Conv2D feature maps
├── dataset/                    # Local training & validation image sets
│   ├── training_set/
│   │   ├── cats/
│   │   └── dogs/
│   └── test_set/
│       ├── cats/
│       └── dogs/
├── images/                     # Built-in sample benchmark images for inference testing
├── requirements.txt            # Python dependencies (TensorFlow, Streamlit, OpenCV, Plotly, etc.)
├── ModelArchitecture.md        # Comprehensive technical guide to neural layer designs
├── LearnWithPrompts.md         # Educational prompt-based learning guide
└── README.md                   # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.10 or higher installed.

### 2. Virtual Environment Setup

Using standard `venv`:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS / Linux:
source .venv/bin/activate
```

Or using `uv` for ultra-fast package management:
```bash
pip install uv
uv venv
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
# or with uv:
uv pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```
The application will launch in your default web browser at `http://localhost:8501`.

---

## 🖥️ UI Workflow & Application Tabs

1. **🚀 Model Training & Analytics**:
   - Configure data directories and inspect model specifications.
   - Click **Train Model** to run training with live epoch updates and real-time loss/accuracy charts.
   - Automatically computes and displays the Confusion Matrix Heatmap, Accuracy, Precision, Recall, and F1-Score on test data upon completion.
   - Automatically serializes trained model weights to `dogcatclassifier.keras` and `dogcatclassifier.h5`.

2. **🔍 Real-Time Inference & Upload**:
   - Upload any image (`.jpg`, `.png`, `.jpeg`) from your computer or pick from built-in sample images (`images/cat1.jpg`, `images/dog1.jpg`).
   - Click **Run Inference** to classify the image as Cat or Dog with exact percentage confidence.
   - View the horizontal probability bar and inspect the **Conv2D Feature Map Activations** showing what low-level visual patterns the filters extracted.

3. **📈 Regularization Dynamics**:
   - Interactive comparative charts contrasting unregularized overfitting models against regularized networks using Dropout and Early Stopping.

4. **🧠 Architecture Deep Dive**:
   - Mathematical and theoretical breakdowns of Convolutions, Max Pooling, ReLU activations, Residual Skip Connections, and Binary Cross-Entropy loss.

---

## ⚙️ Connecting to Your Personal GitHub Repository

If you cloned this repository and want to push it to your personal GitHub account:

```bash
# 1. Check your current git remote
git remote -v

# 2. Update remote URL to your personal GitHub repository:
git remote set-url origin https://github.com/somesh5368/KerasLab.git

# 3. Stage, commit, and push your changes:
git add .
git commit -m "Personalize and upgrade NeuralVision Studio with modern multi-architecture pipeline"
git push -u origin main
```

---

## 📜 License & Credits

- Developed and maintained by **Somesh Pandey**.
- Licensed under the **MIT License**.
