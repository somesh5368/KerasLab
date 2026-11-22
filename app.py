# File: app.py
# Purpose: High-Performance Deep Learning Vision Studio
# Features: Custom Deep CNN vs State-of-the-art Transfer Learning (VGG16, ResNet50),
#           Advanced Regularization (Dropout & Early Stopping), Interactive Image Upload,
#           Real-Time Confidence Analytics, Confusion Matrix, and Conv2D Feature Map Visualizer.

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from pyfiles.constants import IMG_SIZE, CLASSES
from pyfiles.data_loader import load_data, check_dataset_status, generate_demo_dataset
from pyfiles.model import create_model
from pyfiles.preprocessing import preprocess_image, load_display_image
from pyfiles.visualization import (
    plot_training_history,
    plot_confusion_matrix,
    visualize_feature_maps
)
from pyfiles.callbacks import StreamlitCallback

# Page setup
st.set_page_config(
    page_title="NeuralVision Studio | Deep Learning Classifier",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #2563EB, #7C3AED, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

def ensure_initial_setup():
    """Ensure dataset and model exist seamlessly on first run."""
    train_path = "dataset/training_set/training_set"
    test_path = "dataset/test_set/test_set"
    status_ok, _ = check_dataset_status(train_path, test_path)
    if not status_ok:
        generate_demo_dataset()

if "setup_done" not in st.session_state:
    ensure_initial_setup()
    st.session_state["setup_done"] = True

# App Header
st.markdown('<div class="main-header">⚡ NeuralVision Studio: Deep Learning Lab</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">End-to-End Image Classification • Custom CNN vs Transfer Learning (VGG16 / ResNet50) • Adaptive Regularization</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Architecture & Hyperparameters")

    model_choice = st.selectbox(
        "🧠 Neural Architecture",
        ["Custom CNN", "VGG16 (Transfer Learning)", "ResNet50 (Transfer Learning)"],
        index=0,
        help="Custom CNN builds from scratch with Conv-Pool blocks. VGG16 & ResNet50 leverage ImageNet pre-trained representations."
    )

    st.markdown("---")
    st.subheader("🛡️ Regularization Controls")

    dropout_rate = st.slider(
        "Dropout Rate",
        min_value=0.0,
        max_value=0.7,
        value=0.5,
        step=0.05,
        help="Randomly deactivates neurons during training to prevent co-adaptation and overfitting."
    )

    use_early_stopping = st.checkbox(
        "Enable Early Stopping",
        value=True,
        help="Monitors validation loss and terminates training at optimal generalization checkpoint."
    )

    patience = 3
    if use_early_stopping:
        patience = st.number_input("Patience (Epochs)", min_value=1, max_value=10, value=3)

    st.markdown("---")
    st.subheader("⚡ Optimization")
    learning_rate = st.select_slider(
        "Learning Rate (Adam)",
        options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
        value=0.001
    )
    epochs = st.slider("Epochs", min_value=1, max_value=30, value=5)
    batch_size = st.select_slider("Batch Size", options=[8, 16, 32, 64], value=16)

    st.markdown("---")
    st.subheader("📁 Data Pipeline")
    train_dir = "dataset/training_set/training_set"
    test_dir = "dataset/test_set/test_set"
    is_ready, msg = check_dataset_status(train_dir, test_dir)

    if is_ready:
        st.success("✅ Dataset Pipeline Active")
    else:
        st.warning(f"⚠️ {msg}")
        if st.button("Initialize Data Pipeline"):
            generate_demo_dataset()
            st.rerun()

# Modern Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Model Training & Analytics",
    "🔍 Real-Time Inference & Upload",
    "📈 Regularization Dynamics",
    "🧠 Architecture Deep Dive"
])

# -------------------------------------------------------------
# TAB 1: MODEL TRAINING & ANALYTICS
# -------------------------------------------------------------
with tab1:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Dataset Paths & Execution")
        train_path_input = st.text_input("Training Directory", "dataset/training_set/training_set")
        test_path_input = st.text_input("Validation Directory", "dataset/test_set/test_set")

        valid_dataset, err_msg = check_dataset_status(train_path_input, test_path_input)
        if not valid_dataset:
            st.error(f"Directory Error: {err_msg}")
            if st.button("Generate Clean Dataset", key="regen_btn"):
                generate_demo_dataset()
                st.success("Dataset synthesized successfully!")
                st.rerun()

        start_training = st.button("🚀 Train Model", type="primary", use_container_width=True, disabled=not valid_dataset)

    with col_right:
        st.subheader("Selected Model Specifications")
        if model_choice == "Custom CNN":
            st.info(
                "**Upgraded Deep CNN with Data Augmentation & Batch Normalization**:\n"
                "- Active Data Augmentation: `RandomFlip`, `RandomRotation(10%)`, `RandomZoom(10%)`\n"
                "- 3 Dual-Convolutional Blocks with `BatchNormalization` (32 -> 64 -> 128 filters)\n"
                "- 3 `MaxPooling2D` spatial downsampling layers with progressive Dropout\n"
                f"- Classification Head: Flatten -> Dense(128) -> BatchNorm -> Dropout({dropout_rate}) -> Sigmoid"
            )
        elif model_choice == "VGG16 (Transfer Learning)":
            st.info(
                "**VGG16 Transfer Learning**:\n"
                "- 13 Convolutional layers pre-trained on ImageNet (Frozen feature extractor)\n"
                "- `GlobalAveragePooling2D` spatial feature aggregation\n"
                f"- Custom Dense classification head with Dropout ({dropout_rate})"
            )
        else:
            st.info(
                "**ResNet50 Transfer Learning**:\n"
                "- 50-layer deep Residual Network with identity skip connections (Frozen base)\n"
                "- Eliminates vanishing gradients in deep representations\n"
                f"- Custom Dense classification head with Dropout ({dropout_rate})"
            )

    st.markdown("---")

    # Training Execution
    if start_training:
        with st.status(f"Training {model_choice}...", expanded=True) as status:
            st.write("📥 Loading & preprocessing image batches (RGB 128x128)...")
            x_train, y_train = load_data(train_path_input, img_size=IMG_SIZE)
            x_test, y_test = load_data(test_path_input, img_size=IMG_SIZE)
            st.write(f"Loaded **{len(x_train)}** training samples and **{len(x_test)}** validation samples.")

            st.write(f"🔨 Compiling {model_choice} with Adam (lr={learning_rate})...")
            model = create_model(
                model_name=model_choice,
                learning_rate=learning_rate,
                dropout_rate=dropout_rate,
                img_size=IMG_SIZE
            )

            callbacks = [StreamlitCallback(epochs, batch_size)]
            if use_early_stopping:
                early_stop_callback = EarlyStopping(
                    monitor='val_loss',
                    patience=patience,
                    restore_best_weights=True,
                    verbose=1
                )
                callbacks.append(early_stop_callback)
                st.write(f"🛡️ EarlyStopping active (monitors `val_loss`, patience={patience}).")

            st.write("🏃 Training model across epochs...")
            history = model.fit(
                x_train, y_train,
                validation_data=(x_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0
            )

            # Save models
            model.save("dogcatclassifier.keras")
            model.save("dogcatclassifier.h5")
            st.session_state["trained_model"] = model
            st.session_state["training_history"] = history
            st.session_state["x_test"] = x_test
            st.session_state["y_test"] = y_test
            st.session_state["model_choice"] = model_choice
            status.update(label="✅ Training Pipeline Complete! Model weights serialized.", state="complete", expanded=False)

    # Display Metrics if model exists
    if "training_history" in st.session_state:
        history = st.session_state["training_history"]
        x_test = st.session_state["x_test"]
        y_test = st.session_state["y_test"]
        model = st.session_state["trained_model"]

        st.subheader("📈 Training Performance Analytics")
        plot_training_history(history)

        st.markdown("---")
        st.subheader("🎯 Model Discrimination & Confusion Matrix")

        eval_col1, eval_col2 = st.columns([1, 1])
        y_pred_prob = model.predict(x_test, verbose=0)

        with eval_col1:
            metrics = plot_confusion_matrix(y_test, y_pred_prob)

        with eval_col2:
            st.markdown("#### Performance Metrics")
            m_col1, m_col2 = st.columns(2)
            m_col1.metric("Overall Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
            m_col2.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
            m_col3, m_col4 = st.columns(2)
            m_col3.metric("Recall (Sensitivity)", f"{metrics['recall'] * 100:.2f}%")
            m_col4.metric("F1-Score", f"{metrics['f1'] * 100:.2f}%")

            st.caption(
                "**Diagnostic Metric Breakdown:** Evaluates classification fidelity across "
                "True Positives, True Negatives, False Positives, and False Negatives."
            )

# -------------------------------------------------------------
# TAB 2: REAL-TIME INFERENCE & UPLOAD
# -------------------------------------------------------------
with tab2:
    st.header("🔍 Real-Time Image Inference")
    st.markdown("Upload any photo or test with pre-packaged samples to evaluate prediction probability and confidence.")

    pred_left, pred_right = st.columns([1, 1])

    with pred_left:
        uploaded_file = st.file_uploader("📤 Upload Image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

        sample_choice = None
        if not uploaded_file:
            st.markdown("**Or choose a built-in benchmark image:**")
            sample_options = ["images/cat1.jpg", "images/dog1.jpg"]
            available_samples = [s for s in sample_options if os.path.exists(s)]
            if available_samples:
                sample_choice = st.selectbox("Benchmark Images", available_samples)
            else:
                sample_choice = st.text_input("Image Path", "images/cat1.jpg")

        target_input = uploaded_file if uploaded_file else sample_choice

        if target_input:
            try:
                display_img = load_display_image(target_input)
                try:
                    st.image(display_img, caption="Input Tensor (RGB)", use_column_width=True)
                except TypeError:
                    st.image(display_img, caption="Input Tensor (RGB)", use_container_width=True)
            except Exception as exc:
                st.error(f"Image load error: {exc}")
                target_input = None

    with pred_right:
        st.subheader("Inference & Probabilistic Classification")
        predict_button = st.button("⚡ Run Inference", type="primary", use_container_width=True, disabled=(target_input is None))

        if predict_button and target_input:
            if "trained_model" in st.session_state:
                model = st.session_state["trained_model"]
            elif os.path.exists("dogcatclassifier.keras"):
                model = tf.keras.models.load_model("dogcatclassifier.keras")
            elif os.path.exists("dogcatclassifier.h5"):
                model = tf.keras.models.load_model("dogcatclassifier.h5")
            else:
                model = create_model("Custom CNN", img_size=IMG_SIZE)

            prepped = preprocess_image(target_input, img_size=IMG_SIZE)
            prob = float(model.predict(prepped, verbose=0)[0][0])

            cat_conf = prob * 100
            dog_conf = (1.0 - prob) * 100
            pred_class = "Cat" if prob >= 0.5 else "Dog"
            confidence = cat_conf if pred_class == "Cat" else dog_conf

            if pred_class == "Cat":
                st.success(f"### Classification: 🐱 **Cat** ({confidence:.1f}% Confidence)")
            else:
                st.success(f"### Classification: 🐶 **Dog** ({confidence:.1f}% Confidence)")

            # Interactive Plotly Confidence Bar
            fig_conf = go.Figure(go.Bar(
                x=[dog_conf, cat_conf],
                y=['Dog', 'Cat'],
                orientation='h',
                marker=dict(color=['#3B82F6', '#EC4899']),
                text=[f"{dog_conf:.1f}%", f"{cat_conf:.1f}%"],
                textposition='inside'
            ))
            fig_conf.update_layout(
                title='Softmax / Sigmoid Probability Distribution',
                xaxis=dict(range=[0, 100], title='Confidence (%)'),
                yaxis=dict(title='Class'),
                height=220,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            st.plotly_chart(fig_conf, use_container_width=True)

            # Conv2D Feature Map Visualization
            st.markdown("---")
            st.subheader("🔬 Convolutional Feature Map Activations")
            st.caption("Visualizing spatial edge, gradient, and texture detectors extracted by the first Conv2D layer:")
            visualize_feature_maps(model, prepped)

# -------------------------------------------------------------
# TAB 3: REGULARIZATION DYNAMICS
# -------------------------------------------------------------
with tab3:
    st.header("📈 Regularization Dynamics: Combating Overfitting")
    st.markdown(
        "High-capacity neural networks can easily overfit by memorizing pixel-level training idiosyncrasies "
        "rather than generalizable semantic patterns. Here is the comparative impact of **Dropout** and **Early Stopping**:"
    )

    reg_col1, reg_col2 = st.columns(2)

    with reg_col1:
        st.subheader("❌ Unregularized Baseline (Overfitting)")
        st.markdown(
            "- **Training Loss**: Approaches zero exponentially\n"
            "- **Validation Loss**: Diverges upwards after initial epochs\n"
            "- **Co-adaptation**: Units develop high inter-dependency, failing on out-of-distribution inputs"
        )
        ep = np.arange(1, 16)
        train_loss_overfit = 1.0 / (ep + 0.5)
        val_loss_overfit = 0.5 + 0.05 * (ep - 4)**2

        fig_overfit = go.Figure()
        fig_overfit.add_trace(go.Scatter(x=list(ep), y=list(train_loss_overfit), mode='lines+markers', name='Train Loss', line=dict(color='#2563EB')))
        fig_overfit.add_trace(go.Scatter(x=list(ep), y=list(val_loss_overfit), mode='lines+markers', name='Validation Loss (Diverging)', line=dict(color='#EF4444', dash='dash')))
        fig_overfit.add_vline(x=4, line_width=2, line_dash="dot", line_color="orange", annotation_text="Overfitting Threshold")
        fig_overfit.update_layout(title="Unregularized Learning Curves", xaxis_title="Epoch", yaxis_title="Loss", height=340)
        st.plotly_chart(fig_overfit, use_container_width=True)

    with reg_col2:
        st.subheader("✅ Regularized Model (Dropout + Early Stopping)")
        st.markdown(
            "- **Dropout Regularization**: Zeroes activations randomly, forcing redundant distributed representations\n"
            "- **Early Stopping**: Halts training at minimum validation loss, preventing noise memorization\n"
            "- **Generalization**: Stable validation convergence with tight generalization gap"
        )
        val_loss_regularized = 0.35 + 0.4 / (ep**0.6)
        train_loss_regularized = 0.25 + 0.5 / (ep**0.7)

        fig_reg = go.Figure()
        fig_reg.add_trace(go.Scatter(x=list(ep), y=list(train_loss_regularized), mode='lines+markers', name='Train Loss', line=dict(color='#2563EB')))
        fig_reg.add_trace(go.Scatter(x=list(ep), y=list(val_loss_regularized), mode='lines+markers', name='Validation Loss (Converging)', line=dict(color='#10B981', dash='dash')))
        fig_reg.update_layout(title="Regularized Learning Curves", xaxis_title="Epoch", yaxis_title="Loss", height=340)
        st.plotly_chart(fig_reg, use_container_width=True)

# -------------------------------------------------------------
# TAB 4: ARCHITECTURE DEEP DIVE
# -------------------------------------------------------------
with tab4:
    st.header("🧠 Neural Network Architecture Fundamentals")
    st.markdown("Technical overview of the convolutional, residual, and transfer learning mechanisms implemented in this studio:")

    with st.expander("🔬 1. Convolutional & Pooling Layer Mechanics", expanded=True):
        st.markdown("""
        - **Convolutional Layer (`Conv2D`)**: Convolves 2D spatial filters ($3\\times3$) across image channels using localized receptive fields and shared kernel weights. Rectified Linear Unit (ReLU) introduces non-linear activations:
          $$f(x) = \\max(0, x)$$
        - **Max Pooling (`MaxPooling2D`)**: Applies non-linear downsampling ($2\\times2$) to reduce spatial dimensionality by 50%, providing spatial translation invariance and cutting parameter complexity.
        - **Feature Aggregation**: Multi-stage Conv-Pool blocks extract hierarchical visual primitives: edges $\\rightarrow$ textures $\\rightarrow$ parts $\\rightarrow$ object semantics.
        """)

    with st.expander("🛡️ 2. Dropout & Early Stopping Mechanism", expanded=True):
        st.markdown("""
        - **Inverted Dropout ($p = 0.5$)**: Randomly drops connections during each forward pass with probability $p$. This acts as an efficient ensemble over $2^N$ sub-architectures, preventing complex co-adaptation of hidden units.
        - **Early Stopping**: Monitors validation loss $\\mathcal{L}_{\\text{val}}$. When $\\mathcal{L}_{\\text{val}}$ fails to decrease for $k$ consecutive epochs (`patience`), training is terminated and the global optimal checkpoint is restored.
        """)

    with st.expander("⚡ 3. Transfer Learning: VGG16 & ResNet50 Architectures", expanded=True):
        st.markdown("""
        - **Transfer Learning**: Leverages pre-trained feature representations learned on the ImageNet dataset (1.4 million images, 1000 categories).
        - **VGG16**: Deep homogeneous architecture composed of stacked $3\\times3$ convolutions with increasing receptive field depth.
        - **ResNet50 & Residual Learning**: Uses identity skip connections to solve the degradation problem in deep networks:
          $$\\mathcal{H}(x) = \\mathcal{F}(x, \\{W_i\\}) + x$$
          Allows gradients to flow uninterrupted during backpropagation, enabling 50+ layer training without gradient vanishing.
        """)

    with st.expander("📊 4. Confusion Matrix & Classification Metrics", expanded=True):
        st.markdown("""
        - **Accuracy**: $\\frac{TP + TN}{TP + TN + FP + FN}$ (Overall correctness)
        - **Precision**: $\\frac{TP}{TP + FP}$ (Positive predictive value)
        - **Recall**: $\\frac{TP}{TP + FN}$ (True positive rate / Sensitivity)
        - **F1-Score**: $2 \\cdot \\frac{\\text{Precision} \\cdot \\text{Recall}}{\\text{Precision} + \\text{Recall}}$ (Harmonic mean of precision and recall)
        """)

# Footer
st.markdown("---")
st.caption("NeuralVision Studio • Developed by Somesh Pandey • TensorFlow, Keras & Streamlit • Production-Ready Deep Learning Pipeline")
