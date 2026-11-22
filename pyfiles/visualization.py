# File: visualization.py
# Purpose: Interactive visualizations for training dynamics, metrics, and feature maps

import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from pyfiles.constants import CLASSES

def plot_training_history(history, early_stopped_epoch=None):
    """
    Plot training and validation accuracy and loss using interactive Plotly figures.
    """
    epochs_range = list(range(1, len(history.history['accuracy']) + 1))

    col1, col2 = st.columns(2)

    with col1:
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(
            x=epochs_range, y=history.history['accuracy'],
            mode='lines+markers', name='Training Accuracy',
            line=dict(color='#2563EB', width=2.5),
            marker=dict(size=6)
        ))
        if 'val_accuracy' in history.history:
            fig_acc.add_trace(go.Scatter(
                x=epochs_range, y=history.history['val_accuracy'],
                mode='lines+markers', name='Validation Accuracy',
                line=dict(color='#10B981', width=2.5, dash='dash'),
                marker=dict(size=6)
            ))
        fig_acc.update_layout(
            title='Accuracy Progression',
            xaxis_title='Epoch',
            yaxis_title='Accuracy',
            template='plotly_white',
            height=360,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    with col2:
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=epochs_range, y=history.history['loss'],
            mode='lines+markers', name='Training Loss',
            line=dict(color='#F59E0B', width=2.5),
            marker=dict(size=6)
        ))
        if 'val_loss' in history.history:
            fig_loss.add_trace(go.Scatter(
                x=epochs_range, y=history.history['val_loss'],
                mode='lines+markers', name='Validation Loss',
                line=dict(color='#EF4444', width=2.5, dash='dash'),
                marker=dict(size=6)
            ))
        fig_loss.update_layout(
            title='Loss Convergence',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            template='plotly_white',
            height=360,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_loss, use_container_width=True)


def plot_confusion_matrix(y_true, y_pred_prob, threshold=0.5):
    """
    Calculate and display an interactive Confusion Matrix Heatmap using Plotly.
    Returns:
      dict: {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
    """
    y_pred = (y_pred_prob >= threshold).astype(int).flatten()
    y_true = y_true.astype(int).flatten()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    class_labels = ["Dog (Class 0)", "Cat (Class 1)"]

    # Annotations
    z_text = [[str(val) for val in row] for row in cm]

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_labels,
        y=class_labels,
        text=z_text,
        texttemplate="%{text}",
        textfont={"size": 18, "color": "white"},
        colorscale='Blues',
        showscale=True,
        hoverongaps=False
    ))

    fig.update_layout(
        title='Confusion Matrix Heatmap',
        xaxis_title='Predicted Class',
        yaxis_title='Actual Class',
        height=380,
        margin=dict(l=40, r=40, t=50, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    acc = float(np.mean(y_true == y_pred))

    return {
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }


def visualize_feature_maps(model, preprocessed_image, max_filters=8):
    """
    Visualizes the internal activations of the first Conv2D layer.
    Shows the distinct spatial features (edges, contours, gradients) captured by learned filters.
    """
    first_conv_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            first_conv_layer = layer
            break
        elif hasattr(layer, "layers"):
            for sub_layer in layer.layers:
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    first_conv_layer = sub_layer
                    break
            if first_conv_layer:
                break

    if first_conv_layer is None:
        st.info("No Conv2D layer found in the chosen model for feature map visualization.")
        return

    activation_model = tf.keras.Model(inputs=model.input, outputs=first_conv_layer.output)
    feature_maps = activation_model.predict(preprocessed_image, verbose=0)

    num_filters = min(max_filters, feature_maps.shape[-1])
    cols = 4
    rows = int(np.ceil(num_filters / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.8 * rows))
    axes = np.array(axes).reshape(-1)

    for i in range(num_filters):
        fmap = feature_maps[0, :, :, i]
        fmap -= fmap.mean()
        std = fmap.std()
        if std > 1e-5:
            fmap /= std
        fmap *= 64
        fmap += 128
        fmap = np.clip(fmap, 0, 255).astype('uint8')

        axes[i].imshow(fmap, cmap='magma')
        axes[i].set_title(f"Kernel Filter #{i+1}", fontsize=10, fontweight='bold')
        axes[i].axis('off')

    for j in range(num_filters, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f"Layer Activations: '{first_conv_layer.name}' ({num_filters} Feature Detectors)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
