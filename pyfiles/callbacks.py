# File: callbacks.py
# Purpose: Define custom Keras callback to update Streamlit UI during training

import streamlit as st
from tensorflow.keras.callbacks import Callback
import plotly.graph_objects as go

class StreamlitCallback(Callback):
    """
    Custom Keras callback to update Streamlit UI during training in real-time.
    """
    def __init__(self, epochs, batch_size):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        self.epoch_progress = st.empty()
        self.loss_chart = st.empty()
        self.accuracy_chart = st.empty()
        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        progress = (epoch + 1) / self.epochs
        self.progress_bar.progress(min(progress, 1.0))
        self.status_text.text(f"Epoch {epoch + 1}/{self.epochs}")

        loss = logs.get('loss', 0.0)
        acc = logs.get('accuracy', 0.0)
        val_loss = logs.get('val_loss', 0.0)
        val_acc = logs.get('val_accuracy', 0.0)

        self.epoch_progress.text(
            f"Loss: {loss:.4f} | Accuracy: {acc:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}"
        )

        self.history['loss'].append(loss)
        self.history['accuracy'].append(acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_accuracy'].append(val_acc)

        self.update_charts()

    def update_charts(self):
        epochs = list(range(1, len(self.history['loss']) + 1))

        # Real-time Loss Chart
        loss_fig = go.Figure()
        loss_fig.add_trace(go.Scatter(x=epochs, y=self.history['loss'], mode='lines+markers', name='Train Loss', line=dict(color='#2ca02c')))
        if len(self.history['val_loss']) > 0:
            loss_fig.add_trace(go.Scatter(x=epochs, y=self.history['val_loss'], mode='lines+markers', name='Val Loss', line=dict(color='#d62728')))
        loss_fig.update_layout(title='Live Training Loss', xaxis_title='Epoch', yaxis_title='Loss', height=320)
        self.loss_chart.plotly_chart(loss_fig, use_container_width=True)

        # Real-time Accuracy Chart
        acc_fig = go.Figure()
        acc_fig.add_trace(go.Scatter(x=epochs, y=self.history['accuracy'], mode='lines+markers', name='Train Accuracy', line=dict(color='#1f77b4')))
        if len(self.history['val_accuracy']) > 0:
            acc_fig.add_trace(go.Scatter(x=epochs, y=self.history['val_accuracy'], mode='lines+markers', name='Val Accuracy', line=dict(color='#ff7f0e')))
        acc_fig.update_layout(title='Live Training Accuracy', xaxis_title='Epoch', yaxis_title='Accuracy', height=320)
        self.accuracy_chart.plotly_chart(acc_fig, use_container_width=True)
