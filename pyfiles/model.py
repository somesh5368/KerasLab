# File: model.py
# Purpose: Define and compile models: Custom CNN, VGG16, and ResNet50

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Conv2D, MaxPooling2D, Dropout, Flatten,
    GlobalAveragePooling2D, Input, BatchNormalization,
    RandomFlip, RandomRotation, RandomZoom
)
from tensorflow.keras.optimizers import Adam
from pyfiles.constants import IMG_SIZE

@tf.keras.utils.register_keras_serializable()
def vgg_preprocess(x):
    from tensorflow.keras.applications.vgg16 import preprocess_input
    return preprocess_input(x)

@tf.keras.utils.register_keras_serializable()
def resnet_preprocess(x):
    from tensorflow.keras.applications.resnet50 import preprocess_input
    return preprocess_input(x)

def load_saved_model(filepath):
    """
    Safely load a trained model with custom preprocessing objects.
    """
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_prep
    from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_prep

    custom_objects = {
        'preprocess_input': resnet_prep,
        'resnet_preprocess': resnet_preprocess,
        'vgg_preprocess': vgg_preprocess,
    }
    try:
        return tf.keras.models.load_model(filepath, custom_objects=custom_objects, safe_mode=False)
    except Exception:
        return tf.keras.models.load_model(filepath, safe_mode=False)

def create_model(model_name="Custom CNN", learning_rate=0.001, dropout_rate=0.5, img_size=IMG_SIZE):
    """
    Create and compile a model based on chosen architecture.

    Options:
      - "Custom CNN" (Deep Multi-layer Convolutional Network)
      - "VGG16 (Transfer Learning)" (Pre-trained ImageNet deep feature extractor)
      - "ResNet50 (Transfer Learning)" (Pre-trained ImageNet residual network with skip connections)
    """
    if model_name == "VGG16 (Transfer Learning)":
        from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
        base_model = VGG16(
            weights="imagenet",
            include_top=False,
            input_shape=(img_size, img_size, 3)
        )
        base_model.trainable = False  # Freeze pretrained convolutional base

        model = Sequential([
            Input(shape=(img_size, img_size, 3)),
            tf.keras.layers.Rescaling(255.0),
            tf.keras.layers.Lambda(vgg_preprocess),
            base_model,
            GlobalAveragePooling2D(),
            Dense(128, activation="relu"),
            Dropout(dropout_rate),
            Dense(1, activation="sigmoid")
        ], name="VGG16_TransferLearning")

    elif model_name == "ResNet50 (Transfer Learning)":
        from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
        base_model = ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=(img_size, img_size, 3)
        )
        base_model.trainable = False  # Freeze pretrained convolutional base

        model = Sequential([
            Input(shape=(img_size, img_size, 3)),
            tf.keras.layers.Rescaling(255.0),
            tf.keras.layers.Lambda(resnet_preprocess),
            base_model,
            GlobalAveragePooling2D(),
            Dense(128, activation="relu"),
            Dropout(dropout_rate),
            Dense(1, activation="sigmoid")
        ], name="ResNet50_TransferLearning")

    else:
        # High-Performance Modern Custom Convolutional Neural Network
        model = Sequential([
            Input(shape=(img_size, img_size, 3)),

            # 1. On-the-fly Data Augmentation (active during training only)
            RandomFlip("horizontal"),
            RandomRotation(0.1),
            RandomZoom(0.1),

            # 2. Block 1 (32 filters)
            Conv2D(32, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.2),

            # 3. Block 2 (64 filters)
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),

            # 4. Block 3 (128 filters)
            Conv2D(128, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.3),

            # 5. Fully Connected Classification Head
            Flatten(),
            Dense(128, activation="relu"),
            BatchNormalization(),
            Dropout(dropout_rate),
            Dense(1, activation="sigmoid")
        ], name="Custom_CNN")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model
