"""CNN autoencoder architecture for ClearPix.

The network is fully convolutional, so it can restore images of any size whose
height and width are multiples of 4 (the app pads inputs automatically).

    input -> Encoder (2 stages) -> Bottleneck -> Decoder (2 stages) -> output

Optional skip connections carry fine spatial detail from encoder to decoder,
which noticeably improves detail preservation for denoising.
"""
from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

MODEL_NAME = "clearpix_autoencoder"


def _conv_block(x, filters: int, name: str):
    """Two 3x3 convolutions, each followed by batch norm and ReLU."""
    for i in (1, 2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv{i}")(x)
        x = layers.BatchNormalization(name=f"{name}_bn{i}")(x)
        x = layers.ReLU(name=f"{name}_relu{i}")(x)
    return x


def build_autoencoder(
    input_shape: tuple = (None, None, 3),
    base_filters: int = 32,
    skip_connections: bool = True,
) -> keras.Model:
    """Build the denoising autoencoder.

    Args:
        input_shape: (H, W, C). Use (None, None, 3) for variable-size images.
        base_filters: Number of filters in the first encoder stage; doubles per stage.
        skip_connections: Concatenate encoder features into the decoder.
    """
    f = base_filters
    inputs = keras.Input(shape=input_shape, name="noisy_image")

    # Encoder
    e1 = _conv_block(inputs, f, "enc1")
    p1 = layers.MaxPooling2D(2, name="enc1_pool")(e1)
    e2 = _conv_block(p1, f * 2, "enc2")
    p2 = layers.MaxPooling2D(2, name="enc2_pool")(e2)

    # Bottleneck
    b = _conv_block(p2, f * 4, "bottleneck")

    # Decoder
    d2 = layers.UpSampling2D(2, interpolation="bilinear", name="dec2_up")(b)
    d2 = layers.Conv2D(f * 2, 2, padding="same", activation="relu", name="dec2_reduce")(d2)
    if skip_connections:
        d2 = layers.Concatenate(name="dec2_skip")([d2, e2])
    d2 = _conv_block(d2, f * 2, "dec2")

    d1 = layers.UpSampling2D(2, interpolation="bilinear", name="dec1_up")(d2)
    d1 = layers.Conv2D(f, 2, padding="same", activation="relu", name="dec1_reduce")(d1)
    if skip_connections:
        d1 = layers.Concatenate(name="dec1_skip")([d1, e1])
    d1 = _conv_block(d1, f, "dec1")

    # Output layer: restored RGB image in [0, 1]
    outputs = layers.Conv2D(input_shape[-1], 3, padding="same", activation="sigmoid",
                            name="restored_image")(d1)
    return keras.Model(inputs, outputs, name=MODEL_NAME)


if __name__ == "__main__":
    build_autoencoder().summary()
