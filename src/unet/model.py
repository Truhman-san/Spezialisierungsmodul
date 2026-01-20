import tensorflow as tf
from tensorflow.keras import layers as L


def conv_block(x, filters: int, name: str):
    x = L.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv1")(x)
    x = L.BatchNormalization(name=f"{name}_bn1")(x)
    x = L.Activation("relu", name=f"{name}_relu1")(x)

    x = L.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv2")(x)
    x = L.BatchNormalization(name=f"{name}_bn2")(x)
    x = L.Activation("relu", name=f"{name}_relu2")(x)
    return x


def build_unet(
    input_shape=(600, 600, 1),
    base_filters: int = 16,
    num_classes: int = 4,
):
    inputs = L.Input(shape=input_shape)

    # Encoder (3x Pooling: /2, /4, /8)
    c1 = conv_block(inputs, base_filters, name="enc1")
    p1 = L.MaxPooling2D(2, name="pool1")(c1)

    c2 = conv_block(p1, base_filters * 2, name="enc2")
    p2 = L.MaxPooling2D(2, name="pool2")(c2)

    c3 = conv_block(p2, base_filters * 4, name="enc3")
    p3 = L.MaxPooling2D(2, name="pool3")(c3)

    # Bottleneck
    bn = conv_block(p3, base_filters * 8, name="bottleneck")

    # Decoder (3x Up: *2, *4, *8 -> zurück auf Input-Size)
    u3 = L.Conv2DTranspose(base_filters * 4, 2, strides=2, padding="same", name="up3")(bn)
    u3 = L.Concatenate(name="concat3")([u3, c3])
    d3 = conv_block(u3, base_filters * 4, name="dec3")

    u2 = L.Conv2DTranspose(base_filters * 2, 2, strides=2, padding="same", name="up2")(d3)
    u2 = L.Concatenate(name="concat2")([u2, c2])
    d2 = conv_block(u2, base_filters * 2, name="dec2")

    u1 = L.Conv2DTranspose(base_filters, 2, strides=2, padding="same", name="up1")(d2)
    u1 = L.Concatenate(name="concat1")([u1, c1])
    d1 = conv_block(u1, base_filters, name="dec1")

    # EINZIGER Head: Multiclass-Segmentierung
    outputs = L.Conv2D(
        num_classes,
        1,
        activation="softmax",   # weil du from_logits=False verwendest
        name="main",            # Name ist egal, Output ist trotzdem ein Tensor, kein Dict
    )(d1)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="unet_multiclass_single",
    )
    return model
