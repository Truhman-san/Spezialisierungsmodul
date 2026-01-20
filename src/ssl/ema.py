from __future__ import annotations
import tensorflow as tf

class EMATeacher:
    def __init__(self, student: tf.keras.Model, decay: float = 0.999):
        self.decay = tf.constant(decay, tf.float32)
        self.teacher = tf.keras.models.clone_model(student)
        self.teacher.set_weights(student.get_weights())
        self.teacher.trainable = False

    @tf.function
    def update(self, student: tf.keras.Model):
        sw = student.weights
        tw = self.teacher.weights
        for s, t in zip(sw, tw):
            t.assign(self.decay * t + (1.0 - self.decay) * s)
