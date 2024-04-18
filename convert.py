import tensorflow as tf

# Load your Functional model
model = tf.keras.models.load_model('model.keras', compile=False)

# Workaround: Subclass the Functional model


class MyModel(tf.keras.Model):
    def __init__(self, original_model):
        super().__init__(inputs=original_model.inputs, outputs=original_model.outputs)
        self.call = original_model.call

    def _set_save_spec(self, inputs):
        self.inputs = inputs

    def _get_save_spec(self):  # Remove the dynamic_batch_size argument
        input_specs = [tf.TensorSpec(
            [None] + self.inputs.shape[1:], self.inputs.dtype)]
        return input_specs, None


# Create an instance of your subclassed model
model = MyModel(model)

# Proceed with TensorFlow Lite conversion
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the converted model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
