import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

csv_path = 'starts.csv'
data = pd.read_csv(csv_path)

X = data.drop(['status'], axis=1)
y = data['status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(units=1, activation='sigmoid', input_shape=(X_train.shape[1],))
])

model.compile(optimizer='adam',  # Common optimizer
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Adjust epochs and batch size as per your dataset
model.fit(X_train, y_train, epochs=20, batch_size=20)

_, accuracy = model.evaluate(X_test, y_test)
print("ACCURACY:", accuracy)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
