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


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


model.fit(X_train, y_train, epochs=20, batch_size=20)


_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy * 100))

model.save("model.keras")


loaded_model = load_model('model.keras')


# Create a dictionary with the same keys as your feature names
# 1
new_data = {
    "left_shoulder_elbow_angle": [
        30
    ],
    "left_elbow_wrist_angle": [
        83
    ],
    "right_shoulder_elbow_angle": [
        92
    ],
    "right_shoulder_left_shoulder": [
        174
    ],
    "right_elbow_wrist_angle": [
        16
    ],
    "left_shoulder_x_rel": [
        0.08904109589041095
    ],
    "left_shoulder_y_rel": [
        0.10252365930599369
    ],
    "left_elbow_x_rel": [
        -0.005870841487279843
    ],
    "left_elbow_y_rel": [
        0.08675078864353312
    ],
    "left_wrist_x_rel": [
        0.09980430528375733
    ],
    "left_wrist_y_rel": [
        0.2444794952681388
    ],
    "right_shoulder_x_rel": [
        -0.0019569471624266144
    ],
    "right_shoulder_y_rel": [
        0.23974763406940064
    ],
    "right_elbow_x_rel": [
        0.10176125244618395
    ],
    "right_elbow_y_rel": [
        0.37697160883280756
    ],
    "right_wrist_x_rel": [
        -0.014677103718199608
    ],
    "right_wrist_y_rel": [
        0.3470031545741325
    ]
}
# Convert the dictionary to a DataFrame
X_new = pd.DataFrame(new_data)
predictions = loaded_model.predict(X_new)
print(predictions)
