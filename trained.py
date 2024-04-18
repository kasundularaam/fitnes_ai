
import pandas as pd
from tensorflow.keras.models import load_model

loaded_model = load_model('model.keras')


# Create a dictionary with the same keys as your feature names
# 1
new_data_1 = {
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


# Create a dictionary with the same keys as your feature names
# 1
new_data_2 = {
    "left_shoulder_elbow_angle": [
        16
    ],
    "left_elbow_wrist_angle": [
        66
    ],
    "right_shoulder_elbow_angle": [
        65
    ],
    "right_shoulder_left_shoulder": [
        178
    ],
    "right_elbow_wrist_angle": [
        14
    ],
    "left_shoulder_x_rel": [
        0.08383233532934131
    ],
    "left_shoulder_y_rel": [
        0.0636215334420881
    ],
    "left_elbow_x_rel": [
        -0.023952095808383235
    ],
    "left_elbow_y_rel": [
        0.06688417618270799
    ],
    "left_wrist_x_rel": [
        0.11776447105788423
    ],
    "left_wrist_y_rel": [
        0.19249592169657423
    ],
    "right_shoulder_x_rel": [
        -0.05588822355289421
    ],
    "right_shoulder_y_rel": [
        0.18270799347471453
    ],
    "right_elbow_x_rel": [
        0.1506986027944112
    ],
    "right_elbow_y_rel": [
        0.3360522022838499
    ],
    "right_wrist_x_rel": [
        -0.10479041916167664
    ],
    "right_wrist_y_rel": [
        0.2805872756933116
    ]
}
# Convert the dictionary to a DataFrame
X_new = pd.DataFrame(new_data_2)
print(X_new.shape)
predictions = loaded_model.predict(X_new)

if predictions[0][0] >= 0.5:
    predicted_class = 1
else:
    predicted_class = 0
print(predicted_class)
