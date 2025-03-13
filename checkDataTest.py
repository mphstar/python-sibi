import cv2
import mediapipe as mp
import os
import csv
import copy
import itertools
import numpy as np
import tensorflow as tf


# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
model_save_path = 'output/mymodel.h5'

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

abjads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

def abjadToNomor(abjad):
    return abjads.index(abjad)

model = tf.keras.models.load_model(model_save_path)

# Load semua gambar dari folder dataset A dan B
with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    for folder_dataset in abjads:
        for filename in os.listdir(f'data-test/{folder_dataset}'):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                # Baca gambar
                image = cv2.imread(os.path.join(f'data-test/{folder_dataset}', filename))

                # Flip the image horizontally
                # image = cv2.flip(image, 1)

                # Tampilkan gambar
                # if folder_dataset == 'A':
                #     cv2.namedWindow('Hand Detection', cv2.WINDOW_NORMAL)
                #     cv2.resizeWindow('Hand Detection', 500, 500)
                #     cv2.imshow('Hand Detection', image)
                #     cv2.waitKey(0)  # Wait for a key press to close the window
                # Konversi gambar BGR ke RGB untuk MediaPipe
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                debug_image = copy.deepcopy(image)

                results = hands.process(rgb_image)

                if results.multi_hand_landmarks:
                    landmark_list = []
                    for hand_landmarks in results.multi_hand_landmarks:
                        for landmark in hand_landmarks.landmark:
                            landmark_list.append(landmark.x)
                            landmark_list.append(landmark.y)
                            landmark_list.append(landmark.z)
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    final = pre_process_landmark(landmark_list)
                    result = model.predict(np.array([final]))

                    squeezed_result = np.squeeze(result)
                    predicted_label = np.argmax(squeezed_result)

                    accuracy = squeezed_result[predicted_label]

                    print(f'Predicted Label: {folder_dataset} ({abjads[predicted_label]}), Accuracy: {accuracy:.4f}')

            

print("Proses deteksi tangan selesai. Hasil disimpan di output/hasil_deteksi_tangan.csv")
