import cv2
import mediapipe as mp
import os
import pandas as pd
import csv
import copy
import itertools

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands

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


abjads = ['C']
def abjadToNomor(abjad):
    return abjads.index(abjad)


# Load semua gambar dari folder dataset A dan B
for folder_dataset in abjads:
    for filename in os.listdir(f'simulate/{folder_dataset}'):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Baca gambar
            image = cv2.imread(os.path.join(f'simulate/{folder_dataset}', filename))

            # Flip the image horizontally
            # image = cv2.flip(image, 1)

            # Konversi gambar BGR ke RGB untuk MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            debug_image = copy.deepcopy(image)

            # Proses gambar dengan MediaPipe Hands
            with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
                # Deteksi tangan dalam gambar
                results = hands.process(rgb_image)

                # Periksa apakah tangan terdeteksi
                if results.multi_hand_landmarks is not None:

                    # Hitung jumlah jari
                    jumlah_jari = 0
                    # Dapatkan posisi jari
                    landmark_list = []
                    for hand_landmarks in results.multi_hand_landmarks:
                        for landmark in hand_landmarks.landmark:
                            landmark_list.append(landmark.x)
                            landmark_list.append(landmark.y)
                            landmark_list.append(landmark.z)
                   

                    # Landmark calculation
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    # finalResult
                    final = pre_process_landmark(landmark_list)
                    # Simulate result as an image
                    for point in landmark_list:
                        cv2.circle(debug_image, (point[0], point[1]), 5, (0, 255, 0), -1)

                    # Display the image with landmarks
                    cv2.imshow('Hand Landmarks', debug_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    print(final)


print("Proses deteksi tangan selesai. Hasil disimpan di output/hasil_deteksi_tangan.csv")
