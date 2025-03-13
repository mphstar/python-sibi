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

def logging_csv(landmark_list, label):
    csv_path = 'output/hasil_deteksi_tangan.csv'
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([label, *landmark_list])

    return

abjads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']


def abjadToNomor(abjad):
    return abjads.index(abjad)

# Buat folder output jika belum ada
if not os.path.exists('output'):
    os.makedirs('output')

# Buka file CSV untuk menulis hasil
with open('output/hasil_deteksi_tangan.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

# Load semua gambar dari folder dataset A dan B
for folder_dataset in abjads:
    for filename in os.listdir(f'preview-dataset/{folder_dataset}'):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Baca gambar
            image = cv2.imread(os.path.join(f'preview-dataset/{folder_dataset}', filename))

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
                   

                    # print(landmark_list)
                    # print(hand_landmarks)
                    # print("=============================")
                    # print(landmark_list)

                    # Landmark calculation
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    # print("=================================")
                    # print(landmark_list)

                    # finalResult
                    final = pre_process_landmark(landmark_list)
                    # print("=================================")
                    # print(final)

                    # Write to the dataset file
                    logging_csv(pre_process_landmark(landmark_list), abjadToNomor(folder_dataset))


print("Proses deteksi tangan selesai. Hasil disimpan di output/hasil_deteksi_tangan.csv")
