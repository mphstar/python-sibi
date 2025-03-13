import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import copy
import itertools


# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
drawing_utils = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Membuka webcam dengan indeks 0 (webcam default)
cap = cv2.VideoCapture(0)

model = tf.keras.models.load_model('output/mymodel.h5')
abjads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
# abjads = ['A', 'B', 'C', 'D']

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


while True:
    # Membaca frame dari webcam
    ret, frame = cap.read()

    # Konversi gambar BGR ke RGB untuk MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        # Deteksi tangan dalam gambar
        results = hands.process(rgb_frame)

        # Periksa apakah tangan terdeteksi
        if results.multi_hand_landmarks is not None:

            # Tentukan apakah tangan kanan atau kiri
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                if label == 'Right':
                    hand_label = 'Right Hand'
                else:
                    hand_label = 'Left Hand'
                
                # Tampilkan label tangan di frame
                cv2.putText(frame, hand_label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Hitung jumlah jari
            jumlah_jari = 0
            # Dapatkan posisi jari
            landmark_list = []
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    landmark_list.append(landmark.x)
                    landmark_list.append(landmark.y)
                    landmark_list.append(landmark.z)
                
                # draw
                   
            landmark_list = calc_landmark_list(rgb_frame, hand_landmarks)
            
            result = model.predict(np.array([pre_process_landmark(landmark_list)]))
            print(np.squeeze(result))
            print(np.argmax(np.squeeze(result)))

            # tampilkan di opencv
            drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # tampilkan putText

            predicted_label = abjads[np.argmax(np.squeeze(result))]
            accuracy = np.squeeze(result)[np.argmax(np.squeeze(result))]
            cv2.putText(frame, f'{predicted_label} ({accuracy:.4f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)




    # Menampilkan frame di jendela baru
    cv2.imshow('Deteksi Tangan', frame)

    # Menekan tombol 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Menutup jendela dan membebaskan sumber daya
cap.release()
cv2.destroyAllWindows()
