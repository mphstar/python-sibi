import cv2
import mediapipe as mp
import os
import copy

abjads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
mp_hands = mp.solutions.hands

def abjadToNomor(abjad):
    return abjads.index(abjad)

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    for folder_dataset in abjads:
        for filename in os.listdir(f'dataset/{folder_dataset}'):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                # Baca gambar
                image = cv2.imread(os.path.join(f'dataset/{folder_dataset}', filename))

                # Flip the image horizontally
                # image = cv2.flip(image, 1)

                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                debug_image = copy.deepcopy(image)

                results = hands.process(rgb_image)

                if results.multi_hand_landmarks:
                    output_folder = f'clean_dataset/{folder_dataset}'
                    os.makedirs(output_folder, exist_ok=True)
                    output_path = os.path.join(output_folder, f"{filename.split('.')[0]}_2.jpg")
                    cv2.imwrite(output_path, debug_image)

print("Proses selesai")
