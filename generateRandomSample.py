import numpy as np
import pandas as pd
import os

# Jumlah sampel gerakan non-abjad
num_samples = 300

# Jumlah fitur (42 kolom setelah label)
num_features = 42

# Rentang nilai berdasarkan contoh dataset
min_value, max_value = -1.0, 1.0

# Generate data acak untuk gerakan non-abjad
non_abjad_data = np.random.uniform(min_value, max_value, size=(num_samples, num_features))

# Tambahkan label 24 di awal setiap baris
labels = np.full((num_samples, 1), 24)
dataset_non_abjad = np.hstack((labels, non_abjad_data))

# Simpan ke CSV baru
output_dir = "output"
output_file = os.path.join(output_dir, "gerakan_non_abjad.csv")

# Pastikan direktori ada
os.makedirs(output_dir, exist_ok=True)

# Simpan sebagai CSV
df_non_abjad = pd.DataFrame(dataset_non_abjad)
df_non_abjad.to_csv(output_file, index=False, header=False)

# Kembalikan path file
output_file
