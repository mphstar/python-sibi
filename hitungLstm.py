import numpy as np

# Fungsi aktivasi
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

# Data input pertama (timestep pertama)
x_t = np.array([0.2599, -0.0734])

# Asumsi h_t-1 dan C_t-1 bernilai nol di awal
h_prev = np.zeros(4)  # Hidden state sebelumnya (asumsi ukuran 4)
C_prev = np.zeros(4)  # Cell state sebelumnya (asumsi ukuran 4)

# Asumsi bobot dan bias (dengan ukuran sesuai hidden state)
W_f = np.array([[0.2, -0.1], [0.3, 0.25], [-0.2, 0.1], [0.15, -0.3]])
b_f = np.array([0.1, -0.2, 0.05, 0.0])

W_i = np.array([[0.1, 0.2], [-0.2, 0.1], [0.15, -0.25], [0.3, 0.05]])
b_i = np.array([-0.1, 0.2, -0.05, 0.1])

W_c = np.array([[0.25, -0.15], [0.2, 0.3], [-0.3, 0.1], [0.1, -0.2]])
b_c = np.array([0.05, -0.1, 0.15, 0.0])

W_o = np.array([[0.05, -0.2], [0.1, 0.25], [-0.15, 0.3], [0.2, -0.05]])
b_o = np.array([0.0, 0.1, -0.2, 0.05])

# Perhitungan Forget Gate
f_t = sigmoid(np.dot(W_f, x_t) + b_f)

# Perhitungan Input Gate
i_t = sigmoid(np.dot(W_i, x_t) + b_i)

# Perhitungan Candidate Cell State
C_tilde = tanh(np.dot(W_c, x_t) + b_c)

# Perhitungan Cell State
C_t = f_t * C_prev + i_t * C_tilde

# Perhitungan Output Gate
o_t = sigmoid(np.dot(W_o, x_t) + b_o)

# Perhitungan Hidden State
h_t = o_t * tanh(C_t)

# Output hasil perhitungan tiap tahap
f_t, i_t, C_tilde, C_t, o_t, h_t


# Output:
# (array([0.52497919, 0.47502081, 0.5124974 , 0.5       ]),
print(f_t)
