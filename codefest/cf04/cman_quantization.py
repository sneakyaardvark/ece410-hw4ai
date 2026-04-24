import numpy as np

w = np.asarray([
    [0.85, -1.20, 0.34, 2.10],
    [-0.07, 0.91, -1.88, 0.12],
    [1.55, 0.03, -0.44, -2.31],
    [-0.18, 1.03, 0.77, 0.55],
    ])

w_max = np.max(np.abs(w))
print(f"MAX={w_max}\n")
s = w_max / 127
s = 0.01
w_q = np.clip(np.round(w / s), -128, 127)

print("Quantized:\n", w_q)

print("Dequanitzing...")
w_deq = w_q * s

err = np.abs(np.subtract(w, w_deq))
print(err)
max_err_idx = np.argmax(np.abs(err))
print(f"Largest error: element {max_err_idx} with value {err.flatten()[max_err_idx]}")
mae = (np.mean(err))
print(f"MAE = {mae}")
