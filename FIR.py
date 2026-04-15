import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, lfilter

# ------------------------------------------------------------
# FIR Filter Demonstration
# ------------------------------------------------------------

# Example FIR filter: 5-point moving average
b = np.ones(5) / 5.0   # numerator coefficients
a = [1.0]              # FIR => denominator is just 1

print("FIR coefficients b =", b)

# ------------------------------------------------------------
# Part 1: Impulse response
# For an FIR filter, the impulse response is just the coefficients
# ------------------------------------------------------------
n_imp = np.arange(len(b))
h = b.copy()

plt.figure(figsize=(8, 4))
plt.stem(n_imp, h)
plt.title("Impulse Response of the FIR Filter")
plt.xlabel("n")
plt.ylabel("h[n]")
plt.grid(True)
plt.tight_layout()

# ------------------------------------------------------------
# Part 2: Frequency response
# ------------------------------------------------------------
w, H = freqz(b, a, worN=1024)

plt.figure(figsize=(8, 4))
plt.plot(w / np.pi, np.abs(H))
plt.title("Magnitude Response")
plt.xlabel("Normalized Frequency (×π rad/sample)")
plt.ylabel("|H(e^{jω})|")
plt.grid(True)
plt.tight_layout()

plt.figure(figsize=(8, 4))
plt.plot(w / np.pi, np.unwrap(np.angle(H)))
plt.title("Phase Response")
plt.xlabel("Normalized Frequency (×π rad/sample)")
plt.ylabel("Phase (radians)")
plt.grid(True)
plt.tight_layout()

# ------------------------------------------------------------
# Part 3: Filter a noisy signal
# ------------------------------------------------------------
fs = 500                    # sample rate
t = np.arange(0, 1, 1/fs)   # 1 second

# clean signal = two sinusoids
x_clean = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*40*t)

# noisy signal
np.random.seed(0)
noise = 0.5 * np.random.randn(len(t))
x_noisy = x_clean + noise

# filter the noisy signal
y = lfilter(b, a, x_noisy)

plt.figure(figsize=(10, 5))
plt.plot(t, x_noisy, label="Noisy input", alpha=0.7)
plt.plot(t, y, label="Filtered output", linewidth=2)
plt.plot(t, x_clean, label="Clean signal", linestyle="--")
plt.title("FIR Filtering of a Noisy Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()

# ------------------------------------------------------------
# Part 4: Show linear-phase symmetry
# A symmetric FIR filter has linear phase
# ------------------------------------------------------------
b2 = np.array([1, 2, 3, 2, 1], dtype=float)
b2 = b2 / np.sum(b2)

print("Symmetric FIR coefficients b2 =", b2)
print("Is b2 symmetric?", np.allclose(b2, b2[::-1]))

w2, H2 = freqz(b2, [1.0], worN=1024)

plt.figure(figsize=(8, 4))
plt.plot(w2 / np.pi, np.unwrap(np.angle(H2)))
plt.title("Phase Response of a Symmetric FIR Filter")
plt.xlabel("Normalized Frequency (×π rad/sample)")
plt.ylabel("Phase (radians)")
plt.grid(True)
plt.tight_layout()

# ------------------------------------------------------------
# Part 5: Compare original and filtered signals for symmetric FIR
# ------------------------------------------------------------
y2 = lfilter(b2, [1.0], x_noisy)

plt.figure(figsize=(10, 5))
plt.plot(t, x_noisy, label="Noisy input", alpha=0.6)
plt.plot(t, y2, label="Symmetric FIR output", linewidth=2)
plt.plot(t, x_clean, label="Clean signal", linestyle="--")
plt.title("Filtering with a Linear-Phase FIR")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()