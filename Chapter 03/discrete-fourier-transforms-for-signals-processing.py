import jax.numpy as np
from jax import random
import matplotlib.pyplot as plt

key = random.PRNGKey(12345)

def signal(t, freq_1=4.0, freq_2=7.0):
    return np.sin(freq_1 * 2 * np.pi * t) + np.sin(freq_2 * 2 * np.pi * t)

sample_size = 2 ** 7
sample_t = np.linspace(0, 4, sample_size, dtype=np.float32)
sample_y = signal(sample_t) + random.normal(key, (sample_size,))
sample_d = 4. / (sample_size - 1)  # Spacing for linspace array
true_signal = signal(sample_t)

from jax.numpy import fft

fig1, ax1 = plt.subplots()
ax1.plot(sample_t, sample_y, "k.", label="Noisy signal")
ax1.plot(sample_t, true_signal, "k--", label="True signal")

ax1.set_title("Sample signal with noise")
ax1.set_xlabel("Time")
ax1.set_ylabel("Amplitude")
ax1.legend()

spectrum = fft.fft(sample_y)

freq = fft.fftfreq(sample_size, sample_d)
pos_freq_i = np.arange(1, sample_size // 2, dtype=int)

psd = np.abs(spectrum[pos_freq_i]) ** 2 + np.abs(spectrum[-pos_freq_i]) ** 2

fig2, ax2 = plt.subplots()
ax2.plot(freq[pos_freq_i], psd, "k")
ax2.set_title("PSD of the noisy signal")
ax2.set_xlabel("Frequency")
ax2.set_ylabel("Density")

filtered = pos_freq_i[psd > 2e3]

new_spec = np.zeros_like(spectrum)
new_spec = new_spec.at[filtered].set(spectrum[filtered])
new_spec = new_spec.at[-filtered].set(spectrum[-filtered])

new_sample = np.real(fft.ifft(new_spec))

fig3, ax3 = plt.subplots()
ax3.plot(sample_t, true_signal, color="#8c8c8c", linewidth=1.5, label="True signal")
ax3.plot(sample_t, new_sample, "k--", label="Filtered signal")
ax3.legend()
ax3.set_title("Plot comparing filtered signal and true signal")
ax3.set_xlabel("Time")
ax3.set_ylabel("Amplitude")

plt.show()
