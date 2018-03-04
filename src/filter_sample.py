# -*- coding: utf-8 -*-
import numpy as np
from scipy import fftpack

from matplotlib import pyplot as plt


def main():
    low_pass_freq = 0.5
    num_of_sample = 1024
    sampling_time = 100.0
    sampling_freq = num_of_sample / sampling_time
    print("sampling freq is {0} [Hz]".format(sampling_freq))

    ts = np.linspace(0, sampling_time, num=num_of_sample)
    np.random.seed(1234)  # pylint: disable=no-member
    src_waves = create_source_wave(ts)

    # prepare FFT
    time_step = ts[1] - ts[0]
    sample_freq = fftpack.fftfreq(src_waves.size, d=time_step)

    # FFT
    y_fft = fftpack.fft(src_waves)
    pidxs = np.where(sample_freq > 0)
    freqs, power = sample_freq[pidxs], np.abs(y_fft)[pidxs]
    # freq = freqs[power.argmax()]

    y_fft = lowpas_filter(y_fft, sample_freq, low_pass_freq)

    # IFFT
    filtered_src_waves = fftpack.ifft(y_fft).real

    # FFT2
    y_fft2 = fftpack.fft(filtered_src_waves)
    pidxs2 = np.where(sample_freq > 0)
    freqs2, power2 = sample_freq[pidxs2], np.abs(y_fft2)[pidxs2]
    # freq2 = freqs[power2.argmax()]


    plt.figure(figsize=(8, 10))
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(ts, src_waves, 'b-', linewidth=0.5, label="src")
    # ax1.plot(ts, filtered_src_waves, 'r-', linewidth=1, label='filtered')
    setup_plot_wave(ax1)

    ax2 = plt.subplot(212)
    ax2.loglog(freqs, power, 'b.-', lw=1)
    # ax2.loglog(freqs2, power2, 'r.-', lw=1)
    setup_plot_power(ax2)

    plt.show()


def create_source_wave(t):
    return create_wave(t, 0.1, 1) + create_wave(t, 1, 1, 90)


def create_wave(ts, freq, gain, theta=None):
    rads = freq * ts + theta / 360 if theta else freq * ts
    return gain * np.sin(2 * np.pi * rads)


def lowpas_filter(y_fft, sample_freq, low_pass_freq):
    y_fft[np.abs(sample_freq) > low_pass_freq] = 0
    return y_fft


def setup_plot_wave(ax):
    ax.set_xlim(0., 20.)
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Ydata')
    ax.legend()
    ax.grid(True)


def setup_plot_power(ax):
    ax.set_ylim(0.1, 1E+4)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power')
    ax.grid(True)


if __name__ == '__main__':
    main()
