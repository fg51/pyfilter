# -*- coding: utf-8 -*-
import numpy as np
from scipy import signal

from matplotlib import pyplot as plt


def main():
    sampling_frequency = 200
    target_frequency = 60
    Q = 30
    numer, denom = notch_filter(sampling_frequency, target_frequency, Q)
    # w, mag, phases = signal.bode(numer, denom)
    freq, h = to_frequence_response(sampling_frequency, [numer, denom])
    freq = w * sampling_frequency / (2 * np.pi)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    axs[0].plot(freq, to_dB(h))
    axs[1].plot(freq, to_degree(h))

    setup_plot_amplitude(axs[0])
    setup_plot_phase(axs[1])
    fig.suptitle("Frequency Response")

    plt.show()


def notch_filter(sampling_frequency, target_frequency, Q):
    numer, denom = signal.iirnotch(
        target_frequency / (sampling_frequency / 2),
        Q)
    return numer, denom


def to_frequence_response(sampling_freq, xs):
    w, h = signal.freqz(xs[0], xs[1], worN=np.logspace(-1, 2, 1000))
    freq = w * sampling_freq / (2 * np.pi)
    return freq, h


def to_dB(xs):
    return 20 * np.log10(abs(xs))


def to_degree(xs):
    return np.unwrap(np.angle(xs)) * 180 / np.pi


def setup_plot_amplitude(ax):
    ax.set_ylabel("Amplitude [dB]")
    ax.set_xlim([0, 100])
    ax.set_ylim([-25, 10])
    ax.grid()


def setup_plot_phase(ax):
    ax.set_ylabel("Angle [degrees]")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_xlim([0, 100])
    ax.set_yticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    ax.set_ylim([-200, 200])
    ax.grid()


if __name__ == '__main__':
    main()
