# -*- coding: utf-8 -*-
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt


def main():
    freqs, mag, phases = signal.bode([1, [1, 1]])

    ax1 = plt.subplot(2, 1, 1)
    ax1.semilogx(freqs, mag)
    setup_gain(ax1)

    ax2 = plt.subplot(2, 1, 2)
    ax2.semilogx(freqs, phases)
    setup_phase(ax2)

    plt.show()

    # numer, denom = signal.butter(4, 100, "low", analog=True)
    # w, h = signal.freqs(numer, denom)
    # freqs = from_radian_to_Hz(w)

    # ax1 = plt.subplot(2, 1, 1)
    # ax1.semilogx(freqs, to_dB(h))
    # setup_gain(ax1)

    # ax2 = plt.subplot(2, 1, 2)
    # ax2.semilogx(freqs, to_degree(h))
    # setup_phase(ax2)

    # plt.show()


def from_radian_to_Hz(xs):
    return xs / (2 * np.pi)


def to_dB(xs):
    return 20 * np.log10(abs(xs))


def to_degree(xs):
    # return np.angle(xs) * 180 / np.pi  # pylint: disable=no-member
    return np.unwrap(np.angle(xs)) * 180 / np.pi  # pylint: disable=no-member
    return np.unwrap(np.angle(xs, deg=True))  # pylint: disable=no-member


def cheby1():
    numer, denom = signal.iirfilter(4, [1, 10], 1, 60, analog=True, ftype="cheby1")
    w, h = signal.freqs(numer, denom, worN=np.logspace(-1, 2, 1000))

    plt.semilogx(w, 20 * np.log10(abs(h)))
    plt.ylabel("amp response [dB]")
    plt.grid()
    plt.show()


def setup_axis(ax):
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("amp response [dB]")
    ax.grid()


def setup_gain(ax):
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("amp response [dB]")
    ax.grid()


def setup_phase(ax):
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("amp response [dB]")
    # ax.set_ylim(-200, 200)
    # ax.set_yticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    ax.grid()


if __name__ == '__main__':
    main()
