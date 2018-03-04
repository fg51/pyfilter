# -*- coding: utf-8 -*-
from scipy import signal

from matplotlib import pyplot as plt


def main():
    ts = phase_lead_compensator()
    freqs, mag, phases = signal.bode(ts)

    _, axs = plt.subplots(2, 1)
    axs[0].semilogx(freqs, mag)
    setup_gain(axs[0])

    axs[1].semilogx(freqs, phases)
    setup_phase(axs[1])

    plt.show()


def phase_lead_compensator():
    r"""
    \frac{K (T + s)}{\alpha T + s}
    """
    K = 1.5
    T = 0.3
    alpha = 0.075
    return [K * T, 1], [alpha * T, 1]


def setup_gain(ax):
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("amp response [dB]")
    ax.grid()


def setup_phase(ax):
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Phase [deg]")
    ax.set_ylim(-200, 200)
    ax.set_yticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    ax.grid()


if __name__ == '__main__':
    main()
