# -*- coding: utf-8 -*-
"""\
[Python NumPy SciPy : 周波数応答と伝達関数 | org-技術](https://org-technology.com/posts/frequency-response.html)

m ddx + c dx + kx = f(t)
(- \omega ^ 2 m + j \omega c + k) X e ^( j \omega t)

m/k = 1/\Omega
c/k = \frac{2c}{2\sqrt(mk)} sqrt\frac{m}{k} = 2 c/c_c 1/\Omega = 2 \jita \frac{1}{\Omega}

(-(\frac{\omega}{\Omega}^2 + 2 j \zeta {\omega}{\Omega} + 1) X = \frac{F}{k}

\frac{X}{X_st} = \frac{1}{1 - \Beata ^ 2 + 2j zeta \Beta}
"""
import numpy as np
from scipy import fftpack
from scipy import signal

from matplotlib import pyplot as plt


def main():
    # example1()
    example2()


def example1():
    num = 1024
    dt = 1E-3

    m = 1
    c = 1
    k = 400

    freq = fftpack.fftfreq(num, dt)
    zeta = c / (2 * np.sqrt(m * k))
    omega = np.sqrt(k / m)
    beta = freq / omega

    tf = 1 / (1 - beta ** 2 + 2 * 1j * zeta * beta)

    _, axs = plt.subplots(2, 1)
    axs[0].loglog(freq[1:int(num / 2)], np.abs(tf[1:int(num / 2)]))
    axs[0].set_ylabel("Amplitude")
    axs[0].axis("tight")

    axs[1].semilogx(freq[1:int(num / 2)], np.angle(tf[1:int(num / 2)]) * 180 / np.pi)
    axs[1].set_xlabel("Frequency [rad]")
    axs[1].set_ylabel("Phase[deg]")
    axs[1].axis("tight")
    axs[1].set_ylim(-180, 180)
    plt.show()
    plt.close()


def example2():
    """
    Laplace
    """
    m = 1
    c = 1
    k = 400

    numer = [k]
    denom = [m, c, k]
    s1 = signal.lti(numer, denom)
    omega, mag, phase = signal.bode(s1, np.logspace(-1, 4, 100))
    freqs = omega / (2 * np.pi)

    _, axs = plt.subplots(2, 1)
    axs[0].semilogx(freqs, mag)
    axs[0].set_xlim(1, 1000)
    axs[0].set_ylabel("Amplitude [dB]")
    axs[0].axis("tight")

    axs[1].semilogx(freqs, phase)
    axs[1].set_xlabel("Frequency [Hz]")
    axs[1].set_ylabel("Phase [deg]")
    axs[1].axis("tight")
    axs[1].set_xlim(1, 1000)
    axs[1].set_ylim(-180, 180)

    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
