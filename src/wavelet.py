import numpy as np
from numba import njit
from scipy.special import gamma
from abc import ABC, abstractmethod
from typing import Union, Iterable

import mesaclip
import utils


real = Union[float, int]
R = Union[np.ndarray, Iterable, real]
C = Union[np.ndarray, Iterable, complex, real]


def make_scales_s(intervals_per_octave, duration_s, dt, **_):
    log2_scale_max = np.log2(duration_s / 2)
    log2_scale_min = np.log2(dt * 2)  # Nyquist (half frequency, or double scale)
    num_scale_octaves = int(np.ceil(log2_scale_max - log2_scale_min))
    scales_s = 2 ** np.linspace(log2_scale_min, log2_scale_max, intervals_per_octave * num_scale_octaves + 1)
    return scales_s


def make_freqs_hz(intervals_per_octave, freq_min_cpm, freq_max_cpm, **_):
    log2_freq_min_hz = np.log2(freq_min_cpm / 60)
    log2_freq_max_hz = np.log2(freq_max_cpm / 60)
    num_freq_octaves = int(np.round(log2_freq_max_hz - log2_freq_min_hz))
    freqs_hz = 2 ** np.linspace(log2_freq_min_hz, log2_freq_max_hz, intervals_per_octave * num_freq_octaves + 1)
    return freqs_hz


def make_log2_freqs_cpm(intervals_per_octave, freq_min_cpm, freq_max_cpm, **_):
    log2_freq_min_cpm = np.log2(freq_min_cpm)
    log2_freq_max_cpm = np.log2(freq_max_cpm)
    num_freq_octaves = int(np.round(log2_freq_max_cpm - log2_freq_min_cpm))
    log2_freqs_cpm = np.linspace(log2_freq_min_cpm, log2_freq_max_cpm, intervals_per_octave * num_freq_octaves + 1)
    return log2_freqs_cpm


class Wavelet(ABC):

    @abstractmethod
    def time_domain(self, t: R) -> C:
        """
        Wavelet in the time-domain
        :param t: an array of times (seconds)
        :return C: array of the same size as t
        """

    @abstractmethod
    def freq_domain(self, omega: R) -> R:
        """
        Wavelet in the frequency-domain
        :param omega: an array of frequencies (radians/second)
        :return: array of the same size as f
        """

    @abstractmethod
    def convert_freq_scale(self, freq_or_scale: R) -> R:
        """
        Convert between frequency (radians/second) and scale (seconds)
        :param freq_or_scale: an scalar or array of frequencies or scales
        :return: scales corresponding to the supplied frequencies, or frequencies corresponding to supplied scales
        """

    @abstractmethod
    def efoldtime(self) -> real:
        """
        The e-folding time corresponding to a scale of 1, see Torrence and Compo 1998. The e-folding time can be
        multiplied by a scale to obtain the time for that scale.
        :return: e-folding time scalar
        """

    def coi(self, scales: R, dt: real, nx: int) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Generates Cone-of-Influence as a mask and indexes into time and scale
        :param scales: an array of scales
        :param dt: time step
        :param nx: number of samples in the signal
        :return: triple containing the 01 mask and indexes into the time and scale arrays
        """
        ns = len(scales)
        coi_time_idxs = np.ceil(scales * self.efoldtime() / dt).astype(np.int32)
        coi_scale_idxs = np.arange(ns)[coi_time_idxs < nx // 2]
        coi_time_idxs = coi_time_idxs[coi_time_idxs < nx // 2]
        coi_mask = np.ones((ns, nx))
        for i in range(len(scales)):
            if i < len(coi_time_idxs):
                coi_mask[i, :coi_time_idxs[i]] = 0
                coi_mask[i, nx - coi_time_idxs[i]:] = 0
            else:
                coi_mask[i, :] = 0
        return coi_mask, coi_time_idxs, coi_scale_idxs


class Morse(Wavelet):

    def __init__(self, beta=1.58174, gam=3):
        self.beta = float(beta)
        self.gam = float(gam)
        self.a = 2 * (np.e * gam / beta) ** (beta / gam)

    def time_domain(self, t):
        dt = t[1] - t[0]
        m = len(t)
        n = m // 8
        f0 = 1.0 / (8 * dt)
        c = [self.freq_domain(2 * f0 * j / float(n)) for j in range(n // 2 + 1)]
        x = np.zeros(len(t), dtype='c16')

        # lilly2009higher equation (?)
        for i in range(len(t)):
            for j in range(n // 2):
                x[i] += c[j] * np.exp(1j * 2.0 * np.pi * j * i / m)
            x[i] /= (m * dt)

        # rotate to center at 0
        x = np.roll(x, len(x) // 2)

        return x

    def freq_domain(self, omega):
        omega = np.clip(omega, 0, np.inf)
        return self.a * omega ** self.beta * np.exp(-(omega ** self.gam))

    def convert_freq_scale(self, freq_or_scale):
        peak_omega = (self.beta / self.gam) ** (1 / self.gam)
        peak_freq = peak_omega / (2 * np.pi)
        return peak_freq / freq_or_scale

    def efoldtime(self):
        # formula (6) in
        #   Suwansawang, S. & Halliday, D.
        #   Time-frequency based Coherence and Phase Locking Value Analysis
        #    of Human Locomotion Data using Generalized Morse Wavelets.
        #   BIOSIGNALS, 2017, 34-41
        peak_omega = (self.beta / self.gam) ** (1 / self.gam)
        duration = np.sqrt(self.beta * self.gam)
        return np.sqrt(2) * duration / peak_omega

    def __str__(self):
        return f'Morse(β={self.beta:g}, γ={self.gam:g})'


class Morlet(Wavelet):
    def __init__(self, omega0=6):
        self.omega0 = omega0

    def time_domain(self, t):
        return (np.pi ** -0.25) * np.exp(1j * self.omega0 * t - t ** 2.0 / 2.0)

    def freq_domain(self, omega):
        return (omega > 0) * (np.pi ** -0.25) * np.exp(-0.5 * (omega - self.omega0) ** 2.0)

    def convert_freq_scale(self, freq_or_scale):
        scales = 1.0 / (4 * np.pi * freq_or_scale / (self.omega0 + np.sqrt(2 + self.omega0 ** 2)))
        return scales

    def efoldtime(self):
        return np.sqrt(2)

    def __str__(self):
        return f'Morlet($ω_0={self.omega0:g}$)'


class Paul(Wavelet):
    def __init__(self, m=4):
        self.m = m
        self.coeff = 2 ** m * 1j ** m * gamma(m + 2) / np.sqrt(np.pi * gamma(2 * m + 2))
        self.coeff_ft = 2 ** m / np.sqrt(m * gamma(2 * m + 2))

    def time_domain(self, t):
        return self.coeff * (1 - 1j * t) ** (-(self.m + 1))

    def freq_domain(self, omega):
        pos_f = omega > 0
        return pos_f * self.coeff_ft * omega ** self.m * np.exp(-omega * pos_f)  # pos_f in exp prevents exp overflow

    def convert_freq_scale(self, freq_or_scale):
        scales = 1.0 / (4 * np.pi * freq_or_scale / (2 * self.m + 1))
        return scales

    def efoldtime(self):
        return 1 / np.sqrt(2)

    def __str__(self):
        return f'Paul(m={self.m:g})'


@njit(nogil=True)
def synchrosqueeze_histogram(t: R, w: C, omega_log: R, freqs_hz_log: R) -> None:
    nscales, ntimes = omega_log.shape
    nfreqs = len(freqs_hz_log)
    for ti in range(ntimes):
        for src_idx in range(nscales):
            min_diff = np.inf
            min_k = 0
            for dst_idx in range(nfreqs):
                diff = np.abs(omega_log[src_idx, ti] - freqs_hz_log[dst_idx])
                if diff < min_diff:
                    min_diff = diff
                    min_k = dst_idx
            t[min_k, ti] += w[src_idx, ti]


def synchrosqueeze(w: C, dt: real, freqs_hz: R) -> C:
    # use the more numerically stable version (no division by w) based on (3.6) in
    # "Daubechies, I.; Lu, J. & Wu, H.-T. Synchrosqueezed wavelet transforms: an empirical mode decomposition-like tool.
    # Applied and computational harmonic analysis, Elsevier, 2011, 30, 243-261"
    # also used in the Synchrosqueezing Matlab toolbox https://github.com/ebrevdo/synchrosqueezing
    phi = np.unwrap(np.angle(w), axis=1)  # radians
    omega = np.abs((np.gradient(phi, dt)[1]) / (2 * np.pi))  # Hz (instantaneous normalised frequency)

    # ignore divide by 0 error in log since log(0) just returns -inf which is what we want
    divide_err_handling = np.geterr()['divide']
    np.seterr(divide='ignore')

    log_omega = np.log(omega)

    np.seterr(divide=divide_err_handling)

    # create padded low and high frequencies to absorb edge effects of synchrosqueezing due to simple histogram method
    freqs_hz_log = np.log(freqs_hz)
    df = freqs_hz_log[1] - freqs_hz_log[0]
    freqs_hz_log = np.r_[freqs_hz_log[0] - df, freqs_hz_log, freqs_hz_log[-1] + df]

    t = np.zeros((len(freqs_hz_log), w.shape[1]), dtype=w.dtype)

    # for splitting data up into blocks for parallel synchrosqueeze_histogram computation
    blocksize = 1024
    nblocks = int(np.ceil(t.shape[1] / blocksize))

    def exec_func(i):
        slc = slice(i * blocksize, (i + 1) * blocksize)
        synchrosqueeze_histogram(t[:, slc], w[:, slc], log_omega[:, slc], freqs_hz_log)

    utils.parexec(exec_func, nblocks)

    # return with padding removed
    return t[1:-1, :]


def make_nondecreasing(x):
    dx = np.diff(x)
    dx[dx < 0] = 0
    return x[0] + np.r_[0, np.cumsum(dx)]


def mesaclip_filter(w: C, k: real = 2) -> C:
    nscale, nsamp = w.shape

    amp = np.abs(w)
    phase = np.angle(w)
    cycles = np.unwrap(phase)

    k_rad = 2 * np.pi * k

    for i in range(nscale):
        c = make_nondecreasing(cycles[i, :])
        mesaclip.mesaclip(c, amp[i, :], k_rad)

    return amp * np.exp(1j * phase)


def reconstruct(w: C, mother: Wavelet, scales: R) -> R:
    nx = w.shape[1]
    nfft = 2 ** (int(np.log2(nx - 1)) + 1)  # nearest non-smaller power of 2

    freqs_hz = mother.convert_freq_scale(scales)

    # problem with Dirac delta is that there might not be sufficiently short scales supplied to account for the high
    # frequencies at high omegas, so rather than using all available frequencies up to Nyquist of the signal, only
    # consider frequencies available from scales
    omega = 2 * np.pi * np.linspace(freqs_hz[0], 0.5 * freqs_hz[-1], nfft // 2)

    # wavelet transform of Dirac delta function at 0
    c_delta = 0
    for i, s in enumerate(scales):
        psi_hat = np.conjugate(mother.freq_domain(s * omega))
        w_delta = np.sum(psi_hat) / nfft
        c_delta += w_delta.real

    # normally we would need to divide w.real inside sum by sqrt(scales), but already done in cwt due to rectification
    x = np.sum(w.real, axis=0) / c_delta

    return x


def cwt(x: np.ndarray, dt: float, scales: np.ndarray, mother: Wavelet,
        syncsqz_freqs: np.ndarray = None, min_cycles: real = 0, cplx_dtype=None, apply_coi=False):
    """
    Continuous wavelet transform with synchrosqueezing. To plot the coi_time_idxs,
    use plot(t[coi_time_idxs], freqs[-len(coi_time_idxs):]).
    :param x: signal array
    :param dt: time step between elements in x
    :param scales: wavelet scales in units of dt
    :param mother: class describing the mother wavelet, see Morlet for an example
    :param syncsqz_freqs: destination frequency bin centers for synchrosqueezing,
        None to skip synchrosqueezing
    :param min_cycles: k-parameter for pulse filtering
    :param cplx_dtype: dtype of the return array
    :param apply_coi: whether to apply COI zeroing
    :return: returns an array of shape (len(scales), len(x)), coi_mask of same size,
        and coi_time_idxs time-indexes per scale, and coi_freq_idxs of remaining freqs
    """

    if cplx_dtype is None:
        cplx_dtype = np.promote_types(x.dtype, np.complex64)

    nx = len(x)
    nfft = 2 ** (int(np.log2(nx - 1)) + 1)  # nearest non-smaller power of 2

    fx = np.fft.fft(x, nfft)

    omega = 2 * np.pi * np.fft.fftfreq(nfft, dt)  # radians per second

    # calculate wavelet transform
    w = np.empty((len(scales), nfft), cplx_dtype)

    def exec_func(i_):
        s = scales[i_]
        psi_hat = np.sqrt(s) * np.conjugate(mother.freq_domain(s * omega))
        w[i_, :] = np.fft.ifft(fx * psi_hat, nfft)

    utils.parexec(exec_func, len(scales))

    # remove power-of-2 padding
    w = w[:, :len(x)]

    # calculate cone-of-influence
    coi_mask, coi_time_idxs, coi_freq_idxs = mother.coi(scales, dt, nx)

    if apply_coi:
        w *= coi_mask

    if min_cycles > 0:
        w = mesaclip_filter(w, min_cycles)

    # rectify (liu2007rectification)
    # this should be done inside synchrosqueezing, but it is same operation with synchsqz or without,
    # so it's been removed from synchrosqueeze and applied either either way
    w /= np.sqrt(scales[:, None])

    if syncsqz_freqs is not None:
        w = synchrosqueeze(w, dt, syncsqz_freqs)

        # recreate coi for syncsqz_freqs so that it can be applied to syncsqzed w
        syncsqz_scales = mother.convert_freq_scale(syncsqz_freqs)
        coi_mask, coi_time_idxs, coi_freq_idxs = mother.coi(syncsqz_scales, dt, nx)

        if apply_coi:
            w *= coi_mask

    return w, coi_mask, coi_time_idxs, coi_freq_idxs


def gauss_smooth(x, sigma):

    n = len(x)
    n_nextpow2 = int(2 ** np.ceil(np.log2(n)))

    f = 2 * np.pi * np.fft.fftfreq(n_nextpow2)
    ft = np.exp(-0.5 * (f * sigma) ** 2)
    x_smooth = np.fft.ifft(ft * np.fft.fft(x, n_nextpow2))[:n]

    if np.isrealobj(x):
        return x_smooth.real
    else:
        return x_smooth


def coherence(wx, wy, sigmas):

    wxy = wx * wy.conj()
    coh = np.zeros_like(wx, dtype=wx.dtype)

    def amp2(x): return np.abs(x) ** 2

    for i, sigma in enumerate(sigmas):

        def smooth(x): return gauss_smooth(x, sigma)

        # denominator for coherence calculation
        coh_amp_z = smooth(amp2(wx[i, :])) * smooth(amp2(wy[i, :]))

        # for 0-valued amplitudes, define coherence as 0 (hence inf in the denominator)
        coh_amp_z[coh_amp_z == 0] = np.inf

        wxy[i, :] = smooth(wxy[i, :])
        coh_amp = amp2(wxy[i]) / coh_amp_z
        coh_phi = np.angle(wxy[i])

        # remove artefacts due to pathological smoothing for extremely small or large sigmas
        coh_amp = np.clip(coh_amp, 0, 1)
        coh[i, :] = coh_amp * np.exp(1j * coh_phi)

    return coh
