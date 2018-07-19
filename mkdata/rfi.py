import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import astropy.units as u
from skimage import draw


class RFI(object):
    """
    Base RFI class.
    """
    def __init__(self, background, min_freq=400*u.MHz, max_freq=800*u.MHz,
                 duration=1000*u.ms):
        """
        background (ndarray): The rfi background data.
        min_freq (Quantity):  The lower limit of the bandwidth.
        max_freq (Quantity):  The upper limit of the bandwidth.
        duration (Quantity):  The amount of time contained in background.
        """
        nfreq = background.shape[0]
        ntime = background.shape[1]

        try:
            duration = duration.to(u.ms)
        except AttributeError:
            duration = duration * u.ms

        try:
            min_freq = min_freq.to(u.MHz)
        except AttributeError:
            min_freq = min_freq * u.MHz

        try:
            max_freq = max_freq.to(u.MHz)
        except AttributeError:
            max_freq = max_freq * u.MHz

        frequency = np.linspace(max_freq.value, min_freq.value, nfreq)
        time = np.linspace(0, duration.value, ntime)

        self.frequency_array = np.vstack([frequency] * ntime).T
        self.time_array = np.vstack([time] * nfreq)
        self.rfi = background

    def apply_function(self, func, input='value', freq_range=None,
                       time_range=None, **params):
        """
        Applies the function to the current rfi to update the rfi

        Params:
            func (function): The function that will be applied to the data,
                             must take three positional arguments x (The data
                             the function acts on), rfi (The rfi before the
                             function is applied) and boolean (A boolean array
                             that dictates where the function is applies)
                             Function must return the updated rfi.
            input (str):     Determines what x will be for func. If 'time' than
                             x will be an array containing the time values of
                             each index regardless of frequency. If 'freq' than
                             x will be an array containing the frequency values
                             of each in dex regardless of time. Otherwise x
                             will be the values of rfi.
            freq_range (list or None): Determines the range(s) of frequencies
                                       that are affected by the function. If
                                       None the entire bandwidth is affected.
                                       Otherwise each element of the list should
                                       be a tuple specifying a start and stop
                                       value that indicates a range that will be
                                       affected.
            time_range (list or None): Determine the range(s) if time values
                                       that are affected by the function.
                                       Behaves the same as freq_range.
            **params (optional dict):  Additional keyword arguments to pass to
                                       the function.
        Returns: None
        """
        if input == 'freq':
            x = self.frequency_array
        elif input == 'time':
            x = self.time_array
        else:
            x = self.rfi
        if freq_range is None:
            freq_coefs = np.ones_like(self.rfi)
        else:
            freq_coefs = np.zeros_like(self.rfi)
            for start, stop in freq_range:
                freq_coefs[start:stop, :] = 1
        if time_range is None:
            time_coefs = np.ones_like(self.rfi)
        else:
            time_coefs = np.zeros_like(self.rfi)
            for start, stop in time_range:
                time_coefs[:, start:stop] = 1
        coefs = (time_coefs * freq_coefs).astype(bool)
        self.rfi = func(x, self.rfi, coefs, **params)


    def plot(self):
        """
        Plots self.rfi with a colorbar
        """
        plt.imshow(self.rfi, interpolation='nearest', aspect='auto')
        plt.colorbar()
        plt.show()


class NormalRFI(RFI):
    """
    Creates an RFI object with a gaussian/normally distributed background.
    """
    def __init__(self, shape=(1024, 2**15), min_freq=400*u.MHz,
                 max_freq=800*u.MHz, duration=100*u.ms, sigma=0, mu=1):
        bg = np.random.normal(loc=sigma, scale=mu, size=shape)
        super(NormalRFI, self).__init__(bg, min_freq, max_freq, duration)


class UniformRFI(RFI):
    """
    Creates an RFI object with a uniformly distributed background.
    """
    def __init__(self, shape=(1024, 2**15), min_freq=400*u.MHz,
                 max_freq=800*u.MHz, duration=1000*u.ms, low=-3, high=3):
        bg = np.random.uniform(low=low, high=high, size=shape)
        super(UniformRFI, self).__init__(bg, min_freq, max_freq, duration)


class PoissonRFI(RFI):
    """
    Creates an RFI object with a Poisson distribution as the background.
    """
    def __init__(self, shape=(1024, 2**15), min_freq=400*u.MHz,
                 max_freq=800*u.MHz, duration=1000*u.ms, lam=1):
        bg = np.random.poisson(lam=lam, size=shape)
        super(PoissonRFI, self).__init__(bg, min_freq, max_freq, duration)


class SolidRFI(RFI):
    """
    Creates an RFI object with a background full of a single value
    """
    def __init__(self, shape=(1024, 2**15), min_freq=400*u.MHz,
                 max_freq=800*u.MHz, duration=1000*u.ms, val=0):
        bg = np.full(shape, fill_value=val, dtype='float64')
        super(SolidRFI, self).__init__(bg, min_freq, max_freq, duration)


def fourier_lowpass_filter(x, rfi, boolean, cutoff):
    """
    fourier transforms data, sets all values to 0 that are higher than cutoff,
    then returns the inverse fourier transform of that
    """
    fourier_data = np.fft.fft(x)
    filtered_data = np.where(fourier_data < cutoff, fourier_data, 0)
    real_data = np.real(np.fft.ifft(filtered_data))
    rfi = np.where(boolean, real_data, rfi)
    return rfi


def butterworth_lowpass_filter(x, rfi, boolean, cutoff, sr, N):
    """
    Applies an Nth order butterworth lowpass filter to data, with a cutoff
    frequency of cutoff and a sample rate of sr.
    """
    nyquist = 0.5 * sr
    cutoff /= nyquist
    b, a = signal.butter(N, cutoff)
    filtered_data = signal.lfilter(b, a, x)
    rfi = np.where(boolean, filtered_data, rfi)
    return rfi



def one_over_f(x, rfi, boolean, alpha, beta):
    """
    Returns the function f(x) = beta / x**alpha
    """
    # TODO: implement the more complicated 1/f rfi outlined in this paper:
    # https://arxiv.org/pdf/1711.07843.pdf
    def func(x):
        x = np.where(x != 0, x, 1e-9)
        return beta / (x ** alpha)
    x = func(x)
    rfi = np.where(boolean, x, rfi)
    return rfi


def rand_sinusoid(amp, freq, phase):
    """
    Creates a random sinusoidal function with amp, freq, and phase drawn
    from random normal distributions centered at the given values.
    """
    amp = np.random.normal(amp)
    freq = 2 * np.pi * np.random.normal(freq)
    phase = np.random.normal(phase)
    sinusoid = np.sin if np.random.normal() > 0 else np.cos
    return lambda x: amp * sinusoid((x * freq) + phase)


def output(x, rfi, boolean, add):
    """
    Helper function for functions that have an add to and multiple by method.
    """
    if add:
        x = np.where(boolean, x, 0)
        rfi += x
    else:
        x = np.where(boolean, x, 1)
        rfi *= x
    return rfi


def sinusoid(x, rfi, boolean, amp=1.0, freq=0.5, phase=0.0, add=True):
    """
    Applies a random sinusoid function to x. Where boolean is true either
    multiplies or adds the result to rfi (depending on if add is True or False)
    and returns the result of that operation.
    """
    func = rand_sinusoid(amp, freq, phase)
    x = func(x)
    return output(x, rfi, boolean, add)


def sinusoidal_sum(x, rfi, boolean, n=2, amp=1, freq=0.5, phase=0, add=True):
    """
    Applies a function that is the sum of n rand_sinusoid functions
    """
    def sum(x, f1, f2):
        return f1(x) + f2(x)
    def func(x):
        for _ in range(n):
            x = sum(x, rand_sinusoid(amp, freq, phase),
                    rand_sinusoid(amp, freq, phase))
        return x
    x = func(x)
    return output(x, rfi, boolean, add)


def sinusoidal_product(x, rfi, boolean, n=2, amp=1, freq=0.5, phase=0,
                       add=True):
    """
    Applies a function that is the product of n rand_sinusoid functions
    """
    def product(x, f1, f2):
        return f1(x) * f2(x)
    def func(x):
        for _ in range(n):
            x = product(x, rand_sinusoid(a, f, p), rand_sinusoid(a, f, p))
        return x
    x = func(x)
    return output(x, rfi, boolean, add)


def changing_sinusoid(x, rfi, boolean, func=lambda x: 1/np.exp(x), b=100,
                      amp=1, freq=0.5, phase=0, add=True):
    """
    Applies a function that is a random sinusoid that changes over time. Where
    f(x) is the randomized sinusoid, and g(x) is func the output is g(f(x)).
    This is either multiplied or added to x, depending on the value of add.
    """
    f1 = rand_sinusoid(amp, freq, phase)
    x = func(f1(x))
    return output(x, rfi, boolean, add)


def patches(data, rfi, boolean, N=5, min_size=2, max_size=20, patch_size=2000,):
    """
    Creates N random "patches" at random locations in data
    """
    xs = np.random.uniform(0, data.shape[0], size=N)
    ys = np.random.uniform(0, data.shape[1], size=N)
    rs = np.random.uniform(min_size, max_size, size=N)
    for idx, x in enumerate(xs):
        y = ys[idx]
        r = rs[idx]
        xx = [np.random.normal(x)]
        yy = [np.random.normal(y)] * patch_size
        for i in range(patch_size - 1):
            xx.append(np.random.normal(xx[i]) - np.abs((np.random.normal(0, 0.1))))
        rr = np.random.normal(r, 10, patch_size)
        x, y, r1, r2 = xx[i], yy[i], rr[i], rr[i]/16
        patches = [draw.ellipse(x, y, r1, r2) for i in range(patch_size)]
        for X, Y in patches:
            X = np.where(X < data.shape[0], X, data.shape[0]-1)
            X = np.where(X > 0, X, 0)
            Y = np.where(Y < data.shape[1], Y, data.shape[1]-1)
            Y = np.where(Y > 0, Y, 0)
            data[X, Y] = np.abs(data[X, Y]) * 1.001
    rfi = np.where(boolean, data, rfi)
    return rfi


def masked_delta(data, rfi, boolean, width=50, height=50, loc=100, rand=1):
    def delta(x, width, height, loc):
        loc = np.random.normal(loc, rand, size=x.shape[0])
        loc = np.vstack([loc] * x.shape[1]).T
        A = height / (np.abs(width) * np.sqrt(np.pi))
        B = np.exp(-((x - loc) / width) ** 2)
        return A * B
    def mask(x, a1, a2):
        x = np.where(x < a1 - 1, 1, x)

        b1 = np.greater_equal(x, a1 - 1)
        b2 = np.less_equal(x, a1)
        x = np.where(np.logical_and(b1, b2), (x - a1) ** 4, x)

        b1 = np.greater(x, a1)
        b2 = np.less(x, a2)
        x = np.where(np.logical_and(b1, b2), 0.01, x)

        b1 = np.greater_equal(x, a2)
        b2 = np.less_equal(x, a2+1)
        x = np.where(np.logical_and(b1, b2), (x - a2) ** 4, x)

        x = np.where(x > a2 + 1, 1, x)
        return x
    a1 = 0.9999 * loc
    a2 = 1.0001 * loc
    data = delta(data, width, height, loc) * mask(data, a1, a2)
    rfi = np.where(boolean, data, rfi)
    return rfi


if __name__ == '__main__':
    rfi = NormalRFI(shape=(1024, 1000))
    rfi.apply_function(sinusoidal_sum, input='freq')
    rfi.apply_function(patches)
    rfi.plot()
