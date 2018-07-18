import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import astropy.units as u
from skimage import draw


class RFI(object):
    """
    Base RFI class.
    """
    def __init__(self, background, min_freq, max_freq, duration):
        """
        background (ndarray): The rfi background data.
        min_freq (Quantity):  The lower limit of the bandwidth.
        max_freq (Quantity):  The upper limit of the bandwidth.
        duration (Quantity):  The amount of time contained in background.
        """
        self.nfreq = background.shape[0]
        self.ntime = background.shape[1]
        try:
            self.duration = duration.to(u.ms)
        except AttributeError:
            self.duration = duration * u.ms
        try:
            self.min_freq = min_freq.to(u.MHz)
        except AttributeError:
            self.min_freq = min_freq * u.MHz
        try:
            self.max_freq = max_freq.to(u.MHz)
        except AttributeError:
            self.max_freq = max_freq * u.MHz
        self.bw = self.max_freq - self.min_freq
        self.rfi = background

    def freq_func(self, func, multiply=True, params=None):
        """
        Applies the function func to the frequency values of self.rfi,
        multiplies self.rfi by the result.
        """
        params = params or []
        x = np.linspace(self.max_freq, self.min_freq, self.bw)
        x = np.vstack([x] * self.ntime).T
        if multiply:
            self.rfi *= func(x.value, *params)
        else:
            self.rfi += func(x.value, *params)

    def time_func(self, func, multiply=True, params=None):
        """
        Applies the function func to the time value of self.rfi, multiplies
        self.rfi by the result.
        """
        params = params or []
        x = np.linspace(0, self.duration, self.ntime)
        x = np.vstack([x] * self.nfreq)
        if multiply:
            self.rfi *= func(x.value, *params)
        else:
            self.rfi += func(x.value, *params)

    def apply_func(self, func, params=None):
        """
        Passes self.rfi as the first parameter to the function func, updates
        self.rfi to equal the value returned by func.
        """
        params = params or []
        self.rfi = func(self.rfi, *params)

    def display(self):
        """
        Plots self.rfi
        """
        plt.imshow(self.rfi, interpolation='nearest', aspect='auto')
        plt.colorbar()
        plt.show()


class NormalRFI(RFI):
    """
    Creates an RFI object with a gaussian/normally distributed background.
    """
    def __init__(self, shape, min_freq, max_freq, duration, sigma=0, mu=1):
        background = np.random.normal(loc=sigma, scale=mu, size=shape)
        RFI.__init__(self, background, min_freq, max_freq, duration)


class UniformRFI(RFI):
    """
    Creates an RFI object with a uniformly distributed background.
    """
    def __init__(self, shape, min_freq, max_freq, duration, low=-3, high=3):
        background = np.random.uniform(low=low, high=high, size=shape)
        RFI.__init__(self, background, min_freq, max_freq, duration)


class PoissonRFI(RFI):
    """
    Creates an RFI object with a Poisson distribution background.
    """
    def __init__(self, shape, min_freq, max_freq, duration, lam=1):
        background = np.random.poisson(lam=lam, size=shape)
        RFI.__init__(self, biackground, min_freq, max_freq, duration)


class SolidRFI(RFI):
    """
    Creates an RFI object with a background full of a single value
    """
    def __init__(self, shape, min_freq, max_freq, duration, val=0):
        background = np.full(shape, fill_value=val, dtype='float64')
        RFI.__init__(self, background, min_freq, max_freq, duration)


def fourier_lowpass_filter(data, cutoff):
    """
    fourier transforms data, sets all values to 0 that are higher than cutoff,
    then returns the inverse fourier transform of that
    """
    fourier_data = np.fft.fft(data)
    filtered_data = np.where(data < cutoff, data, 0)
    return np.real(np.fft.ifft(filtered_data))


def butterworth_lowpass_filter(data, cutoff, sr, N):
    """
    Applies an Nth order butterworth lowpass filter to data, with a cutoff
    frequency of cutoff and a sample rate of st.
    """
    nyquist = 0.5 * sr
    cutoff /= nyquist
    b, a = signal.butter(N, cutoff)
    return signal.lfilter(b, a, data)


def one_over_f(alpha, beta):
    """
    Returns the function f(x) = beta / x**alpha
    """
    # TODO: implement the more complicated 1/f rfi outlined in this paper:
    # https://arxiv.org/pdf/1711.07843.pdf
    return lambda x: beta / (np.where(x != 0, x, 1e-9)**alpha)


def rand_sinusoid(amplitude, freq, phase):
    """
    Returns a random sinusoidal function with ampltidude, freq, and phase drawn
    from random normal distributions centered at their given values.
    """
    A = np.random.normal(amplitude)
    f = 2 * np.pi * np.random.normal(freq)
    p = np.random.normal(phase)
    func = np.sin if np.random.normal() > 0 else np.cos
    return lambda x: A * func((f * x) + p)


def sinusoidal_sum(n, a, f, p):
    """
    Returns a function that is the sum of n rand_sinusoid functions
    """
    def sum(x, f1, f2):
        return f1(x) + f2(x)
    def func(x):
        for _ in range(n):
            x = sum(x, rand_sinusoid(a, f, p), rand_sinusoid(a, f, p))
        return x
    return func


def sinusoidal_product(n, a, f, p):
    """
    Returns a function that is the product of n rand_sinusoid functions
    """
    def product(x, f1, f2):
        return f1(x) * f2(x)
    def func(x):
        for _ in range(n):
            x = product(x, rand_sinusoid(a, f, p), rand_sinusoid(a, f, p))
        return x
    return func


def patches(data, N, min_size=2, max_size=20, patch_size=1000):
    """
    Creates N random "patches" at random locations in data
    """
    xs = np.random.uniform(0, data.shape[0], N)
    ys = np.random.uniform(0, data.shape[1], N)
    rs = np.random.uniform(min_size, max_size, N)
    for idx, x in enumerate(xs):
        y = ys[idx]
        r = rs[idx]
        xx = [np.random.normal(x)]
        yy = [np.random.normal(y)]
        for i in range(patch_size - 1):
            xx.append(np.random.normal(xx[i]) + np.random.normal(0)*2)
            yy.append(np.random.normal(yy[i]) + np.random.normal(0)*2)
        rr = np.random.normal(r, 10, patch_size)
        circles = [draw.circle(xx[i], yy[i], rr[i]) for i in range(patch_size)]
        for X, Y in circles:
            X = np.where(X < data.shape[0], X, data.shape[0]-1)
            X = np.where(X > 0, X, 0)
            Y = np.where(Y < data.shape[1], Y, data.shape[1]-1)
            Y = np.where(Y > 0, Y, 0)
            data[X, Y] = np.abs(data[X, Y]) * 1.01
    return data


def masked_delta(data, width=50, height=50, loc=100, rand=1):
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
    return data


if __name__ == '__main__':
    rfi = NormalRFI((1024, 1000), 400*u.MHz, 800*u.MHz, 200*u.ms)
    rfi.time_func(masked_delta, False, [5, 100, 75, 0.1])
    rfi.display()
