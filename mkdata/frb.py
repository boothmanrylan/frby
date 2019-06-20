import random
import glob
import os
import string
import numpy as np
import pandas as pd
from baseband import vdif
from baseband.helpers import sequentialfile as sf
import astropy.units as u
from scipy import signal
import matplotlib
# matplotlib.use('Agg') # Needed when running on scinet
import matplotlib.pyplot as plt
from math import gcd
from tools import read_vdif

k_dm = 4.148808e6 * (u.MHz**2 * u.cm**3 * u.ms / u.pc)

# TODO: check the mean and std of real background data
# TODO: save directly to TFRecords

class FRB(object):
    """
    Generate a simulated fast radio burst.
    Based on https://github.com/liamconnor/single_pulse_ml which is in turn
    based on https://github.com/kiyo-masui/burst_search

    All parameters that are quantities can be passed as unitless floats, which
    will be converted to quantities with the default units.

    If any of t_ref, f_ref, scintillate, or window are passed as "None", a
    uniformly sampled random value will be assigned to them from a range that
    makes sense with respect to the other parameters.

    Any of dm, fluence, scat_factor, width, and spec_ind can be passed as a
    single value or a 2-tuple of values. If one of them is passed as a tuple a
    final value will be drawn uniformly from the range in the tuple and
    assigned to the parameter.

    Any parameters whose final value is determined randomly will have that
    final value saved in the parameter, not the range the value was drawn from.

    Parameters
    ----------
    t_ref : Quantity (default units: ms)
        The reference time used when determining where the burst is located.
    f_ref : Quantity (default units: MHz)
        The reference frequency used when determining where the burst is
        located.
    NFREQ : int
        The number of frequency bins.
    NTIME : int
        The number of time bins.
    delta_t : Quantity (default units: ms)
        The time step between two time bins.
    dm : Quantity (default units: pc cm^-3, default range: 50 - 2500)
        The dispersion measure (DM) of the pulse.
    fluence : Quantity (default units: Jy ms, default range: 0.02 - 200)
        The fluence of the pulse.
    freq : tuple of Quantities (MHz)
        The maximum and minimum frequency values.
    rate : Quantity (default units: MHz)
        The rate at which samples were taken.
    scat_factor : int (default range: -5 - -3),
        Used in determing how much scatter occurs in the burst.
    width : Quantity (default units: ms, default range: 0.05 - 40)
        The width in time of the pulse.
    scintillate : bool
        If true scintills are added to the pulse.
    spec_ind : int (default range: -10 - 10)
        the spectral index of the burst.
    background : vdif file, list of vdif files, numpy ndarray
        See _see_background
    rfi_type: string
        Description of the background if given as an ndarray.
    max_size : int
        The size to reduce/bin the data to in time.
    bg_mean : float
        Mean used when creating gaussian background.
    bg_std : float
        Standard deviation used when creating gaussian background.
    window : bool
        If true a "window" of random size is placed on the burst.
    normalize : bool
        If true frequency bands are normalized to have 0 mean and 1 std

    Attributes
    ----------
    sample : ndarray
        The simulated frb in the background noise.
    background : ndarray
        The background noise.
    signal : ndarray
        The simulated frb without any background noise.
    bandwith : Quantity
        The frequency range of the samples.
    snr : float
        The approximate signal-to-noise ratio of the burst.
    """

    def __init__(self, t_ref=0*u.ms, f_ref=None, NFREQ=1024, NTIME=8192,
                delta_t=0.16*u.ms, dm=(50, 2500)*(u.pc/u.cm**3),
                fluence=(0.02, 200)*(u.Jy*u.ms), freq=(0.8, 0.4)*u.GHz,
                rate=1000*u.Hz, scat_factor=(-5, -3),
                #rate=(0.4/1024)*u.GHz, scat_factor=(-5, -4),
                width=(0.05, 40)*u.ms, scintillate=None, spec_ind=(-10, 10),
                background=None, rfi_type=None, max_size=2**15,
                bg_mean=1, bg_std=1, window=True, normalize=True):

        if t_ref is None and f_ref is None:
            print("Warning selecting both the reference time and the reference" \
                  " frequency randomly can create samples that do not exist in" \
                  " the window and have a signal-to-noise ratio of 0.")

        self.label = 'frb'

        self.max_size = max_size

        self._set_rate_and_delta_t(rate, delta_t)
        self._set_background(background, NFREQ, NTIME, bg_mean, bg_std, rfi_type)
        self._set_freq(freq)
        self._set_f_ref(f_ref)
        self._set_t_ref(t_ref)
        self._set_dm(dm)
        self._set_fluence(fluence)
        self._set_width(width)
        self._set_spec_ind(spec_ind)
        self._set_scat_factor(scat_factor)
        self._set_scintillate(scintillate)
        self._set_window(window)

        self.signal = self._simulate_frb()

        if self.window:
            self.signal = apply_window(self.signal)

        self.sample = self.signal + self.background
        self.snr = self._get_snr()

        if normalize:
            self.sample = self._normalize()

        # add pulsar period attributes for consistency between classes
        self.period = 0 * u.ms
        self.n_periods = 0

    def __repr__(self):
        return '\n'.join(str(self._get_parameters())[1:-1].split(', '))

    def _set_scintillate(self, scintillate):
        """
        Sets class attribute scintillate.
        """
        if not isinstance(scintillate, bool): # 50% chance scintillation turned on
            scintillate = True if np.random.randint(10) % 2 == 0 else False
        self.scintillate = scintillate

    def _set_window(self, window):
        """
        Sets class attirbute window.
        """
        if not isinstance(window, bool): # 50% chance a window is applied
            window = True if np.random.randint(10) % 2 == 0 else False
        self.window = window

    def _set_freq(self, freq):
        """
        Sets class attributes freq and bandwidth. Also sets f_ref as the median
        of the bandwidth if f_ref has not yet been set.
        """
        try:
            temp = self.NFREQ
        except AttributeError as E:
            raise AttributeError("_set_freq called before _set_background")
        if not isinstance(freq, u.Quantity):
            freq *= u.MHz
        self.bandwidth = (max(freq) - min(freq)).to(u.MHz)
        self.freq = np.linspace(freq[0], freq[1], self.NFREQ).to(u.MHz)
        try:
            temp = self.f_ref
        except AttributeError as E:
            self.f_ref = np.median(self.freq)

    def _set_f_ref(self, f_ref):
        """
        Sets class attribute f_ref.
        """
        try:
            temp = self.freq
        except AttributeError as E:
            raise AttributeError("_set_f_ref called before _set_freq")
        if f_ref is None:
            f_ref = random.uniform(*(max(self.freq) * 0.99,
                                     min(self.freq) * 1.01))
        if not isinstance(f_ref, u.Quantity):
            f_ref *= u.MHz
        self.f_ref = f_ref.to(u.MHz)

    def _set_rate_and_delta_t(self, rate, delta_t):
        """
        Sets class attributes rate and delta_t. If rate is not None, delta_t
        will be based on rate not the given parameter.
        """
        if rate is not None:
            if not isinstance(rate, u.Quantity):
                rate *= u.MHz
            self.rate = rate.to(u.MHz)
            self.delta_t = (1 / self.rate).to(u.ms)
        else:
            if not isinstance(delta_t, u.Quantity):
                delta_t *= u.ms
            self.delta_t = delta_t
            self.rate = (1 / self.delta_t).to(u.MHz)

    def _set_t_ref(self, t_ref):
        """
        Sets class attirbute t_ref.
        """
        try:
            temp = self.NTIME
        except AttributeError as E:
            raise AttributeError("_set_t_ref called before NTIME was set")
        if t_ref is None:
            t_ref = (-.9 * self.NTIME, .9 * self.NTIME) * u.ms
            t_ref = random.uniform(*t_ref)
        if not isinstance(t_ref, u.Quantity):
            t_ref *= u.ms
        self.t_ref = t_ref.to(u.ms)

    def _set_dm(self, dm):
        """
        Sets class attribute dm.
        """
        if not isinstance(dm, u.Quantity):
            dm *= (u.pc * u.cm ** -3)
        try:
            self.dm = random.uniform(*dm).to(u.pc * u.cm ** -3)
        except TypeError:
            self.dm = dm.to(u.pc/u.cm**3)

    def _set_fluence(self, fluence):
        """
        Sets class attribute fluence.
        """
        if not isinstance(fluence, u.Quantity):
            fluence *= (u.Jy * u.ms)
        try:
            self.fluence = random.uniform(*fluence).to(u.Jy * u.ms)
        except TypeError:
            self.fluence = fluence.to(u.Jy * u.ms)

    def _set_width(self, width):
        """
        Sets class attribute width.
        """
        try:
            temp = self.delta_t
        except AttributeError:
            msg = "_set_width called before _set_rate_and_delta_t"
            raise AttributeError(msg)
        try:
            temp = self.f_ref
        except AttributeError:
            msg = "_set_width called before f_ref was set"
            raise AttributeError(msg)
        try:
            temp = self.dm
        except AttributeError:
            msg = "_set_width called before _set_dm"
            raiseAttribute(msg)
        try:
            temp = self.bandwidth
        except AttributeError:
            msg = "_set_width called before _set_freq"
            raise AttributeError(msg)
        try:
            temp = self.NFREQ
        except AttributeError:
            msg = "_set_width called before NFREQ was set"
            raise AttributeError(msg)
        if width is None:
            width =  (2, 5) * self.delta_t
        if not isinstance(width, u.Quantity):
            width *= u.ms
        units = width.unit
        value = width.value
        try:
            logvalue = np.random.lognormal(np.log(value[0]), value[1])
            value = value[0]
        except TypeError:
             logvalue = np.random.lognormal(np.log(value), value)
        width = max(min(logvalue, 100*value), 0.5*value) * units
        self.width = self._calc_width(width)

    def _set_spec_ind(self, spec_ind):
        """
        Sets class attribute spec_ind.
        """
        try:
            self.spec_ind = random.uniform(*spec_ind)
        except TypeError:
            self.spec_ind = spec_ind

    def _set_scat_factor(self, scat_factor):
        """
        Sets class attribute scat_factor.
        """
        try:
            self.scat_factor = np.exp(random.uniform(*scat_factor))
        except TypeError:
            self.scat_factor = np.exp(scat_factor)
        self.scat_factor = min(1, self.scat_factor + 1e-18) # quick bug fix hack

    def _set_background(self, noise, NFREQ, NTIME, mean, std, bg_description):
        """
        Generates the background noise the simulated FRB will be injected into.

        Args:
            noise: can be vdif file(s) containg the background data, a 2D numpy
                   array containing the background data background, or None at
                   which point gaussian noise will be used as the background data.
            NFREQ: int, number of frequency channels to use when creating gaussian
                   background. Ignored if noise is not None.
            NTIME: int, number of time bins to use when creating gaussian
                   background. Ignored if noise is not None.
            mean: float, the mean to use when creating a gaussian background.
                  Ignored if noise is not None.
            std: float, the standard deviation to use when creating gaussian
                 background. Ignored if noise is not None.
        Returns:
            np.float32 numpy array containing the background noise. Also sets the object
            attributes NFREQ, NTIME, rate, delta_t, files
        """
        try:
            temp = self.max_size
        except AttributeError as E:
            msg = "_set_background called before max_size was set"
            raise AttributeError(msg)
        try:
            temp = self.rate
        except AttributeError as E:
            msg = "_set_background called before _set_rate_and_delta_t"
            raise AttributeError(msg)
        try: # noise is a file or list of files
            noise, files, rate = read_vdif(noise, self.rate)
            self.NFREQ = noise.shape[0]
            self.NTIME = noise.shape[1]
            self.files = files
            self.rate = rate
            self.delta_t = (1/rate).to(u.ms)
            self.rfi_type = "from_vdif_files"
        except TypeError: # noise isn't a file or the file doesn't exist
            try: # noise is a numpy array 
                self.NFREQ = noise.shape[0]
                self.NTIME = noise.shape[1]
                self.files = ''
                self.rfi_type = bg_description
            except AttributeError: # noise isn't an array
                noise = np.random.normal(mean, std, size=(NFREQ, NTIME))
                self.NFREQ = noise.shape[0]
                self.NTIME = noise.shape[1]
                self.files = ''
                self.rfi_type = "gaussian_noise"

            if self.NTIME > self.max_size: # Reduce detail in time
                width = int(self.max_size)
                while self.NTIME % width != 0:
                    width -= 1
                x = self.NTIME//width
                noise = noise.reshape(self.NFREQ, x, width)
                noise = noise.mean(1)
                self.NFREQ = noise.shape[0]
                self.NTIME = noise.shape[1]
                self.delta_t = self.NTIME * self.delta_t / width
                self.NTIME = width

        self.background =  noise.astype(np.float32)

    def _disp_delay(self, f):
        """
        Calculate dispersion delay at frequency f
        """
        return k_dm * self.dm * (f**-2)

    def _arrival_time(self, f):
        """
        Calculate the burst arrival time at frequency f:
        arrival time = reference time + dispersion delay at f - dispersion delay at reference frequency
        """
        return self.t_ref + self._disp_delay(f) - self._disp_delay(self.f_ref)

    def _calc_width(self, width):
        """
        Calculate effective width of pulse including DM smearing, sample time,
        etc.
        """
        delta_freq = self.bandwidth/self.NFREQ

        # taudm in milliseconds
        tau_dm = 2*k_dm * self.dm * delta_freq / (self.f_ref)**3
        tI = np.sqrt(width**2 + self.delta_t**2 + tau_dm**2)
        t_index = max(1, int(tI / self.delta_t))

        return t_index

    def _scintillation(self):
        """
        Include spectral scintillation across the band. Approximate effect as
        a sinusoid, with a random phase and a random decorrelation bandwidth.
        """
        # Make location of peaks / troughs random
        scint_phi = np.random.rand()
        f = np.linspace(0, 1, len(self.freq))

        # Make number of scintils between 0 and 10 (ish)
        nscint = np.exp(np.random.uniform(np.log(1e-3), np.log(7)))
        envelope = np.cos(2*np.pi*nscint*f + scint_phi)
        envelope[envelope<0] = 0
        envelope += 0.1

        return envelope

    def _gaussian_profile(self, t0):
        """
        Create normalized Gaussian window for the pulse at time t0.
        """
        t = np.linspace(-self.NTIME//2, self.NTIME//2, self.NTIME)
        g = np.exp(-(t-t0)**2 / self.width**2)

        if not np.all(g > 0):
            g += 1e-18

        g /= g.max()

        return g

    def _scat_profile(self, f):
        """
        Create exponential scattering profile for the pulse at frequency f.
        """
        tau_nu = self.scat_factor * (f / self.f_ref)**-4.
        t = np.linspace(0., self.NTIME//2, self.NTIME)

        prof = 1 / tau_nu * np.exp(-t / tau_nu)
        return prof / prof.max()

    def _pulse_profile(self, f, t):
        """
        Convolve the gaussian and scattering profiles for final pulse shape at
        frequency f and time t. Pulse intensity multiplied  by fluence and
        (f/f_ref)^(spectral index) then normalized.
        """
        gaus_prof = self._gaussian_profile(t)
        scat_prof = self._scat_profile(f)
        pulse_prof = signal.fftconvolve(gaus_prof, scat_prof)[:self.NTIME]
        pulse_prof *= self.fluence.value
        pulse_prof *= (f / self.f_ref).value ** self.spec_ind
        pulse_prof /= (pulse_prof.max())
        return pulse_prof

    def _simulate_frb(self):
        """
        Create a simulated frb, including frequency-dependent width (smearing,
        scattering, etc.) and amplitude (scintillation, spectral index).
        """
        tmid = self.NTIME//2

        signal = np.zeros_like(self.background)
        self.arrival_indices = np.zeros(self.NFREQ) # used to dedisperse pulse later

        if self.scintillate:
            scint_amp = self._scintillation()

        for ii, f in enumerate(self.freq):
            # calculate the arrival time index
            t = int(self._arrival_time(f) / self.delta_t)
            self.arrival_indices[ii] = t

            # ensure that edges of data are not crossed
            if abs(t) >= tmid:
                continue

            p = self._pulse_profile(f, t)

            if self.scintillate is True:
                p *= scint_amp[ii]

            signal[ii] += p

        return np.abs(signal)

    def plot(self, save=None):
        f, axis = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(18, 30))
        axis[0].imshow(self.signal, interpolation="nearest", aspect="auto")
        axis[0].set_title("Simulated FRB DM:{:.3f}".format(self.dm))
        axis[0].set_xlabel("Time ({})".format(self.delta_t.unit))
        axis[0].set_ylabel("Frequency ({})".format(self.freq.unit))
        yticks = np.linspace(0, self.NFREQ, 10)
        ylabels = max(self.freq)-(self.bandwidth/self.NFREQ*yticks)
        ylabels = np.around(ylabels.value, 2)
        xticks = np.linspace(0, self.NTIME, 10)
        xlabels = (self.delta_t * xticks).astype(int)
        xlabels = xlabels.value
        axis[0].set_yticks(yticks)
        axis[0].set_yticklabels(ylabels)
        axis[0].set_xticks(xticks)
        axis[0].set_xticklabels(xlabels)
        axis[1].imshow(self.sample, interpolation='nearest', aspect='auto')
        axis[1].set_title("FRB S/N: {:.2f}".format(self.snr))
        axis[1].set_xlabel("Time ({})".format(self.delta_t.unit))
        axis[1].set_xticks(xticks)
        axis[1].set_xticklabels(xlabels)

        if save is None:
            plt.show()
        else:
            plt.savefig(save)

    def _dedisperse(self):
        output = np.copy(self.sample)
        for i in range(output.shape[0]):
            output[i, :] = np.roll(output[i, :], int(self.arrival_indices[i]*-1))
        return output

    def save(self, output):
        """
        Save the simulated FRB as a binary .npy file. If output already exists,
        it will be over written. If the file extension of output is not .npy,
        .npy will be appended to the end of output. For example if output is
        foobar.txt the data will be saved to a file named foobar.txt.npy

        Params:
            output (str): Path to where you want to save the simulation.

        Returns: None
        """
        # TODO: update to save to TFRecord
        output_dir = '/'.join(output.split('/')[:-1])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        np.save(output, self.sample)

    def _get_parameters(self):
        """
        Return the parameters of the event as a dictionary.
        """
        params = {'t_ref': self.t_ref, 'scintillate': self.scintillate,
                  'bandwidth': self.bandwidth, 'f_ref': self.f_ref,
                  'rate': self.rate, 'delta_t': self.delta_t,
                  'NFREQ': self.NFREQ, 'NTIME': self.NTIME, 'dm': self.dm,
                  'fluence': self.fluence, 'width': self.width,
                  'spec_ind': self.spec_ind, 'scat_factor': self.scat_factor,
                  'max_freq': max(self.freq), 'min_freq': min(self.freq),
                  'files': self.files, 'label': self.label, 'snr': self.snr,
                  'window': self.window, 'period': self.period,
                  'n_periods': self.n_periods}
        return params

    def _normalize(self):
        std = np.std(self.sample, axis=1, keepdims=True)
        std = np.where(std != 0, std, 1)
        return ((self.sample - np.mean(self.sample, axis=1, keepdims=True)) / std)

    def _get_snr(self):
        if np.sum(self.background) == 0:
            return np.inf
        else:
            return np.sqrt(np.sum(self.signal**2))/np.median(self.background)


def apply_window(A):
    # location of max value in each row, will follow the frb path
    centres = list(zip(np.arange(A.shape[0]), np.argmax(A, axis=1)))

    # remove edges, ensuring window is not centered at top/bottom edges of sample
    centres = centres[int(0.1*len(centres)):-int(0.1*len(centres))]

    # remove locations close to left/right edges of sample
    centres = [x for x in centres if
               x[1] > 0.1 * A.shape[1] and x[1] < 0.9 * A.shape[1]]

    # remove locations where burst does not exist
    centres = [x for x in centres if A[x] > 0]

    # nothing to be done; no valid window locations
    if len(centres) == 0:
        return A

    centre = centres[np.random.randint(len(centres))]

    # create window dimensions randomly
    max_left_width = max(0, centre[1] - int(0.05 * A.shape[1]))
    left = np.random.randint(0, max_left_width)

    min_right_width = min(A.shape[1], centre[1] + int(0.05 * A.shape[1]))
    right = np.random.randint(min_right_width, A.shape[1])

    max_top_width = max(0, centre[0] - int(0.05 * A.shape[0]))
    top = np.random.randint(0, max_top_width)

    min_bottom_width = min(A.shape[0], centre[0] + int(0.05 * A.shape[0]))
    bottom = np.random.randint(min_bottom_width, A.shape[0])

    def decay(width, reverse=False):
        x = np.arange(width) + 1
        x = 1 / (x**2)
        if reverse:
            x = np.flip(x, axis=0)
        return x

    window = np.ones_like(A)

    left_decay = np.vstack([decay(left, True)] * window.shape[0])
    window[:, :left] *= left_decay

    right_decay = np.vstack(
        [decay(window.shape[1] - right)] * window.shape[0]
    )
    window[:, right:] *= right_decay

    top_decay = np.vstack([decay(top, True)] * window.shape[1])
    window[:top, :] *= top_decay.T

    bottom_decay = np.vstack(
        [decay(window.shape[0] - bottom)] * window.shape[1]
    )
    window[bottom:, :] *= bottom_decay.T

    return A * window

if __name__ == "__main__":
    d = '/scratch/r/rhlozek/rylan/aro_rfi/000010'
    files = [d + str(x) + '.vdif' for x in range(10)]
    #files = [d + str(x) + '.vdif' for x in range(10)]
    #d = '/home/rylan/dunlap/data/vdif/00001'
    #files.extend([d + str(x) + '.vdif' for x in range(16)[10:]])
    event = FRB(background=files, dm=250*u.pc/u.cm**3)
    print(event)
    event.plot()

