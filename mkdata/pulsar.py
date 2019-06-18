import random
import numpy as np
import astropy.units as u
# import matplotlib # Needed when running on scinet
# matplotlib.use('Agg') # Needed when running on scinet
import matplotlib.pyplot as plt
from frb import FRB, k_dm, apply_window

class Pulsar(FRB):
    """
    Generate a simulated pulsar by stacking together simulated FRBs
    periodically.

    All parameters that are quantities can be passed as unitless floats, which
    will be converted to quantities with the default units.

    If any of scintillate, or window are passed as "None", a uniformly sampled
    random value will be assigned to them from a range that makes sense with
    respect to the other parameters.

    Any of dm, fluence, scat_factor, width, and spec_ind can be passed as a
    single value or a 2-tuple of values. If one of them is passed as a tuple a
    final value will be drawn uniformly from the range in the tuple and
    assigned to the parameter.

    Any parameters whose final value is determined randomly will have that
    final value saved in the parameter, not the range the value was drawn from.

    Parameters
    ----------
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
    pulsar : ndarray
        The simulated pulsar in background noise
    background : ndarray
        The background noise.
    signal : ndarray
        The simulated pulsar without any background noise.
    frb : ndarray
        A singla period of the pulsar, without any background noise.
    bandwith : Quantity
        The frequency range of the samples.
    snr : float
        The approximate signal-to-noise ratio of the burst.
    attributes : dict
        Key parameters used to make the simulation.
    """

    def __init__(self, NFREQ=1024, NTIME=8192,
                 delta_t=0.16*u.ms, dm=(2, 25)*(u.pc/u.cm**3),
                 fluence=(0.02, 15)*(u.Jy*u.ms), freq=(800, 400)*u.MHz,
                 rate=1000*u.Hz, scat_factor=(-5, -4),
                 width=(0.05, 5)*u.ms, scintillate=False, spec_ind=(-1, 1),
                 background=None, max_size=2**15, period=(4, 500)*u.ms,
                 bg_mean=1, bg_std=1, window=True, normalize=True):

        # set the pulse attrributes
        self.max_size = max_size
        self.NFREQ = NFREQ
        super()._set_dm(dm)
        super()._set_fluence(fluence)
        super()._set_rate_and_delta_t(rate, delta_t)
        super()._set_freq(freq)
        super()._set_width(width)
        super()._set_spec_ind(spec_ind)
        super()._set_scat_factor(scat_factor)
        super()._set_scintillate(scintillate)
        self.t_ref = 0
        self.window = False
        self.pulse_duration = super()._disp_delay(max(self.freq)) * 10
        self.NTIME = int(self.pulse_duration / self.delta_t) + 1
        self.background = np.zeros((self.NFREQ, self.NTIME))

        # simulate one period of the pulsar
        self.frb = super()._simulate_frb()

        # set background for entire sample not just one burst
        super()._set_background(background, NFREQ, NTIME, bg_mean, bg_std)

        # Crop burst so that it runs from edge to edge
        self.frb = self.frb[:, np.argmax(self.frb[0]):np.argmax(self.frb[-1])]

        self.duration = self.NTIME * self.delta_t
        self._set_period(period)

        # pad frb so that it has the width of exactly one period
        pad = int(self.period / self.delta_t) + 1 - self.frb.shape[1]
        self.frb = np.hstack([self.frb, np.zeros((self.NFREQ, pad))])

        self.n_periods = int(self.duration / self.period) + 1

        # stack the pulse n_periods times
        self.window = window
        if self.window:
            # apply different window to each burst
            signals = [apply_window(self.frb) for _ in range(self.n_periods)]
        else:
            signals = [self.frb] * self.n_periods
        self.signal = np.hstack(signals)

        # crop the pulses so that the duration is exactly NTIME, randomly crop
        # partially from left and right side of pulses
        if self.signal.shape[1] > self.NTIME:
            diff = self.signal.shape[1] - self.NTIME
            ldiff = np.random.randint(0, diff)
            rdiff = diff - ldiff
            self.signal = self.signal[:, ldiff:-rdiff]

        self.pulsar = self.background + self.signal

        self.snr = super()._get_snr()

        if normalize:
            self.pulsar = self._normalize()

        self.attributes = self._get_parameters()

    def __repr__(self):
        return '\n'.join(str(self._get_parameters())[1:-1].split(', '))

    def _normalize(self):
        std = np.std(self.pulsar, axis=1, keepdims=True)
        std = np.where(std != 0, std, 1)
        return ((self.pulsar - np.mean(self.pulsar, axis=1, keepdims=True)) / std)

    def _set_dm(self, dm):
        if not isinstance(dm, u.Quantity):
            dm *= (u.pc * u.cm ** -3)
        try:
            self.dm = random.uniform(*dm).to(u.pc * u.cm ** -3)
        except TypeError:
            self.dm = dm.to(u.pc * u.cm ** -3)

    def _set_period(self, period):
        if not isinstance(period, u.Quantity):
            period *= u.ms
        try:
            self.period = random.uniform(*period).to(u.ms)
        except TypeError as E:
            self.period = period.to(u.ms)
        try:
            assert self.duration / 2 >= self.pulse_duration
        except AssertionError:
            raise AssertionError("DM is too high for the given period")
        # minimum possible period is self.frb.shape[1]
        # maximum possible period is self.duration / 2
        self.period = min(self.duration/2, max(self.pulse_duration, self.period))

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
                  'files': self.files, 'class': 'pulsar', 'snr': self.snr,
                  'window': self.window, 'period': self.period,
                  'n_periods': self.n_periods}
        return params

    def plot(self, save=None):
        plt.imshow(self.pulsar, interpolation='nearest', aspect='auto')
        plt.colorbar()
        plt.show()

if __name__ == '__main__':
    pulsar = Pulsar()
    pulsar.plot()
