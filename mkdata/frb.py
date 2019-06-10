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

class FRB(object):
    """
    Generate a simulated fast radio burst.
    Based on https://github.com/liamconnor/single_pulse_ml which is in turn
    based on https://github.com/kiyo-masui/burst_search
    """
    def __init__(self, t_ref=0*u.ms, f_ref=600*u.MHz, NFREQ=1024, NTIME=1024,
                 delta_t=0.16*u.ms, dm=(50, 3750)*(u.pc/u.cm**3),
                 fluence=(0.02, 200)*(u.Jy*u.ms), freq=(0.8, 0.4)*u.GHz,
                 rate=1000*u.Hz, scat_factor=(-5, -3),
                 #rate=(0.4/1024)*u.GHz, scat_factor=(-5, -4),
                 width=(0.05, 40)*u.ms, scintillate=True, spec_ind=(-10, 10),
                 background=None, max_size=2**15, bg_mean=1, bg_std=1):
        """
        t_ref :         The reference time used when computing the DM.
        f_ref :         The reference frequency used when computing the DM.
        NFREQ :         The number of frequency bins.
        NTIME :         The number of time bins.
        delta_t :       The time step between two time bins.
        dm :            The dispersion measure (DM) of the pulse.
        fluence :       The fluence of the pulse.
        freq :          The range of freqeuncies in the data.
        rate :          The rate at which samples were taken.
        scat_factor :   How much scattering occurs in the pulse.
        width :         The width in time of the pulse.
        scintillate :   If true scintills are added to the data.
        spec_ind :      The spectral index.
        background :    The background the pulse will be injected into.
        max_size :      The size to reduce/bin the data to in time.
        """
        # TODO: If inputs are not quantities, give them default units
        self.scintillate = scintillate
        self.bandwidth = (max(freq) - min(freq)).to(u.MHz)
        self.max_size = max_size

        if f_ref is None:
            f_ref = (max(freq.value)*0.99, min(freq.value)*1.01) * freq.unit
            f_ref = random.uniform(*f_ref)
        self.f_ref = f_ref.to(u.MHz)

        try:
            self.rate=rate.to(u.MHz)
            self.delta_t = (1/rate).to(u.ms)
        except AttributeError:
            self.rate = rate
            self.delta_t = delta_t.to(u.ms)

        if t_ref is None:
            half_time = NTIME * self.delta_t.value // 2
            t_ref = (-half_time, 1.9 * half_time) * u.ms
            t_ref = random.uniform(*t_ref)
        self.t_ref = t_ref.to(u.ms)

        self.make_background(background, NFREQ, NTIME, bg_mean, bg_std)

        self.stds = np.std(self.background)

        try:
            self.dm = random.uniform(*dm).to(u.pc/u.cm**3)
        except TypeError:
            self.dm = dm.to(u.pc/u.cm**3)

        try:
            self.fluence = random.uniform(*fluence)
        except TypeError:
            self.fluence = fluence

        if width is None:
            width =  (2, 5) * self.delta_t
        units = width.unit
        value = width.value
        try:
            logvalue = np.random.lognormal(np.log(value[0]), value[1])
            value = value[0]
        except TypeError:
             logvalue = np.random.lognormal(np.log(value), value)
        width = max(min(logvalue, 100*value), 0.5*value) * units
        self.width = self.calc_width(width)

        try:
            self.spec_ind = random.uniform(*spec_ind)
        except TypeError:
            self.spec_ind = spec_ind

        try:
            self.scat_factor = np.exp(random.uniform(*scat_factor))
        except TypeError:
            self.scat_factor = np.exp(scat_factor)
        self.scat_factor = min(1, self.scat_factor + 1e-18) # quick bug fix hack

        self.freq = np.linspace(freq[0], freq[1], self.NFREQ).to(u.MHz)
        self.frb = self.simulate_frb()
        self.attributes = self.get_parameters()

    def __repr__(self):
        return '\n'.join(str(self.get_parameters())[1:-1].split(', '))

    def make_background(self, background, NFREQ, NTIME, mean, std):
        try: # background is a file or list of files
             data, files, rate = read_vdif(background, self.rate)
             self.background = data
             self.NFREQ = self.background.shape[0]
             self.NTIME = self.background.shape[1]
             self.files = files
             self.rate = rate
             self.delta_t = (1/rate).to(u.ms)
        except TypeError as E: # background isn't a file or the file doesn't exist
            try: # background is a numpy array 
                self.background = background
                self.NFREQ = background.shape[0]
                self.NTIME = background.shape[1]
                self.files = ''
            except AttributeError: # background isn't an array
                self.background = np.random.normal(mean, std, size=(NFREQ, NTIME))
                self.NFREQ = NFREQ
                self.NTIME = NTIME
                self.files = ''
            if self.NTIME > self.max_size: # Reduce detail in time
                width = int(self.max_size)
                while self.NTIME % width != 0:
                    width -= 1
                x = self.NTIME//width
                self.background = self.background.reshape(self.NFREQ, x, width)
                self.background = self.background.mean(1)
                self.delta_t = self.NTIME * self.delta_t / width
                self.NTIME = width

    def disp_delay(self, f):
        """
        Calculate dispersion delay in seconds for frequency,f, in MHz, _dm in
        pc cm**-3, and a dispersion index, _disp_ind.
        """
        return k_dm * self.dm * (f**-2)

    def arrival_time(self, f):
        return self.t_ref + self.disp_delay(f) - self.disp_delay(self.f_ref)

    def calc_width(self, width):
        """
        Calculated effective width of pulse including DM smearing, sample time,
        etc.  Input/output times are in seconds.
        """

        delta_freq = self.bandwidth/self.NFREQ

        # taudm in milliseconds
        tau_dm = 2*k_dm * self.dm * delta_freq / (self.f_ref)**3
        tI = np.sqrt(width**2 + self.delta_t**2 + tau_dm**2)
        t_index = max(1, int(tI / self.delta_t))

        return t_index

    def scintillation(self, freq):
        """
        Include spectral scintillation across the band. Apulse_profroximate effect as
        a sinusoid, with a random phase and a random decorrelation bandwidth.
        """
        # Make location of peaks / troughs random
        scint_phi = np.random.rand()
        f = np.linspace(0, 1, len(freq))

        # Make number of scintils between 0 and 10 (ish)
        nscint = np.exp(np.random.uniform(np.log(1e-3), np.log(7)))
        envelope = np.cos(2*np.pi*nscint*f + scint_phi)
        envelope[envelope<0] = 0
        envelope += 0.1

        return envelope

    def gaussian_profile(self, t0):
        """
        Use a normalized Gaussian window for the pulse, rather than a boxcar.
        """
        t = np.linspace(-self.NTIME//2, self.NTIME//2, self.NTIME)
        g = np.exp(-(t-t0)**2 / self.width**2)

        if not np.all(g > 0):
            g += 1e-18

        g /= g.max()

        return g

    def scat_profile(self, f):
        """
        Include exponential scattering profile.
        """
        tau_nu = self.scat_factor * (f / self.f_ref)**-4.
        t = np.linspace(0., self.NTIME//2, self.NTIME)

        prof = 1 / tau_nu * np.exp(-t / tau_nu)
        return prof / prof.max()

    def pulse_profile(self, f, t):
        """
        Convolve the gaussian and scattering profiles for final pulse shape at
        frequency channel f.
        """
        gaus_prof = self.gaussian_profile(t)
        scat_prof = self.scat_profile(f)
        pulse_prof = signal.fftconvolve(gaus_prof, scat_prof)[:self.NTIME]
        pulse_prof *= self.fluence.value
        pulse_prof *= (f / self.f_ref).value ** self.spec_ind
        pulse_prof /= (pulse_prof.max()*self.stds)
        # pulse_prof /= (self.width / self.delta_t.value)
        return pulse_prof

    def simulate_frb(self):
        """
        Method to add already-dedispersed pulse to background noise data.
        Includes frequency-dependent width (smearing, scattering, etc.) and
        amplitude (scintillation, spectral index).
        """
        tmid = self.NTIME//2

        self.signal = np.zeros_like(self.background)
        self.arrival_indices = np.zeros(self.NFREQ)

        if self.scintillate:
            scint_amp = self.scintillation(self.freq)

        for ii, f in enumerate(self.freq):
            # calculate the arrival time index
            t = int(self.arrival_time(f) / self.delta_t)
            self.arrival_indices[ii] = t

            # ensure that edges of data are not crossed
            if abs(t) >= tmid:
                continue

            p = self.pulse_profile(f, t)

            if self.scintillate is True:
                p *= scint_amp[ii]

            self.signal[ii] += p

        self.snr = np.max(self.signal) / np.median(self.background)
        return self.background + self.signal


    def plot(self, save=None):
        f, axis = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(18, 30))
        axis[0].imshow(self.signal, interpolation="nearest", aspect="auto")
        axis[0].set_title("Simulated FRB")
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
        axis[1].imshow(self.frb, interpolation="nearest", aspect="auto")
        axis[1].set_title("After Injection")
        axis[1].set_xlabel("Time ({})".format(self.delta_t.unit))
        axis[1].set_xticks(xticks)
        axis[1].set_xticklabels(xlabels)

        if save is None:
            plt.show()
        else:
            plt.savefig(save)

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
        # TODO: update to save to hdf5
        output_dir = '/'.join(output.split('/')[:-1])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        np.save(output, self.frb)

    def get_parameters(self):
        """
        Return the parameters of the event as a dictionary.
        """
        params = {'t_ref': self.t_ref, 'scintillate': self.scintillate,
                  'bandwidth': self.bandwidth, 'f_ref': self.f_ref,
                  'rate': self.rate, 'delta_t': self.delta_t,
                  'NFREQ': self.NFREQ, 'NTIME': self.NTIME, 'stds': self.stds,
                  'dm': self.dm, 'fluence': self.fluence, 'width': self.width,
                  'spec_ind': self.spec_ind, 'scat_factor': self.scat_factor,
                  'max_freq': max(self.freq), 'min_freq': min(self.freq),
                  'files': self.files, 'class': 'frb', 'snr': self.snr}
        return params


if __name__ == "__main__":
    d = '/scratch/r/rhlozek/rylan/aro_rfi/000010'
    files = [d + str(x) + '.vdif' for x in range(10)]
    #files = [d + str(x) + '.vdif' for x in range(10)]
    #d = '/home/rylan/dunlap/data/vdif/00001'
    #files.extend([d + str(x) + '.vdif' for x in range(16)[10:]])
    event = FRB(background=files, dm=250*u.pc/u.cm**3)
    print(event)
    event.plot()

