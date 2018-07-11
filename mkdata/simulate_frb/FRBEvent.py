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
matplotlib.use('Agg') # Needed when running on scinet
import matplotlib.pyplot as plt
from math import gcd

k_dm = 4.148808e6 * (u.MHz**2 * u.cm**3 * u.ms / u.pc)

class FRBEvent(object):
    """
    Class to generate a realistic fast radio burst and add the event to data,
    including scintillation, temporal scattering, spectral index variation, and
    DM smearing .This class was expanded from real-time FRB injection in
    Kiyoshi Masui's https://github.com/kiyo-masui/burst\_search
    """
    def __init__(self, t_ref=0*u.ms, f_ref=0.6*u.GHz, NFREQ=1024, NTIME=2**15,
                 delta_t=0.16*u.ms, dm=(100, 500)*(u.pc/u.cm**3),
                 fluence=(0.02, 150)*(u.Jy*u.ms), freq=(0.8, 0.4)*u.GHz,
                 rate=(0.4/1024)*u.GHz, scat_factor=(-5, -4),
                 width=(0.05, 30)*u.ms, scintillate=True, spec_ind=(-10, 15),
                 background=None, max_size=2**15):
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
        # TODO: If input are not quantities, give them default units
        self.t_ref = t_ref.to(u.ms)
        self.scintillate = scintillate
        self.bandwidth = (max(freq) - min(freq)).to(u.MHz)
        self.output_file = None
        self.max_size = max_size

        if f_ref is None:
            f_ref = (max(freq.value)*0.9, min(freq.value)*1.1) * freq.unit
            f_ref = random.uniform(*f_ref)
        self.f_ref = f_ref.to(u.MHz)

        try:
            self.rate=rate.to(u.MHz)
            self.delta_t = (1/rate).to(u.ms)
        except AttributeError:
            self.rate = rate
            self.delta_t = delta_t.to(u.ms)

        self.make_background(background, NFREQ, NTIME)

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
        self.simulated_frb = self.simulate_frb()

    def __repr__(self):
        repr = str(self.get_parameters())
        repr = repr[1:-1]
        repr = repr.split(', ')
        repr = '\n'.join(repr)
        return repr

    def make_background(self, background, NFREQ, NTIME):
        try: # background is a file or list of files
             data, files, rate = read_vdif(background, self.rate)
             self.background = data
             self.NFREQ = self.background.shape[0]
             self.NTIME = self.background.shape[1]
             self.input = files
             self.rate = rate
             self.delta_t = (1/rate).to(u.ms)
        except TypeError as E: # background isn't a file or the file doesn't exist
            raise(E)
            try: # background is a numpy array 
                self.background = background
                self.NFREQ = background.shape[0]
                self.NTIME = background.shape[1]
                self.input = 'ndarray'
            except AttributeError: # background isn't an array
                self.background = np.random.normal(0, 1, size=(NFREQ, NTIME))
                self.NFREQ = NFREQ
                self.NTIME = NTIME
                self.input = 'None'
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
        t = self.disp_delay(f)
        t = t - self.disp_delay(self.f_ref)
        return self.t_ref + t

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
        # pulse_prof /= (pulse_prof.max()*self.stds)
        # pulse_prof /= (self.width / self.delta_t.value)

        return pulse_prof

    def simulate_frb(self):
        """
        Method to add already-dedispersed pulse to background noise data.
        Includes frequency-dependent width (smearing, scattering, etc.) and
        amplitude (scintillation, spectral index).
        """
        data = np.copy(self.background)
        tmid = self.NTIME//2

        if self.scintillate:
            scint_amp = self.scintillation(self.freq)

        for ii, f in enumerate(self.freq):
            # calculate the arrival time index
            t = int(self.arrival_time(f) / self.delta_t)

            # ensure that edges of data are not crossed
            if abs(t) >= tmid:
                continue

            p = self.pulse_profile(f, t)

            if self.scintillate is True:
                p *= scint_amp[ii]

            data[ii] += p
        return data

    # TODO: Make this more efficient 
    def dm_transform(self, data, NDM=50):
        dm = np.linspace(-self.dm, self.dm, NDM)
        dm_data = np.zeros([NDM, self.NTIME])

        for ii, dm in enumerate(dm):
            for jj, f in enumerate(self.freq):
                t = int(self.arrival_time(f) / self.delta_t)
                data_rot = np.roll(data[jj], t, axis=-1)
                dm_data[ii] += data_rot

        return data

    def plot(self, save=None):
        f, axis = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(18, 30))
        axis[0].imshow(self.background, vmin=-1.0, vmax=1.0, interpolation="nearest",
                       aspect="auto")
        axis[0].set_title("Background Noise")
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
        axis[1].imshow(self.simulated_frb, vmin=-1.0, vmax=1.0, interpolation="nearest",
                   aspect="auto")
        axis[1].set_title("Simulated FRB")
        axis[1].set_xlabel("Time (ms)")
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
        output_dir = '/'.join(output.split('/')[:-1])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        np.save(output, self.simulated_frb)
        self.output_file = output

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
                  'Name': self.output_file, 'Input': self.input, 'FRB': True}
        return params

    def get_headers(self):
        params = self.get_parameters()
        headers = sorted(params.keys())
        return headers

    def get_params(self):
        params = self.get_parameters()
        headers = self.get_headers()
        params = [params[x] for x in headers]
        return params


def read_vdif(vdif_files, rate_in, rate_out=1e-3*u.MHz):
    """
    Reads and reduces the data in vdif_files. Will reduce the size of the data
    so that the sample rate is as close to rate_out as possible.

    Args:
        vdif_files:             The vdif files to be read. Can be either a
                                single file (str), a list of files (list(str))
                                or a sequentialfile object.
        rate_in (Quantity):     The sampling rate of the vdifs.
        rate_out (Quantity):    The desired sampling rate after reducing
                                data size. Default: 0.001 MHz

    Returns:
        complete_data (array):  The reduced data contained in vdif_files.
        file_names (str):       The name of the first and last file in
                                vdif_files, saved to use in metadata later.
        rate_out (Quantity):    The sample out of the data after being reduced.
    """
    # Ensure that vdif_files is a sequentialfile object, convert if not
    try:
        temp = vdif_files.file_size # Will fail if not a sequentialfile
    except AttributeError:
        try:
            temp = vdif_files.split() # Will split if str, fail if list
            vdif_files = [vdif_files] # Convert to a list with 1 element
        except AttributeError:
            pass # Do nothing because vdif_files is already a list
        vdif_files = sf.open(vdif_files)

    with vdif.open(vdif_files, 'rs', sample_rate=rate_in) as fh:
        # Determine the number of samples and the sample rate
        nfreq = fh.shape[-1]
        n_samples_in = fh.shape[0]
        n_samples_out = int(((n_samples_in/rate_in)*rate_out).value)
        n_samples_in_trunc = (n_samples_in//n_samples_out)*n_samples_out
        chunk_width = n_samples_in_trunc // n_samples_out
        rate_out = (n_samples_out * rate_in) / n_samples_in
        n_chunks = n_samples_in_trunc // chunk_width

        # Read chunks in groups of chunks no larger than 2**15 samples, to
        # reduce the number of reads necessary to read all the data
        max_samples_per_read = 2**15
        chunks_per_read = max_samples_per_read//chunk_width
        n_reads = n_chunks//chunks_per_read

        # Read the file in chunks, reducing each chunk after reading
        complete_data = np.zeros((nfreq, n_samples_out))
        for i in range(n_reads):
            # Ensure enough data is left to do a complete read
            if chunk_width * chunks_per_read < n_samples_in_trunc:
                data = fh.read(chunk_width * chunks_per_read)
            else:
                # TODO: find method to read remaining data
                break

            # Get the power from data
            data = (np.abs(data) ** 2).mean(1)
            data -= np.nanmean(data, axis=1, keepdims=True)

            # Reshape to take mean across time of each chunk
            data = data.reshape(chunks_per_read, chunk_width, nfreq).mean(1)
            data = data.T

            # Save chunk to complete_data
            start = i * chunks_per_read
            stop = (i + 1) * chunks_per_read
            complete_data[:, start:stop] = data

    # Get the first and last input filenames without their path or suffix 
    files = vdif_files.files
    files = ['.'.join(x.split('/')[-1].split('.')[:-1]) for x in files]
    file_names = '-'.join([files[0], files[-1]])

    return complete_data, file_names, rate_out


if __name__ == "__main__":
    d = '/scratch/r/rhlozek/rylan/aro_rfi/000010'
    files = [d + str(x) + '.vdif' for x in range(10)]
    #files = [d + str(x) + '.vdif' for x in range(10)]
    #d = '/home/rylan/dunlap/data/vdif/00001'
    #files.extend([d + str(x) + '.vdif' for x in range(16)[10:]])
    event = FRBEvent(background=files, dm=250*u.pc/u.cm**3)
    print(event)
    event.plot()

