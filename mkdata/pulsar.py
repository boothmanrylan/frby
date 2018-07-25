import random
import numpy as np
import astropy.units as u
# import matplotlib # Needed when running on scinet
# matplotlib.use('Agg') # Needed when running on scinet
import matplotlib.pyplot as plt
from frb import FRB
from frb import k_dm

class Pulsar(FRB):
    def __init__(self, t_ref=0*u.ms, f_ref=600*u.MHz, NFREQ=1024, NTIME=1000,
                 delta_t=3.2*u.ms, dm=(2, 25)*(u.pc/u.cm**3),
                 fluence=(2.5, 15)*(u.Jy*u.ms), freq=(800, 400)*u.MHz,
                 rate=None, scat_factor=(-5, -4),
                 width=(0.5, 5)*u.ms, scintillate=False, spec_ind=(-1, 1),
                 background=None, max_size=2**15, period=(4, 5e2)*u.ms):
        # Initialize rate and delta_t because they are required for
        # super.make_background to function
        try:
            self.rate = rate.to(u.MHz)
            self.delta_t = (1 / self.rate).to(u.ms)
        except AttributeError:
            self.delta_t = delta_t.to(u.ms)
            self.rate = (1 / self.delta_t).to(u.MHz)

        try:
            self.period = random.uniform(*period).to(u.ms)
        except TypeError:
            self.period = period.to(u.ms)

        try:
            self.dm = random.uniform(*dm).to(u.pc/u.cm**3)
        except TypeError:
            self.dm = dm.to(u.pc/u.cm**3)

        self.max_size = max_size

        # use FRB.make_background to initialize NTIME NFREQ and background
        super(Pulsar, self).make_background(background, NFREQ, NTIME)

        duration = NTIME * self.delta_t

        # Ensure that more than one period of the pulsar is saved
        min_pulse_period = duration / 2
        if self.period > min_pulse_period:
            self.period = min_pulse_period

        # Extra addition to ensure full coverage i.e. ensure
        # n_periods * spp >= NTIME
        self.n_periods = int(duration / self.period) + 1
        self.spp = (self.NTIME // self.n_periods) + 1 # samples per period

        # Determine the amount of time required for the pulse to appear at
        # every frequency in the bandwidth based on the dispersion measure.
        # Use the amount of samples over that time as NTIME when generating the
        # first pulse.
        pulse_time = k_dm * self.dm * ((0.75*min(freq))**-2)
        pulse_samples = int(pulse_time.to(u.s) * self.rate.to(1/u.s))

        # The pulse needs fewer than spp samples, generate the pulse over spp
        # samples to avoid having to add a buffer later on.
        pulse_samples = max(pulse_samples, self.spp)

        # Save attributes that will be changed by the call to super
        temp_bg = np.copy(self.background)
        temp_NTIME = self.NTIME
        temp_delta_t = self.delta_t
        temp_rate = self.rate
        temp_files = self.files

        # The background of a single pulse will be all zeros
        bg = np.zeros((self.NFREQ, pulse_samples), dtype=float)

        max_pulse_size = (self.max_size // self.n_periods) + 1

        # Call FRB.__init__ to generate a single period of the pulsar
        super(Pulsar, self).__init__(t_ref=t_ref, f_ref=f_ref,
                                     NFREQ=self.NFREQ, delta_t=self.delta_t,
                                     NTIME=pulse_samples, freq=freq,
                                     fluence=fluence, rate=self.rate,
                                     dm=self.dm, scat_factor=scat_factor,
                                     width=width, scintillate=scintillate,
                                     spec_ind=spec_ind, background=bg,
                                     max_size=max_pulse_size)

        # Restore attributes that are changed by the call to super
        self.background = temp_bg
        self.NTIME = temp_NTIME
        self.delta_t = temp_delta_t
        self.rate = temp_rate
        self.files = temp_files

        # To account for the dispersion measure the pulse may have more than
        # the allowed number of samples, if this is the case fold the excess
        # samples back onto the pulse and sum over the extra dimension.
        pulse_len = self.frb.shape[1]
        if pulse_len > self.spp:
            # If pulse cannot be divided evenly by spp buffer it with zeros
            pulse = np.copy(self.frb)
            while pulse.shape[1] > self.spp:
                valid = pulse[:, :self.spp]
                over = pulse[:, self.spp:]
                if valid.shape[1] > over.shape[1]:
                    extra = valid.shape[1] - over.shape[1]
                    buf = np.zeros((self.NFREQ, extra), dtype=float)
                    over = np.hstack([over, buf])
                    pulse = valid + over
                elif over.shape[1] > valid.shape[1]:
                    extra = over.shape[1] - valid.shape[1]
                    buf = np.zeros((self.NFREQ, extra), dtype=float)
                    valid = np.hstack([valid, buf])
                    pulse = valid + over
                else:
                    pulse = valid + over
        else:
            pulse = self.frb

        # Combine the single pulse n_period times to create the pulsar
        foreground = np.hstack([pulse] * self.n_periods)
        foreground = foreground[:, :self.NTIME]

        # Inject the pulsar into the background rfi
        self.pulsar = self.background + foreground

        self.attributes = super(Pulsar, self).get_parameters()
        self.attributes['class'] = 'pulsar'
        self.attributes['n_periods'] = self.n_periods
        self.attributes['period'] = self.period

    def plot(self, save=None):
        plt.imshow(self.pulsar, interpolation='nearest', aspect='auto')
        plt.colorbar()
        plt.show()

if __name__ == '__main__':
    pulsar = Pulsar()
    pulsar.plot()
