import random
import numpy as np
import astropy.units as u
# import matplotlib # Needed when running on scinet
# matplotlib.use('Agg') # Needed when running on scinet
import matplotlib.pyplot as plt
from frb import FRB

class Pulsar(FRB):
    def __init__(self, t_ref=0*u.ms, f_ref=0.6*u.GHz, NFREQ=1024, NTIME=1000,
                 delta_t=3.2*u.ms, dm=(2.5, 35)*(u.pc/u.cm**3),
                 fluence=(2.5, 15)*(u.Jy*u.ms), freq=(0.8, 0.4)*u.GHz,
                 rate=None, scat_factor=(-5, -4),
                 width=(0.5, 5)*u.ms, scintillate=False, spec_ind=(-1, 1),
                 background=None, max_size=2**15, period=(2, 5e3)*u.ms):
        # Initialize rate and delta_t because they are required for
        # super.make_background to function
        try:
            self.rate = rate.to(u.MHz)
            self.delta_t = (1 / rate).to(u.ms)
        except AttributeError:
            self.rate = rate
            self.delta_t = delta_t.to(u.ms)

        try:
            self.period = random.uniform(*period).to(u.ms)
        except TypeError:
            self.period = period.to(u.ms)

        self.max_size = max_size
        duration = NTIME * self.delta_t

        # Extra addition to ensure full coverage
        # i.e. ensure n_periods * ntime_per_period >= NTIME
        self.n_periods = int(duration / self.period) + 1
        self.ntime_per_period = (NTIME // self.n_periods) + 1
        max_period_size = self.max_size // self.n_periods + 1

        # use FRB.make_background to initialize NTIME NFREQ and background
        super(Pulsar, self).make_background(background, NFREQ, NTIME)

        # Save background and NTIME to ensure not overwritten by super
        temp_bg = self.background
        temp_NTIME = self.NTIME

        # Create the background of a single pulse
        bg = np.zeros((self.NFREQ, self.ntime_per_period), dtype=float)

        # Call FRB __init__ to generate a single period of the pulsar
        super(Pulsar, self).__init__(t_ref=t_ref, f_ref=f_ref,
                                     NFREQ=self.NFREQ, delta_t=self.delta_t,
                                     NTIME=self.ntime_per_period, freq=freq,
                                     fluence=fluence, rate=self.rate, dm=dm,
                                     scat_factor=scat_factor, width=width,
                                     scintillate=scintillate, spec_ind=spec_ind,
                                     background=bg, max_size=max_period_size)
        # Restore background and NTIME from before the call to super
        self.background = temp_bg
        self.NTIME = temp_NTIME

        # Combine the single pulse n_period times to create the pulsar
        foreground = np.hstack([self.simulated_frb] * self.n_periods)
        foreground = foreground[:, :self.NTIME]

        # Inject the pulsar into the background rfi
        self.pulsar = self.background + foreground

    def plot(self, save=None):
        plt.imshow(self.pulsar, interpolation='nearest', aspect='auto')
        plt.colorbar()
        plt.show()

if __name__ == '__main__':
    pulsar = Pulsar()
    pulsar.plot()
