import numpy as np
import astropy.units as u
from rfi import fourier_lowpass_filter, butterworth_lowpass_filter, one_over_f
from rfi import sinusoid, sinusoidal_sum, sinusoidal_product
from rfi import changing_sinusoid, patches, delta

single_mods = [[{'func': fourier_lowpass_filter,
                 'name': 'fourier_lowpass_filter',
                 'input': 'value',
                 'freq_range': None,
                 'time_range': None,
                 'params': {'cutoff': None
                     }
                 }],
                [{'func': butterworth_lowpass_filter,
                  'name': 'butterworth_lowpass_filter',
                  'input': 'value',
                  'freq_range': None,
                  'time_range': None,
                  'params': {'cutoff': np.random.uniform(0.01, 0.1),
                             'sr': 1.004 * u.ms,
                             'N': np.random.uniform(2, 10)
                             }
                 }],
                [{'func': one_over_f,
                  'name': 'one_over_f',
                  'input': 'freq',
                  'freq_range': None,
                  'time_range': None,
                  'params': {'alpha': np.random.uniform(1, 4),
                             'beta': np.random.uniform(1, 6)
                             }
                 }],
                [{'func': patches,
                  'name': 'patches',
                  'input': 'value',
                  'freq_range': None,
                  'time_range': None,
                  'params': {'N': np.random.uniform(5, 10),
                             'min_size': np.random.uniform(1, 4),
                             'max_size': np.random.uniform(15, 25),
                             'patch_size': np.random.uniform(1500, 2500)
                             }
                  }],
                [{'func': patches,
                  'name': 'patches',
                  'input': 'value',
                  'freq_range': None,
                  'time_range': None,
                  'params': {'N': np.random.uniform(1, 5),
                             'min_size': np.random.uniform(10, 18),
                             'max_size': np.random.uniform(20, 25),
                             'patch_size': np.random.uniform(1500, 2500)
                             }
                  }],
                [{'func': patches,
                  'name': 'patches',
                  'input': 'value',
                  'freq_range': None,
                  'time_range': None,
                  'params': {'N': np.random.uniform(75, 125),
                             'min_size': np.random.uniform(1, 3),
                             'max_size': np.random.uniform(5, 10),
                             'patch_size': np.random.uniform(3500, 4500)
                             }
                  }],
                [{'func': patches,
                  'name': 'patches',
                  'input': 'value',
                  'freq_range': None,
                  'time_range': None,
                  'params': {'N': np.random.uniform(1, 4),
                             'min_size': np.random.uniform(1, 3),
                             'max_size': np.random.uniform(4, 8),
                             'patch_size': np.random.uniform(7500, 8500)
                             }
                  }],
                [{'func': sinusoidal_sum,
                  'name': 'sinusoidal_sum',
                  'input': 'time',
                  'freq_range': None,
                  'time_range': None,
                  'params': {'n': np.random.uniform(2, 4),
                             'amp': np.random.normal(1),
                             'freq': np.random.normal(0.5),
                             'phase': np.random.normal(),
                             'add': False
                             }
                  }],
                [{'func': sinusoidal_sum,
                  'name': 'sinusoidal_sum',
                  'input': 'freq',
                  'freq_range': None,
                  'time_range': None,
                  'params': {'n': np.random.uniform(2, 4),
                             'amp': np.random.normal(1),
                             'freq': np.random.normal(0.5),
                             'phase': np.random.normal(),
                             'add': False
                             }
                  }],
                [{'func': sinusoidal_sum,
                  'name': 'sinusoidal_sum',
                  'input': 'freq',
                  'freq_range': [(750, 850), (150, 200)],
                  'time_range': None,
                  'params': {'n': np.random.uniform(2, 4),
                             'amp': np.random.normal(1),
                             'freq': np.random.normal(0.5),
                             'phase': np.random.normal(),
                             'add': False
                             }
                  }],
                [{'func': sinusoidal_sum,
                  'name': 'sinusoidal_sum',
                  'input': 'time',
                  'freq_range': None,
                  'time_range': [(np.random.uniform(0, 500),
                                  np.random.uniform(500, 600))],
                  'params': {'n': np.random.uniform(2, 4),
                             'amp': np.random.normal(1),
                             'freq': np.random.normal(0.5),
                             'phase': np.random.normal(),
                             'add': False
                             }
                  }],
                [{'func': changing_sinusoid,
                  'name': 'changing_sinusoid',
                  'input': 'time',
                  'freq_range': None,
                  'time_range': None,
                  'params': {'amp': np.random.normal(1),
                             'freq': np.random.normal(0.5),
                             'phase': np.random.normal(),
                             'add': False
                             }
                  }]
                 ]

double_mods = []
for m in single_mods:
    if not 'patches' in m[0]['name']:
        double_mods.append([m[0], {'func': patches,
                                   'name': 'patches',
                                   'input': 'value',
                                   'freq_range': None,
                                   'time_range': None,
                                   'params':{'N': np.random.uniform(1, 50),
                                             'min_size': np.random.uniform(1, 5),
                                             'max_size': np.random.uniform(6, 25),
                                             'patch_size': np.random.normal(2000, 10)
                                            }
                                   }])
    if not 'filter' in m[0]['name']:
        double_mods.append([m[0], {'func': butterworth_lowpass_filter,
                                   'name': 'butterworth_lowpass_filter',
                                   'input': 'value',
                                   'freq_range': None,
                                   'time_range': None,
                                   'params': {'cutoff': np.random.uniform(0.01, 0.1),
                                              'sr': 1.004 * u.ms,
                                              'N': np.random.uniform(2, 10)
                                            }
                                   }])
        double_mods.append([m[0], {'func': fourier_lowpass_filter,
                                   'name': 'fourier_lowpass_filter',
                                   'input': 'value',
                                   'freq_range': None,
                                   'time_range': None,
                                   'params': {'cutoff': None }
                                   }])
    if not 'one_over_f' in m[0]['name']:
        double_mods.append([m[0], {'func': one_over_f,
                                   'name': 'one_over_f',
                                   'input': 'freq',
                                   'freq_range': None,
                                   'time_range': None,
                                   'params': {'offset': 400,
                                              'coef': np.random.uniform(1, 4),
                                              'scale': np.random.uniform(1, 4)
                                              }
                                   }])

modifications = single_mods + double_mods
