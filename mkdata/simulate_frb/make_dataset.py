import os
import signal
import pickle
from mpi4py import MPI
from FRBEvent import FRBEvent
import numpy as np

interrupted = False
def signal_handler(signum, frame):
    global interrupted
    interrupted = True

def find_file(path, suffix):
    letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRTSUVWXYZ'
    count = 0
    while os.path.exists(path):
        if count == 0:
            b = '{}.{}'.format(letters[count], suffix)
            path = path.replace(suffix, b)
        elif count > 0 and count < len(letters):
            a = '{}.{}'.format(letters[count - 1], suffix)
            b = '{}.{}'.format(letters[count], suffix)
            path = path.replace(a, b)
        else:
            a = '{}.{}'.format(letters[count - 1], suffix)
            b = '{}.{}'.format(np.random.rand(), suffix)
            path = path.replace(a, b)
        count += 1
    return path

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

in_dir = '/scratch/r/rhlozek/rylan/aro_rfi/'
output = '/scratch/r/rhlozek/rylan/simulations/'

group = 10

data = sorted([in_dir + x for x in os.listdir(in_dir)])

signal.signal(signal.SIGINT, signal_handler)

metadata = []
for i in range(len(data) // group):
    # Ensure that the metadata are saved if executions ends prematurely
    if interrupted:
        break
    curr_data = data[i * group:(i + 1) * group]
    frb = FRBEvent(background=curr_data)
    path = '{}/{}/{}-V{}.npy'.format(output, rank, frb.input, rank)
    path = find_file(path, 'npy')
    frb.save(path)
    metadata.append(frb.get_params())

# Save the metadata for the simulations
out = [frb.get_headers()]
out.extend(metadata)
path = find_file('{}/{}/metadata{}.csv'.format(output, rank, rank), 'csv')
with open(path, 'wb') as f:
    pickle.dump(out, f)
