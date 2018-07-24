import numpy as np
import astropy.units as u
from baseband import vdif
from baseband.helpers import sequentialfile as sf

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

        # How many samples will be left after changing the sample rate
        n_samples_out = int(((n_samples_in/rate_in) * rate_out).value)

        # Truncate the # of input samples so that the number of input samples
        # can be divded evenly by the number of output samples
        n_samples_in_trunc = (n_samples_in // n_samples_out) * n_samples_out

        # How many samples are in each chunk
        chunk_width = n_samples_in_trunc // n_samples_out

        # The true rate out, the desired rate will not be acieved perfectly
        rate_out = (n_samples_out * rate_in) / n_samples_in

        # Read chunks in groups of chunks no larger than 2**15 samples, to
        # reduce the number of reads necessary to read all the data
        max_samples_per_read = 2**15
        chunks_per_read = max_samples_per_read // chunk_width
        n_reads = n_samples_out // chunks_per_read

        # The true number of samples after changinf sample rate, will bot be
        # perfectly n_samples_out because everything needs to divide nicely
        true_samples_out = n_reads * chunks_per_read

        # Read the file in chunks, reducing each chunk after reading
        complete_data = np.zeros((nfreq, true_samples_out))
        for i in range(n_reads):
            # Ensure enough data is left to do a complete read
            data = fh.read(chunk_width * chunks_per_read)

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
