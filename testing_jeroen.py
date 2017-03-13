'''
small test file
'''

import nabu.processing.prepare_data as prepare_data
from collections import OrderedDict
import scipy.io.wavfile as writewav
from scipy.signal import resample
import numpy as np

def read_wavfiles(filename):
    '''
     read the wav.scp file used in kaldi

     Args:
        filename: path to the wav scp file

    Returns:
        a dictionary containing:
            - key: the utterance ID
            - value: a pair containing
                - the filenames
                - bool wheter or not the filename is extended (with a
                    read command)
    '''

    with open(filename) as fid:
        files = OrderedDict()
        counter = 0
        for line in fid:
            data = line.strip().split(' ')
            #wav.scp contains filenames
            if len(data) == 2:
                #utterance:(filename, not extended)
                files[data[0]] = (data[1], False)
            #wav.scp contains extended filenames
            else:
                #utterance: (extended filename, extended)
                files[data[0]] = (line[len(data[0])+1:len(line)-1], True)
            counter = counter+1
            if counter>10:
                break
    return files


#read the wavfiles
wavfiles = read_wavfiles('/esat/spchdisk/scratch/vrenkens/data/timit/train' + '/wav.scp')

#read all the wav files
rate_utt = {utt: prepare_data.read_wav(wavfiles[utt]) for utt in wavfiles}
directory = '/esat/spchtemp/scratch/jbosmans/audiofile'
for key, value in rate_utt.iteritems():
    rate, audio = value
    filenamex = directory + '/' + key + '_original.wav'
    writewav.write(filenamex, rate, audio)

    seconds = float(audio.shape[0])/float(rate)
    new_rate = 1000
    new_nbr_elements = int(new_rate*seconds)
    new_audio = resample(audio, new_nbr_elements)
    new_audio = new_audio.astype(np.int16)

    filenamex = directory + '/' + key + '_resampled.wav'
    writewav.write(filenamex, new_rate, new_audio)
