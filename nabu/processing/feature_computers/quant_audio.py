'''@file quant_audio.py
contains the quantizer of the audio samples'''

import numpy as np
import feature_computer

class Quant_Audio(feature_computer.FeatureComputer):
    ''' a pseudo-feature computer that actually is used to store the audio
    sampleso of a signal after the quantization with mu-law
    '''

    def comp_feat(self, sig, rate):
        '''
        compute the features. Here we will simply return the
        speech signal after having quantized with the mu-law

        Args:
            sig: the audio signal as a 1-D numpy array with seq_length elements
            rate: the sampling rate

        Returns:
            the samples as a [seq_length x 1] numpy array
                (2D to be conform with other feature computers)
        '''

        # define the parameters
        num_levels = int(self.conf['quant_levels'])
        mu = num_levels-1
        # convert the speech signal to a float representation
        inputs = sig.astype(np.float32)
        #inputs should first be translated to -1,1 range
        ####
        # UPDATE NEEDED: it might be better to divide here by a constant, such that all of the utterances are handled equally?
        #####
        inputs = inputs / np.max(abs(inputs))
        # mu-law transformation
        transformed = np.sign(inputs)*np.log(1+mu*np.abs(inputs))/np.log(1+mu)
        # quantization to an integer
        quantized = ((transformed+1)*num_levels/2+0.5).astype(np.int32)
        # because the output is expected to be in a twodimensional matrix
        quantized = quantized.reshape(quantized.shape[0],1)

        return quantized

    def get_dim(self):
        '''the feature dimemsion'''

        dim = int(self.conf['quant_levels'])

        return dim
