'''
#@package batchdispenser
# contain the functionality for read features and batches
# of features for neural network training and testing
'''

from abc import ABCMeta, abstractmethod, abstractproperty
import copy
import text_reader
import numpy as np

## Class that dispenses batches of data for mini-batch training
class BatchDispenser(object):
    ''' BatchDispenser interface cannot be created but gives methods to its
    child classes.'''
    __metaclass__ = ABCMeta

    def __init__(self, size):
        '''
        batchDispenser constructor

        Args:
            size: Specifies how many utterances should be contained
                  in each batch.
        '''

        #store the batch size
        self.size = size

    def get_batch(self, pos=None):
        '''
        Get a batch of features and targets.

        Args:
            pos: position in the reader, if None will remain unchanged

        Returns:
            A pair containing:
                - The features: a list of feature matrices
                - The targets: a list of target tupples(!), where the second
                    element of each tupple can be used for varying reasons
                        example: quantized audio samples
        '''

        #set up the data lists.
        batch_inputs = []
        batch_targets = []

        if pos is not None:
            self.pos = pos

        while len(batch_inputs) < self.size:
            #read the next pair
            inputs, targets = self.get_pair()

            if targets[0] is not None:
                batch_inputs.append(inputs)
                batch_targets.append(targets)

        return batch_inputs, batch_targets

    @abstractmethod
    def split(self, num_utt):
        '''take a number of utterances from the batchdispenser to make a new one

        Args:
            num_utt: the number of utterances in the new batchdispenser

        Returns:
            a batch dispenser with the requested number of utterances'''

    @abstractmethod
    def get_pair(self):
        '''get the next input-target pair'''

    @property
    def num_batches(self):
        '''
        The number of batches in the given data.

        The number of batches is not necessarily a whole number
        '''

        return float(self.num_utt)/self.size

    @abstractproperty
    def num_utt(self):
        '''The number of utterances in the given data'''

    @abstractproperty
    def num_labels(self):
        '''the number of output labels'''

    @abstractproperty
    def max_input_length(self):
        '''the maximal sequence length of the features'''

    @abstractproperty
    def max_target_length(self):
        '''the maximal length of the targets'''

    #pylint: disable=E0202
    @abstractproperty
    def pos(self):
        '''the current position in the data'''

    @pos.setter
    @abstractmethod
    def pos(self, pos):
        '''setter for the current position in the data'''

class AsrTextBatchDispenser(BatchDispenser):
    '''a batch dispenser, used for ASR training'''

    def __init__(self, feature_reader, target_coder, size, target_path):
        '''
        batchDispenser constructor

        Args:
            feature_reader: Kaldi ark-file feature reader instance.
            target_coder: a TargetCoder object to encode and decode the target
                sequences
            size: Specifies how many utterances should be contained
                  in each batch.
            target_path: path to the file containing the targets
        '''

        #store the feature reader
        self.feature_reader = feature_reader

        #save the target coder
        self.target_coder = target_coder

        #get a dictionary connecting training utterances and targets.
        self.target_dict = {}

        with open(target_path, 'r') as fid:
            for line in fid:
                splitline = line.strip().split(' ')
                self.target_dict[splitline[0]] = ' '.join(splitline[1:])

        super(AsrTextBatchDispenser, self).__init__(size)

    def split(self, num_utt):
        '''take a number of utterances from the batchdispenser to make a new one

        Args:
            num_utt: the number of utterances in the new batchdispenser

        Returns:
            a batch dispenser with the requested number of utterances'''

        #create a copy of self
        dispenser = copy.deepcopy(self)

        #split of a part of the feature reader
        dispenser.feature_reader = self.feature_reader.split(num_utt)

        #get a list of keys in the featutre readers
        dispenser_ids = dispenser.feature_reader.reader.scp_data.keys()
        self_ids = self.feature_reader.reader.scp_data.keys()

        #split the target dicts
        dispenser.target_dict = {key: dispenser.target_dict[key] for key in
                                 dispenser_ids}
        self.target_dict = {key: self.target_dict[key] for key in self_ids}

        return dispenser

    def get_pair(self):
        '''get the next input-target pair'''

        utt_id, inputs, _ = self.feature_reader.get_utt()

        if utt_id in self.target_dict and self.target_dict[utt_id] != '':
            texttargets = self.target_coder.encode(self.target_dict[utt_id])
        else:
            # when doing supervised training on a dataset where a lot of the
            # utterances have no labels, we cant do anything with these utterances
    #        print 'WARNING no targets for %s' % utt_id
            texttargets = None
            inputs = None

        return inputs, (texttargets, np.zeros([1,1]))

    @property
    def num_utt(self):
        '''The number of utterances in the given data
        (returns only the one where text targets are available)'''
        utterances_with_text_targets = [utt for utt in self.target_dict if self.target_dict[utt] != '']
        return len(utterances_with_text_targets)

    @property
    def num_labels(self):
        '''the number of output labels'''

        return self.target_coder.num_labels

    @property
    def max_input_length(self):
        '''the maximal sequence length of the features'''

        return self.feature_reader.max_length

    @property
    def max_target_length(self):
        '''the maximal length of the targets'''
        part1 = max([len(targets.split(' '))
                    for targets in self.target_dict.values()])
        return(part1,1)


    @property
    def pos(self):
        '''the current position in the data'''

        return self.feature_reader.pos

    @pos.setter
    def pos(self, pos):
        '''setter for the current position in the data'''

        self.feature_reader.pos = pos

class LmBatchDispenser(BatchDispenser):
    '''a batch dispenser, used for language model training'''

    def __init__(self, target_coder, size, textfile, max_length,
                 num_utt):
        '''
        BatchDispenser constructor

        Args:
            size: Specifies how many utterances should be contained
                  in each batch.
            textfile: path to the file containing the text
            max_length: the maximum length of the text sequences
            num_utt: number of lines in the textfile
        '''

        #create a text reader object
        self.textreader = text_reader.TextReader(textfile, max_length,
                                                 target_coder)

        #the total number of utterances
        self._num_utt = num_utt

        super(LmBatchDispenser, self).__init__(size)

    def split(self, num_utt):
        '''take a number of utterances from the batchdispenser to make a new one

        Args:
            num_utt: the number of utterances in the new batchdispenser

        Returns:
            a batch dispenser with the requested number of utterances'''

        #create a copy of self
        dispenser = copy.deepcopy(self)

        #split the textreader
        dispenser.textreader = self.textreader.split(num_utt)
        dispenser.unm_utt = num_utt
        self._num_utt -= num_utt

        return dispenser

    def get_pair(self):
        '''get the next input-target pair'''

        _, targets, _ = self.textreader.get_utt()

        return targets, (targets[:, 0],np.zeros(1))

    @property
    def num_labels(self):
        '''the number of output labels'''

        return self.textreader.coder.num_labels

    @property
    def max_input_length(self):
        '''the maximal sequence length of the features'''

        return self.textreader.max_length

    @property
    def max_target_length(self):
        '''the maximal length of the targets'''

        return (self.textreader.max_length,1)

    @property
    def num_utt(self):
        '''The number of utterances in the given data'''

        return self._num_utt

    @property
    def pos(self):
        '''the current position in the data'''

        return self.textreader.pos

    @pos.setter
    def pos(self, pos):
        '''setter for the current position in the data'''

        self.textreader.pos = pos

class AsrTextAndAudioBatchDispenser(BatchDispenser):
    '''a batch dispenser, used for ASR training, when working with
    (partly) non-supervised data and audio samples'''

    def __init__(self, feature_reader, audio_reader, target_coder, size, target_path):
        '''
        batchDispenser constructor

        Args:
            feature_reader: Kaldi ark-file feature reader instance.
            audio_reader: Kaldi ark-file feature reader for the audio samples
            target_coder: a TargetCoder object to encode and decode the target
                sequences
            size: Specifies how many utterances should be contained
                  in each batch.
            target_path: path to the file containing the targets
        '''

        #store the feature reader
        self.feature_reader = feature_reader

        #store the audio reader
        self.audio_reader = audio_reader

        #save the target coder
        self.target_coder = target_coder

        #get a dictionary connecting training utterances and targets.
        self.target_dict = {}

        with open(target_path, 'r') as fid:
            for line in fid:
                splitline = line.strip().split(' ')
                self.target_dict[splitline[0]] = ' '.join(splitline[1:])

        super(AsrTextAndAudioBatchDispenser, self).__init__(size)

    def split(self, num_utt):
        '''take a number of utterances from the batchdispenser to make a new one

        Args:
            num_utt: the number of utterances in the new batchdispenser

        Returns:
            a batch dispenser with the requested number of utterances'''

        #create a copy of self
        dispenser = copy.deepcopy(self)

        #split of a part of the feature reader
        dispenser.feature_reader = self.feature_reader.split(num_utt)

        #get a list of keys in the featutre readers
        dispenser_ids = dispenser.feature_reader.reader.scp_data.keys()
        self_ids = self.feature_reader.reader.scp_data.keys()

        #split the target dicts
        dispenser.target_dict = {key: dispenser.target_dict[key] for key in
                                 dispenser_ids}
        self.target_dict = {key: self.target_dict[key] for key in self_ids}

        return dispenser

    def get_pair(self):
        '''get the next input-target pair'''

        utt_id, inputs, _ = self.feature_reader.get_utt()

        if utt_id in self.target_dict:
            # normally the unlabeled data is simply encoded with '' as targets
            text_targets = self.target_coder.encode(self.target_dict[utt_id])
        else:
            #When something went wrong, we can still simply take empty string as target
            text_targets = self.target_coder.encode('')

        audio_samples = self.audio_reader.get_utt_with_id(utt_id)

        # the targets should now be a pair of the real targets and the audio samples
        targets = (text_targets, audio_samples)

        return inputs, targets

    @property
    def num_utt(self):
        '''The number of utterances in the given data'''

        return len(self.target_dict)

    @property
    def num_labels(self):
        '''the number of output labels'''

        return self.target_coder.num_labels

    @property
    def max_input_length(self):
        '''the maximal sequence length of the features'''

        return self.feature_reader.max_length

    @property
    def max_target_length(self):
        '''the maximal length of the targets'''
        part1 = max([len(targets.split(' '))
                    for targets in self.target_dict.values()])
        part2 = self.audio_reader.max_length
        return(part1, part2)


    @property
    def pos(self):
        '''the current position in the data'''

        return self.feature_reader.pos

    @pos.setter
    def pos(self, pos):
        '''setter for the current position in the data'''

        self.feature_reader.pos = pos

class AsrTextAndFeatureBatchDispenser(BatchDispenser):
    '''a batch dispenser, used for ASR training, when working with
    (partly) non-supervised data'''

    def __init__(self, feature_reader, target_coder, size, target_path):
        '''
        batchDispenser constructor

        Args:
            feature_reader: Kaldi ark-file feature reader instance.
            target_coder: a TargetCoder object to encode and decode the target
                sequences
            size: Specifies how many utterances should be contained
                  in each batch.
            target_path: path to the file containing the targets
        '''

        #store the feature reader
        self.feature_reader = feature_reader

        #save the target coder
        self.target_coder = target_coder

        #get a dictionary connecting training utterances and targets.
        self.target_dict = {}

        with open(target_path, 'r') as fid:
            for line in fid:
                splitline = line.strip().split(' ')
                self.target_dict[splitline[0]] = ' '.join(splitline[1:])

        super(AsrTextAndFeatureBatchDispenser, self).__init__(size)

    def split(self, num_utt):
        '''take a number of utterances from the batchdispenser to make a new one

        Args:
            num_utt: the number of utterances in the new batchdispenser

        Returns:
            a batch dispenser with the requested number of utterances'''

        #create a copy of self
        dispenser = copy.deepcopy(self)

        #split of a part of the feature reader
        dispenser.feature_reader = self.feature_reader.split(num_utt)

        #get a list of keys in the featutre readers
        dispenser_ids = dispenser.feature_reader.reader.scp_data.keys()
        self_ids = self.feature_reader.reader.scp_data.keys()

        #split the target dicts
        dispenser.target_dict = {key: dispenser.target_dict[key] for key in
                                 dispenser_ids}
        self.target_dict = {key: self.target_dict[key] for key in self_ids}

        return dispenser

    def get_pair(self):
        '''get the next input-target pair'''

        utt_id, inputs, _ = self.feature_reader.get_utt()

        if utt_id in self.target_dict:
            # normally the unlabeled data is simply encoded with '' as targets
            text_targets = self.target_coder.encode(self.target_dict[utt_id])
        else:
            #When something went wrong, we can still simply take empty string as target
            text_targets = self.target_coder.encode('')

        # the targets should now be a pair of the real targets and the feature inputs
        targets = (text_targets, inputs)

        return inputs, targets

    @property
    def num_utt(self):
        '''The number of utterances in the given data'''

        return len(self.target_dict)

    @property
    def num_labels(self):
        '''the number of output labels'''

        return self.target_coder.num_labels

    @property
    def max_input_length(self):
        '''the maximal sequence length of the features'''

        return self.feature_reader.max_length

    @property
    def max_target_length(self):
        '''the maximal length of the targets'''
        part1 = max([len(targets.split(' '))
                    for targets in self.target_dict.values()])
        part2 = self.feature_reader.max_length
        return(part1, part2)


    @property
    def pos(self):
        '''the current position in the data'''

        return self.feature_reader.pos

    @pos.setter
    def pos(self, pos):
        '''setter for the current position in the data'''

        self.feature_reader.pos = pos
