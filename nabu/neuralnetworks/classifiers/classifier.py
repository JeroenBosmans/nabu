'''@file classifier.py
The abstract class for a neural network classifier'''

from abc import ABCMeta, abstractmethod
import tensorflow as tf

class Classifier(object):
    '''This an abstract class defining a neural net classifier'''
    __metaclass__ = ABCMeta

    def __init__(self, conf, output_dim, name=None):
        '''classifier constructor

        Args:
            conf: The classifier configuration
            output_dim: the classifier output dimension
                    This is a tuple, each element representing the output_dim
                    for one kind of targets
            name: the classifier name
        '''

        self.conf = conf

        # if there is only a add_labels in the config, we suppose that only the
        # first element of this tuple is important
        if 'add_labels' in conf:
            self.output_dim = output_dim[0] + int(conf['add_labels'])

        # if there is only an add_labels_reconstruction but not an
        # add_labels_prediction in config, assume only second element to be of
        # importance
        elif 'add_labels_reconstruction' in conf and \
            not 'add_labels_prediction' in conf:
            self.output_dim = output_dim[1] + int(
                conf['add_labels_reconstruction'])

        # if both present, both elements of the tuple will be of importance
        elif 'add_labels_reconstruction' in conf and \
            'add_labels_prediction' in conf:
            outdim1 = output_dim[0] + int(conf['add_labels_prediction'])
            outdim2 = output_dim[1] + int(conf['add_labels_reconstruction'])
            self.output_dim = (outdim1, outdim2)

        else:
            raise Exception(
                'Wrong kind of add_labels information in the config')

        #create the variable scope for the classifier
        self.scope = tf.VariableScope(False, name or type(self).__name__)

    def __call__(self, inputs, input_seq_length, targets,
                 target_seq_length, is_training):
        '''
        Add the neural net variables and operations to the graph

        Args:
            inputs: the inputs to the neural network, this is a
                [batch_size x max_input_length x feature_dim] tensor
            input_seq_length: The sequence lengths of the input utterances, this
                is a [batch_size] vector
            targets: a tupple of targets, the first one being a
                [batch_size, max_target_length] tensor containing the real
                targets, the second one being [batch_size, max_audioseq_length]
                tensor containing the audio samples or other extra information.
            target_seq_length: the length of all the target sequences as a
                tupple of two [batch_size] vectors, both for one of the elements
                in the targets tupple
            is_training: whether or not the network is in training mode

        Returns:
            A pair containing:
                - output logits as a tuple of tensors
                - the output logits sequence lengths as a tuple of vectors
        '''

        with tf.variable_scope(self.scope):
            outputs, output_seq_lengths = self._get_outputs(
                inputs, input_seq_length, targets, target_seq_length,
                is_training)

        #put the reuse flag to true in the scope to make sure the variables are
        #reused in the next call
        self.scope.reuse_variables()

        return outputs, output_seq_lengths

    @abstractmethod
    def _get_outputs(self, inputs, input_seq_length, targets,
                     target_seq_length, is_training):

        '''
        Add the neural net variables and operations to the graph

        Args:
            inputs: the inputs to the neural network, this is a
                [batch_size x max_input_length x feature_dim] tensor
            input_seq_length: The sequence lengths of the input utterances, this
                is a [batch_size] vector
            targets: the targets to the neural network, this is a
                [batch_size x max_output_length] tensor. The targets can be
                used during training
            target_seq_length: The sequence lengths of the target utterances,
                this is a [batch_size] vector
            is_training: whether or not the network is in training mode

        Returns:
            A pair containing:
                - output logits in a tuple (one of the elements can be None)
                - the output logits sequence lengths as a tuple of vectors
        '''

        raise NotImplementedError("Abstract method")
