'''@file cross_enthropytrainer.py
contains the CrossEnthropyTrainer'''

import tensorflow as tf
from nabu.neuralnetworks.trainers import trainer
from nabu.neuralnetworks import ops

class CrossEntropyTrainer(trainer.Trainer):
    '''A trainer that minimises the cross-entropy loss between given text logits
    and text targets. The output sequences must be of the same length as the
    input sequences'''

    def compute_loss(self, targets, logits, logit_seq_length,
                     target_seq_length):
        '''
        Compute the loss

        Creates the operation to compute the cross-entropy loss for every input
        frame (if you want to have a different loss function, overwrite this
        method)

        Args:
            targets: a tupple of targets, the first one being a
                [batch_size, max_target_length] tensor containing the real text
                targets, the second one being a [batch_size, max_length x dim]
                tensor containing the reconstruction features.
            logits: a tuple of [batch_size, max_logit_length, dim] tensors
                containing the logits for the text and the reconstruction
            logit_seq_length: the length of all the logit sequences as a tuple
                of [batch_size] vectors
            target_seq_length: the length of all the target sequences as a
                tuple of two [batch_size] vectors, both for one of the elements
                in the targets tuple

        Returns:
            a scalar value containing the total loss
        '''

        with tf.name_scope('cross_entropy_loss'):
            # extract the text logits out of the tuple
            logits = logits[0]
            logit_seq_length = logit_seq_length[0]
            targets = targets[0]
            target_seq_length = target_seq_length[0]

            loss = ops.cross_entropy_integers_logits_with_appending_eos(
                targets, logits, logit_seq_length, target_seq_length)

        return loss
