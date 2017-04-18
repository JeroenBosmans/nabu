'''@file cross_enthropytrainer_rec.py
contains the CrossEnthropyTrainerRec for reconstruction of the audio samples'''

import tensorflow as tf
import trainer
from nabu.neuralnetworks import ops

class CrossEntropyTrainer(trainer.Trainer):
    '''A trainer that minimises the cross-enthropy loss, the output sequences
    must be of the same length as the input sequences'''

    def compute_loss(self, targets, logits, logit_seq_length,
                     target_seq_length):
        '''
        Compute the loss

        Creates the operation to compute the cross-entropy loss for every input
        frame (if you want to have a different loss function, overwrite this
        method)

        Args:
            targets: a tupple of targets, the first one being a
                [batch_size x max_target_length] tensor containing the real
                targets, the second one being a [batch_size x max_audioseq_length x 1]
                tensor containing the audio samples or other extra information.
            logits: a tuple of [batch_size x max_logit_length x dim] tensors,
                where in this case the second element will contain the actual information
            logit_seq_length: the length of all the logit sequences as a tuple
                [batch_size] vectors, where in this case the second element will
                contain the actual information
            target_seq_length: the length of all the target sequences as a
                tupple of two [batch_size] vectors, both for one of the elements
                in the targets tupple

        Returns:
            a scalar value containing the loss
        '''

        with tf.name_scope('cross_entropy_loss'):
            #extract the logits and the lengths out of the tuple
            logits = logits[1]
            logit_seq_length = logit_seq_length[1]
            targets = targets[1]
            target_seq_length = target_seq_length[1]

            targets_int = tf.cast(targets, tf.int32)

            loss = ops.cross_entropy_integers_logits(targets_int, logits,
                                            logit_seq_length, target_seq_length)

        return loss
