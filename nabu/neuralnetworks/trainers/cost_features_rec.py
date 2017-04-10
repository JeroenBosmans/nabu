'''@file cross_enthropytrainer_rec.py
contains the CrossEnthropyTrainerRec for reconstruction of the audio samples'''

import tensorflow as tf
import trainer

class CostFeaturesRec(trainer.Trainer):
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
                [batch_size, max_target_length] tensor containing the real
                targets, the second one being a [batch_size, max_audioseq_length]
                tensor containing the audio samples or other extra information.
            logits: a tuple of  [batch_size, max_logit_length, dim] tensors
                containing the logits for the text and the audio samples
            logit_seq_length: the length of all the logit sequences as a tuple of
                [batch_size] vectors
            target_seq_length: the length of all the target sequences as a
                tupple of two [batch_size] vectors, both for one of the elements
                in the targets tupple

        Returns:
            a scalar value containing the loss
        '''

        with tf.name_scope('cross_entropy_loss'):

            #compute the mean squared variance of the reconstruction
            loss = tf.nn.l2_loss(targets[1] - logits[1])

        return loss
