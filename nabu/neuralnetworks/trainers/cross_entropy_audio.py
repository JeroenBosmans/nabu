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

            output_dim = int(logits.get_shape()[2])

            # we know the targets are integers when working with audio samples
            targets_int = tf.cast(targets[1], tf.int32)

            #put all the targets on top of each other
            split_targets = tf.unstack(targets_int)
            for i, target in enumerate(split_targets):
                #only use the real data
                split_targets[i] = target[:target_seq_length[1][i]]

            #concatenate the targets
            nonseq_targets = tf.concat(split_targets,0)

            #convert the logits to non sequential data
            nonseq_logits = ops.seq2nonseq(logits, logit_seq_length)

            #one hot encode the targets
            #pylint: disable=E1101
            nonseq_targets = tf.one_hot(nonseq_targets, output_dim)

            #compute the cross-enthropy loss
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=nonseq_logits, labels=nonseq_targets))

        return loss
