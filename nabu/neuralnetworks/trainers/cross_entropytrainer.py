'''@file cross_enthropytrainer.py
contains the CrossEnthropyTrainer'''

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
                [batch_size, max_target_length] tensor containing the real
                targets, the second one being a [batch_size, max_audioseq_length]
                tensor containing the audio samples or other extra information.
            logits: a [batch_size, max_logit_length, dim] tensor containing the
                logits
            logit_seq_length: the length of all the logit sequences as a
                [batch_size] vector
            target_seq_length: the length of all the target sequences as a
                tupple of two [batch_size] vectors, both for one of the elements
                in the targets tupple

        Returns:
            a scalar value containing the loss
        '''

        with tf.name_scope('cross_entropy_loss'):
            output_dim = int(logits.get_shape()[2])

            #put all the targets on top of each other
            split_targets = tf.unpack(targets[0])
            for i, target in enumerate(split_targets):
                #only use the real data
                split_targets[i] = target[:target_seq_length[0][i]]

                #append an end of sequence label
                split_targets[i] = tf.concat(
                    0, [split_targets[i], [output_dim-1]])

            #concatenate the targets
            nonseq_targets = tf.concat(0, split_targets)

            #convert the logits to non sequential data
            nonseq_logits = ops.seq2nonseq(logits, logit_seq_length)

            #one hot encode the targets
            #pylint: disable=E1101
            nonseq_targets = tf.one_hot(nonseq_targets, output_dim)

            #compute the cross-enthropy loss
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                nonseq_logits, nonseq_targets))

        return loss
