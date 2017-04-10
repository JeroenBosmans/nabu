'''@file cross_enthropytrainer.py
contains the CrossEnthropyTrainer'''

import tensorflow as tf
import trainer
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
                [batch_size, max_target_length x dim] tensor containing the real
                targets, the second one being a [batch_size, max_audioseq_length x dim]
                tensor containing a form of reconstruction features.
            logits: a tuple of [batch_size, max_logit_length, dim] tensors,
                representing the text logits and the reconstruction logits
            logit_seq_length: the length of all the logit sequences as a tuple
                [batch_size] vectors, first element corresponding to the text
                logits and the second to the reconstruction logits
            target_seq_length: the length of all the target sequences as a
                tuple of two [batch_size] vectors, both for one of the elements
                in the targets tuple

        Returns:
            a scalar value containing the loss
        '''

        with tf.name_scope('cross_entropy_loss'):
            # extract the text logits out of the tuple
            logits = logits[0]
            logit_seq_length = logit_seq_length[0]

            output_dim = int(logits.get_shape()[2])

            #for the logits where there were no acutal targets, the targets in
            #here will simply be a eos target. So we could replace the corresponding
            # logits with just and eos logit.
            eos_logits = tf.expand_dims(tf.constant([0.0]*(output_dim-1) +
                [1]), 0)
            eos_logits = tf.pad(eos_logits,
                             [[0, int(logits.get_shape()[1])-1], [0, 0]])
            eos_logits = tf.stack([eos_logits]*int(logits.get_shape()[0]), 0)

            empty_targets = tf.equal(target_seq_length[0], 0)
            logits = tf.where(empty_targets, eos_logits, logits)

            #put all the targets on top of each other
            split_targets = tf.unstack(targets[0])

            for i, target in enumerate(split_targets):
                #only use the real data
                split_targets[i] = target[:target_seq_length[0][i]]

                #append an end of sequence label
                split_targets[i] = tf.concat(
                    [split_targets[i], [output_dim-1]], 0)

            #concatenate the targets
            nonseq_targets = tf.concat(split_targets, 0)

            #convert the logits to non sequential data
            nonseq_logits = ops.seq2nonseq(logits, logit_seq_length)

            #one hot encode the targets
            #pylint: disable=E1101
            nonseq_targets = tf.one_hot(nonseq_targets, output_dim)

            #compute the cross-enthropy loss
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=nonseq_logits, labels=nonseq_targets))

        return loss
