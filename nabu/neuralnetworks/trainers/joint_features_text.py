'''@file cross_enthropytrainer_rec.py
contains the CrossEnthropyTrainerRec for reconstruction of the audio samples'''

import tensorflow as tf
import trainer
from nabu.neuralnetworks import ops

class JointFeaturesTextCost(trainer.Trainer):
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

            ## first process text logits and targets

            # extract the text logits out of the tuple
            text_logits = logits[0]
            text_logit_seq_length = logit_seq_length[0]

            output_dim = int(text_logits.get_shape()[2])

            #for the logits where there were no acutal targets, the targets in
            #here will simply be a eos target. So we could replace the corresponding
            # logits with just and eos logit.
            eos_logits = tf.expand_dims(tf.constant([0.0]*(output_dim-1) +
                [1]), 0)
            eos_logits = tf.pad(eos_logits,
                             [[0, int(text_logits.get_shape()[1])-1], [0, 0]])
            eos_logits = tf.stack([eos_logits]*int(text_logits.get_shape()[0]), 0)

            empty_targets = tf.equal(target_seq_length[0], 0)
            text_logits = tf.where(empty_targets, eos_logits, text_logits)

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
            nonseq_logits = ops.seq2nonseq(text_logits, text_logit_seq_length)

            #one hot encode the targets
            #pylint: disable=E1101
            nonseq_targets = tf.one_hot(nonseq_targets, output_dim)

            #compute the cross-enthropy loss
            loss_text = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=nonseq_logits, labels=nonseq_targets))

            ## next process reconstruction targets and logits

            #compute the mean squared variance of the reconstruction
            dim = targets[1].get_shape()[2]

            #compute the mean squared variance of the reconstruction
            errors = targets[1] - logits[1]
            errors_squared = errors**2

            errors_list = tf.unstack(errors_squared)
            loss_features = tf.zeros([])
            for i, error in enumerate(errors_list):
                error = error[:target_seq_length[1][i],:]
                error = tf.reduce_sum(error)/tf.cast((target_seq_length[1][i]*dim), dtype=tf.float32)
                loss_features = loss_features + error

            ## finally combine the two loss functions

            # get the tradeoff parameters
            tradeoff = float(self.conf['loss_trade_off'])
            if tradeoff<0 or tradeoff>1:
                raise Exception('Trade off parameter for the loss function should be between 0 and 1')

            # make a combination of the two loss functions
            loss = tradeoff*loss_text + (1-tradeoff)*loss_features


        return loss
