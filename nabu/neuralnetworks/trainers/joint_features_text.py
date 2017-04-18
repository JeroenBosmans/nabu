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
            text_targets = targets[0]
            text_target_seq_length = target_seq_length[0]

            loss_text = ops.cross_entropy_integers_logits_with_appending_eos(
                                text_targets, text_logits,
                                text_logit_seq_length,text_target_seq_length)

            ## next process reconstruction targets and logits

            #compute the mean squared variance of the reconstruction
            rec_targets = targets[1]
            rec_logits = logits[1]
            rec_target_length = target_seq_length[1]

            #compute the mean squared variance of the reconstruction
            loss_features = ops.mse(rec_targets, rec_logits, rec_target_length)

            ## finally combine the two loss functions

            # get the tradeoff parameters
            tradeoff = float(self.conf['loss_trade_off'])
            if tradeoff<0 or tradeoff>1:
                raise Exception('Trade off parameter for the loss function should be between 0 and 1')

            # make a combination of the two loss functions
            loss = tradeoff*loss_text + (1-tradeoff)*loss_features


        return loss
