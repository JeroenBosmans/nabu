'''@file test_reconstruction.py
this file will test the reconstruction on its own by computing loss on test set'''

import os
from six.moves import configparser
import tensorflow as tf
import numpy as np
from nabu.neuralnetworks.classifiers.asr import asr_factory
from nabu.neuralnetworks import ops
from nabu.processing import feature_reader


tf.app.flags.DEFINE_string('expdir', 'expdir', 'The experiments directory')
FLAGS = tf.app.flags.FLAGS

def main(_):
    '''does everything for testing of pure reconstruction with the simple loss function'''

    #read the database config file
    parsed_database_cfg = configparser.ConfigParser()
    parsed_database_cfg.read(os.path.join(FLAGS.expdir, 'database.cfg'))
    database_cfg = dict(parsed_database_cfg.items('database'))

    #read the features config file
    parsed_feat_cfg = configparser.ConfigParser()
    parsed_feat_cfg.read(os.path.join(FLAGS.expdir, 'model', 'features.cfg'))
    feat_cfg = dict(parsed_feat_cfg.items('features'))

    #read the asr config file
    parsed_nnet_cfg = configparser.ConfigParser()
    parsed_nnet_cfg.read(os.path.join(FLAGS.expdir, 'model', 'asr.cfg'))
    nnet_cfg = dict(parsed_nnet_cfg.items('asr'))

    # read the trainer config file
    parsed_trainer_cfg = configparser.ConfigParser()
    parsed_trainer_cfg.read(os.path.join(FLAGS.expdir, 'trainer.cfg'))
    trainer_cfg = dict(parsed_trainer_cfg.items('trainer'))

    if 'reconstruction_features' in trainer_cfg:
        pass
    else:
        raise Exception('There are no reconstruction features specified, something is wrong')

    if trainer_cfg['reconstruction_features'] == 'audio_samples':
        audio_used = True
    else:
        audio_used = False

    #read the quantization config file
    if audio_used:
        parsed_quant_cfg = configparser.ConfigParser()
        parsed_quant_cfg.read(os.path.join(FLAGS.expdir, 'model', 'quantization.cfg'))
        quant_cfg = dict(parsed_quant_cfg.items('features'))

    #create a feature reader
    featdir = os.path.join(database_cfg['test_dir'], feat_cfg['name'])

    with open(os.path.join(featdir, 'maxlength'), 'r') as fid:
        max_length_feat = int(fid.read())

    feat_reader = feature_reader.FeatureReader(
        scpfile=os.path.join(featdir, 'feats.scp'),
        cmvnfile=os.path.join(featdir, 'cmvn.scp'),
        utt2spkfile=os.path.join(featdir, 'utt2spk'),
        max_length=max_length_feat)

    #create an audio sample reader
    if audio_used:
        audiodir = os.path.join(database_cfg['test_dir'], quant_cfg['name'])

        with open(os.path.join(audiodir, 'maxlength'), 'r') as fid:
            max_length_audio = int(fid.read())

        audio_reader = feature_reader.FeatureReader(
            scpfile=os.path.join(audiodir, 'feats.scp'),
            cmvnfile=None,
            utt2spkfile=None,
            max_length=max_length_audio)

    #check number of test examples
    number_examples = feat_reader.num_utt

    # set a batch_size to determine how many test examples are processed in each steps
    # this doesn't really matter, only for memory issues
    # take the same one as used in training
    batch_size = int(trainer_cfg['batch_size'])

    #create a ndarray of all of the features
    _,features,_ = feat_reader.get_utt()
    features = features.reshape(1,-1,features.shape[1])
    features_lengths = features.shape[1]*np.ones([1], dtype=np.int32)
    features = np.concatenate([features, np.zeros([features.shape[0],max_length_feat-features.shape[1],features.shape[2]])],1)
    looped = False
    while not looped:
        _,temp,looped = feat_reader.get_utt()
        temp = temp.reshape(1,-1,temp.shape[1])
        features_lengths = np.concatenate([features_lengths,temp.shape[1]*np.ones([1], dtype=np.int32)],0)
        temp = np.concatenate([temp, np.zeros([temp.shape[0],max_length_feat-temp.shape[1],temp.shape[2]])],1)
        features = np.concatenate([features, temp],0)

    #create a ndarray of all of the targets
    if audio_used:
        _,audio,_ = audio_reader.get_utt()
        audio = audio.reshape(1,-1,audio.shape[1])
        audio_lengths = audio.shape[1]*np.ones([1], dtype=np.int32)
        audio = np.concatenate([audio, np.zeros([audio.shape[0],max_length_audio-audio.shape[1],audio.shape[2]])],1)
        looped = False
        while not looped:
            _,temp,looped = audio_reader.get_utt()
            temp = temp.reshape(1,-1,temp.shape[1])
            audio_lengths = np.concatenate([audio_lengths,temp.shape[1]*np.ones([1], dtype=np.int32)],0)
            temp = np.concatenate([temp, np.zeros([temp.shape[0],max_length_audio-temp.shape[1],temp.shape[2]])],1)
            audio = np.concatenate([audio, temp],0)

    # store dimensions
        max_audio_length = audio.shape[1]

    else:
        audio = np.zeros([number_examples,1,1])
        max_audio_length = 1
        audio_lengths = np.ones([number_examples])

    # store dimensions
    max_feature_length = features.shape[1]
    feature_dim = features.shape[2]

    #create a graph
    graph = tf.Graph()

    with graph.as_default():
        #create the classifier
        if audio_used:
            outputdim = int(quant_cfg['quant_levels'])
        else:
            outputdim = feature_dim
        classifier = asr_factory.factory(
            conf=nnet_cfg,
            output_dim=(1,outputdim))

        # create placeholders for reconstruction and features
        features_ph = tf.placeholder(
            tf.float32,
            shape = [batch_size, max_feature_length, feature_dim],
            name = 'features')

        audio_ph = tf.placeholder(
            tf.int32,
            shape = [batch_size, max_audio_length,1],
            name = 'audio')

        audio_lengths_ph = tf.placeholder(
            tf.int32, shape=[batch_size], name='audio_lenght')

        feature_lengths_ph = tf.placeholder(
            tf.int32, shape=[batch_size], name='feat_lenght')

        # decide what to give as targets
        if audio_used:
            rec_ph = audio_ph
            rec_l_ph = audio_lengths_ph
        else:
            rec_ph = features_ph
            rec_l_ph = audio_lengths_ph

        #create the logits for reconstructed audio samples
        logits, logits_lengths = classifier(
            inputs=features_ph,
            input_seq_length=feature_lengths_ph,
            targets = (None, rec_ph),
            target_seq_length= (None, rec_l_ph),
            is_training=False)

        #compute the loss score
        if audio_used:
            score = compute_loss_audio((None, rec_ph), logits, logits_lengths, (None, rec_l_ph))
        else:
            score = compute_loss_features((None, rec_ph), logits, logits_lengths, (None, rec_l_ph))

        saver = tf.train.Saver(tf.trainable_variables())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    config.allow_soft_placement = True

    with tf.Session(graph=graph, config=config) as sess:

        #create a saver and load the model
        saver.restore(sess, os.path.join(FLAGS.expdir, 'model', 'network.ckpt'))

        all_processed = False
        number_batch = 0
        total_elements=0
        avrg_loss = 0.0

        total_steps = int(np.ceil(number_examples/batch_size))

        # process the loss on the test set batch by batch
        while not all_processed:
            # put a part of the features and audio samples in a batch
            start = number_batch*batch_size
            end = (number_batch+1)*batch_size
            if end>=number_examples:
                end = number_examples
                all_processed = True
            part_features = features[start:end,:,:]
            part_features_lengths = features_lengths[start:end]
            part_audio = audio[start:end,:,:]
            part_audio_lengths = audio_lengths[start:end]

            # pad with zeros if the last batch isn't completely filled
            if all_processed:
                elements_last_batch = end-start
                to_add = batch_size - elements_last_batch
                part_features = np.concatenate([part_features, np.zeros([to_add, max_feature_length, feature_dim])],0)
                part_features_lengths = np.concatenate([part_features_lengths, np.zeros([to_add], dtype=np.int32)],0)
                part_audio = np.concatenate([part_audio, np.zeros([to_add, max_audio_length,1], dtype=np.int32)],0)
                part_audio_lengths = np.concatenate([part_audio_lengths, np.zeros([to_add], dtype=np.int32)],0)

            # number of elements in the current batch
            numel = end-start

            # compute loss on this batch
            loss = sess.run(
                score,
                feed_dict={features_ph:part_features,
                        audio_ph:part_audio,
                        feature_lengths_ph:part_features_lengths,
                        audio_lengths_ph: part_audio_lengths})

            # update the average loss with the result of the loss on the current batch
            avrg_loss = ((total_elements*avrg_loss + numel*loss)/
                         (numel + total_elements))
            total_elements += numel

            number_batch = number_batch+1

            # print some info about how we're proceeding
            print 'Computing loss on test set: step %d of %d' %(number_batch, total_steps)

        #test for correctness
        if not(total_elements == number_examples):
            raise Exception('something went wrong in the loop where test loss of reconstruction is calculated')

    # print eventual result
    print '========================================'
    print 'The loss on the test set: %f' % avrg_loss
    print '========================================'


def compute_loss_audio(targets, logits, logit_seq_length,
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


def compute_loss_features(targets, logits, logit_seq_length,
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


if __name__ == '__main__':
    tf.app.run()
