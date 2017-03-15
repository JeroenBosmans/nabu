'''@file test_asr.py
this file will test the asr on its own'''

import os
from six.moves import configparser
import tensorflow as tf
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

    #read the quantization config file
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
    audiodir = os.path.join(database_cfg['test_dir'], quant_cfg['name'])

    with open(os.path.join(audiodir, 'maxlength'), 'r') as fid:
        max_length_audio = int(fid.read())

    audio_reader = feature_reader.FeatureReader(
        scpfile=os.path.join(audiodir, 'feats.scp'),
        cmvnfile=None,
        utt2spkfile=None,
        max_length=max_length_audio)

    #create the classifier
    classifier = asr_factory.factory(
        conf=nnet_cfg,
        output_dim=(1,int(quant_cfg['quant_levels'])))

    #create a graph
    graph = tf.Graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    config.allow_soft_placement = True

    with tf.Session(graph=graph, config=config) as sess:
        #create a tensor of all of the features
        _,features,_ = feat_reader.get_utt()
        features=tf.constant(features)
        features = tf.reshape(features,[1,-1,int(features.get_shape()[1])])
        features_lengths = tf.constant(int(features.get_shape()[1]),dtype=tf.int32,shape=[1])
        features = tf.concat([features, tf.zeros([int(features.get_shape()[0]),max_length_feat-int(features.get_shape()[1]),int(features.get_shape()[2])])],1)
        looped = False
        while not looped:
            _,temp,looped = feat_reader.get_utt()
            temp = tf.constant(temp)
            temp = tf.reshape(temp,[1,-1,int(temp.get_shape()[1])])
            features_lengths = tf.concat([features_lengths,tf.constant(int(temp.get_shape()[1]),dtype=tf.int32,shape=[1])],0)
            temp = tf.concat([temp, tf.zeros([int(temp.get_shape()[0]),max_length_feat-int(temp.get_shape()[1]),int(temp.get_shape()[2])])],1)
            features = tf.concat([features, temp],0)

        #create a tensor of all of the targets
        _,audio,_ = audio_reader.get_utt()
        audio = tf.constant(audio, dtype=tf.int32)
        audio = tf.reshape(audio,[1,-1,int(audio.get_shape()[1])])
        audio_lengths = tf.constant(int(audio.get_shape()[1]),dtype=tf.int32,shape=[1])
        audio = tf.concat([audio, tf.zeros([int(audio.get_shape()[0]),max_length_audio-int(audio.get_shape()[1]),int(audio.get_shape()[2])], dtype=tf.int32)],1)
        looped = False
        while not looped:
            _,temp,looped = audio_reader.get_utt()
            temp = tf.constant(temp, dtype=tf.int32)
            temp = tf.reshape(temp,[1,-1,int(temp.get_shape()[1])])
            audio_lengths = tf.concat([audio_lengths,tf.constant(int(temp.get_shape()[1]),dtype=tf.int32,shape=[1])],0)
            temp = tf.concat([temp, tf.zeros([int(temp.get_shape()[0]),max_length_audio-int(temp.get_shape()[1]),int(temp.get_shape()[2])], dtype=tf.int32)],1)
            audio = tf.concat([audio, temp],0)

        # reshape the targets and put them in a tuple
        audio = tf.reshape(audio, [int(audio.get_shape()[0]),-1])
        targets = (None, audio)
        target_seq_lengths = (None, audio_lengths)

        #create a tensor of the logits
        logits, logits_lengths = classifier(features, features_lengths, targets, target_seq_lengths, False)

        #create a saver and load the model
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(FLAGS.expdir, 'model', 'network.ckpt'))


        #compute the loss score
        score = compute_loss(targets, logits, logits_lengths, target_seq_lengths)
        score.eval(sess)

    print '========================================'
    print 'The loss on the test set: %f' % score
    print '========================================'


def compute_loss(targets, logits, logit_seq_length,
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
        split_targets = tf.unstack(targets[1])
        for i, target in enumerate(split_targets):
            #only use the real data
            split_targets[i] = target[:target_seq_length[1][i]]

            #append an end of sequence label
            split_targets[i] = tf.concat(
                [split_targets[i], [output_dim-1]],0)

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


if __name__ == '__main__':
    tf.app.run()
