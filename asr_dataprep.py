'''@file dataprep.py
this file will do the dataprep for asr training'''

import os
import random
from six.moves import configparser
from nabu.processing import ark, prepare_data
from nabu.processing.target_normalizers import normalizer_factory

#pointers to the config files
database_cfg_file = 'config/asr_databases/TIMIT50p.conf'
feat_cfg_file = 'config/features/fbank.cfg'

#pointer to the config file for storing of the audio samples
#only needed when audio samples should be extracted
audio_storage_cfg_file = 'config/features/quant_audio_samples.cfg'

#read the database config file
database_cfg = configparser.ConfigParser()
database_cfg.read(database_cfg_file)
database_cfg = dict(database_cfg.items('database'))

#read the features config file
feat_cfg = configparser.ConfigParser()
feat_cfg.read(feat_cfg_file)
feat_cfg = dict(feat_cfg.items('features'))

#read if what kind of training we are aiming at
if database_cfg['prepare_audio'] == 'True':
    prepare_audio = True
else:
    prepare_audio = False

# read the audio storage config file
if prepare_audio:
    audiostore_cfg = configparser.ConfigParser()
    audiostore_cfg.read(audio_storage_cfg_file)
    audiostore_cfg = dict(audiostore_cfg.items('features'))

# read the percentage of the labels to keep if that is specified
if 'part_labeled' in database_cfg:
    percentage_to_keep = float(database_cfg['part_labeled'])
else:
    percentage_to_keep = 1.0

#compute the features of the training set for training
print '------- computing training features ----------'
prepare_data.prepare_data(
    datadir=database_cfg['train_data'],
    featdir=os.path.join(database_cfg['train_dir'], feat_cfg['name']),
    conf=feat_cfg)

print '------- computing cmvn stats ----------'
prepare_data.compute_cmvn(
    featdir=database_cfg['train_dir'] + '/' + feat_cfg['name'])


#get the feature dim
reader = ark.ArkReader(os.path.join(database_cfg['train_dir'],
                                    feat_cfg['name'], 'feats.scp'))
_, features, _ = reader.read_next_utt()
input_dim = features.shape[1]

with open(os.path.join(database_cfg['train_dir'], feat_cfg['name'], 'dim'),
          'w') as fid:
    fid.write(str(input_dim))

#compute the features of the dev set
if 'dev_data' in database_cfg:

    print '------- computing developement features ----------'
    prepare_data.prepare_data(
        datadir=database_cfg['dev_data'],
        featdir=os.path.join(database_cfg['dev_dir'], feat_cfg['name']),
        conf=feat_cfg)

    print '------- computing cmvn stats ----------'
    prepare_data.compute_cmvn(
        featdir=os.path.join(database_cfg['dev_dir'], feat_cfg['name']))

    with open(os.path.join(database_cfg['dev_dir'], feat_cfg['name'],
                           'dim'), 'w') as fid:
        fid.write(str(input_dim))

#compute the features of the test set for testing
print '------- computing testing features ----------'
prepare_data.prepare_data(
    datadir=database_cfg['test_data'],
    featdir=os.path.join(database_cfg['test_dir'], feat_cfg['name']),
    conf=feat_cfg)

print '------- computing cmvn stats ----------'
prepare_data.compute_cmvn(
    featdir=os.path.join(database_cfg['test_dir'], feat_cfg['name']))

with open(os.path.join(database_cfg['test_dir'], feat_cfg['name'], 'dim'),
          'w') as fid:
    fid.write(str(input_dim))

#shuffle the training data on disk
print '------- shuffling examples ----------'
prepare_data.shuffle_examples(os.path.join(database_cfg['train_dir'],
                                           feat_cfg['name']))

#create the text normalizer
normalizer = normalizer_factory.factory(database_cfg['normalizer'])

print '------- normalizing training targets -----------'
sourcefile = database_cfg['traintext']
target_fid = open(os.path.join(database_cfg['train_dir'], 'targets'), 'w')

# how many examples are available
num_examples = sum(1 for line in open(sourcefile))

# shuffle an array of 0's and zeros to decide which labels to keep
labels_to_keep = int(percentage_to_keep * num_examples)
labels_to_delete = num_examples - labels_to_keep
decider = [0]*labels_to_delete + [1]*labels_to_keep
random.shuffle(decider)

#read the textfile line by line, normalize and write in target file
with open(sourcefile) as fid:
    counter = 0
    for line in fid.readlines():
        splitline = line.strip().split(' ')
        utt_id = splitline[0]
        trans = ' '.join(splitline[1:])
        normalized = normalizer(trans)

        #write the result when the decider has a 1 for this entry
        if decider[counter] == 1:
            target_fid.write('%s %s\n' % (utt_id, normalized))
        else:
            target_fid.write('%s %s\n' % (utt_id, ''))

        counter = counter+1

#store the alphabet
with open(os.path.join(database_cfg['train_dir'], 'alphabet'), 'w') as fid:
    fid.write(' '.join(normalizer.alphabet))

print '------- normalizing testing targets -----------'
sourcefile = database_cfg['testtext']
target_fid = open(os.path.join(database_cfg['test_dir'], 'targets'), 'w')

#read the textfile line by line, normalize and write in target file
with open(sourcefile) as fid:
    for line in fid.readlines():
        splitline = line.strip().split(' ')
        utt_id = splitline[0]
        trans = ' '.join(splitline[1:])
        normalized = normalizer(trans)
        target_fid.write('%s %s\n' % (utt_id, normalized))

#store the alphabet
with open(os.path.join(database_cfg['test_dir'], 'alphabet'), 'w') as fid:
    fid.write(' '.join(normalizer.alphabet))

if 'devtext' in database_cfg:
    print '------- normalizing developments targets -----------'
    sourcefile = database_cfg['devtext']
    target_fid = open(os.path.join(database_cfg['dev_dir'], 'targets'),
                      'w')

    #read the textfile line by line, normalize and write in target file
    with open(sourcefile) as fid:
        for line in fid.readlines():
            splitline = line.strip().split(' ')
            utt_id = splitline[0]
            trans = ' '.join(splitline[1:])
            normalized = normalizer(trans)
            target_fid.write('%s %s\n' % (utt_id, normalized))

    #store the alphabet
    with open(os.path.join(database_cfg['dev_dir'], 'alphabet'),
              'w') as fid:
        fid.write(' '.join(normalizer.alphabet))


# when audio is needed, we also need to store the quantized audio samples
if prepare_audio:

    #store the audio samples of the training set
    print '------- computing training audio samples ----------'
    prepare_data.prepare_data(
        datadir=database_cfg['train_data'],
        featdir=os.path.join(database_cfg['train_dir'],
                             audiostore_cfg['name']),
        conf=audiostore_cfg)

    #compute the audio samples of the dev set
    if 'dev_data' in database_cfg:
        print '------- computing developement audio samples ----------'
        prepare_data.prepare_data(
            datadir=database_cfg['dev_data'],
            featdir=os.path.join(database_cfg['dev_dir'],
                                 audiostore_cfg['name']),
            conf=audiostore_cfg)

    #compute the audio samples of the test set for testing
    print '------- computing testing audio samples ----------'
    prepare_data.prepare_data(
        datadir=database_cfg['test_data'],
        featdir=os.path.join(database_cfg['test_dir'], audiostore_cfg['name']),
        conf=audiostore_cfg)

    #shuffle the samples on disk
    print '------- shuffling examples ----------'
    prepare_data.shuffle_examples(os.path.join(database_cfg['train_dir'],
                                               audiostore_cfg['name']))
