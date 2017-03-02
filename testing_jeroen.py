'''
small test file
'''


import nabu.processing.feature_reader as feature_reader

featdir = '/esat/spchtemp/scratch/jbosmans/features/TIMIT/train/quant_audio'

#create a feature reader for the training data
with open(featdir + '/maxlength', 'r') as fid:
    max_length = int(fid.read())

#cmvn = featdir + 'cmvn.scp'
cmvn = None

featreader = feature_reader.FeatureReader(
    scpfile=featdir + '/feats_shuffled.scp',
    cmvnfile=cmvn,
    utt2spkfile=featdir + '/utt2spk',
    max_length=max_length)

for i in range(1):
    utt_id, utt_mat, looped = featreader.get_utt()
    print(utt_id)
    print(looped)
