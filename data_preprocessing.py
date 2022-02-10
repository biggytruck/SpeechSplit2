import os
import pickle
import numpy as np
import soundfile as sf
from numpy.random import RandomState
from utils import *

def make_spect_f0(config):
    fs = 16000
    data_dir = config.data_dir
    feat_dir = config.feat_dir
    wav_dir = os.path.join(feat_dir, config.wav_dir)
    spmel_dir = os.path.join(feat_dir, config.spmel_dir)
    f0_dir = os.path.join(feat_dir, config.f0_dir)
    spk_meta = pickle.load(open('spk_meta.pkl', "rb"))

    dir_name, spk_dir_list, _ = next(os.walk(data_dir))
    state_count = 1

    for spk_dir in sorted(spk_dir_list):
        if spk_dir not in spk_meta:
            print(f'Warning: {spk_dir} not in speaker metadata; skip generating features')
            continue
        print(f'Generating features for speaker {spk_dir}')
        
        for fea_dir in [wav_dir, spmel_dir, f0_dir]:
            if not os.path.exists(os.path.join(fea_dir, spk_dir)):
                os.makedirs(os.path.join(fea_dir, spk_dir))

        _,_, file_list = next(os.walk(os.path.join(dir_name,spk_dir)))

        if spk_meta[spk_dir][1] == 'M':
            lo, hi = 50, 250
        elif spk_meta[spk_dir][1] == 'F':
            lo, hi = 100, 600
        else:
            continue

        prng = RandomState(state_count) 
        wavs, f0s, sps, aps = [], [], [], []
        for filename in sorted(file_list):
            # read audios
            x, _ = sf.read(os.path.join(dir_name,spk_dir,filename))
            if x.shape[0] % 256 == 0:
                x = np.concatenate((x, np.array([1e-06])), axis=0)
            wav = filter_wav(x, prng)

            # get WORLD analyzer parameters
            f0, sp, ap = get_world_params(wav, fs)
            wavs.append(wav)
            f0s.append(f0)
            sps.append(sp)
            aps.append(ap)

        # smooth pitch to synthesize monotonic speech
        f0s = average_f0s(f0s, mode='global')

        for idx, (wav, f0, sp, ap) in enumerate(zip(wavs, f0s, sps, aps)):

            wav_mono = get_monotonic_wav(wav, f0, sp, ap, fs)
            spmel = get_spmel(wav)
            f0_rapt, f0_norm = extract_f0(wav, fs, lo, hi)
            assert len(spmel) == len(f0_rapt)

            # segment feature into trunks with the same length during training
            start_idx = 0
            trunk_len = 49151
            while start_idx*trunk_len < len(wav_mono):
                wav_mono_trunk = wav_mono[start_idx*trunk_len:(start_idx+1)*trunk_len]
                if len(wav_mono_trunk) < trunk_len:
                    wav_mono_trunk = np.pad(wav_mono_trunk, (0, trunk_len-len(wav_mono_trunk)))
                np.save(os.path.join(wav_dir, spk_dir, os.path.splitext(filename)[0]+'_'+str(start_idx)),
                        wav_mono_trunk.astype(np.float32), allow_pickle=False)
                start_idx += 1
            feas = [spmel, f0_norm]
            fea_dirs = [spmel_dir, f0_dir]
            for fea, fea_dir in zip(feas, fea_dirs):
                start_idx = 0
                trunk_len = 192
                while start_idx*trunk_len < len(fea):
                    fea_trunk = fea[start_idx*trunk_len:(start_idx+1)*trunk_len]
                    if len(fea_trunk) < trunk_len:
                        if fea_trunk.ndim==2:
                            fea_trunk = np.pad(fea_trunk, ((0, trunk_len-len(fea_trunk)), (0, 0)))
                        else:
                            fea_trunk = np.pad(fea_trunk, ((0, trunk_len-len(fea_trunk)), ))
                    np.save(os.path.join(fea_dir, spk_dir, os.path.splitext(filename)[0]+'_'+str(start_idx)),
                            fea_trunk.astype(np.float32), allow_pickle=False)
                    start_idx += 1


def make_metadata(config):
    feat_dir = config.feat_dir
    wav_dir = os.path.join(feat_dir, config.wav_dir) # use wav directory simply because all inputs have the same filename
    dir_name, spk_dir_list, _ = next(os.walk(wav_dir))
    spk_meta = pickle.load(open('spk_meta.pkl', "rb"))
    dataset = []

    for spk_dir in sorted(spk_dir_list):
        spk_id, _ = spk_meta[spk_dir]

        # may use generalized speaker embedding for zero-shot conversion
        spk_emb = np.zeros((config.dim_spk_emb,), dtype=np.float32)
        spk_emb[int(spk_id)] = 1.0

        utterances = []
        _, _, file_list = next(os.walk(os.path.join(dir_name, spk_dir)))
        file_list = sorted(file_list)
        for filename in file_list:
            utterances.append(os.path.join(spk_dir,filename))
        for utterance in utterances:
            dataset.append((spk_dir, spk_emb, utterance))

    with open(os.path.join(feat_dir, 'dataset.pkl'), 'wb') as handle:
        pickle.dump(dataset, handle)

def preprocess_data(config):
    print('Start preprocessing...')
    make_spect_f0(config)
    make_metadata(config)
    print('Done')