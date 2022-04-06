# SpeechSplit2
Official implementation of [SpeechSplit 2.0: Unsupervised speech disentanglement for voice conversion Without tuning autoencoder Bottlenecks](https://arxiv.org/pdf/2203.14156.pdf).

## Audio Demo
The audio demo can be found [here](https://biggytruck.github.io/spsp2-demo/).

## Pretrained Models
| | Small Bottleneck | Large Bottleneck |
|----------------|----------------|----------------|
| Generator| [link](https://drive.google.com/uc?export=download&id=1_Eo6_XxcZpk4P0jzjudkgjTKeb3Y-wMu) | [link](https://drive.google.com/uc?export=download&id=1yTVy4BjonLdXW7kTxvEMfDf_RhuDCyBZ) |
| F0 Converter | [link](https://drive.google.com/uc?export=download&id=1MhWkz3UGeZSolKfw0FF0DqhHNN1e5C82) | [link](https://drive.google.com/uc?export=download&id=1th0OFjM1k7y3dtNcijhUy1teKY23bHL8) |

The WaveNet vocoder is the same as in [AutoVC](https://github.com/auspicious3000/autovc). Please refer to the original repo to download the vocoder.

## Demo

To run the demo, first create a new directory called `models` and download the pretrained models and the WaveNet vocoder into this directory. Then, run `demo.ipynb`. The converted results will be saved under `result`.

## Training

### 1. Prepare dataset
Download the [VCTK Corpus](https://datashare.ed.ac.uk/handle/10283/3443) and place it under `data/train`. The data directory should look like:
```
  data
    |__train
    |    |__P225
    |        |__wavfile1
    |        |__wavfile2
    |        ...
    |    |__P226
    |     ...
    |__test
         |__p225_001.wav # source audio for demo
         |__p258_001.wav # target audio for demo
```
**NOTE**: The released models were trained only on a subset of speakers in the VCTK corpus. The full list of speakers for training is encoded as a dictionary and saved in `spk_meta.pkl`. If you want to train with more speakers or use another dataset, please prepare the metadata in the following key-value format:
```
speaker: (id, gender)
```
where `speaker` should be a string, `id` should be a unique integer for each speaker(will be used to generate one-hot speaker vector), and `gender` should either be "M"(for male) and "F"(for female).

### 2. Generate features
To generate features, run
```
python main.py --stage 0
```
By default, all generated features are saved in the `feat` directory.

### 3. Train the model
To train a model from scratch, run
```
python main.py --stage 1 --config_name spsp2-large --model_type G
```
To finetune a pretrained model(make sure all pretrained models are downloaded into `models`), run
```
python main.py --stage 1 --config_name spsp2-large --model_type G --resume_iters 800000
```
If you want to train the variant with smaller bottleneck, replace `spsp2-large` with `spsp2-small`. If you want to train the pitch converter, replace `G` with `F`.
