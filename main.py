from preprocessor import corpus_to_tensor, scores_to_tensor
from vocabulary import Vocabulary
from model import RNNLM
import torch
import argparse
import logging
import numpy as np

if not torch.cuda.is_available():
  DEVICE = 'cpu'
else:
  DEVICE = 'cuda:0'


if __name__ == '__main__':

    # Reading arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', default='zh')
    args = parser.parse_args()
    lang = args.language

    # Logger
    log = logging.getLogger("default-logger")

    # Setting up seed
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Creating vocabularies
    vocab_src = Vocabulary()
    vocab_src.build_from_file('data/en-' + lang + '/train.en' + lang + '.src')
    # TODO: zh vocabulary should work differently as most "characters" are actually words
    vocab_mt = Vocabulary()
    vocab_mt.build_from_file('data/en-' + lang + '/train.en' + lang + '.mt')

    # Datasets from source language
    train_src = corpus_to_tensor(vocab_src, 'data/en-' + lang + '/train.en' + lang + '.src')
    valid_src = corpus_to_tensor(vocab_src, 'data/en-' + lang + '/dev.en' + lang + '.src')
    test_src = corpus_to_tensor(vocab_src, 'data/en-' + lang + '/test.en' + lang + '.src')

    # Datasets from target language
    train_mt = corpus_to_tensor(vocab_mt, 'data/en-' + lang + '/train.en' + lang + '.mt')
    valid_mt = corpus_to_tensor(vocab_mt, 'data/en-' + lang + '/dev.en' + lang + '.mt')
    test_mt = corpus_to_tensor(vocab_mt, 'data/en-' + lang + '/test.en' + lang + '.mt')

    # Getting scores
    train_scores = scores_to_tensor('data/en-' + lang + '/train.en' + lang + '.scores')
    valid_scores = scores_to_tensor('data/en-' + lang + '/dev.en' + lang + '.scores')
    test_scores = scores_to_tensor('data/en-' + lang + '/test.en' + lang + '.scores')


    rnnlm_model = RNNLM(
        vocab_size=len(vocab_src),  # vocabulary size
        emb_dim=128,                # word embedding dim
        hid_dim=128,                # hidden layer dim
        rnn_type='GRU',             # RNN type
        n_layers=1,                 # Number of stacked RNN layers
        clip_gradient_norm=1.0,     # gradient clip threshold
        bptt_steps=35,              # Truncated BPTT window size
        dropout=0.4,                # dropout probability
    )

    # move to device
    rnnlm_model.to(DEVICE)

    # Initial learning rate for the optimizer
    RNNLM_INIT_LR = 0.002

    # Create the optimizer
    rnnlm_optimizer = torch.optim.Adam(rnnlm_model.parameters(), lr=RNNLM_INIT_LR)
    print(rnnlm_model)

    print('Starting training!')
    rnnlm_model.train_model(rnnlm_optimizer, train_src, valid_src, test_src, n_epochs=5, batch_size=16)
