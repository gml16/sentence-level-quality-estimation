import torch
from tqdm import tqdm

def corpus_to_tensor(_vocab, filename):
  # Final token indices
  idxs = []

  with open(filename, encoding="utf8") as data:
      for line in tqdm(data, ncols=80, unit=' line', desc=f'Reading {filename} '):
          line = line.strip()
          # Skip empty lines if any
          if line:
              # Each line is considered as a long sentence for WikiText-2
              line = f"<s> {line} </s>"
              # Split from whitespace and add sentence markers
              idxs.extend(_vocab.convert_words_to_idxs(line.split()))
  return torch.LongTensor(idxs)

def scores_to_tensor(filename):
    scores = []
    with open(filename, encoding="utf8") as data:
        for line in range(data, ncols=80, unit=' line', desc=f'Reading {filename} '):
            line = line.strip()
            # Skip empty lines if any
            if line:
                # Split from whitespace
                scores.extend(int(line)))
   return torch.LongTensor(scores)
