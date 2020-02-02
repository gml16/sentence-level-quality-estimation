# TODO: remove words with too few occurences

class Vocabulary(object):
  """Data structure representing the vocabulary of a corpus."""
  def __init__(self):
    # Mapping from tokens to integers
    self._word2idx = {}

    # Reverse-mapping from integers to tokens
    self.idx2word = []

    # 0-padding token
    self.add_word('<pad>')
    # sentence start
    self.add_word('<s>')
    # sentence end
    self.add_word('</s>')
    # Unknown words
    self.add_word('<unk>')

    self._unk_idx = self._word2idx['<unk>']

  def word2idx(self, word):
    """Returns the integer ID of the word or <unk> if not found."""
    return self._word2idx.get(word, self._unk_idx)

  def add_word(self, word):
    """Adds the `word` into the vocabulary."""
    if word not in self._word2idx:
      self.idx2word.append(word)
      self._word2idx[word] = len(self.idx2word) - 1

  def build_from_file(self, fname):
    """Builds a vocabulary from a given corpus file."""
    with open(fname, encoding="utf8") as f:
      for line in f:
        words = line.strip().split()
        for word in words:
          self.add_word(word)

  def convert_idxs_to_words(self, idxs):
    """Converts a list of indices to words."""
    return ' '.join(self.idx2word[idx] for idx in idxs)

  def convert_words_to_idxs(self, words):
    """Converts a list of words to a list of indices."""
    return [self.word2idx(w) for w in words]

  def __len__(self):
    """Returns the size of the vocabulary."""
    return len(self.idx2word)

  def __repr__(self):
    return "Vocabulary with {} items".format(self.__len__())
