import torch
from torch import nn
import numpy as np
import time

if not torch.cuda.is_available():
  DEVICE = 'cpu'
else:
  DEVICE = 'cuda:0'

def readable_size(n):
    """Returns a readable size string for model parameters count."""
    sizes = ['K', 'M', 'G']
    fmt = ''
    size = n
    for i, s in enumerate(sizes):
        nn = n / (1000 ** (i + 1))
        if nn >= 1:
            size = nn
            fmt = sizes[i]
        else:
            break
    return '%.2f%s' % (size, fmt)

class RNNLM(nn.Module):
  """RNN-based LM module."""
  def __init__(self, vocab_size, emb_dim, hid_dim, rnn_type='RNN',
               n_layers=1, dropout=0.5, clip_gradient_norm=1.0,
               bptt_steps=35):
    super(RNNLM, self).__init__()

    # Store arguments
    self.vocab_size = vocab_size
    self.emb_dim = emb_dim
    self.hid_dim = hid_dim
    self.clip_gradient_norm = clip_gradient_norm
    self.bptt_steps = bptt_steps
    self.n_layers = n_layers
    self.rnn_type = rnn_type.upper()

    # This will be used to store the detached histories for truncated BPTT
    self.prev_histories = None

    # Create the loss, don't sum or average, we'll take care of it
    # in the training loop for logging purposes
    self.loss = nn.CrossEntropyLoss(reduction='none')

    # Create the dropout
    self.drop = nn.Dropout(p=dropout)

    # Create the embedding layer as usual
    self.emb = nn.Embedding(
      num_embeddings=self.vocab_size, embedding_dim=self.emb_dim,
      padding_idx=0)

    # Create the RNN layer
    if self.rnn_type == 'RNN':
      self.rnn = nn.RNN(
          input_size=self.emb_dim, hidden_size=self.hid_dim,
          num_layers=self.n_layers, nonlinearity='tanh')
    elif self.rnn_type == 'GRU':
      self.rnn = nn.GRU(
          input_size=self.emb_dim, hidden_size=self.hid_dim,
          num_layers=self.n_layers)
    elif self.rnn_type == 'LSTM':
      #####################################
      # Q: Fill in to create the LSTM layer
      #####################################
      self.rnn = "<TODO>"

    # Create the output layer: maps the hidden state of the RNN to vocabulary
    self.out = nn.Linear(self.hid_dim, self.vocab_size)

    # Compute number of parameters for information
    self.n_params = 0
    for param in self.parameters():
      self.n_params += np.cumprod(param.data.size())[-1]
    self.n_params = readable_size(self.n_params)

  def init_state(self, batch_size):
    """Returns the initial 0 states."""
    if self.rnn_type != 'LSTM':
      # for every layer and every sample -> 0 hidden state vector
      return torch.zeros(self.n_layers, batch_size, self.hid_dim, device=DEVICE)
    else:
      #################################################################
      # Q: Adapt the above snippet to LSTM. Check PyTorch docs
      # to understand what is the expectation of LSTM's forward() call
      # in terms of initial states.
      #################################################################
      return "<TODO>"

  def clear_hidden_states(self):
    """Set the relevant instance attribute to None."""
    self.prev_histories = None

  def save_hidden_states(self, last_states):
    """Save the detached states into the model for the next batch. `last_states`
    is the second return value of RNN/GRU/LSTM's forward() methods."""
    if isinstance(last_states, tuple):
      # This is true for LSTM
      self.prev_histories = tuple(r.detach() for r in last_states)
    else:
      self.prev_histories = last_states.detach()

  def forward(self, x, y):
    """Forward-pass of the module."""
    # Detached previous histories for a batch. If `None`, we assume
    # start of an epoch or start of an evaluation and create 0
    # vector(s) to start with.
    if self.prev_histories is None:
      self.prev_histories = self.init_state(x.shape[1])

    # Tokens -> Embeddings -> Dropout
    embs = self.drop(self.emb(x))

    # an RNN in PyTorch returns two values:
    # (1) All hidden states of the last RNN layer
    #     Shape -> (bptt_steps, batch_size, hid_dim)
    #     You'll plug the output layer on top of this to obtain
    #     the logits for each prediction.
    # (2) Hidden state h_t of last timestep for EVERY layer
    #     Shape -> (self.n_layers, batch_size, hid_dim)
    #     This is what we'll store as the previous history
    #     (NOTE: this is a tuple for LSTM which contains h_t and c_t)
    all_hids, last_hid = self.rnn(embs, self.prev_histories)

    # Detach the computation graph since we are done with BPTT for this batch
    self.save_hidden_states(last_hid)

    ##########################################################
    # Q: Apply dropout on all_hids and pass it to output layer
    ##########################################################
    logits = "<TODO>"

    # Return the losses per token/position
    return self.loss(logits.view(-1, self.vocab_size), y)

  def get_batches(self, data_tensor, batch_size):
    # NOTE: There is absolutely no shuffling here, which
    # will totally break the histories coming from previous steps.
    # The document is evenly divided into independent `batch_size` portions.
    # At every iteration, the BPTT window will slide over each of these
    # portions, by keeping track of the previous h_t's as discussed
    # in the lecture.

    # Imagine this as `batch_size` pointers running over the text, each
    # processing its share in a continuous. Although the portions may have
    # been splitted in a noisy way (one pointer can be starting from the
    # middle of a sentence for example), this makes training faster.
    # For instance, with the alphabet as the dataset and batch size 4, we'd get
    # ┌ a g m s ┐
    # │ b h n t │
    # │ c i o u │
    # │ d j p v │
    # │ e k q w │
    # └ f l r x ┘.
    # These columns are treated as "independent" by the model, which means that
    # the dependence of 'g' on 'f' can not be learned, but allows more efficient
    # batch processing. The view above will further be splitted into chunks
    # of size `bptt_steps` to apply truncated BPTT. For example, with
    # `bptt_steps == 2`, we'll have the following `x` and `y` tensors. The
    # first batch will be processing "a, b" to predict "b, c",
    # the second batch will be processing "g, h" to predict "h, i", and so on.
    #
    #       X          Y
    #   ----->>------
    #   |           |
    # ┌ a g m s ┐ ┌ b h n t ┐
    # └ b h n t ┘ └ c i o u ┘
    #   |           |
    #   ----->>------

    # Work out how cleanly we can divide the dataset into batch_size parts.
    n_batches = data_tensor.size(0) // batch_size

    # Trim off the remainder tokens to evenly split
    # Evenly divide the data across the batches
    data = data_tensor[:n_batches * batch_size].view(
        batch_size, n_batches).t().contiguous()

    batches = []

    for i in range(0, data.size(0) - 1, self.bptt_steps):
      # seq_len can be less than bptt_steps in the final parts of the data
      seq_len = min(self.bptt_steps, len(data) - i - 1)

      # x shape => (seq_len, batch_size)
      x = data[i: i + seq_len]
      # flatten the ground-truth labels (shifted inputs for LM)
      y = data[i + 1: i + 1 + seq_len].view(-1)
      batches.append((x, y))

    return batches

  def train_model(self, optim, train_tensor, valid_tensor, test_tensor, n_epochs=5,
                 batch_size=64):
    """Trains the model."""
    # Get batches for all splits at once
    train_batches = self.get_batches(train_tensor, batch_size)
    valid_batches = self.get_batches(valid_tensor, batch_size)
    test_batches = self.get_batches(test_tensor, batch_size)

    for eidx in range(1, n_epochs + 1):
      start_time = time.time()
      epoch_loss = 0
      epoch_items = 0

      # Enable training mode
      self.train()

      # Start training
      for iter_count, (x, y) in enumerate(train_batches):
        # Clear the gradients
        optim.zero_grad()

        loss = self.forward(x.to(DEVICE), y.to(DEVICE))

        # Backprop the average loss and update parameters
        loss.mean().backward()

        # Clip the gradients to avoid exploding gradients
        if self.clip_gradient_norm > 0:
          torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_gradient_norm)

        # Update parameters
        optim.step()

        # sum the loss for reporting, along with the denominator
        epoch_loss += loss.detach().sum()
        epoch_items += loss.numel()

        if iter_count % 500 == 0:
          # Print progress
          loss_per_token = epoch_loss / epoch_items
          ppl = math.exp(loss_per_token)
          print(f'[Epoch {eidx:<3}] loss: {loss_per_token:6.2f}, perplexity: {ppl:6.2f}')

      time_spent = time.time() - start_time

      # Clear stale h_t history before evaluation
      self.clear_hidden_states()

      print(f'\n[Epoch {eidx:<3}] ended with train_loss: {loss_per_token:6.2f}, ppl: {ppl:6.2f}')
      # Evaluate on valid set
      valid_loss, valid_ppl = self.evaluate(valid_batches)
      print(f'[Epoch {eidx:<3}] ended with valid_loss: {valid_loss:6.2f}, valid_ppl: {valid_ppl:6.2f}')
      print(f'[Epoch {eidx:<3}] completed in {time_spent:.2f} seconds\n')

    # Evaluate the final model on test set
    test_loss, test_ppl = self.evaluate(test_batches)
    print(f' ---> Final test set performance: {test_loss:6.2f}, test_ppl: {test_ppl:6.2f}')

  def evaluate(self, batches):
    # Clear stale h_t history before evaluation
    self.clear_hidden_states()

    # Switch to eval mode
    self.eval()

    total_loss = 0.
    total_tokens = 0

    with torch.no_grad():
      for iter_count, (x, y) in enumerate(batches):
        loss = self.forward(x.to(DEVICE), y.to(DEVICE))
        total_loss += loss.sum().item()
        total_tokens += loss.size(0)
    total_loss /= total_tokens

    self.clear_hidden_states()
    return total_loss, math.exp(total_loss)

  def __repr__(self):
    s = super(RNNLM, self).__repr__()
    return "{}\n# of parameters: {}".format(s, self.n_params)
