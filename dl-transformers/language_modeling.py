import torch
import math
from torch import Tensor
from torch import nn

from third_party import batchify, get_batch, evaluate, generate_square_subsequent_mask, EncoderLayer

# TODO: you are supposed to implement a language model and the training function to support the notebook. This is
# the last assignment of the class, so you should have known the pipeline of training a deep model, so this time
# the minimum starter code is given.

# NOTE 1: this time you ARE allowed to copy-paste some code from ONLY the following two sources. You need the
# put the code in classes or functions in `third_party.py` and import it from there, just as the commented line
# shown above. You should uncomment the commented importing line if you need these functions

# * d2l Chapter 11.7 [link](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html)
# * Torch tutorial on language modeling [link](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)

# This code file should only contain the given code and code you will type in. NO code should copy-pasted from
# ANY source.

# NOTE 2: You cannot import a transformer mdoel entirely from torch. You are supposed to construct a Transformer model
# with torch layers.


# The `PositionalEncoding` class is provided here.
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class SmallLanguageModel(nn.Module):
    """
    A small language model using the transformer architecture
    """

    def __init__(self, vocabulary):
        """
        args:
            vocabulary: a list of characters. The list also provide the vocabulary size
        """
        super().__init__()

        # Suggestion: my suggestion is to use the embedding layer to get vector representations of
        # integers (indices of characters). An alternative is to convert indices to one-hot encodings
        # and then apply a dense layer.

        vocab_size = len(vocabulary)
        embedding_dim = 256
        hidden_dim = 256
        num_layers = 5
        num_heads = 4
        dropout = 0.01

        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.pos = PositionalEncoding(embedding_dim, dropout)

        layers = [EncoderLayer(embedding_dim, hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        self.enc = nn.Sequential(*layers)

        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, X):
        """
        The forward function of the model. We will follow the format of the `nn.Transformer` and assume
        `X` has the shape `[seq_len, batch_size]`.
        args:
            X: a tensor with shape `[seq_len, batch_size]`. Each entry of X is an index to a char in vocabulary.
        returns:
            out: a tensor with shape `[seq_len, batch_size, len(vocabulary)]`. The fiber `X[t, b, :]` should be the logits
                 for the prediction of the `(t+1)`-th char in the input sequence.
        """

        # NOTE: please do not forget to add positional encoding here
        # NOTE: please do not forget to turn on the causal flag here.
        # You will need a mask for causal modeling, and `generate_square_subsequent_mask`
        # will get you the mask

        out = self.emb(X) * math.sqrt(self.emb.embedding_dim)
        out = self.pos(out)

        mask = generate_square_subsequent_mask(X.size(0)).to(X.device)
        for layer in self.enc:
            out = layer(out, mask)

        out = self.linear(out)

        return out


# TODO: please implement the following function to train the language model

def train(model, train_data, val_data, loss_func, optimizer, scheduler, num_epochs = 2, bptt = 50):
    """
    The training function for language modeling

    args:
        model: a language model. Given (c_0, c_2, ..., c_{k-1}), then model should output logits for (c_1, c_2, ..., c_k)
        train_data: a 1-d tensor containing the training data
        val_data: a 1-d tensor containing the validation data
        loss_func: the loss function
        optimizer: the torch opimizer
        schedular: the torch schedular for learning rate
        num_epochs: int, the maximum number of training epochs
        bptt: int, the window size of the input, or it is the sequence length in one batch.
    """

    # Suggestion: you may want to use `batchify` to reshape the data to `batch_size` sequences, so the data will be reshaped
    # in a tensor with size `[seq_len, batch_size]`. Later you can use batches to get a bptt rows from the tensor to get a batch
    # This is how the data is processed in the tutorial


    # TODO: implement the training loop

    train_data = batchify(train_data.to('mps'), 32)
    val_data = batchify(val_data.to('mps'), 32)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        num_batches = train_data.size(0)
        for i in range(0, num_batches - 1, bptt):
            data, targets = get_batch(train_data, i, bptt)
            targets = targets.long()

            output = model(data)
            voc_size = output.shape[-1]
            output_flattened = output.view(-1, voc_size)

            loss = loss_func(output_flattened, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            if i % 100 == 0 and i > 0:
                scheduler.step()
                cur_loss = total_loss / 100
                print(f'epoch {epoch}/{num_epochs}, batch {i}/{num_batches}, loss {cur_loss}', end='\r')
                total_loss = 0

        val_loss = evaluate(model, val_data, loss_func, bptt)
        print(f'epoch {epoch}, val loss {val_loss}\n')

    return model