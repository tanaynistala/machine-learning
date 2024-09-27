import torch
import math
from torch import Tensor
from torch import nn

def batchify(data: Tensor, bsz: int) -> Tensor:
    """Divides the data into ``bsz`` separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Arguments:
        data: Tensor, shape ``[N]``
        bsz: int, batch size

    Returns:
        Tensor of shape ``[N // bsz, bsz]``
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data


def get_batch(source, i, bptt):
    """
    Args:
        source: Tensor, shape ``[full_seq_len, batch_size]``
        i: int

    Returns:
        tuple (data, target), where data has shape ``[seq_len, batch_size]`` and
        target has shape ``[seq_len * batch_size]``
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

generate_square_subsequent_mask = torch.nn.Transformer.generate_square_subsequent_mask


def evaluate(model, eval_data, loss_func, bptt = 50):

    model.eval()  # turn on evaluation mode

    total_loss = 0.

    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):

            data, targets = get_batch(eval_data, i, bptt)
            targets = targets.long()
            seq_len = data.size(0)

            output = model(data)
            voc_size = output.shape[-1]

            output_flat = output.view(-1, voc_size)
            total_loss += seq_len * loss_func(output_flat, targets).item()

    return total_loss / (len(eval_data) - 1)


# From D2L

class FFN(nn.Module):
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.Linear(ffn_num_outputs, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class Norm(nn.Module):
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class EncoderLayer(nn.Module):
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(num_hiddens, num_heads, dropout, bias=False)
        self.norm1 = Norm(num_hiddens, dropout)
        self.ffn = FFN(ffn_num_hiddens, num_hiddens)
        self.norm2 = Norm(num_hiddens, dropout)

    def forward(self, X, mask):
        Y = self.norm1(X, self.attn(X, X, X, attn_mask=mask, is_causal=True)[0])
        return self.norm2(Y, self.ffn(Y))
