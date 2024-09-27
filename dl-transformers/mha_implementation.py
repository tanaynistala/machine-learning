import torch
import math


def mha_137(query, key, value, in_wbs, out_wb):
    """ An implementation of the MHA calculation.

        args:
            query: a tensor with shape (seq_len1, batch_size, emb_dim)
            key: a tensor with shape (seq_len2, batch_size, emb_dim)
            value: a tensor with shape (seq_len2, batch_size, emb_dim)
            in_wbs: weights and bias used in linear transformations for the three input
            out_wb: weights and bias used in the last linear transformation for computing the output
        returns:
            output: a tensor with shape (seq_len1, batch_size, emb_dim)
    """

    # You are supposed to implement multihead attention in this function.


    # TODO: Please check the documentation of MHA before you implement this function
    # https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html

    # TODO: please read the comments below carefully to understand the provided parameters


    # We will need the following parameters to do linear transformations
    # `wqs` is a list of weight matrices, each for an attention head.
    # The length of `wqs` is the number of attention heads. Each element
    # should be applied for the linear transformation of the query `query`.

    # The list `bqs` contains bias vectors for transoformations
    # of the query in different attention heads. The transformation of the query should be

    # query * wqs[head].transpose() + bqs[head]

    # Here * is the matrix multiplication. Please consider the function
    # `torch.nn.functional.linear`
    #
    # Similarly, `(wks, bks)` is for the transformations of `key`, and `(wvs, bvs)` is
    # for the transformation of `value`

    wqs, wks, wvs, bqs, bks, bvs = in_wbs
    out_w, out_b = out_wb

    # Suggestion: you may want to first transpose the tensor such that the last two ranks are [seq_len, emb_dim],
    # which is convenient for matrix calculation later.

    query = query.transpose(0, 1)
    key = key.transpose(0, 1)
    value = value.transpose(0, 1)

    # loop over attention heads
    output = []
    for head in range(len(wqs)):

        # TODO: run linear transformation on query with (wqs[head], bqs[head])
        query_head = query @ wqs[head].transpose(0, 1) + bqs[head]
        # TODO: run linear transformation on key with (wks[head], bks[head])
        key_head = key @ wks[head].transpose(0, 1) + bks[head]
        # TODO: run linear transformation on value with (wvs[head], bvs[head])
        value_head = value @ wvs[head].transpose(0, 1) + bvs[head]

        # TODO: calculate attention logits using inner product
        # Please remember to scale these logits weight the sqrt of the dimmension of the transformed queries/keys
        output_head = query_head @ key_head.transpose(1, 2) / math.sqrt(query_head.shape[-1])

        # TODO: apply softmax to compute attention weights
        output_head = torch.softmax(output_head, dim=-1)

        # TODO: use attention weights to pool values. Note that this step can be done with matrix multiplication
        output_head = output_head @ value_head

        output.append(output_head)

    # TODO: concatenate attention outputs from different heads
    attn_output = torch.cat(output, dim=-1)

    # TODO: apply the last linear transformation with (out_w, out_b)
    attn_output = attn_output @ out_w.transpose(0, 1) + out_b

    # Suggestion: if necessary, please permute ranks
    attn_output = attn_output.transpose(0, 1)

    return attn_output
