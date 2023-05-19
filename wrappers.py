import torch
import transformers

import models


class AttentiveModelConfig(transformers.PretrainedConfig):
    """A wrapper for the configuration of a ``models.AttentiveModel``, required by ``transformers``/``trl``."""
    def __init__(
        self,
        n_tokens: int,
        enc_size: int,
        dec_size: int,
        emb_size: int,
        attn_size: int,
        device: str,
        **kwargs
    ):
        self.n_tokens = n_tokens
        self.enc_size = enc_size
        self.dec_size = dec_size
        self.emb_size = emb_size
        self.attn_size = attn_size
        self.device = device
        super().__init__(**kwargs)


class AttentiveModelWrapper(transformers.PreTrainedModel):
    """
    A wrapper for a ``models.AttentiveModel`` to make it behave like a ``transformers.PreTrainedModel``
    for the purposes of training it with the ``trl`` library.
    Note that we only need to implement the ``forward`` method to make ``trl`` happy.
    """
    def __init__(self, cfg: AttentiveModelConfig):
        super().__init__(cfg)
        self.model = models.AttentiveModel(
            cfg.n_tokens,
            cfg.enc_size,
            cfg.dec_size,
            cfg.emb_size,
            cfg.attn_size,
            cfg.device
        )

    def forward(self, input_ids, decoder_input_ids, **_):
        """
        This method is called internally by ``trl.PPOTrainer``.
        It probably shouldn't be called directly -- use this object's ``model`` attribute
        to access the underlying ``models.AttentiveModel`` instead.
        """
        h_0, enc_seq = unflatten_batched_queries(input_ids, self.config.dec_size, self.config.enc_size)
        logits_seq = self.model(h_0, enc_seq, decoder_input_ids)
        return transformers.modeling_outputs.Seq2SeqLMOutput(
            # trl only needs 2 attributes of this output -- logits and decoder_hidden_states
            logits=logits_seq,
            decoder_hidden_states=(torch.stack(self.model.state.dec_states[:-1], dim=1),)
            # decoder_hidden_states should be a tuple of tensors with shape (batch_size, sequence_length, hidden_size)
            # trl only looks at the last element of the tuple, therefore we don't need to provide the others
        )


def flatten_queries(queries: list[list[torch.Tensor]]) -> list[torch.Tensor]:
    """
    Flattens a list of input data for the model into 1D tensors.
    This is done because ``trl`` requires each query to the model to be a 1D tensor.
    It batches the queries internally and passes the resulting tensor to ``AttentiveModelWrapper.forward``.

    This function also pads each tensor to the same length with zeros.
    This is done because normally ``trl`` tries to pad both the queries and the responses by itself,
    but unfortunately it does not support 2 different padding values for the input and output of the model,
    which is what we need.
    Therefore, we let ``trl`` do the padding for the **responses** by setting the ``pad_token_id`` for the "tokenizer"
    (see ``ResponsePadder`` in ``train.py``)
    and we pad the **queries** ourselves in this function.

    :param queries: a list of input data for the model.
                    Each element of the list represents information about one ML problem as a list of 2 tensors --
                    the first is the embedding of the entire description with the shape ``(dec_size,)``
                    and the second is the sequence of embeddings of the tokens in the description
                    with the shape ``(n_tokens, enc_size)``.
    :return: a list of tensors with the shape ``(dec_size + max_n_tokens * enc_size)``.
             The i-th element of this list is the flattened, concatenated and padded tensors at ``queries[i]``.
             ``max_n_tokens`` is the maximum value of ``n_tokens`` among all input sequences.
    """
    flat = [torch.cat([h_0.flatten(), enc_seq.flatten()]) for h_0, enc_seq in queries]
    return list(torch.nn.utils.rnn.pad_sequence(flat, batch_first=True, padding_value=0))


def unflatten_batched_queries(tsr: torch.Tensor, dec_size: int, enc_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reshapes a batched tensor of flattened queries that were obtained with ``flatten_queries``
    so that the results can be passed into ``AttentiveModel.forward`` as the 1st and 2nd argument respectively.

    :param tsr: a batched tensor of flattened queries with the shape ``(batch_size, dec_size + n_tokens * enc_size)``.
    :param dec_size: the length of the embedding of the entire problem description
    :param enc_size: the length of the embedding of each individual token in the description
    :return: 2 tensors with the shapes ``(batch_size, dec_size)`` and ``(batch_size, n_tokens, enc_size)`` respectively.
    """
    return tsr[:, :dec_size], tsr[:, dec_size:].reshape(tsr.shape[0], -1, enc_size)
