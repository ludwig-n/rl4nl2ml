import numpy as np
import torch
import torch.nn as nn


SOS_ID = 75
EOS_ID = 76
PAD_ID = 77


class AttentionLayer(nn.Module):
    def __init__(self, name, enc_size, dec_size, hid_size, activ=torch.tanh):
        """ A layer that computes additive attention response and weights """
        super().__init__()
        self.name = name
        self.enc_size = enc_size  # num units in encoder state
        self.dec_size = dec_size  # num units in decoder state
        self.hid_size = hid_size  # attention layer hidden units
        self.activ = activ  # attention layer hidden nonlinearity

        self.linear_e = nn.Linear(enc_size, hid_size)
        self.linear_d = nn.Linear(dec_size, hid_size)

        self.linear_o = nn.Linear(hid_size, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, enc, dec, inp_mask):
        """
        Computes attention response and weights
        :param enc: encoder activation sequence, float32[batch_size, ninp, enc_size]
        :param dec: single decoder state used as "query", float32[batch_size, dec_size]
        :param inp_mask: mask on enc activatons (0 after first eos), float32 [batch_size, ninp]
        :returns: attn[batch_size, enc_size], probs[batch_size, ninp]
            - attn - attention response vector (weighted sum of enc)
            - probs - attention weights after softmax
        """
        # Compute logits
        logits = self.linear_o(self.activ(self.linear_e(enc) + self.linear_d(dec.reshape((dec.shape[0], 1, -1)))))

        # Apply mask - if mask is 0, logits should be -inf or -1e9
        # You may need torch.where
        logits = logits.reshape((logits.shape[0], logits.shape[1]))
        logits[~inp_mask] = -1e9

        # Compute attention probabilities (softmax)
        probs = self.softmax(logits)

        # Compute attention response using enc and probs
        attn = torch.sum(enc * probs.reshape(probs.shape[0], -1, 1), dim=1)

        return attn, probs


class State:
    def __init__(self, dec_state, attn_probs, enc_seq, mask):
        self.dec_states = [dec_state]
        self.attn_probs = attn_probs
        self.enc_seq = enc_seq
        self.mask = mask


class AttentiveModel(nn.Module):
    def __init__(self, n_tokens, enc_size, dec_size, emb_size=128, attn_size=128, device='cuda'):
        """ Translation model that uses attention. See instructions above. """
        super().__init__()
        self.n_tokens = n_tokens  # output size
        self.device = device

        self.emb_out = nn.Embedding(n_tokens, emb_size)

        self.attn = AttentionLayer('attention', enc_size, enc_size, attn_size)

        self.dec_start = nn.Linear(dec_size, enc_size)
        self.dec0 = nn.GRUCell(emb_size, enc_size)
        self.logits = nn.Linear(enc_size, n_tokens)
        self.state = None

    def encode(self, enc_seq, last_state, mask):
        """
        Takes symbolic input sequence, computes initial state
        :param inp: matrix of input tokens [batch, time]
        :return: a list of initial decoder state tensors
        """
        # encode input sequence, create initial decoder states
        dec_start = self.dec_start(last_state)

        # apply attention layer from initial decoder hidden state
        first_attn_result, first_attn_probs = self.attn(enc_seq, dec_start, mask)
        self.state = State(first_attn_result + dec_start, [first_attn_probs], enc_seq, mask)

    def forward(self, h_0, enc_seq, input):
        mask = (enc_seq == torch.zeros(enc_seq.shape[-1]).to(self.device))[:, :, 0]
        self.encode(enc_seq, h_0, mask)
        return self.decode(input)

    def decode_step(self, prev_tokens, **flags):
        """
        Takes previous decoder state and tokens, returns new state and logits for next tokens
        :param prev_state: a list of previous decoder state tensors
        :param prev_tokens: previous output tokens, an int vector of [batch_size]
        :return: a list of next decoder state tensors, a tensor of logits [batch, n_tokens]
        """
        prev_gru0_state = self.state.dec_states[-1]
        prev_emb = self.emb_out(prev_tokens)  # embedding of the true pipelines
        new_dec_state = self.dec0(prev_emb, prev_gru0_state)
        output_logits = self.logits(new_dec_state)

        attn_result, attn_probs = self.attn(self.state.enc_seq, new_dec_state, self.state.mask)
        self.state.dec_states.append(new_dec_state + attn_result)
        self.state.attn_probs.append(attn_probs)

        return output_logits

    def decode(self, out_tokens, **flags):
        """ Iterate over reference tokens (out_tokens) with decode_step """
        logits_sequence = []
        for i in range(out_tokens.shape[1]):
            logits = self.decode_step(out_tokens[:, i])
            logits_sequence.append(logits)
        return torch.stack(logits_sequence, dim=1)

    @torch.inference_mode()
    def generate(self, h_0, graph=None, temperature=1.0, max_len=100, return_tokens=False, return_states=False):
        """ Generate pipelines with temperature """
        assert graph is not None or return_tokens or return_states

        initial_state, enc_seq = h_0[0][None, :].to(self.device), h_0[1][None, :, :].to(self.device)
        self.encode(enc_seq, initial_state, torch.full([1, enc_seq.shape[1]], True).to(self.device))

        prefix = [SOS_ID]

        for i in range(max_len):
            logits = self.decode_step(torch.as_tensor([prefix[-1]], dtype=torch.int64).to(self.device))
            probs = torch.softmax(logits, dim=-1).cpu().detach().numpy()[0, :]

            if temperature == 0:
                next_token = np.argmax(probs)
            else:
                probs = np.array([p ** (1. / temperature) for p in probs])
                probs /= sum(probs)
                next_token = np.random.choice(range(self.n_tokens), p=probs)

            prefix.append(next_token)
            if next_token == EOS_ID:
                break

        if return_tokens:
            return prefix
        elif return_states:
            return self.state.attn_probs
        else:
            classes = []
            for token in prefix[1:]:
                if token == EOS_ID:
                    break
                elif token == SOS_ID:
                    classes.append('sos')
                elif token == PAD_ID:
                    classes.append('pad')
                else:
                    classes.append(graph.loc[token, 'graph_vertex_subclass'])
            return ' '.join(classes)
