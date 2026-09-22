"""Model tests: padding invariance (section 14 item 9), chronological sensitivity (item 10),
dedup path == per-hand path, masked softmax, torch g map == numpy g map."""
import numpy as np
import torch

from leduc_decision_repr.game.leduc_tree import get_tree
from leduc_decision_repr.game.sequence_form import get_sequence_form
from leduc_decision_repr.game.symmetry import get_symmetry
from leduc_decision_repr.game.policy_utils import RankPolicyToG, rank_infoset_features
from leduc_decision_repr.data.tokenizer import get_token_table
from leduc_decision_repr.models.encoder import OpponentEncoder
from leduc_decision_repr.models.heads import ReconstructionHead, DecisionHead
from leduc_decision_repr.models.torch_g import TorchRankPolicyToG

T = get_tree(); S = get_sequence_form(); SYM = get_symmetry(); TAB = get_token_table()
torch.manual_seed(0)
ENC = OpponentEncoder(TAB.cat, TAB.num, TAB.length).eval()
RNG = np.random.default_rng(0)


def test_padding_invariance_hands_and_events():
    ids = torch.as_tensor(RNG.integers(0, TAB.n_types, size=(3, 20)))
    with torch.no_grad():
        z = ENC(ids)
        # pad to 50 hand slots with arbitrary ids but mask=False
        pad_ids = torch.cat([ids, torch.as_tensor(RNG.integers(0, TAB.n_types, size=(3, 30)))], 1)
        mask = torch.cat([torch.ones(3, 20, dtype=torch.bool), torch.zeros(3, 30, dtype=torch.bool)], 1)
        z_pad = ENC(pad_ids, mask)
        assert torch.allclose(z, z_pad, atol=1e-5)
        # padded events inside a hand: corrupt token fields beyond each hand's length
        enc2 = OpponentEncoder(TAB.cat, TAB.num, TAB.length).eval()
        enc2.load_state_dict(ENC.state_dict())
        cat = enc2.type_cat.clone(); num = enc2.type_num.clone()
        for t in range(TAB.n_types):
            cat[t, TAB.length[t]:] = 2; num[t, TAB.length[t]:] = 0.7
        enc2.type_cat.copy_(cat); enc2.type_num.copy_(num)
        assert torch.allclose(z, enc2(ids), atol=1e-5)


def test_dedup_path_equals_direct_path():
    ids = torch.as_tensor(RNG.integers(0, TAB.n_types, size=(4, 37)))
    with torch.no_grad():
        assert torch.allclose(ENC(ids), ENC.encode_hands_direct(ids), atol=1e-5)


def test_chronological_sensitivity():
    ids = torch.as_tensor(RNG.integers(0, TAB.n_types, size=(8, 50)))
    perm = torch.as_tensor(RNG.permutation(50))
    with torch.no_grad():
        z = ENC(ids); zp = ENC(ids[:, perm])
    assert (z - zp).abs().max() > 1e-4
    # and different content changes z
    ids2 = ids.clone(); ids2[:, 0] = (ids2[:, 0] + 1) % TAB.n_types
    with torch.no_grad():
        assert (ENC(ids2) - z).abs().max() > 1e-6


def test_reconstruction_head_masked_softmax_and_loss():
    feat = rank_infoset_features(T, SYM, 1)
    head = ReconstructionHead(feat, SYM.rank_legal_mask[1])
    z = torch.randn(5, 128)
    q = head(z)
    assert q.shape == (5, 144, 3)
    assert torch.allclose(q.sum(-1), torch.ones(5, 144), atol=1e-6)
    assert torch.all(q[:, ~torch.as_tensor(SYM.rank_legal_mask[1])] == 0)
    q_true = torch.as_tensor(np.stack([RNG.dirichlet(np.ones(3), size=144) for _ in range(5)]), dtype=torch.float32)
    q_true = q_true * head.mask[None]; q_true = q_true / q_true.sum(-1, keepdim=True)
    loss = head.loss(z, q_true)
    assert torch.isfinite(loss) and loss > 0


def test_decision_head_fixed_dims_and_torch_g_map():
    r2g = RankPolicyToG(T, S, SYM)
    qs = np.stack([RNG.dirichlet(np.ones(3), size=144) * SYM.rank_legal_mask[1] for _ in range(6)])
    qs = qs / qs.sum(-1, keepdims=True)
    G = r2g.g(qs)
    tg = TorchRankPolicyToG(r2g)
    Gt = tg(torch.as_tensor(qs, dtype=torch.float32)).numpy()
    assert np.abs(Gt - G).max() < 1e-4
    mean, std = G.mean(0), G.std(0)
    valid = std > 1e-3 * std.max()
    head = DecisionHead(mean, std, valid)
    z = torch.randn(6, 128)
    g_hat = head(z).detach().numpy()
    assert np.allclose(g_hat[:, ~valid], mean[~valid])
    assert torch.isfinite(head.loss(z, torch.as_tensor(G, dtype=torch.float32)))
