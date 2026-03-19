"""Tests for DP Federated LoRA."""
import torch
import torch.nn as nn
from dp_federated_lora import LoRALayer, PrivacyAccountant
from dp_federated_lora.federated import (
    FederatedConfig, FederatedClient, FederatedServer, run_federation
)


def _make_classifier():
    return nn.Sequential(nn.Linear(16, 32), LoRALayer(32, 4, rank=4, alpha=16.0))


def _data(n=60, seed=0):
    torch.manual_seed(seed)
    return torch.randn(n, 16), torch.randint(0, 4, (n,))


class TestLoRALayer:
    def test_output_shape(self):
        out = LoRALayer(16, 8, rank=4)(torch.randn(5, 16))
        assert out.shape == (5, 8)

    def test_lora_A_shape(self):
        assert LoRALayer(16, 8, rank=4).lora_A.shape == (4, 16)

    def test_lora_B_shape(self):
        assert LoRALayer(16, 8, rank=4).lora_B.shape == (8, 4)

    def test_lora_B_init_zero(self):
        assert torch.all(LoRALayer(16, 8, rank=4).lora_B == 0)

    def test_trainable(self):
        layer = LoRALayer(16, 8, rank=4)
        assert layer.lora_A.requires_grad and layer.lora_B.requires_grad

    def test_gradient_flows(self):
        layer = LoRALayer(8, 4, rank=2)
        layer(torch.randn(3, 8)).sum().backward()
        assert layer.lora_A.grad is not None

    def test_different_ranks(self):
        for r in [1, 4, 8]:
            assert LoRALayer(16, 8, rank=r)(torch.randn(2, 16)).shape == (2, 8)


class TestPrivacyAccountant:
    def test_epsilon_nonneg(self):
        acc = PrivacyAccountant(noise_multiplier=1.0, sample_rate=0.01)
        assert acc.get_epsilon() >= 0.0

    def test_epsilon_increases(self):
        acc = PrivacyAccountant(noise_multiplier=1.0, sample_rate=0.1)
        acc.step(); e1 = acc.get_epsilon()
        acc.step(); e2 = acc.get_epsilon()
        assert e2 >= e1

    def test_higher_noise_lower_epsilon(self):
        a = PrivacyAccountant(noise_multiplier=2.0, sample_rate=0.1)
        b = PrivacyAccountant(noise_multiplier=0.5, sample_rate=0.1)
        for _ in range(10): a.step(); b.step()
        assert a.get_epsilon() < b.get_epsilon()

    def test_get_budget_used(self):
        acc = PrivacyAccountant(noise_multiplier=1.0, sample_rate=0.1)
        acc.step()
        budget = acc.get_budget_used()
        assert "epsilon" in budget and budget["steps"] == 1


class TestFederated:
    cfg = FederatedConfig(n_rounds=2, clients_per_round=2, local_epochs=1,
                          learning_rate=1e-3, noise_multiplier=0.5, batch_size=16)

    def _setup(self, n=3):
        m = _make_classifier()
        srv = FederatedServer(m, self.cfg)
        clients = [FederatedClient(i, m, _data(seed=i), self.cfg) for i in range(n)]
        return m, srv, clients, _data(n=40, seed=99)

    def test_lora_params_nonempty(self):
        _, _, clients, _ = self._setup()
        assert len(clients[0].get_lora_params()) > 0

    def test_train_round_tuple(self):
        _, _, clients, _ = self._setup()
        p, loss, eps = clients[0].train_round()
        assert isinstance(p, dict) and isinstance(loss, float) and eps >= 0

    def test_fedavg_changes_model(self):
        _, srv, clients, _ = self._setup()
        before = {k: v.clone() for k, v in srv.get_global_lora_params().items()}
        updates = []
        for c in clients[:2]:
            p = {k: v + torch.randn_like(v)*0.1 for k, v in c.get_lora_params().items()}
            updates.append(p)
        srv.fedavg(updates)
        after = srv.get_global_lora_params()
        assert any(not torch.allclose(before[k], after[k]) for k in before)

    def test_evaluate_valid(self):
        _, srv, _, val = self._setup()
        acc = srv.evaluate(*val)
        assert 0.0 <= acc <= 1.0

    def test_run_federation(self):
        m, srv, clients, val = self._setup()
        results = run_federation(m, clients, srv, val, self.cfg)
        assert len(results) == self.cfg.n_rounds
        for r in results:
            assert r.epsilon >= 0 and 0.0 <= r.val_accuracy <= 1.0
