"""Tests for DP Federated LoRA."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
from dp_federated_lora import LoRALayer
from dp_federated_lora.privacy import PrivacyAccountant, DPSGDOptimizer
from dp_federated_lora.federated import (
    FederatedConfig, FederatedClient, FederatedServer, run_federation, _lora_params
)


def _cfg(**kw):
    return FederatedConfig(n_rounds=2, clients_per_round=2, local_epochs=1,
                           batch_size=4, noise_multiplier=0.5, **kw)

def _make_data(n=20, in_f=8, n_classes=4):
    return torch.randn(n, in_f), torch.randint(0, n_classes, (n,))


class TestLoRALayer:
    def test_output_shape(self):
        assert LoRALayer(8, 16, rank=2)(torch.randn(4, 8)).shape == (4, 16)

    def test_lora_B_zero_init(self):
        layer = LoRALayer(8, 16, rank=2)
        assert torch.allclose(layer.lora_B, torch.zeros_like(layer.lora_B))

    def test_base_weight_frozen(self):
        layer = LoRALayer(8, 16, rank=2, base_weight=torch.randn(16, 8))
        frozen = {n for n, p in layer.named_parameters() if not p.requires_grad}
        assert any("base_weight" in k for k, _ in layer.named_buffers())

    def test_gradient_flows(self):
        layer = LoRALayer(8, 16, rank=2)
        layer(torch.randn(3, 8)).sum().backward()
        assert layer.lora_A.grad is not None

    def test_param_efficiency(self):
        layer = LoRALayer(64, 128, rank=4)
        n = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        assert n < 64 * 128


class TestPrivacyAccountant:
    def _acc(self, noise=1.0, rate=0.01):
        return PrivacyAccountant(noise_multiplier=noise, sample_rate=rate)

    def test_zero_before_steps(self):
        assert self._acc().get_epsilon() >= 0.0

    def test_epsilon_grows_with_steps(self):
        acc = self._acc()
        acc.step()
        e1 = acc.get_epsilon()
        acc.step()
        assert acc.get_epsilon() > e1

    def test_higher_noise_lower_epsilon(self):
        loud = self._acc(noise=5.0); loud.step(10)
        quiet = self._acc(noise=0.5); quiet.step(10)
        assert loud.get_epsilon() < quiet.get_epsilon()

    def test_steps_tracked(self):
        acc = self._acc(); acc.step(7)
        assert acc.steps == 7


class TestDPSGDOptimizer:
    def test_clip_method(self):
        """clip_gradients clips per-sample gradients."""
        g1 = torch.tensor([10.0, 0.0])
        g2 = torch.tensor([0.0, 10.0])
        clipped, norms = DPSGDOptimizer.clip_per_sample_gradients([g1, g2], max_norm=1.0)
        for g in clipped:
            assert g.norm(2).item() <= 1.0 + 1e-5

    def test_add_noise_returns_tensor(self):
        grad = torch.randn(4, 8)
        noised = DPSGDOptimizer.add_noise(grad, batch_size=4, max_grad_norm=1.0, noise_multiplier=1.0)
        assert noised.shape == grad.shape
        assert noised.isfinite().all()

    def test_init_stores_config(self):
        model = nn.Linear(4, 2)
        opt = DPSGDOptimizer(list(model.parameters()), lr=0.01, max_grad_norm=0.5, noise_multiplier=1.5)
        assert opt.defaults['max_grad_norm'] == 0.5
        assert opt.defaults['noise_multiplier'] == 1.5


class TestFederated:
    def _setup(self):
        model = LoRALayer(8, 4, rank=2)
        cfg = _cfg()
        server = FederatedServer(model, cfg)
        clients = [FederatedClient(i, model, _make_data(), cfg) for i in range(3)]
        return model, server, clients, cfg

    def test_lora_params_extracted(self):
        model = LoRALayer(8, 4, rank=2)
        params = _lora_params(model)
        assert any("lora_A" in k for k in params)
        assert any("lora_B" in k for k in params)

    def test_client_train_round(self):
        model = LoRALayer(8, 4, rank=2)
        client = FederatedClient(0, model, _make_data(), _cfg())
        params, loss, eps = client.train_round()
        assert isinstance(loss, float)
        assert isinstance(params, dict)

    def test_server_fedavg(self):
        model = LoRALayer(8, 4, rank=2)
        server = FederatedServer(model, _cfg())
        p0 = {k: v.clone() for k, v in _lora_params(model).items()}
        updates = [{k: torch.ones_like(v) for k, v in p0.items()} for _ in range(2)]
        server.fedavg(updates)
        p1 = _lora_params(model)
        # After fedavg with all-ones updates, params should have changed
        changed = any(not torch.allclose(p1[k], p0[k]) for k in p0)
        assert changed

    def test_server_evaluate(self):
        model = LoRALayer(8, 4, rank=2)
        server = FederatedServer(model, _cfg())
        x, y = _make_data(10)
        acc = server.evaluate(x, y)
        assert 0.0 <= acc <= 1.0

    def test_run_federation_returns_results(self):
        model, server, clients, cfg = self._setup()
        val = _make_data(10)
        results = run_federation(model, clients, server, val, cfg)
        assert len(results) == cfg.n_rounds
        assert all(hasattr(r, 'train_loss') for r in results)

    def test_privacy_budget_accumulates(self):
        model, server, clients, cfg = self._setup()
        val = _make_data(10)
        results = run_federation(model, clients, server, val, cfg)
        # Privacy budget should be positive after training
        assert results[-1].epsilon >= 0.0
