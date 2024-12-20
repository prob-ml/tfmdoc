import torch
from hydra import compose, initialize
from hydra.utils import instantiate

from tfmdoc.aipw import aipw_estimator
from tfmdoc.datasets import random_mask


def test_bert():
    with initialize(config_path=".."):
        cfg = compose(config_name="config")
        bert = instantiate(cfg.bert, n_tokens=6)
        t = torch.randint(low=20, high=60, size=(4, 32))
        v = torch.randint(high=3, size=(4, 32))
        x = torch.randint(high=6, size=(4, 32))
        y = []
        for i, records in enumerate(x):
            _, labels = random_mask(records, 6, i)
            y.append(torch.tensor(labels))
        y = torch.stack(y)
        assert bert.configure_optimizers().defaults
        assert bert.training_step((t, v, None, x, y), 0) > 0
        assert bert.validation_step((t, v, None, x, y), 0) > 0


def test_model():
    with initialize(config_path=".."):
        cfg = compose(config_name="config")
        model = instantiate(cfg.model, n_tokens=6)
        t = torch.randint(low=20, high=60, size=(4, 32))
        v = torch.randint(high=3, size=(4, 32))
        w = torch.randn(size=(4, 2))
        # fmt: off
        w[:, 1] = (w[:, 1] >= 0)
        # fmt: on
        x = torch.randint(high=6, size=(4, 32))
        y = torch.LongTensor([0, 1, 1, 0])
        assert model.configure_optimizers().defaults
        assert model.training_step((t, v, w, x, y), 0) > 0
        assert model.validation_step((t, v, w, x, y), 0) > 0


def test_bow():
    with initialize(config_path=".."):
        cfg = compose(
            config_name="config",
            overrides=["model.transformer=False", "model.d_model=40"],
        )
        model = instantiate(cfg.model, n_tokens=200)
        t = torch.randint(low=20, high=60, size=(4, 100))
        v = None
        w = torch.randn(size=(4, 2))
        # fmt: off
        w[:, 1] = (w[:, 1] >= 0)
        # fmt: on
        x = torch.randint(high=4, size=(4, 200)).float()
        y = torch.LongTensor([0, 1, 1, 0])
        assert model.configure_optimizers().defaults
        assert model.training_step((t, v, w, x, y), 0) > 0
        assert model.validation_step((t, v, w, x, y), 0) > 0


def test_aipw():
    torch.manual_seed(35)
    # ey_x contains both treatment and control estimates
    y, t, pi_x, ey_x = simulate_causality(1000, 40, 32, 22, 4)
    est_ate = aipw_estimator(y, t, pi_x, ey_x)

    # simulate true ATE empirically over 100 trials
    assigned_treatment = torch.bernoulli(ey_x[:, 1].repeat(100, 1)).mean(1)
    assigned_control = torch.bernoulli(ey_x[:, 0].repeat(100, 1)).mean(1)
    simulated_ates = assigned_treatment - assigned_control
    assert abs(est_ate - simulated_ates.mean()) < 0.05


# helpers


def simulate_causality(n, n_codes, seq_length, treatment_event, outcome_event):
    x = torch.randint(n_codes, (n, seq_length))
    t_mask = (x == treatment_event).any(1)
    prop_score = torch.empty((n,)).fill_(0.8)
    prop_score = prop_score.masked_fill_(t_mask, 0.2)
    treatment = torch.bernoulli(prop_score)
    outcome_mask = (x == outcome_event).any(1)
    outcome_probs = torch.empty((n,)).fill_(0.8)
    outcome_probs.masked_fill_(torch.logical_and(outcome_mask, treatment), 0.2)
    outcome = torch.bernoulli(outcome_probs)

    ey_x0 = torch.empty((n,)).fill_(0.8)
    ey_x1 = torch.empty((n,)).fill_(0.8)
    ey_x0.masked_fill_(
        torch.logical_and(outcome_mask, torch.empty((n,)).fill_(False)), 0.2
    )
    ey_x1.masked_fill_(
        torch.logical_and(outcome_mask, torch.empty((n,)).fill_(True)), 0.2
    )

    estimated_outcomes = torch.cat([ey_x0.unsqueeze(1), ey_x1.unsqueeze(1)], dim=1)

    return outcome, treatment, prop_score, estimated_outcomes
