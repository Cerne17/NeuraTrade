"""Testes de score_curves (PR-AUC/ROC-AUC) e expected_cost (ADR-0015)."""

import numpy as np
import pytest

from src.evaluate import expected_cost, score_curves


def test_score_curves_separacao_perfeita():
    # escores que separam perfeitamente: PR-AUC e ROC-AUC = 1.
    labels = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
    out = score_curves(scores, labels)
    assert out["pr_auc"] == pytest.approx(1.0)
    assert out["roc_auc"] == pytest.approx(1.0)
    assert out["prevalence"] == pytest.approx(2 / 5)


def test_score_curves_roc_infla_em_classe_rara():
    # 1 positivo entre 100; o positivo é alto mas vários negativos o superam →
    # ROC permanece alta (vence a maioria) mas PR despenca (classe rara).
    rng = np.random.default_rng(0)
    labels = np.zeros(100, dtype=int); labels[0] = 1
    scores = rng.uniform(0, 1, 100); scores[0] = 0.8     # ~20% dos negativos ficam acima
    out = score_curves(scores, labels)
    assert out["prevalence"] == pytest.approx(0.01)
    assert out["roc_auc"] > 0.6                          # ROC parece "boa"
    assert out["pr_auc"] < out["roc_auc"]                # mas PR expõe a fraqueza (rara)
    assert out["pr_lift"] == pytest.approx(out["pr_auc"] / out["prevalence"])


def test_score_curves_uma_classe_retorna_nan():
    out = score_curves(np.array([0.1, 0.2, 0.3]), np.array([0, 0, 0]))
    assert np.isnan(out["pr_auc"]) and np.isnan(out["roc_auc"])


def test_expected_cost_assimetrico():
    labels = np.array([1, 1, 0, 0])
    flags = np.array([1, 0, 1, 0])   # tp=1, fn=1, fp=1, tn=1
    base = expected_cost(flags, labels, cost_fp=1.0, cost_fn=1.0)
    assert (base["tp"], base["fn"], base["fp"], base["tn"]) == (1, 1, 1, 1)
    assert base["cost"] == pytest.approx(2.0)
    # FN 5x mais caro → custo sobe só pela parcela de FN
    pesado = expected_cost(flags, labels, cost_fp=1.0, cost_fn=5.0)
    assert pesado["cost"] == pytest.approx(1.0 + 5.0)
