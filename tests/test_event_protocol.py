"""Testes da avaliação por evento (ADR-0005/0015).

Protocolo sintético com janelas sobrepostas espalha cada injeção por
``~window_size`` janelas, inflando a prevalência ao nível de janela. A
avaliação por evento agrupa janelas positivas contíguas num único evento e
conta a detecção no nível de evento — refletindo o cenário raro real.
"""

import numpy as np
import pytest

from src.evaluate import event_metrics, group_events


def test_group_events_runs_contiguos():
    labels = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1])
    assert group_events(labels) == [(1, 3), (5, 6), (7, 10)]


def test_group_events_bordas():
    # evento colado no início e no fim
    assert group_events(np.array([1, 1, 0, 1])) == [(0, 2), (3, 4)]
    assert group_events(np.array([0, 0, 0])) == []
    assert group_events(np.array([1, 1, 1])) == [(0, 3)]


def test_event_metrics_recall_por_evento_nao_por_janela():
    # 1 evento de 4 janelas; o detector pega 1 janela dele → recall de evento = 1.
    labels = np.array([0, 1, 1, 1, 1, 0])
    flags = np.array([0, 0, 1, 0, 0, 0])
    out = event_metrics(flags, labels)
    assert out["n_true_events"] == 1
    assert out["tp_events"] == 1
    assert out["recall"] == pytest.approx(1.0)


def test_event_metrics_evento_perdido_e_falso_alarme():
    # 2 eventos verdadeiros; detector pega 1 e dispara 1 alarme espúrio.
    labels = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    flags = np.array([0, 1, 0, 0, 0, 0, 1, 0])
    out = event_metrics(flags, labels)
    assert out["n_true_events"] == 2
    assert out["n_pred_events"] == 2
    assert out["tp_events"] == 1          # só o 1º evento foi pego
    assert out["recall"] == pytest.approx(0.5)
    assert out["precision"] == pytest.approx(0.5)   # 1 de 2 alarmes acerta
    assert out["f1"] == pytest.approx(0.5)


def test_event_metrics_sem_flags_zera():
    labels = np.array([0, 1, 1, 0])
    flags = np.array([0, 0, 0, 0])
    out = event_metrics(flags, labels)
    assert out["recall"] == 0.0
    assert out["precision"] == 0.0
    assert out["f1"] == 0.0


def test_event_metrics_reduz_prevalencia_vs_janela():
    # Janela: 8 de 10 positivas (prev 0.8). Evento: 1 evento só.
    labels = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 0])
    assert labels.mean() == pytest.approx(0.8)
    assert len(group_events(labels)) == 1
