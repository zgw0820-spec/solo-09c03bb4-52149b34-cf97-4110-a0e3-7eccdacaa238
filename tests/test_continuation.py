"""Continuation (derived retraining) tests.

Covers:
* the raw-scale transfer maths, including constant source columns;
* the API contract (derivation, lineage, refusal cases);
* that a derived job trains from exactly its transferred initial weights and
  starts with zero momentum;
* that an interrupted derived job finishes bit-identically to an uninterrupted
  one, and that the source job is untouched.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import lease as lm
from app import store
from app.config import Settings
from app.db import init_db, transaction
from app.features import (
    InputError,
    normalise,
    normalise_samples,
    rescale_weights,
)
from app.main import app
from app.model import sgd_step


def make_settings(tmp_path, **overrides) -> Settings:
    base = dict(
        db_path=tmp_path / "training.db",
        lease_seconds=0.6,
        heartbeat_interval=0.1,
        poll_interval=0.01,
        batch_delay_seconds=0.0,
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture()
def settings(tmp_path):
    s = make_settings(tmp_path)
    s.validate()
    init_db(s.db_path)
    return s


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db = tmp_path / "api.db"
    monkeypatch.setenv("TRAINING_DB_PATH", str(db))
    monkeypatch.setenv("TRAINING_LEASE_SECONDS", "0.6")
    monkeypatch.setenv("TRAINING_HEARTBEAT_INTERVAL", "0.1")

    import app.main as main_module

    monkeypatch.setattr(main_module, "settings", Settings.from_env())
    with TestClient(app) as test_client:
        yield test_client


def sample(records: int = 16, seed: int = 0, columns: int = 3):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(records, columns))
    labels = [int(i) % 2 for i in range(records)]
    return features.tolist(), labels


def raw_logits(
    weights: np.ndarray, features: list[list[float]], stats=None
) -> np.ndarray:
    """Model logits on raw records, optionally standardised by ``stats``."""
    matrix = np.asarray(features, dtype=np.float64)
    if stats is not None:
        matrix = (matrix - stats.mean) / np.where(stats.constant, 1.0, stats.std)
        matrix[:, stats.constant] = 0.0
    design = np.concatenate(
        [matrix, np.ones((matrix.shape[0], 1), dtype=np.float64)], axis=1
    )
    return design @ np.asarray(weights, dtype=np.float64)


def drain(settings, lease, limit=1000):
    steps = []
    while True:
        result = store.advance_job(settings.db_path, lease)
        steps.append(result["step"])
        if result["status"] == "completed":
            return steps
        assert len(steps) < limit


def complete(settings, job_id, worker_id="w"):
    lm.register_worker(settings.db_path, worker_id)
    lease = lm.acquire(settings.db_path, job_id, worker_id, settings.lease_seconds)
    assert lease is not None
    drain(settings, lease)
    lm.release(settings.db_path, lease)


# ---------------------------------------------------------------------------
# Transfer maths
# ---------------------------------------------------------------------------


def test_transfer_preserves_raw_predictions_between_samples():
    source_features, source_labels = sample(20, seed=1)
    target_features, target_labels = sample(24, seed=2)

    design_source, targets_source, stats_source = normalise_samples(
        source_features, source_labels
    )
    design_target, _, stats_target = normalise_samples(target_features, target_labels)

    source_weights = np.linspace(-0.4, 0.9, design_source.shape[1])
    transferred = rescale_weights(source_weights, stats_source, stats_target)

    # The source logit on a raw record computed through its own standardised
    # design equals the transferred logit through the target standardisation.
    source_logits = design_source @ source_weights
    raw = np.asarray(source_features, dtype=np.float64)
    target_style = (
        (raw - stats_target.mean) / stats_target.std
    ) @ transferred[:-1] + transferred[-1]
    assert np.allclose(source_logits, target_style, rtol=0, atol=1e-12)

    # ... and on records from the *new* sample as well.
    raw_target = np.asarray(target_features, dtype=np.float64)
    raw_space_source = (
        (raw_target - stats_source.mean) / stats_source.std
    ) @ source_weights[:-1] + source_weights[-1]
    raw_space_target = (
        (raw_target - stats_target.mean) / stats_target.std
    ) @ transferred[:-1] + transferred[-1]
    assert np.allclose(raw_space_source, raw_space_target, rtol=0, atol=1e-12)

    # Interval predictions on the new sample also agree.
    design_target_logits = design_target @ transferred
    assert np.allclose(
        raw_space_source, design_target_logits, rtol=0, atol=1e-12
    )


def test_transfer_with_constant_source_column_keeps_bias():
    features, labels = sample(16, seed=3)
    for row in features:
        row[1] = 5.0  # constant column in the source sample
    design, _, stats = normalise_samples(features, labels)
    assert stats.constant.tolist() == [False, True, False]

    weights = np.array([0.3, 0.0, -0.7, 1.1])
    target_features, target_labels = sample(12, seed=4)
    _, _, stats_t = normalise_samples(target_features, target_labels)
    design_t, _ = normalise(target_features, target_labels)
    transferred = rescale_weights(weights, stats, stats_t)

    assert transferred[1] == 0.0  # constant source column stays zero
    # For any raw record both scorings agree - here, the new sample's records.
    raw = np.asarray(target_features, dtype=np.float64)
    source_logits = (
        (raw - stats.mean) / np.where(stats.constant, 1.0, stats.std)
    ) @ weights[:-1] + weights[-1]
    target_logits = design_t @ transferred
    assert np.allclose(source_logits, target_logits, rtol=0, atol=1e-12)


def test_transfer_is_exact_identity_for_identical_samples():
    features, labels = sample(16, seed=31)
    _, _, stats = normalise_samples(features, labels)
    weights = np.array([0.11, -0.27, 0.83, -1.4])
    assert rescale_weights(weights, stats, stats).tolist() == weights.tolist()


def test_transfer_folds_constant_target_column_into_bias():
    """A constant column in the new sample cannot carry a term.

    Every new record has the same value there, so the source contribution is a
    constant and is folded into the bias: the transfer weight stays zero while
    predictions on the new sample's raw measurements are unchanged.
    """
    source_features, source_labels = sample(16, seed=5)
    _, _, stats_source = normalise_samples(source_features, source_labels)
    target_features, target_labels = sample(12, seed=6)
    for row in target_features:
        row[2] = 1.5
    design_t, _, stats_target = normalise_samples(target_features, target_labels)
    assert stats_target.constant.tolist() == [False, False, True]

    weights = np.array([0.2, -0.5, 0.4, 0.1])
    transferred = rescale_weights(weights, stats_source, stats_target)
    assert transferred[2] == 0.0

    raw = np.asarray(target_features, dtype=np.float64)
    source_logits = (
        (raw - stats_source.mean) / stats_source.std
    ) @ weights[:-1] + weights[-1]
    assert np.allclose(design_t @ transferred, source_logits, rtol=0, atol=1e-12)


def test_transfer_with_varying_target_column_after_constant_source_column():
    """Source column constant but the new sample varies: weight stays zero."""
    source_features, source_labels = sample(16, seed=35)
    for row in source_features:
        row[1] = 5.0
    _, _, stats_source = normalise_samples(source_features, source_labels)
    assert stats_source.constant.tolist() == [False, True, False]

    target_features, target_labels = sample(12, seed=36)
    design_t, _, stats_target = normalise_samples(target_features, target_labels)
    assert stats_target.constant.tolist() == [False, False, False]

    weights = np.array([0.3, 0.0, -0.7, 1.1])
    transferred = rescale_weights(weights, stats_source, stats_target)
    assert transferred[1] == 0.0

    # The source model never looked at column 1, so both models agree on the
    # new sample even though that column now varies.
    raw = np.asarray(target_features, dtype=np.float64)
    source_logits = (
        (raw - stats_source.mean) / np.where(stats_source.constant, 1.0, stats_source.std)
    ) @ weights[:-1] + weights[-1]
    assert np.allclose(design_t @ transferred, source_logits, rtol=0, atol=1e-12)


def test_transfer_refuses_dimension_mismatch():
    _, _, stats_a = normalise_samples(*sample(10, seed=7, columns=2))
    _, _, stats_b = normalise_samples(*sample(10, seed=8, columns=3))
    with pytest.raises(InputError):
        rescale_weights(np.zeros(3), stats_a, stats_b)


def test_derived_job_with_constant_target_column_preserves_predictions(settings):
    """A derived job whose new sample has a constant column still starts equal."""
    source_features, source_labels = sample(20, seed=32)
    target_features, target_labels = sample(16, seed=33)
    for row in target_features:
        row[1] = -3.0  # constant column in the new sample
    source = store.create_job(settings.db_path, source_features, source_labels, 2)
    complete(settings, source["job_id"])
    source = store.get_job(settings.db_path, source["job_id"])

    derived = store.create_continuation_job(
        settings.db_path, source["job_id"], target_features, target_labels, 2
    )
    assert derived["column_stats"]["constant"] == [False, True, False]
    assert derived["weights"][1] == 0.0
    assert derived["initial_weights"] == derived["weights"]
    assert derived["momentum_velocity"] == [0.0] * derived["feature_dim"]

    _, _, stats_source = normalise_samples(source_features, source_labels)
    _, _, stats_target = normalise_samples(target_features, target_labels)
    assert np.allclose(
        raw_logits(derived["weights"], target_features, stats_target),
        raw_logits(source["final_weights"], target_features, stats_source),
        rtol=0,
        atol=1e-12,
    )
    complete(settings, derived["job_id"])
    assert store.get_job(settings.db_path, derived["job_id"])["status"] == "completed"


def test_derived_training_starts_from_transferred_weights_and_zero_momentum(settings):
    source_features, source_labels = sample(20, seed=9)
    target_features, target_labels = sample(16, seed=10)

    source = store.create_job(settings.db_path, source_features, source_labels, 2)
    complete(settings, source["job_id"])
    source = store.get_job(settings.db_path, source["job_id"])

    derived = store.create_continuation_job(
        settings.db_path, source["job_id"], target_features, target_labels, 2
    )
    assert derived["status"] == "queued"
    assert derived["completed_steps"] == 0
    assert derived["source_job_id"] == source["job_id"]
    assert derived["initial_weights_source"] == "scaled_from_source"
    assert derived["initial_weights"] == derived["weights"]
    assert derived["weights"] != source["final_weights"]
    assert derived["momentum_velocity"] == [0.0] * derived["feature_dim"]

    # The queued weights already reproduce the source model on the new sample's
    # raw measurements.
    _, _, source_stats = normalise_samples(source_features, source_labels)
    _, _, target_stats = normalise_samples(target_features, target_labels)
    assert np.allclose(
        raw_logits(derived["weights"], target_features, target_stats),
        raw_logits(source["final_weights"], target_features, source_stats),
        rtol=0,
        atol=1e-12,
    )

    # Training then proceeds from exactly the initial weights vector.
    lm.register_worker(settings.db_path, "w")
    lease = lm.acquire(settings.db_path, derived["job_id"], "w", settings.lease_seconds)
    result = store.advance_job(settings.db_path, lease)
    design, targets = normalise(target_features, target_labels)
    start = design[:8]
    expected_w, expected_v, _ = sgd_step(
        np.asarray(derived["weights"]),
        np.zeros(derived["feature_dim"]),
        start,
        targets[:8],
        derived["learning_rate"],
        derived["momentum"],
    )
    checkpoint = store.list_checkpoints(settings.db_path, derived["job_id"])[0]
    assert checkpoint["weights"] == expected_w.tolist()
    assert np.allclose(checkpoint["momentum_velocity"], expected_v, rtol=0, atol=1e-15)
    assert result["step"] == 1


def test_derived_job_matches_direct_replay(settings):
    """A derived job's whole trajectory equals an independent replay."""
    source_features, source_labels = sample(18, seed=11)
    target_features, target_labels = sample(21, seed=12)
    source = store.create_job(settings.db_path, source_features, source_labels, 2)
    complete(settings, source["job_id"])
    source = store.get_job(settings.db_path, source["job_id"])
    derived = store.create_continuation_job(
        settings.db_path, source["job_id"], target_features, target_labels, 3
    )

    _, _, target_stats = normalise_samples(target_features, target_labels)
    _, _, source_stats = normalise_samples(source_features, source_labels)
    init = rescale_weights(
        np.asarray(source["final_weights"]), source_stats, target_stats
    )
    design, targets = normalise(target_features, target_labels)
    weights, velocity = init.copy(), np.zeros_like(init)
    expected = []
    for _ in range(3):
        for start in range(0, design.shape[0], 8):
            weights, velocity, _ = sgd_step(
                weights, velocity,
                design[start:start + 8], targets[start:start + 8],
                derived["learning_rate"], derived["momentum"],
            )
            expected.append(weights.copy())

    complete(settings, derived["job_id"])
    final = store.get_job(settings.db_path, derived["job_id"])
    checkpoints = store.list_checkpoints(settings.db_path, derived["job_id"])
    assert len(checkpoints) == len(expected)
    for ckpt, want in zip(checkpoints, expected):
        assert np.allclose(ckpt["weights"], want, rtol=0, atol=1e-12)
    assert np.allclose(final["final_weights"], expected[-1], rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# Refusals and source isolation
# ---------------------------------------------------------------------------


def test_refuses_unfinished_source(settings):
    features, labels = sample(16, seed=13)
    source = store.create_job(settings.db_path, features, labels, 2)
    with pytest.raises(store.SourceConflict):
        store.create_continuation_job(settings.db_path, source["job_id"], features, labels, 2)


def test_refuses_missing_source(settings):
    features, labels = sample(16, seed=14)
    with pytest.raises(store.JobNotFound):
        store.create_continuation_job(
            settings.db_path, "nope", features, labels, 2
        )


def test_refuses_dimension_mismatch(settings):
    source_features, source_labels = sample(16, seed=15, columns=3)
    source = store.create_job(settings.db_path, source_features, source_labels, 2)
    complete(settings, source["job_id"])
    other_features, other_labels = sample(16, seed=16, columns=4)
    with pytest.raises(InputError, match="columns"):
        store.create_continuation_job(
            settings.db_path, source["job_id"], other_features, other_labels, 2
        )


def test_refuses_source_without_raw_stats(settings):
    source_features, source_labels = sample(16, seed=17)
    source = store.create_job(settings.db_path, source_features, source_labels, 2)
    complete(settings, source["job_id"])
    with transaction(settings.db_path) as conn:
        conn.execute(
            "UPDATE jobs SET column_mean_json=NULL, column_std_json=NULL, "
            "column_constant_json=NULL WHERE job_id=?",
            (source["job_id"],),
        )
    with pytest.raises(store.SourceConflict, match="no stored raw column statistics"):
        store.create_continuation_job(
            settings.db_path, source["job_id"], source_features, source_labels, 2
        )


def test_refuses_source_without_committed_final_checkpoint(settings):
    source_features, source_labels = sample(16, seed=34)
    source = store.create_job(settings.db_path, source_features, source_labels, 2)
    complete(settings, source["job_id"])
    with transaction(settings.db_path) as conn:
        conn.execute("DELETE FROM checkpoints WHERE job_id=?", (source["job_id"],))
    with pytest.raises(store.SourceConflict, match="final checkpoint"):
        store.create_continuation_job(
            settings.db_path, source["job_id"], source_features, source_labels, 2
        )


def test_all_jobs_persist_raw_stats(settings):
    features, labels = sample(12, seed=18, columns=4)
    features[0] = [features[0][0], features[0][1], 7.0, features[0][3]]
    for row in features:
        row[2] = 7.0
    job = store.create_job(settings.db_path, features, labels, 2)
    matrix = np.asarray(features, dtype=np.float64)
    assert job["column_stats"]["mean"] == pytest.approx(matrix.mean(axis=0).tolist())
    assert job["column_stats"]["std"] == pytest.approx(matrix.std(axis=0).tolist())
    assert job["column_stats"]["constant"] == [False, False, True, False]
    assert job["source_job_id"] is None
    assert job["initial_weights"] == [0.0] * job["feature_dim"]
    assert job["initial_weights_source"] == "zeros"


def test_continuation_leaves_source_untouched(settings):
    source_features, source_labels = sample(20, seed=19)
    target_features, target_labels = sample(16, seed=20)
    source = store.create_job(settings.db_path, source_features, source_labels, 2)
    complete(settings, source["job_id"])
    before = store.get_job(settings.db_path, source["job_id"])
    before_events = store.list_events(settings.db_path, source["job_id"])
    before_checkpoints = store.list_checkpoints(settings.db_path, source["job_id"])

    derived = store.create_continuation_job(
        settings.db_path, source["job_id"], target_features, target_labels, 2
    )
    complete(settings, derived["job_id"])

    after = store.get_job(settings.db_path, source["job_id"])
    assert after == before
    assert store.list_events(settings.db_path, source["job_id"]) == before_events
    assert store.list_checkpoints(settings.db_path, source["job_id"]) == before_checkpoints
    derived_events = store.list_events(settings.db_path, derived["job_id"])
    derived_event = [e for e in derived_events if e["event"] == "job_derived"][0]
    assert derived_event["detail"]["source_job_id"] == source["job_id"]


def test_interrupted_derived_job_resumes_identically(settings):
    source_features, source_labels = sample(24, seed=21)
    target_features, target_labels = sample(30, seed=22)
    source = store.create_job(settings.db_path, source_features, source_labels, 3)
    complete(settings, source["job_id"])

    reference = store.create_continuation_job(
        settings.db_path, source["job_id"], target_features, target_labels, 3
    )
    complete(settings, reference["job_id"], "ref")
    expected = store.get_job(settings.db_path, reference["job_id"])

    derived = store.create_continuation_job(
        settings.db_path, source["job_id"], target_features, target_labels, 3
    )
    lm.register_worker(settings.db_path, "w1")
    lease = lm.acquire(settings.db_path, derived["job_id"], "w1", settings.lease_seconds)
    for _ in range(4):
        store.advance_job(settings.db_path, lease)
    lm.release(settings.db_path, lease)  # process died
    time.sleep(settings.lease_seconds + 0.05)
    resumed = lm.acquire(settings.db_path, derived["job_id"], "w2", settings.lease_seconds)
    assert resumed is not None and resumed.epoch > lease.epoch
    drain(settings, resumed)

    final = store.get_job(settings.db_path, derived["job_id"])
    assert final["status"] == "completed"
    assert final["final_weights"] == expected["final_weights"]
    assert final["momentum_velocity"] == expected["momentum_velocity"]
    assert final["loss"] == expected["loss"]
    assert [c["step"] for c in store.list_checkpoints(settings.db_path, derived["job_id"])] == list(
        range(1, expected["total_steps"] + 1)
    )


def test_chained_continuations(settings):
    features, labels = sample(16, seed=23)
    first = store.create_job(settings.db_path, features, labels, 2)
    complete(settings, first["job_id"])
    second = store.create_continuation_job(
        settings.db_path, first["job_id"], features, labels, 2
    )
    complete(settings, second["job_id"])
    second = store.get_job(settings.db_path, second["job_id"])
    third = store.create_continuation_job(
        settings.db_path, second["job_id"], features, labels, 2
    )
    assert third["source_job_id"] == second["job_id"]
    assert third["initial_weights"] == second["final_weights"]
    assert third["initial_weights_source"] == "scaled_from_source"


# ---------------------------------------------------------------------------
# API contract
# ---------------------------------------------------------------------------


def _http_payload(features, labels, epochs=2):
    return {"features": features, "labels": labels, "epochs": epochs}


def _complete_via_api(client, job_id, worker_id):
    settings = Settings.from_env()
    lm.register_worker(settings.db_path, worker_id)
    lease = lm.acquire(settings.db_path, job_id, worker_id, settings.lease_seconds)
    drain(settings, lease)
    lm.release(settings.db_path, lease)


def test_api_creates_continuation(client):
    source_features, source_labels = sample(20, seed=24)
    target_features, target_labels = sample(16, seed=25)
    source = client.post("/jobs", json=_http_payload(source_features, source_labels)).json()
    _complete_via_api(client, source["job_id"], "w1")
    source = client.get(f"/jobs/{source['job_id']}").json()

    response = client.post(
        f"/jobs/{source['job_id']}/continuations",
        json=_http_payload(target_features, target_labels),
    )
    assert response.status_code == 201, response.text
    derived = response.json()
    assert derived["status"] == "queued"
    assert derived["source_job_id"] == source["job_id"]
    assert derived["completed_steps"] == 0
    assert derived["initial_weights"] == derived["weights"]
    assert derived["initial_weights_source"] == "scaled_from_source"
    assert derived["weights"] != source["final_weights"]

    _complete_via_api(client, derived["job_id"], "w2")
    done = client.get(f"/jobs/{derived['job_id']}").json()
    assert done["status"] == "completed"
    assert done["final_weights"] is not None
    events = client.get(f"/jobs/{derived['job_id']}/events").json()
    assert any(e["event"] == "job_derived" for e in events)

    # The original job still reads back exactly as before.
    unchanged = client.get(f"/jobs/{source['job_id']}").json()
    assert unchanged["final_weights"] == source["final_weights"]
    assert unchanged["completed_steps"] == source["total_steps"]


def test_api_rejects_incomplete_source(client):
    features, labels = sample(16, seed=26)
    source = client.post("/jobs", json=_http_payload(features, labels)).json()
    response = client.post(
        f"/jobs/{source['job_id']}/continuations",
        json=_http_payload(features, labels),
    )
    assert response.status_code == 422
    assert "not completed" in response.json()["detail"]


def test_api_404_for_missing_source(client):
    features, labels = sample(16, seed=27)
    response = client.post(
        "/jobs/missing/continuations", json=_http_payload(features, labels)
    )
    assert response.status_code == 404


def test_api_rejects_dimension_mismatch(client):
    source_features, source_labels = sample(16, seed=28, columns=3)
    other_features, other_labels = sample(16, seed=29, columns=2)
    source = client.post("/jobs", json=_http_payload(source_features, source_labels)).json()
    _complete_via_api(client, source["job_id"], "w1")
    response = client.post(
        f"/jobs/{source['job_id']}/continuations",
        json=_http_payload(other_features, other_labels),
    )
    assert response.status_code == 422


def test_api_rejects_source_without_stats(client):
    source_features, source_labels = sample(16, seed=30)
    source = client.post("/jobs", json=_http_payload(source_features, source_labels)).json()
    _complete_via_api(client, source["job_id"], "w1")
    settings = Settings.from_env()
    with transaction(settings.db_path) as conn:
        conn.execute(
            "UPDATE jobs SET column_mean_json=NULL, column_std_json=NULL, "
            "column_constant_json=NULL WHERE job_id=?",
            (source["job_id"],),
        )
    response = client.post(
        f"/jobs/{source['job_id']}/continuations",
        json=_http_payload(source_features, source_labels),
    )
    assert response.status_code == 422
    assert "raw column statistics" in response.json()["detail"]


def test_api_validates_new_sample(client):
    source_features, source_labels = sample(20, seed=31)
    source = client.post("/jobs", json=_http_payload(source_features, source_labels)).json()
    _complete_via_api(client, source["job_id"], "w1")
    too_few = _http_payload(source_features[:7], source_labels[:7])
    assert client.post(
        f"/jobs/{source['job_id']}/continuations", json=too_few
    ).status_code == 422
    bad_epochs = _http_payload(source_features, source_labels, epochs=6)
    assert client.post(
        f"/jobs/{source['job_id']}/continuations", json=bad_epochs
    ).status_code == 422
    mismatched = _http_payload(source_features, source_labels, epochs=2)
    mismatched["source_job_id"] = "other"
    assert client.post(
        f"/jobs/{source['job_id']}/continuations", json=mismatched
    ).status_code == 422


def test_swagger_exposes_continuation_endpoint(client):
    schema = client.get("/openapi.json").json()
    assert "/jobs/{source_job_id}/continuations" in schema["paths"]
    operation = schema["paths"]["/jobs/{source_job_id}/continuations"]["post"]
    assert operation["responses"]["201"]
    assert "ContinuationRequest" in json.dumps(schema["components"]["schemas"])


def test_migration_adds_columns_to_old_database(tmp_path):
    """An existing pre-feature database is upgraded additively on startup."""
    import sqlite3

    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY, status TEXT NOT NULL,
            feature_dim INTEGER NOT NULL, epochs INTEGER NOT NULL,
            batch_size INTEGER NOT NULL, record_count INTEGER NOT NULL,
            total_batches INTEGER NOT NULL, learning_rate REAL NOT NULL,
            momentum REAL NOT NULL, features_json TEXT NOT NULL,
            labels_json TEXT NOT NULL, weights_json TEXT NOT NULL,
            velocity_json TEXT NOT NULL, loss REAL NOT NULL,
            completed_steps INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL, updated_at REAL NOT NULL, finished_at REAL
        );
        """
    )
    conn.close()
    init_db(db)
    conn = sqlite3.connect(db)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
    conn.close()
    assert {"column_mean_json", "column_std_json", "column_constant_json"} <= columns
    assert {
        "source_job_id",
        "initial_weights_json",
        "initial_weights_source",
    } <= columns
