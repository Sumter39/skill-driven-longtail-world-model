from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from skilldrive.data import build_cvae_schema, cvae_schema_fingerprint
from skilldrive.generation.inference import (
    file_sha256,
    load_active_cvae,
    load_configured_cvae,
    load_repair_cvae,
    standard_normal_from_seeds,
    validate_active_checkpoint_promotion,
)
from skilldrive.generation.config import ActiveCheckpointConfig
from skilldrive.generation.contracts import canonical_sha256
from skilldrive.models import ConditionalCVAE
from skilldrive.training.checkpoint import TrainingProgress, save_checkpoint


CANDIDATE_EPOCH = 45
CANDIDATE_GLOBAL_STEP = 4095


def _model_kwargs() -> dict[str, int | float]:
    schema = build_cvae_schema()
    return {
        "actor_feature_dim": 6,
        "map_feature_dim": 4,
        "num_actor_types": len(schema.actor_type_vocabulary.tokens),
        "num_actor_roles": len(schema.role_vocabulary.tokens),
        "num_map_types": len(schema.map_type_vocabulary.tokens),
        "num_skills": len(schema.skill_vocabulary.tokens),
        "parameter_dim": schema.parameter_schema.dimension,
        "actor_type_embedding_dim": 4,
        "actor_role_embedding_dim": 4,
        "history_hidden_dim": 8,
        "map_type_embedding_dim": 4,
        "map_hidden_dim": 8,
        "interaction_hidden_dim": 8,
        "interaction_layers": 1,
        "interaction_heads": 2,
        "skill_embedding_dim": 4,
        "parameter_hidden_dim": 4,
        "latent_dim": 4,
        "decoder_hidden_dim": 8,
        "future_steps": 60,
        "dropout": 0.2,
    }


def _checkpoint_and_manifest(
    tmp_path: Path,
    *,
    stage: str = "formal",
    repair_contract: str | None = None,
) -> tuple[Path, Path]:
    kwargs = _model_kwargs()
    model = ConditionalCVAE(**kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    fingerprints = {"contract": "test"}
    checkpoint = tmp_path / "best.pt"
    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        progress=TrainingProgress(
            epoch=1,
            global_step=2,
            next_batch_index=0,
            best_metric=None,
            best_epoch=None,
        ),
        fingerprints=fingerprints,
    )
    manifest = tmp_path / "run_manifest.json"
    value = {
        "stage": stage,
        "fingerprints": fingerprints,
        "model": kwargs,
    }
    if repair_contract is not None:
        value["repair_contract"] = repair_contract
    manifest.write_text(
        json.dumps(value, sort_keys=True),
        encoding="utf-8",
    )
    return checkpoint, manifest


def _baseline_active_checkpoint(
    tmp_path: Path,
    *,
    stage: str = "formal",
    repair_contract: str | None = None,
) -> ActiveCheckpointConfig:
    checkpoint, manifest = _checkpoint_and_manifest(
        tmp_path,
        stage=stage,
        repair_contract=repair_contract,
    )
    return ActiveCheckpointConfig(
        path=checkpoint,
        sha256=file_sha256(checkpoint),
        run_manifest=manifest,
        run_manifest_sha256=file_sha256(manifest),
        schema_sha256=cvae_schema_fingerprint(build_cvae_schema()),
    )


def _promoted_repair_active_checkpoint(
    tmp_path: Path,
) -> tuple[ActiveCheckpointConfig, Path, Path]:
    kwargs = _model_kwargs()
    fingerprints = {"contract": "test-repair-formal"}
    epoch_root = tmp_path / "epoch_candidates"
    epoch_root.mkdir(parents=True)
    manifest = tmp_path / "run_manifest.json"
    manifest_value = {
        "stage": "repair-formal",
        "repair_contract": "cvae_generation_repair_v1",
        "schema_sha256": cvae_schema_fingerprint(build_cvae_schema()),
        "fingerprints": fingerprints,
        "model": kwargs,
        "formal_selection": {
            "active_checkpoint_gate": (
                "heldout_generation_capability_gate_required"
            ),
            "epoch_candidate_directory": epoch_root.as_posix(),
        },
    }
    manifest.write_text(json.dumps(manifest_value, sort_keys=True), encoding="utf-8")
    checkpoint = epoch_root / "epoch-0045-step-00004095.pt"
    model = ConditionalCVAE(**kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        progress=TrainingProgress(
            epoch=CANDIDATE_EPOCH,
            global_step=CANDIDATE_GLOBAL_STEP,
            next_batch_index=0,
            best_metric=None,
            best_epoch=None,
        ),
        fingerprints=fingerprints,
        extra={
            "run_manifest_sha256": canonical_sha256(manifest_value),
            "checkpoint": {
                "role": "epoch_validation_candidate",
                "active_checkpoint": False,
                "candidate_epoch": CANDIDATE_EPOCH,
                "selection_status": "unpromoted_epoch_candidate",
                "active_checkpoint_gate": (
                    "heldout_generation_capability_gate_required"
                ),
            },
        },
    )
    checkpoint_sha = file_sha256(checkpoint)
    manifest_sha = file_sha256(manifest)
    rebind = tmp_path / "rebind.json"
    rebind.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "repair_heldout_rebind_contract",
                "contract": "repair_heldout_gate_v1",
                "status": "rebound_pending_execution",
                "checkpoint": {
                    "path": checkpoint.as_posix(),
                    "sha256": checkpoint_sha,
                    "candidate_epoch": CANDIDATE_EPOCH,
                    "global_step": CANDIDATE_GLOBAL_STEP,
                },
                "run_manifest": {
                    "path": manifest.as_posix(),
                    "sha256": manifest_sha,
                    "stage": "repair-formal",
                    "repair_contract": "cvae_generation_repair_v1",
                },
                "formal_active": False,
                "active_config_modified": False,
                "validation_manifests_opened": False,
                "final_validation_accessed": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    heldout = tmp_path / "heldout.json"
    heldout.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "repair_heldout_gate_summary",
                "status": "passed",
                "checkpoint_sha256": checkpoint_sha,
                "run_manifest_sha256": manifest_sha,
                "candidate_epoch": CANDIDATE_EPOCH,
                "ability_gates": {
                    "complete": True,
                    "conditioned_exceeds_control": True,
                },
                "validation_manifests_opened": False,
                "final_validation_accessed": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    repair_dev = tmp_path / "repair-dev.json"
    repair_dev.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "repair_dev_candidate_gate",
                "status": "passed",
                "checkpoint_sha256": checkpoint_sha,
                "run_manifest_sha256": manifest_sha,
                "candidate_epoch": CANDIDATE_EPOCH,
                "source_partition": "repair_dev_from_formal_train",
                "formal_training_complete": True,
                "gates": [{"name": "repair_dev", "passed": True}],
                "validation_manifests_opened": False,
                "final_validation_accessed": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    promotion = tmp_path / "promotion.json"
    promotion.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "repair_checkpoint_promotion_recommendation",
                "contract": "repair_heldout_gate_v1",
                "status": "completed",
                "recommendation": "recommend_promotion",
                "failure_reasons": [],
                "formal_active": False,
                "active_config_modified": False,
                "requires_separate_active_config_update": True,
                "checkpoint": {
                    "path": checkpoint.as_posix(),
                    "sha256": checkpoint_sha,
                    "candidate_epoch": CANDIDATE_EPOCH,
                    "global_step": CANDIDATE_GLOBAL_STEP,
                },
                "run_manifest": {
                    "path": manifest.as_posix(),
                    "sha256": manifest_sha,
                    "stage": "repair-formal",
                    "repair_contract": "cvae_generation_repair_v1",
                },
                "evidence": {
                    "rebind_contract": {
                        "path": rebind.as_posix(),
                        "sha256": file_sha256(rebind),
                    },
                    "heldout_gate_summary": {
                        "path": heldout.as_posix(),
                        "sha256": file_sha256(heldout),
                        "status": "passed",
                    },
                    "repair_dev_candidate_gate": {
                        "path": repair_dev.as_posix(),
                        "sha256": file_sha256(repair_dev),
                        "status": "passed",
                    },
                },
                "validation_manifests_opened": False,
                "final_validation_accessed": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    active = ActiveCheckpointConfig(
        path=checkpoint,
        sha256=checkpoint_sha,
        run_manifest=manifest,
        run_manifest_sha256=manifest_sha,
        schema_sha256=cvae_schema_fingerprint(build_cvae_schema()),
        run_manifest_stage="repair-formal",
        repair_contract="cvae_generation_repair_v1",
        promotion_recommendation=promotion,
        promotion_recommendation_sha256=file_sha256(promotion),
    )
    return active, checkpoint, promotion


def _mutate_promotion_evidence(
    active: ActiveCheckpointConfig,
    promotion_path: Path,
    evidence_name: str,
    mutation: Callable[[dict[str, object]], None],
) -> ActiveCheckpointConfig:
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    descriptor = promotion["evidence"][evidence_name]
    evidence_path = Path(descriptor["path"])
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    mutation(evidence)
    evidence_path.write_text(json.dumps(evidence, sort_keys=True), encoding="utf-8")
    descriptor["sha256"] = file_sha256(evidence_path)
    promotion_path.write_text(json.dumps(promotion, sort_keys=True), encoding="utf-8")
    return replace(
        active,
        promotion_recommendation_sha256=file_sha256(promotion_path),
    )


def test_standard_normal_from_seeds_is_batch_independent() -> None:
    seeds = np.array([[1, 2], [3, 4]], dtype=np.int64)
    together = standard_normal_from_seeds(seeds, latent_dim=4)
    split = np.concatenate(
        [
            standard_normal_from_seeds(seeds[:1], latent_dim=4),
            standard_normal_from_seeds(seeds[1:], latent_dim=4),
        ],
        axis=0,
    )
    np.testing.assert_array_equal(together, split)


def test_load_active_cvae_validates_hashes_and_sets_eval(tmp_path: Path) -> None:
    checkpoint, manifest = _checkpoint_and_manifest(tmp_path)
    schema = build_cvae_schema()

    runtime = load_active_cvae(
        checkpoint_path=checkpoint,
        run_manifest_path=manifest,
        schema=schema,
        expected_checkpoint_sha256=file_sha256(checkpoint),
        expected_run_manifest_sha256=file_sha256(manifest),
        expected_schema_sha256=cvae_schema_fingerprint(schema),
        device="cpu",
    )

    assert runtime.model.training is False
    assert all(not parameter.requires_grad for parameter in runtime.model.parameters())
    assert runtime.progress.global_step == 2
    assert runtime.manifest_stage == "formal"


def test_load_configured_cvae_accepts_legacy_baseline_checkpoint(
    tmp_path: Path,
) -> None:
    active = _baseline_active_checkpoint(tmp_path)

    runtime = load_configured_cvae(
        active_checkpoint=active,
        schema=build_cvae_schema(),
        device="cpu",
        repository_root=tmp_path,
    )

    assert runtime.manifest_stage == "formal"
    assert runtime.checkpoint_sha256 == active.sha256


@pytest.mark.parametrize(
    ("mode", "stage"),
    (("diagnostic-overfit", "repair-overfit"), ("formal", "repair-formal")),
)
def test_load_repair_cvae_enforces_explicit_stage_policy(
    tmp_path: Path,
    mode: str,
    stage: str,
) -> None:
    checkpoint, manifest = _checkpoint_and_manifest(
        tmp_path,
        stage=stage,
        repair_contract="cvae_generation_repair_v1",
    )
    schema = build_cvae_schema()

    runtime = load_repair_cvae(
        checkpoint_path=checkpoint,
        run_manifest_path=manifest,
        schema=schema,
        expected_checkpoint_sha256=file_sha256(checkpoint),
        expected_run_manifest_sha256=file_sha256(manifest),
        expected_schema_sha256=cvae_schema_fingerprint(schema),
        device="cpu",
        checkpoint_mode=mode,
    )

    assert runtime.manifest_stage == stage


def test_load_configured_cvae_accepts_only_promoted_repair_checkpoint(
    tmp_path: Path,
) -> None:
    active, _, _ = _promoted_repair_active_checkpoint(tmp_path)
    schema = build_cvae_schema()

    promotion = validate_active_checkpoint_promotion(
        active,
        repository_root=tmp_path,
    )
    runtime = load_configured_cvae(
        active_checkpoint=active,
        schema=schema,
        device="cpu",
        repository_root=tmp_path,
    )

    assert promotion is not None
    assert promotion["recommendation"] == "recommend_promotion"
    assert runtime.manifest_stage == "repair-formal"
    assert runtime.checkpoint_sha256 == active.sha256


@pytest.mark.parametrize(
    ("configured_stage", "manifest_stage", "manifest_contract"),
    [
        ("formal", "repair-formal", "cvae_generation_repair_v1"),
        ("repair-formal", "formal", None),
    ],
)
def test_configured_load_rejects_manifest_stage_disguise_before_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured_stage: str,
    manifest_stage: str,
    manifest_contract: str | None,
) -> None:
    active = _baseline_active_checkpoint(
        tmp_path,
        stage=manifest_stage,
        repair_contract=manifest_contract,
    )
    if configured_stage == "repair-formal":
        active = replace(
            active,
            run_manifest_stage="repair-formal",
            repair_contract="cvae_generation_repair_v1",
        )
    monkeypatch.setattr(
        "skilldrive.generation.inference.ConditionalCVAE",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("model load reached")),
    )

    with pytest.raises(ValueError, match="stage differs from config"):
        load_configured_cvae(
            active_checkpoint=active,
            schema=build_cvae_schema(),
            device="cpu",
            repository_root=tmp_path,
        )


def test_configured_repair_requires_promotion_before_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active, _, _ = _promoted_repair_active_checkpoint(tmp_path)
    active = replace(
        active,
        promotion_recommendation=None,
        promotion_recommendation_sha256=None,
    )
    monkeypatch.setattr(
        "skilldrive.generation.inference.ConditionalCVAE",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("model load reached")),
    )

    with pytest.raises(ValueError, match="lacks promotion evidence"):
        load_configured_cvae(
            active_checkpoint=active,
            schema=build_cvae_schema(),
            device="cpu",
            repository_root=tmp_path,
        )


def test_configured_repair_rejects_semantic_promotion_failure_before_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active, _, promotion_path = _promoted_repair_active_checkpoint(tmp_path)
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    promotion["recommendation"] = "reject"
    promotion["failure_reasons"] = ["conditioned_acceptance_exceeds_control"]
    promotion_path.write_text(json.dumps(promotion, sort_keys=True), encoding="utf-8")
    active = replace(
        active,
        promotion_recommendation_sha256=file_sha256(promotion_path),
    )
    monkeypatch.setattr(
        "skilldrive.generation.inference.ConditionalCVAE",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("model load reached")),
    )

    with pytest.raises(ValueError, match="did not pass"):
        load_configured_cvae(
            active_checkpoint=active,
            schema=build_cvae_schema(),
            device="cpu",
            repository_root=tmp_path,
        )


def test_configured_repair_rejects_stale_evidence_hash_before_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active, _, promotion_path = _promoted_repair_active_checkpoint(tmp_path)
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    heldout_path = Path(promotion["evidence"]["heldout_gate_summary"]["path"])
    heldout = json.loads(heldout_path.read_text(encoding="utf-8"))
    heldout["status"] = "failed"
    heldout_path.write_text(json.dumps(heldout, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(
        "skilldrive.generation.inference.ConditionalCVAE",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("model load reached")),
    )

    with pytest.raises(ValueError, match="heldout_gate_summary SHA-256 mismatch"):
        load_configured_cvae(
            active_checkpoint=active,
            schema=build_cvae_schema(),
            device="cpu",
            repository_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("evidence_name", "mutation", "message"),
    [
        (
            "heldout_gate_summary",
            lambda value: value.update({"checkpoint_sha256": "0" * 64}),
            "content is invalid",
        ),
        (
            "repair_dev_candidate_gate",
            lambda value: value.update({"run_manifest_sha256": "0" * 64}),
            "content is invalid",
        ),
        (
            "heldout_gate_summary",
            lambda value: value.update({"candidate_epoch": CANDIDATE_EPOCH + 1}),
            "content is invalid",
        ),
        (
            "heldout_gate_summary",
            lambda value: value["ability_gates"].update({"complete": False}),
            "ability gates did not all pass",
        ),
        (
            "repair_dev_candidate_gate",
            lambda value: value["gates"][0].update({"passed": False}),
            "Repair Dev gate is invalid",
        ),
        (
            "rebind_contract",
            lambda value: value.update({"status": "completed"}),
            "rebind contract is invalid",
        ),
        (
            "rebind_contract",
            lambda value: value["checkpoint"].update({"sha256": "0" * 64}),
            "rebind contract is invalid",
        ),
    ],
)
def test_configured_repair_rejects_evidence_drift_before_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evidence_name: str,
    mutation: Callable[[dict[str, object]], None],
    message: str,
) -> None:
    active, _, promotion_path = _promoted_repair_active_checkpoint(tmp_path)
    active = _mutate_promotion_evidence(
        active,
        promotion_path,
        evidence_name,
        mutation,
    )
    monkeypatch.setattr(
        "skilldrive.generation.inference.ConditionalCVAE",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("model load reached")),
    )

    with pytest.raises(ValueError, match=message):
        load_configured_cvae(
            active_checkpoint=active,
            schema=build_cvae_schema(),
            device="cpu",
            repository_root=tmp_path,
        )


def test_configured_repair_rejects_non_candidate_checkpoint_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active, checkpoint_path, promotion_path = _promoted_repair_active_checkpoint(
        tmp_path
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload["extra"]["checkpoint"]["role"] = "formal_best"
    torch.save(payload, checkpoint_path)
    checkpoint_sha = file_sha256(checkpoint_path)
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    promotion["checkpoint"]["sha256"] = checkpoint_sha
    promotion_path.write_text(json.dumps(promotion, sort_keys=True), encoding="utf-8")
    active = replace(
        active,
        sha256=checkpoint_sha,
        promotion_recommendation_sha256=file_sha256(promotion_path),
    )
    monkeypatch.setattr(
        "skilldrive.generation.inference.ConditionalCVAE",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("model load reached")),
    )

    with pytest.raises(ValueError, match="epoch candidate metadata mismatch"):
        load_configured_cvae(
            active_checkpoint=active,
            schema=build_cvae_schema(),
            device="cpu",
            repository_root=tmp_path,
        )


def test_formal_repair_loader_rejects_overfit_before_model_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, manifest = _checkpoint_and_manifest(
        tmp_path,
        stage="repair-overfit",
        repair_contract="cvae_generation_repair_v1",
    )
    schema = build_cvae_schema()
    monkeypatch.setattr(
        "skilldrive.generation.inference.ConditionalCVAE",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("model load reached")),
    )

    with pytest.raises(ValueError, match="repair-formal"):
        load_repair_cvae(
            checkpoint_path=checkpoint,
            run_manifest_path=manifest,
            schema=schema,
            expected_checkpoint_sha256=file_sha256(checkpoint),
            expected_run_manifest_sha256=file_sha256(manifest),
            expected_schema_sha256=cvae_schema_fingerprint(schema),
            device="cpu",
            checkpoint_mode="formal",
        )


def test_repair_loader_rejects_missing_repair_contract(tmp_path: Path) -> None:
    checkpoint, manifest = _checkpoint_and_manifest(
        tmp_path,
        stage="repair-overfit",
    )
    schema = build_cvae_schema()
    with pytest.raises(ValueError, match="contract mismatch"):
        load_repair_cvae(
            checkpoint_path=checkpoint,
            run_manifest_path=manifest,
            schema=schema,
            expected_checkpoint_sha256=file_sha256(checkpoint),
            expected_run_manifest_sha256=file_sha256(manifest),
            expected_schema_sha256=cvae_schema_fingerprint(schema),
            device="cpu",
            checkpoint_mode="diagnostic-overfit",
        )


def test_load_active_cvae_rejects_checkpoint_hash_mismatch(tmp_path: Path) -> None:
    checkpoint, manifest = _checkpoint_and_manifest(tmp_path)
    schema = build_cvae_schema()
    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        load_active_cvae(
            checkpoint_path=checkpoint,
            run_manifest_path=manifest,
            schema=schema,
            expected_checkpoint_sha256="0" * 64,
            expected_run_manifest_sha256=file_sha256(manifest),
            expected_schema_sha256=cvae_schema_fingerprint(schema),
            device="cpu",
        )


def test_load_active_cvae_rejects_run_manifest_hash_mismatch(tmp_path: Path) -> None:
    checkpoint, manifest = _checkpoint_and_manifest(tmp_path)
    schema = build_cvae_schema()
    with pytest.raises(ValueError, match="run manifest SHA-256 mismatch"):
        load_active_cvae(
            checkpoint_path=checkpoint,
            run_manifest_path=manifest,
            schema=schema,
            expected_checkpoint_sha256=file_sha256(checkpoint),
            expected_run_manifest_sha256="0" * 64,
            expected_schema_sha256=cvae_schema_fingerprint(schema),
            device="cpu",
        )
