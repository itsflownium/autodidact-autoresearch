from __future__ import annotations

from autodidact.ledger import WriterRole
from autodidact.records import (
    ArtifactManifest,
    ArtifactRef,
    ArtifactRetention,
    CandidateRecord,
    ComputeRecord,
    DecisionRecord,
    DecisionVerdict,
    DownstreamPrediction,
    ExperimentRecord,
    ExperimentStage,
    LineageRecord,
    PatchProposal,
    ResourceLimits,
    RunArm,
    RunResult,
    RunStatus,
    TrialSpec,
    build_effect_estimate,
    build_paired_result,
)

PARENT_COMMIT = "a" * 40
CANDIDATE_COMMIT = "b" * 40


def digest(character: str) -> str:
    return character * 64


def evidence_records() -> dict[str, ExperimentRecord]:
    proposal = PatchProposal(
        proposal_id="proposal-001",
        parent_commit=PARENT_COMMIT,
        title="Tune warmup",
        hypothesis="A smoother warmup should reduce early optimization error.",
        mechanism="Delay the peak learning rate until gradients stabilize.",
        change="Increase warmup from two to five percent of training tokens.",
        expected_effect_bpb=0.004,
        minimum_useful_gain_bpb=0.001,
        resource_risk="Negligible runtime change.",
        failure_signal="Held-out BPB is unchanged or worse.",
        interaction_risk="May interact with the terminal learning-rate floor.",
    )
    candidate = CandidateRecord(
        candidate_id="candidate-001",
        proposal_id=proposal.proposal_id,
        parent_commit=PARENT_COMMIT,
        candidate_commit=CANDIDATE_COMMIT,
        diff_sha256=digest("1"),
        changed_paths=("train.py",),
        trainer_sha256=digest("2"),
        policy_sha256=digest("3"),
        parameter_count=1_016_960,
    )
    trial = TrialSpec(
        trial_id="trial-001",
        candidate_id=candidate.candidate_id,
        parent_commit=PARENT_COMMIT,
        candidate_commit=CANDIDATE_COMMIT,
        stage=ExperimentStage.CHEAP,
        seed=11,
        token_budget=2_000_000,
        eval_tokens=250_000,
        batch_size=64,
        eval_batch_size=64,
        execution_order=(RunArm.PARENT, RunArm.CANDIDATE),
        data_config_sha256=digest("4"),
        tokenizer_sha256=digest("5"),
        parent_trainer_sha256=digest("6"),
        candidate_trainer_sha256=candidate.trainer_sha256,
        evaluator_sha256=digest("7"),
        runner_sha256=digest("8"),
        environment_sha256=digest("9"),
        order_assignment_sha256=digest("a"),
        device="mps",
        limits=ResourceLimits(
            timeout_seconds=600,
            max_peak_process_rss_bytes=900_000_000,
            max_peak_device_bytes=300_000_000,
            min_training_tokens_per_second=10_000.0,
            max_training_throughput_regression_fraction=0.1,
            max_peak_process_rss_regression_fraction=0.1,
            max_peak_device_regression_fraction=0.2,
        ),
    )
    parent_run = RunResult(
        run_id="run-parent-001",
        trial_id=trial.trial_id,
        arm=RunArm.PARENT,
        status=RunStatus.SUCCEEDED,
        seed=trial.seed,
        target_tokens=trial.token_budget,
        tokens_seen=trial.token_budget,
        evaluation_tokens=trial.eval_tokens or 0,
        parameter_count=1_016_960,
        validation_bpb=1.100,
        mean_train_loss=1.20,
        training_tokens_per_second=20_000.0,
        evaluation_tokens_per_second=12_000.0,
        peak_process_rss_bytes=600_000_000,
        peak_device_allocated_bytes=100_000_000,
        peak_device_reserved_bytes=120_000_000,
        training_seconds=100.0,
        evaluation_seconds=10.0,
        wall_seconds=110.0,
        data_order_sha256=digest("b"),
    )
    candidate_run = RunResult(
        run_id="run-candidate-001",
        trial_id=trial.trial_id,
        arm=RunArm.CANDIDATE,
        status=RunStatus.SUCCEEDED,
        seed=trial.seed,
        target_tokens=trial.token_budget,
        tokens_seen=trial.token_budget,
        evaluation_tokens=trial.eval_tokens or 0,
        parameter_count=candidate.parameter_count,
        validation_bpb=1.095,
        mean_train_loss=1.19,
        training_tokens_per_second=21_000.0,
        evaluation_tokens_per_second=12_500.0,
        peak_process_rss_bytes=620_000_000,
        peak_device_allocated_bytes=110_000_000,
        peak_device_reserved_bytes=130_000_000,
        training_seconds=95.0,
        evaluation_seconds=9.0,
        wall_seconds=104.0,
        data_order_sha256=parent_run.data_order_sha256,
    )

    def manifest(run: RunResult, suffix: str) -> ArtifactManifest:
        return ArtifactManifest(
            manifest_id=f"manifest-{suffix}-001",
            run_id=run.run_id,
            artifacts=(
                ArtifactRef(
                    artifact_id=f"checkpoint-{suffix}-001",
                    kind="checkpoint",
                    relative_path=f"runs/{run.run_id}/checkpoint.pt",
                    sha256=digest("c" if suffix == "parent" else "d"),
                    size_bytes=8_000_000,
                    retention=ArtifactRetention.EPHEMERAL,
                ),
                ArtifactRef(
                    artifact_id=f"metrics-{suffix}-001",
                    kind="metrics",
                    relative_path=f"runs/{run.run_id}/metrics.jsonl",
                    sha256=digest("e" if suffix == "parent" else "f"),
                    size_bytes=4_000,
                    retention=ArtifactRetention.COMPACT,
                ),
            ),
        )

    parent_manifest = manifest(parent_run, "parent")
    candidate_manifest = manifest(candidate_run, "candidate")
    paired = build_paired_result(
        "pair-001",
        trial=trial,
        candidate_id=candidate.candidate_id,
        parent=parent_run,
        candidate=candidate_run,
    )
    effect = build_effect_estimate(
        "estimate-001",
        candidate_id=candidate.candidate_id,
        stage=trial.stage,
        pairs=(paired,),
        minimum_useful_gain_bpb=proposal.minimum_useful_gain_bpb,
        probability_exceeds_minimum=0.97,
        estimator_version="paired-normal-v1",
    )
    prediction = DownstreamPrediction(
        prediction_id="prediction-001",
        candidate_id=candidate.candidate_id,
        source_trial_ids=(trial.trial_id,),
        source_stages=(trial.stage,),
        target_stage=ExperimentStage.FULL,
        expected_gain_bpb=0.003,
        predictive_standard_deviation=0.001,
        interval_lower_bpb=0.001,
        interval_upper_bpb=0.005,
        minimum_useful_gain_bpb=proposal.minimum_useful_gain_bpb,
        probability_exceeds_minimum=0.95,
        model_version="bayesian-curve-v1",
        full_budget_label_count=40,
    )
    decision = DecisionRecord(
        decision_id="decision-001",
        candidate_id=candidate.candidate_id,
        stage=trial.stage,
        verdict=DecisionVerdict.PROMOTE,
        effect_estimate_id=effect.estimate_id,
        downstream_prediction_id=prediction.prediction_id,
        minimum_useful_gain_bpb=proposal.minimum_useful_gain_bpb,
        probability_threshold=0.9,
        constraints_passed=True,
        reasons=("Paired and predicted gains exceed the promotion threshold.",),
        resulting_parent_commit=candidate.candidate_commit,
    )
    lineage = LineageRecord(
        lineage_id="lineage-001",
        generation=1,
        previous_lineage_id=None,
        parent_commit=proposal.parent_commit,
        candidate_id=candidate.candidate_id,
        candidate_commit=candidate.candidate_commit,
        decision_id=decision.decision_id,
    )
    parent_compute = ComputeRecord(
        compute_id="compute-parent-001",
        trial_id=trial.trial_id,
        run_id=parent_run.run_id,
        device=trial.device,
        wall_seconds=parent_run.wall_seconds,
        accelerator_seconds=parent_run.wall_seconds,
        training_tokens=parent_run.tokens_seen,
        evaluation_tokens=parent_run.evaluation_tokens,
        attempts=1,
        estimated_cost_usd=0.0,
    )
    candidate_compute = ComputeRecord(
        compute_id="compute-candidate-001",
        trial_id=trial.trial_id,
        run_id=candidate_run.run_id,
        device=trial.device,
        wall_seconds=candidate_run.wall_seconds,
        accelerator_seconds=candidate_run.wall_seconds,
        training_tokens=candidate_run.tokens_seen,
        evaluation_tokens=candidate_run.evaluation_tokens,
        attempts=1,
        estimated_cost_usd=0.0,
    )
    return {
        "proposal": proposal,
        "candidate": candidate,
        "trial": trial,
        "parent_run": parent_run,
        "candidate_run": candidate_run,
        "parent_manifest": parent_manifest,
        "candidate_manifest": candidate_manifest,
        "paired": paired,
        "effect": effect,
        "prediction": prediction,
        "decision": decision,
        "lineage": lineage,
        "parent_compute": parent_compute,
        "candidate_compute": candidate_compute,
    }


def lifecycle_entries() -> tuple[tuple[ExperimentRecord, WriterRole], ...]:
    records = evidence_records()
    return (
        (records["proposal"], WriterRole.RESEARCH_AGENT),
        (records["candidate"], WriterRole.CONTROLLER),
        (records["trial"], WriterRole.CONTROLLER),
        (records["parent_run"], WriterRole.EVALUATOR),
        (records["parent_manifest"], WriterRole.EVALUATOR),
        (records["parent_compute"], WriterRole.EVALUATOR),
        (records["candidate_run"], WriterRole.EVALUATOR),
        (records["candidate_manifest"], WriterRole.EVALUATOR),
        (records["candidate_compute"], WriterRole.EVALUATOR),
        (records["paired"], WriterRole.EVALUATOR),
        (records["effect"], WriterRole.EVALUATOR),
        (records["prediction"], WriterRole.EVALUATOR),
        (records["decision"], WriterRole.CONTROLLER),
        (records["lineage"], WriterRole.CONTROLLER),
    )
