#!/usr/bin/env python3
"""Synthetic algebra witnesses for a proposed conditional message contract.

This file is deliberately independent of the model implementation. It imports
only NumPy and the Python standard library, uses synthetic float64 arrays, and
does not read traffic observations, targets, or checkpoints. Passing these
checks is not an integrated-model validation or a performance result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def masked_softmax(logits: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Normalize source-node and position axes, defining empty rows as zero."""
    logits = np.asarray(logits, dtype=np.float64)
    allowed = np.asarray(allowed, dtype=bool)
    require(logits.ndim == 3 and logits.shape == allowed.shape, "Invalid kernel axes")
    require(np.isfinite(logits).all(), "Non-finite synthetic logits")
    nonempty = allowed.any(axis=(1, 2), keepdims=True)
    row_max = np.max(np.where(allowed, logits, -np.inf), axis=(1, 2), keepdims=True)
    row_max = np.where(nonempty, row_max, 0.0)
    unnormalized = np.exp(np.where(allowed, logits - row_max, -np.inf))
    denominator = unnormalized.sum(axis=(1, 2), keepdims=True)
    return np.divide(
        unnormalized, denominator, out=np.zeros_like(unnormalized), where=denominator > 0
    )


def conditional_message(
    base_logits: np.ndarray,
    base_value: np.ndarray,
    allowed: np.ndarray,
    event_raw: np.ndarray,
    value_raw: np.ndarray,
    node_support: np.ndarray,
    *,
    enabled: bool = True,
    available: bool = True,
    beta: float = 0.5,
    eta: float = 0.1,
) -> dict[str, np.ndarray]:
    """One fixed-history call, with a final gate after arbitrary biased maps."""
    base_logits = np.asarray(base_logits, dtype=np.float64)
    base_value = np.asarray(base_value, dtype=np.float64)
    event_raw = np.asarray(event_raw, dtype=np.float64)
    value_raw = np.asarray(value_raw, dtype=np.float64)
    node_support = np.asarray(node_support, dtype=bool)
    require(base_logits.shape == event_raw.shape, "Invalid event-score axes")
    require(base_value.shape == value_raw.shape, "Invalid event-value axes")
    require(base_value.shape[:3] == base_logits.shape, "Invalid message-value axes")
    require(base_logits.shape[:2] == (len(node_support), len(node_support)), "Node-axis mismatch")
    require(np.isfinite(base_value).all() and np.isfinite(event_raw).all()
            and np.isfinite(value_raw).all(), "Non-finite synthetic message input")
    require(np.isfinite(beta) and np.isfinite(eta) and beta >= 0 and eta >= 0,
            "Perturbation bounds must be finite and nonnegative")

    gate = (node_support[:, None] | node_support[None, :])[:, :, None]
    gate = gate.astype(np.float64) * float(enabled and available)
    # event_raw/value_raw represent outputs of maps that may have nonzero bias.
    # The final multiplication is essential for exact OFF/unavailable fallback.
    event_score = gate * beta * np.tanh(event_raw)
    value_delta = gate[..., None] * eta * np.tanh(value_raw)
    kernel = masked_softmax(base_logits + event_score, allowed)
    message = np.sum(kernel[..., None] * (base_value + value_delta), axis=(1, 2))
    return {"kernel": kernel, "message": message,
            "event_score": event_score, "value_delta": value_delta}


def verify_contract() -> dict[str, Any]:
    rng = np.random.default_rng(20261015)
    checks: list[dict[str, Any]] = []

    def passed(name: str, **details: Any) -> None:
        checks.append({"name": name, "status": "PASS", **details})

    nodes, positions, channels = 5, 3, 4
    base = rng.normal(size=(nodes, nodes, positions))
    values = rng.normal(size=(nodes, nodes, positions, channels))
    mask = rng.random(base.shape) > 0.25
    event_raw = rng.normal(size=base.shape) + 3.0
    value_raw = rng.normal(size=values.shape) + 2.0
    support = np.array([True, False, True, False, False])
    history_kernel = masked_softmax(base, mask)
    history_message = np.sum(history_kernel[..., None] * values, axis=(1, 2))

    for name, switches, current_support in (
        ("report_OFF_exact_history_fallback", {"enabled": False}, support),
        ("unavailable_exact_history_fallback", {"available": False}, support),
        ("zero_report_support_exact_history_fallback", {}, np.zeros(nodes, dtype=bool)),
    ):
        result = conditional_message(base, values, mask, event_raw, value_raw,
                                     current_support, **switches)
        require(np.array_equal(result["kernel"], history_kernel), name + ": kernel differs")
        require(np.array_equal(result["message"], history_message), name + ": message differs")
        require(np.count_nonzero(result["event_score"]) == 0
                and np.count_nonzero(result["value_delta"]) == 0, name + ": gate leaked")
        passed(name, biased_map_outputs_used=True, exact_array_equality=True)

    common = masked_softmax(base + 7.125, mask)
    common_error = float(np.max(np.abs(common - history_kernel)))
    require(common_error < 1e-14, "Common-logit offset changed weights")
    passed("common_logit_offset_is_ineffective", kernel_abs_max=common_error)

    all_allowed = np.ones_like(mask)
    homogeneous = np.full_like(values, 1.75)
    blind = conditional_message(base, homogeneous, all_allowed, event_raw,
                                value_raw, support, eta=0.0)
    blind_history = np.sum(masked_softmax(base, all_allowed)[..., None]
                           * homogeneous, axis=(1, 2))
    blind_error = float(np.max(np.abs(blind["message"] - blind_history)))
    changed_weights = float(np.max(np.abs(blind["kernel"] - masked_softmax(base, all_allowed))))
    require(blind_error < 1e-14 and changed_weights > 1e-5,
            "Did not construct pure-logit homogeneous-value blindness")
    passed("pure_logit_homogeneous_value_blindness_counterexample",
           message_abs_max=blind_error, kernel_change_abs_max=changed_weights)

    for name, current_values in (
        ("value_delta_can_respond_with_homogeneous_history", homogeneous),
        ("value_delta_can_respond_with_zero_history", np.zeros_like(values)),
    ):
        result = conditional_message(base, current_values, all_allowed,
                                     np.zeros_like(base), np.ones_like(values), support)
        reference = np.sum(masked_softmax(base, all_allowed)[..., None]
                           * current_values, axis=(1, 2))
        response = float(np.max(np.abs(result["message"] - reference)))
        require(response > 0.01, name + ": no event-value response")
        passed(name, message_change_abs_max=response,
               scope="constructive_expression_witness_not_learned_use")

    # Hold history, support, availability, kernel scores, and perturbation
    # amplitudes fixed. Only the content of supported-node report vectors
    # changes. This explicitly witnesses more than an ON/support response.
    report_one = 0.25 * support.astype(np.float64)
    report_two = 0.75 * support.astype(np.float64)
    channel_direction = np.array([1.0, -0.5, 0.25, -0.75])

    def direct_report_value_map(report: np.ndarray) -> np.ndarray:
        # Match the contract's bounded pair report map r_ij, then a linear
        # value map and the final operator tanh/gate. R also stays in [-1,1].
        pair_content = np.tanh(0.5 * report[:, None] + 0.75 * report[None, :])
        return np.broadcast_to(pair_content[:, :, None, None]
                               * channel_direction[None, None, None, :], values.shape).copy()

    content_one = conditional_message(base, np.zeros_like(values), all_allowed,
                                      np.zeros_like(base), direct_report_value_map(report_one),
                                      support, enabled=True, available=True, beta=0.5, eta=0.1)
    content_two = conditional_message(base, np.zeros_like(values), all_allowed,
                                      np.zeros_like(base), direct_report_value_map(report_two),
                                      support, enabled=True, available=True, beta=0.5, eta=0.1)
    content_difference = float(np.max(np.abs(content_two["message"] - content_one["message"])))
    require(np.array_equal(content_one["kernel"], content_two["kernel"]),
            "Fixed-support content witness unexpectedly changed the kernel")
    require(np.count_nonzero(report_one[~support]) == 0
            and np.count_nonzero(report_two[~support]) == 0,
            "Fixed-support report witness changed unsupported-node content")
    require(content_difference > 1e-3, "Fixed-support report content did not change the message")
    passed("constructive_fixed_support_report_content_witness",
           message_content_difference_abs_max=content_difference,
           value_raw_map="tanh(0.5 * R_i + 0.75 * R_j) * channel_direction",
           supported_report_values=[0.25, 0.75], unchanged_kernel=True,
           scope="synthetic_fixed_support_expression_not_learned_semantic_use")

    # Explicit source x position x report multiplication. Four allowed cells
    # suffice to witness a nonseparable event contribution to log odds.
    cross_base = np.array([[[0.2, -0.4], [0.6, 0.1]],
                           [[-0.1, 0.3], [0.4, -0.2]]], dtype=np.float64)
    source_factor = np.array([0.6, -0.9])[None, :, None]
    position_factor = np.array([-0.7, 0.8])[None, None, :]
    report_factor = 0.75
    cross_raw = np.broadcast_to(source_factor * position_factor * report_factor,
                                cross_base.shape).copy()
    cross_mask = np.ones_like(cross_base, dtype=bool)
    cross_values = np.zeros((*cross_base.shape, 1), dtype=np.float64)
    cross = conditional_message(cross_base, cross_values, cross_mask, cross_raw,
                                cross_values, np.array([True, False]), beta=0.8, eta=0.0)

    def cross_difference(matrix: np.ndarray) -> float:
        return float(matrix[0, 0] - matrix[0, 1] - matrix[1, 0] + matrix[1, 1])

    expected_cross = cross_difference(cross["event_score"][0])
    observed_cross = cross_difference(np.log(cross["kernel"][0])) - cross_difference(
        np.log(masked_softmax(cross_base, cross_mask)[0]))
    require(abs(expected_cross) > 0.1 and abs(expected_cross - observed_cross) < 1e-14,
            "Joint source-position event log-odds identity failed")
    passed("source_position_report_cross_log_odds_identity",
           event_cross_difference=expected_cross,
           observed_cross_log_odds_change=observed_cross,
           residual=abs(expected_cross - observed_cross))

    maximum_bound_excess = -float("inf")
    for _ in range(128):
        trial_base = rng.normal(size=base.shape) * 4.0
        trial_values = rng.normal(size=values.shape) * 3.0
        trial_mask = rng.random(base.shape) > 0.35
        trial_mask[0] = False
        trial_support = rng.random(nodes) > 0.5
        beta, eta = float(rng.uniform(0, 3)), float(rng.uniform(0, 1))
        trial = conditional_message(trial_base, trial_values, trial_mask,
                                    rng.normal(size=base.shape) * 5,
                                    rng.normal(size=values.shape) * 5,
                                    trial_support, beta=beta, eta=eta)
        trial_history = np.sum(masked_softmax(trial_base, trial_mask)[..., None]
                               * trial_values, axis=(1, 2))
        response = np.max(np.abs(trial["message"] - trial_history), axis=-1)
        value_bound = np.max(np.abs(trial_values), axis=(1, 2, 3))
        local_bound = 2 * value_bound * min(1.0, beta) + eta
        maximum_bound_excess = max(maximum_bound_excess,
                                   float(np.max(response - local_bound)))
        require(np.all(response <= local_bound + 1e-12), "Fixed-history local bound failed")
    passed("fixed_history_local_event_perturbation_bound", trials=128,
           bound="2 * V * min(1, beta) + eta", maximum_excess=maximum_bound_excess,
           scope="single_fixed_history_operator_no_projection_or_rollout")

    empty_mask = mask.copy()
    empty_mask[0] = False
    empty = conditional_message(base, values, empty_mask, event_raw, value_raw, support)
    require(np.isfinite(empty["kernel"]).all() and np.isfinite(empty["message"]).all(),
            "Empty row produced non-finite output")
    require(np.count_nonzero(empty["kernel"][~empty_mask]) == 0,
            "Forbidden edge received weight")
    require(np.count_nonzero(empty["kernel"][0]) == 0
            and np.count_nonzero(empty["message"][0]) == 0, "Empty row did not return zero")
    row_sum = empty["kernel"].sum(axis=(1, 2))
    require(np.allclose(row_sum[1:], 1.0, atol=1e-15, rtol=0), "Nonempty row not normalized")
    passed("hard_forbidden_edges_and_empty_row_contract", empty_row_sum=float(row_sum[0]),
           forbidden_nonzero_count=int(np.count_nonzero(empty["kernel"][~empty_mask])))

    disconnected = np.zeros((4, 4, 2), dtype=bool)
    disconnected[:2, :2] = True
    disconnected[2:, 2:] = True
    disconnected_base = np.zeros(disconnected.shape, dtype=np.float64)
    disconnected_values = np.zeros((*disconnected.shape, 2), dtype=np.float64)
    isolated = conditional_message(disconnected_base, disconnected_values, disconnected,
                                   disconnected_base, np.ones_like(disconnected_values),
                                   np.array([True, False, False, False]))
    require(np.count_nonzero(isolated["message"][2:]) == 0,
            "Report crossed disconnected fixed-history support")
    require(np.max(np.abs(isolated["message"][:2])) > 0.01,
            "Disconnected witness has no response in the report component")
    passed("disconnected_fixed_history_report_isolation",
           outside_component_abs_max=float(np.max(np.abs(isolated["message"][2:]))),
           scope="synthetic_fixed_graph_only_not_integrated_dynamic_graph")

    original = conditional_message(base, values, mask, event_raw, value_raw, support)
    permutation = rng.permutation(nodes)

    def permute_nodes(array: np.ndarray) -> np.ndarray:
        return array[permutation][:, permutation]

    permuted = conditional_message(permute_nodes(base), permute_nodes(values),
                                  permute_nodes(mask), permute_nodes(event_raw),
                                  permute_nodes(value_raw), support[permutation])
    equivariance_error = float(np.max(np.abs(permuted["message"]
                                            - original["message"][permutation])))
    require(equivariance_error < 1e-14, "Joint node-relabeling equivariance failed")
    require(np.allclose(permuted["kernel"], permute_nodes(original["kernel"]),
                        atol=1e-14, rtol=0), "Kernel node-relabeling equivariance failed")
    passed("joint_node_permutation_equivariance", message_abs_max=equivariance_error,
           scope="relabel_all_node_indexed_inputs_together")

    # Row normalization constrains receiving mixtures, not the global quantity.
    nonconserving_mask = np.array([[[True], [False]], [[True], [False]]])
    quantity = np.array([1.0, 0.0])
    quantity_values = np.broadcast_to(quantity[None, :, None, None], (2, 2, 1, 1)).copy()
    nonconserving = conditional_message(np.zeros((2, 2, 1)), quantity_values,
                                       nonconserving_mask, np.zeros((2, 2, 1)),
                                       np.zeros_like(quantity_values),
                                       np.zeros(2, dtype=bool), enabled=False)
    output_total = float(nonconserving["message"].sum())
    require(output_total == 2.0 and float(quantity.sum()) == 1.0,
            "Did not construct row-normalized nonconservation")
    passed("row_normalization_does_not_conserve_total_counterexample",
           input_total=float(quantity.sum()), output_total=output_total,
           scope="abstract_quantity_not_certified_vehicle_flow")

    history_lengths = [12]
    for _ in range(5):
        history_lengths.append(history_lengths[-1] - 3 + 1)
    require(history_lengths == [12, 10, 8, 6, 4, 2], "Residual history-axis contract failed")
    prefixes: list[list[str]] = []
    observed = ["H[-2]", "H[-1]"]
    generated = ["G0"]  # First generated block comes from history-encoded DIF.
    for block_index in range(1, 4):
        window = (observed + generated)[-3:]
        require(len(window) == 3, "Generated-prefix window length failed")
        require(all(not token.startswith("G") or int(token[1:]) < block_index
                    for token in window), "Read a future generated block")
        prefixes.append(window)
        generated.append("G" + str(block_index))
    require(prefixes == [["H[-2]", "H[-1]", "G0"],
                         ["H[-1]", "G0", "G1"], ["G0", "G1", "G2"]],
            "Four-block generation-prefix contract failed")
    readout = [{"target_step": step + 1, "block": step // 3,
                "head_channel": step % 3} for step in range(12)]
    require([row["block"] for row in readout] == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
            "Twelve-target block expansion failed")
    require(len(generated) == 4 and len(readout) == 12, "Forecast count failed")
    passed("history_reduction_four_block_prefix_and_twelve_target_readout",
           history_lengths=history_lengths, generated_blocks=generated,
           recursive_source_windows=prefixes, readout_mapping=readout,
           scope="index_witness_no_claim_of_uniform_physical_source_spacing")

    return {
        "schema": "min_conditional_propagation_algebra_witnesses_v1",
        "status": "ALGEBRA_WITNESSES_PASS_NOT_INTEGRATED_MODEL",
        "engineering_only": True,
        "dtype": "float64",
        "seed": 20261015,
        "checks": checks,
        "check_count": len(checks),
        "formal_inference_calls": 0,
        "optimizer_updates": 0,
        "raw_traffic_or_target_data_read": False,
        "checkpoint_read": False,
        "model_imported": False,
        "test_evaluated": False,
        "scope": "synthetic_single_operator_algebra_and_index_witnesses_only",
        "not_proven": ["integrated_candidate_correctness", "learned_report_use",
                       "global_network_contraction", "rollout_stability",
                       "physical_conservation", "predictive_gain", "novelty"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON path; existing files are refused")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly allow output replacement")
    args = parser.parse_args()
    if args.output is not None and args.output.exists() and not args.overwrite:
        parser.error(f"Output already exists; use --overwrite explicitly: {args.output}")
    result = verify_contract()
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also prevents an accidental overwrite after a race.
        with args.output.open("w" if args.overwrite else "x", encoding="utf-8") as handle:
            handle.write(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
