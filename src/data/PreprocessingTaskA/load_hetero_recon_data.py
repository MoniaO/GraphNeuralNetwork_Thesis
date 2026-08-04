"""Data loader for Task A: structural link prediction on GSN v3.

The loader produces three HeteroData objects:

    train_data
    valid_data
    test_data

All three objects contain the SAME message-passing graph G_train.

G_train is constructed only from positive edges assigned to the training
split. Positive validation/test edges are hidden from message passing.

The three objects differ only in the candidate pairs that are scored:

    data.link_source_idx : LongTensor [num_candidates]
    data.link_target_idx : LongTensor [num_candidates]
    data.edge_label      : FloatTensor [num_candidates]

Candidate indices are global flat node indices across all node types.
Their ordering must remain consistent with the ordering used by
HeteroReconGNN when flattening node embeddings.

Expected configuration structure:

    cfg.data.dataset.root_dir
    cfg.data.dataset.scenario
    cfg.data.dataset.nodes_file
    cfg.data.dataset.edges_file
    cfg.data.dataset.samples
    cfg.data.dataset.patient_split_file
    cfg.data.dataset.graph_version
    cfg.data.dataset.candidate_id_prefix

    cfg.data.node_feature_profile
    cfg.data.add_reverse_edges
    cfg.data.negative_ratio
    cfg.data.edge_repeat

    cfg.training.seed

Recommended environment variable:

    export GSN_PROJECT_ROOT="$HOME/Desktop/GSN Graphs dysertation 2026"
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from data.PreprocessingTaskA.feature_ablation import (
    apply_feature_ablation,
    feature_fingerprint,
    print_feature_summary,
)
from experiments.taskA_interventions import (
    graph_fingerprint,
    remove_relation_types,
)
from data.PreprocessingTaskA.hetero_data_v2_2 import (
    load_native_heterodata,
)
from utils.task_a_layout import (
    assert_candidate_layout,
    global_layout,
    layout_signature,
)


DEFAULT_NODES_FILE = "synthetic_pharmacotherapy_v3_nodes.csv"
DEFAULT_EDGES_FILE = (
    "synthetic_pharmacotherapy_v3_edges_audited.csv"
)
DEFAULT_PATIENT_SPLIT_FILE = "patient_splits_v3.csv"
DEFAULT_GRAPH_VERSION = "v3_audited"
DEFAULT_CANDIDATE_ID_PREFIX = "v3_c"


def _parse_bool(value: Any, name: str) -> bool:
    """Convert a YAML/configuration value to a strict boolean."""
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()

    if normalized in {"1", "true", "yes", "on"}:
        return True

    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(
        f"{name} must be true or false, received: {value!r}"
    )


def _require_columns(
    frame: pd.DataFrame,
    required_columns: set[str],
    frame_name: str,
) -> None:
    """Raise a clear error when required CSV columns are missing."""
    missing = required_columns.difference(frame.columns)

    if missing:
        raise ValueError(
            f"{frame_name} is missing required columns: "
            f"{sorted(missing)}"
        )


def make_candidates(
    data_dir: Path,
    negative_ratio: int,
    seed: int,
    nodes_file: str = DEFAULT_NODES_FILE,
    edges_file: str = DEFAULT_EDGES_FILE,
    candidate_id_prefix: str = DEFAULT_CANDIDATE_ID_PREFIX,
) -> pd.DataFrame:
    """Build positive and negative candidate pairs.

    Positive candidates are observed audited graph edges.

    Negative candidates are sampled non-edges satisfying the same
    source-node-type and target-node-type combinations that occur among
    positive audited edges.

    Patient-level empirical data are not used to generate candidates
    in this baseline.
    """
    if negative_ratio < 0:
        raise ValueError(
            "negative_ratio must be greater than or equal to 0, "
            f"received: {negative_ratio}"
        )

    nodes_path = data_dir / nodes_file
    edges_path = data_dir / edges_file

    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)

    _require_columns(
        nodes,
        {"node", "node_type", "is_latent"},
        nodes_path.name,
    )

    _require_columns(
        edges,
        {"source", "target", "edge_id"},
        edges_path.name,
    )

    if nodes["node"].isna().any():
        raise ValueError(
            f"{nodes_path.name} contains missing node names."
        )

    if nodes["node"].duplicated().any():
        duplicated_nodes = (
            nodes.loc[nodes["node"].duplicated(keep=False), "node"]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Node names must be unique. Duplicated node names: "
            f"{duplicated_nodes}"
        )

    latent_mask = (
        nodes["is_latent"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )

    observed_nodes = nodes.loc[~latent_mask].copy()

    if observed_nodes.empty:
        raise ValueError(
            "No observable nodes remain after removing latent nodes."
        )

    observed_node_names = set(observed_nodes["node"])

    positives = edges.loc[
        edges["source"].isin(observed_node_names)
        & edges["target"].isin(observed_node_names)
    ].copy()

    if positives.empty:
        raise ValueError(
            "No positive audited edges remain after removing "
            "latent nodes and their incident edges."
        )

    if positives["edge_id"].isna().any():
        raise ValueError(
            "Positive audited edges contain missing edge_id values."
        )

    if positives["edge_id"].duplicated().any():
        duplicated_edge_ids = (
            positives.loc[
                positives["edge_id"].duplicated(keep=False),
                "edge_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "edge_id values must be unique. Duplicated edge IDs: "
            f"{duplicated_edge_ids}"
        )

    # The current decoder predicts binary existence of a source-target pair.
    # The same pair therefore cannot appear multiple times as separate
    # positive examples.
    duplicated_pair_mask = positives.duplicated(
        subset=["source", "target"],
        keep=False,
    )

    if duplicated_pair_mask.any():
        duplicated_pairs = (
            positives.loc[
                duplicated_pair_mask,
                ["source", "target"],
            ]
            .drop_duplicates()
            .to_dict(orient="records")
        )

        raise ValueError(
            "The audited graph contains duplicated source-target pairs. "
            "The current Task A decoder performs binary pair-level link "
            "prediction and requires each source-target pair to be unique. "
            f"Duplicated pairs: {duplicated_pairs}"
        )

    node_type_by_name = (
        observed_nodes
        .set_index("node")["node_type"]
        .to_dict()
    )

    allowed_type_pairs = {
        (
            node_type_by_name[source],
            node_type_by_name[target],
        )
        for source, target in positives[
            ["source", "target"]
        ].itertuples(index=False, name=None)
    }

    positive_pairs = set(
        positives[
            ["source", "target"]
        ].itertuples(index=False, name=None)
    )

    node_names = observed_nodes["node"].tolist()

    negative_pool = [
        (source, target)
        for source in node_names
        for target in node_names
        if source != target
        and (
            node_type_by_name[source],
            node_type_by_name[target],
        )
        in allowed_type_pairs
        and (source, target) not in positive_pairs
    ]

    requested_negative_count = negative_ratio * len(positives)

    negative_count = min(
        len(negative_pool),
        requested_negative_count,
    )

    rng = np.random.default_rng(seed)

    if negative_count > 0:
        sampled_indices = rng.choice(
            len(negative_pool),
            size=negative_count,
            replace=False,
        )

        sampled_negative_pairs = [
            negative_pool[index]
            for index in sampled_indices
        ]
    else:
        sampled_negative_pairs = []

    negatives = pd.DataFrame(
        sampled_negative_pairs,
        columns=["source", "target"],
    )

    node_type_lookup = (
        observed_nodes
        .set_index("node")["node_type"]
        .astype(str)
        .to_dict()
    )

    # Keep audited relation type for per-edge-type analysis (E3).
    if "edge_type" not in positives.columns:
        positives["edge_type"] = "unknown"

    positives = positives[
        ["source", "target", "edge_id", "edge_type"]
    ].copy()
    positives["edge_label"] = 1
    positives["source_node_type"] = positives["source"].map(node_type_lookup)
    positives["target_node_type"] = positives["target"].map(node_type_lookup)

    negatives["edge_id"] = ""
    negatives["edge_type"] = "negative"
    negatives["edge_label"] = 0
    negatives["source_node_type"] = negatives["source"].map(node_type_lookup)
    negatives["target_node_type"] = negatives["target"].map(node_type_lookup)

    candidates = pd.concat(
        [positives, negatives],
        ignore_index=True,
    )

    candidate_ids = [
        f"{candidate_id_prefix}{index:05d}"
        for index in range(len(candidates))
    ]

    candidates.insert(
        0,
        "candidate_id",
        candidate_ids,
    )

    return candidates


def add_split(
    candidates: pd.DataFrame,
    repeat: int,
    seed: int,
) -> pd.DataFrame:
    """Create a stratified 70/15/15 candidate split."""
    if candidates.empty:
        raise ValueError(
            "Cannot split an empty candidate DataFrame."
        )

    _require_columns(
        candidates,
        {"candidate_id", "source", "target", "edge_label"},
        "candidates",
    )

    label_counts = candidates["edge_label"].value_counts()

    if set(label_counts.index) != {0, 1}:
        raise ValueError(
            "Candidate labels must contain both classes 0 and 1. "
            f"Observed counts: {label_counts.to_dict()}"
        )

    indices = np.arange(len(candidates))

    train_indices, held_out_indices = train_test_split(
        indices,
        test_size=0.30,
        random_state=seed + repeat,
        stratify=candidates["edge_label"],
    )

    validation_indices, test_indices = train_test_split(
        held_out_indices,
        test_size=0.50,
        random_state=seed + 1000 + repeat,
        stratify=candidates.iloc[
            held_out_indices
        ]["edge_label"],
    )

    result = candidates.copy()
    result["split"] = ""

    result.loc[train_indices, "split"] = "train"
    result.loc[validation_indices, "split"] = "validation"
    result.loc[test_indices, "split"] = "test"

    result["edge_repeat"] = repeat

    return result


def _attach_candidates(
    data: Any,
    split_df: pd.DataFrame,
    split_name: str,
    lookup: dict[str, int],
) -> Any:
    """Attach candidate indices and binary labels to HeteroData.

    Source/target names from the candidate table are converted with the
    shared ``global_layout`` lookup, then stored again as
    ``candidate_source_name`` / ``candidate_target_name`` so that
    ``assert_candidate_layout`` can prove the round-trip is exact.
    """
    candidates = (
        split_df.loc[split_df["split"].eq(split_name)]
        .copy()
        .reset_index(drop=True)
    )

    if candidates.empty:
        raise ValueError(
            f"No candidates found for split '{split_name}'."
        )

    candidate_source_names = (
        candidates["source"]
        .astype(str)
        .tolist()
    )

    candidate_target_names = (
        candidates["target"]
        .astype(str)
        .tolist()
    )

    unknown_sources = sorted(
        {
            name
            for name in candidate_source_names
            if name not in lookup
        }
    )

    unknown_targets = sorted(
        {
            name
            for name in candidate_target_names
            if name not in lookup
        }
    )

    if unknown_sources:
        raise KeyError(
            "Candidate sources missing from the global layout: "
            f"{unknown_sources}"
        )

    if unknown_targets:
        raise KeyError(
            "Candidate targets missing from the global layout: "
            f"{unknown_targets}"
        )

    data.link_source_idx = torch.tensor(
        [
            lookup[name]
            for name in candidate_source_names
        ],
        dtype=torch.long,
    )

    data.link_target_idx = torch.tensor(
        [
            lookup[name]
            for name in candidate_target_names
        ],
        dtype=torch.long,
    )

    data.edge_label = torch.tensor(
        candidates["edge_label"]
        .astype(float)
        .to_numpy(),
        dtype=torch.float32,
    )

    # Keep the original names so every index can be proved to point to
    # exactly the node that was present in the candidate table.
    data.candidate_source_name = candidate_source_names
    data.candidate_target_name = candidate_target_names

    if "source_node_type" in candidates.columns:
        data.candidate_source_node_type = (
            candidates["source_node_type"].astype(str).tolist()
        )
    if "target_node_type" in candidates.columns:
        data.candidate_target_node_type = (
            candidates["target_node_type"].astype(str).tolist()
        )
    if "edge_type" in candidates.columns:
        data.candidate_edge_type = (
            candidates["edge_type"].astype(str).tolist()
        )

    assert_candidate_layout(data)

    return data


def _resolve_data_dir(cfg: Any) -> Path:
    """Resolve and validate the configured GSN v3 dataset directory."""
    try:
        raw_root_dir = cfg.data.dataset.get(
            "root_dir",
            None,
        )
    except Exception as exc:
        raise ValueError(
            "Could not resolve cfg.data.dataset.root_dir. "
            "Set GSN_PROJECT_ROOT before running, for example:\n"
            'export GSN_PROJECT_ROOT="$HOME/Desktop/'
            'GSN Graphs dysertation 2026"'
        ) from exc

    if raw_root_dir is None or str(raw_root_dir).strip() == "":
        raise ValueError(
            "cfg.data.dataset.root_dir is empty. "
            "Set GSN_PROJECT_ROOT before running, for example:\n"
            'export GSN_PROJECT_ROOT="$HOME/Desktop/'
            'GSN Graphs dysertation 2026"'
        )

    data_dir = (
        Path(str(raw_root_dir))
        .expanduser()
        .resolve()
    )

    if not data_dir.is_dir():
        raise FileNotFoundError(
            "Resolved GSN v3 dataset directory does not exist:\n"
            f"{data_dir}\n"
            "Check GSN_PROJECT_ROOT and "
            "cfg.data.dataset.root_dir."
        )

    return data_dir


def _dataset_settings(cfg: Any) -> dict[str, Any]:
    """Read and validate dataset-related Hydra settings."""
    dataset_cfg = cfg.data.dataset
    scenario = str(dataset_cfg.scenario)

    samples = dataset_cfg.get("samples", {})

    if scenario not in samples:
        available_scenarios = ", ".join(
            sorted(str(key) for key in samples.keys())
        )

        raise KeyError(
            f"Unknown scenario '{scenario}'. "
            f"Available scenarios: {available_scenarios}"
        )

    node_feature_profile = getattr(
        cfg.data,
        "node_feature_profile",
        None,
    )

    if node_feature_profile is None:
        node_feature_profile = dataset_cfg.get(
            "node_feature_profile",
            "empirical",
        )

    raw_add_reverse_edges = getattr(
        cfg.data,
        "add_reverse_edges",
        None,
    )

    if raw_add_reverse_edges is None:
        raw_add_reverse_edges = dataset_cfg.get(
            "add_reverse_edges",
            True,
        )

    return {
        "nodes_file": str(
            dataset_cfg.get(
                "nodes_file",
                DEFAULT_NODES_FILE,
            )
        ),
        "edges_file": str(
            dataset_cfg.get(
                "edges_file",
                DEFAULT_EDGES_FILE,
            )
        ),
        "samples_file": str(samples[scenario]),
        "patient_split_filename": str(
            dataset_cfg.get(
                "patient_split_file",
                DEFAULT_PATIENT_SPLIT_FILE,
            )
        ),
        "graph_version": str(
            dataset_cfg.get(
                "graph_version",
                DEFAULT_GRAPH_VERSION,
            )
        ),
        "candidate_id_prefix": str(
            dataset_cfg.get(
                "candidate_id_prefix",
                DEFAULT_CANDIDATE_ID_PREFIX,
            )
        ),
        "node_feature_profile": str(
            node_feature_profile
        ),
        "add_reverse_edges": _parse_bool(
            raw_add_reverse_edges,
            "cfg.data.add_reverse_edges",
        ),
    }


def _validate_input_files(
    data_dir: Path,
    settings: dict[str, Any],
) -> None:
    """Verify all Task A source files before loading the graph."""
    nodes_path = data_dir / settings["nodes_file"]
    edges_path = data_dir / settings["edges_file"]
    samples_path = data_dir / settings["samples_file"]

    # The native loader resolves the patient split from:
    # data_dir.parent / "splits" / patient_split_filename
    patient_split_path = (
        data_dir.parent
        / "splits"
        / settings["patient_split_filename"]
    )

    expected_files = {
        "nodes file": nodes_path,
        "edges file": edges_path,
        "scenario samples file": samples_path,
        "patient split file": patient_split_path,
    }

    for description, path in expected_files.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Expected {description} was not found:\n"
                f"{path}"
            )



def _candidate_fingerprint(split_df: pd.DataFrame) -> str:
    """Stable hash of candidate pairs + labels + splits (scenario-invariant)."""
    import hashlib

    cols = ["source", "target", "edge_label", "split"]
    missing = set(cols) - set(split_df.columns)
    if missing:
        raise KeyError(f"split_df missing columns for fingerprint: {sorted(missing)}")
    payload = (
        split_df.loc[:, cols]
        .astype(str)
        .sort_values(cols)
        .to_csv(index=False)
        .encode("utf-8")
    )
    return hashlib.sha256(payload).hexdigest()[:16]


def load_recon_heterodata(cfg: Any):
    """Load leakage-safe Task A train/validation/test HeteroData.

    Returns
    -------
    train_data
        HeteroData containing G_train and training candidate pairs.

    valid_data
        HeteroData containing the same G_train and validation candidates.

    test_data
        HeteroData containing the same G_train and test candidates.

    node_to_idx
        Mapping from node name to its global flat index.
    """
    data_dir = _resolve_data_dir(cfg)
    settings = _dataset_settings(cfg)

    _validate_input_files(
        data_dir,
        settings,
    )

    scenario = str(cfg.data.dataset.scenario)

    negative_ratio = int(
        getattr(
            cfg.data,
            "negative_ratio",
            3,
        )
    )

    edge_repeat = int(
        getattr(
            cfg.data,
            "edge_repeat",
            1,
        )
    )

    # training.seed → weight init / dropout only (set in train_taskA).
    # candidate_seed → negatives + candidate split; MUST stay frozen across
    # scenarios so Wave-1 comparisons isolate patient/node-feature effects.
    candidate_seed = int(
        getattr(
            cfg.data,
            "candidate_seed",
            getattr(cfg.training, "seed", 20260722),
        )
    )

    node_feature_profile = settings[
        "node_feature_profile"
    ]

    add_reverse_edges = settings[
        "add_reverse_edges"
    ]

    candidates = make_candidates(
        data_dir=data_dir,
        negative_ratio=negative_ratio,
        seed=candidate_seed,
        nodes_file=settings["nodes_file"],
        edges_file=settings["edges_file"],
        candidate_id_prefix=settings[
            "candidate_id_prefix"
        ],
    )

    split_df = add_split(
        candidates=candidates,
        repeat=edge_repeat,
        seed=candidate_seed,
    )

    positive_train_edge_ids = set(
        split_df.loc[
            split_df["split"].eq("train")
            & split_df["edge_label"].eq(1),
            "edge_id",
        ]
    )

    # Wave 3B motif completion: hold parent_a→gate edges out of G_train MP.
    held_out_motif_edge_ids: list[str] = []
    from experiments.motif_completion import (
        held_out_edge_ids,
        motif_completion_enabled,
    )

    if motif_completion_enabled(cfg):
        # Use train-patient columns so load/continuous parents are excluded.
        from data.patient_matrix import load_patient_matrix_with_split, train_patient_df
        from experiments.motif_completion import hide_tasks
        from hcr.motifs import load_truth_graph

        train_patients = train_patient_df(load_patient_matrix_with_split(cfg))
        held_out_motif_edge_ids = held_out_edge_ids(cfg, samples=train_patients)
        before = len(positive_train_edge_ids)
        positive_train_edge_ids = {
            eid
            for eid in positive_train_edge_ids
            if str(eid) not in set(held_out_motif_edge_ids)
        }

        # Keep visible co-parent → gate edges in G_train even if the candidate
        # split put them in valid/test. Motif completion needs B→G present when
        # A→G is held out (and symmetrically for parent_b masks).
        _, edges_audited = load_truth_graph(cfg)
        forced_visible: list[str] = []
        for task in hide_tasks(cfg, samples=train_patients):
            match = edges_audited[
                (edges_audited["source"].astype(str) == task.visible_parent)
                & (edges_audited["target"].astype(str) == task.candidate_target)
            ]
            if match.empty:
                continue
            eid = str(match.iloc[0]["edge_id"])
            if eid in set(held_out_motif_edge_ids):
                continue
            if eid not in positive_train_edge_ids:
                positive_train_edge_ids.add(eid)
                forced_visible.append(eid)

        print(
            "\nMOTIF COMPLETION"
            f"\n  held_out_edges: {len(held_out_motif_edge_ids)}"
            f"\n  train MP edges: {before} → {len(positive_train_edge_ids)}"
            f"\n  forced_visible_coparent_edges: {len(forced_visible)}"
            f"\n  ids: {held_out_motif_edge_ids}"
        )

    if not positive_train_edge_ids:
        raise ValueError(
            "The training split contains no positive edge IDs."
        )

    # Build G_train only from positive training edges.
    #
    # Positive validation/test edges are not included in message passing.
    # Negative candidates are never included in message passing.
    loaded = load_native_heterodata(
        data_dir,
        scenario,
        train_edge_ids=positive_train_edge_ids,
        nodes_file=settings["nodes_file"],
        edges_file=settings["edges_file"],
        samples_file=settings["samples_file"],
        patient_split_filename=settings[
            "patient_split_filename"
        ],
        graph_version=settings["graph_version"],

        # The native loader currently exposes this parameter as
        # feature_profile. Keep the keyword consistent with its signature.
        feature_profile=node_feature_profile,

        add_reverse_edges=add_reverse_edges,
    )

    base_data = loaded.data

    # Controlled feature ablation AFTER empirical construction, BEFORE splits.
    # Candidates / G-TRAIN stay frozen; only node features `.x` may change.
    feature_ablation_profile = str(
        getattr(cfg.data, "feature_ablation_profile", "empirical")
    ).strip().lower()
    feature_ablation_seed = int(
        getattr(cfg.data, "feature_ablation_seed", 20260722)
    )

    apply_feature_ablation(
        base_data,
        profile=feature_ablation_profile,
        seed=feature_ablation_seed,
    )

    # Freeze audit fields for W&B / cross-scenario checks
    base_data.candidate_seed = int(candidate_seed)
    base_data.training_seed_expected = int(cfg.training.seed)
    base_data.candidate_fingerprint = _candidate_fingerprint(split_df)
    base_data.n_message_train_positives = int(len(positive_train_edge_ids))
    base_data.feature_ablation_profile = feature_ablation_profile
    base_data.feature_ablation_seed = feature_ablation_seed
    base_data.feature_fingerprint = feature_fingerprint(base_data)

    print(
        "\nFEATURE ABLATION"
        f"\n  profile:     {feature_ablation_profile}"
        f"\n  seed:        {feature_ablation_seed}"
        f"\n  fingerprint: {base_data.feature_fingerprint}"
    )
    print_feature_summary(base_data, title="EFFECTIVE NODE FEATURES")

    relation_ablation_cfg = getattr(cfg.experiment, "relation_ablation", None)
    if relation_ablation_cfg is not None and bool(
        getattr(relation_ablation_cfg, "enabled", False)
    ):
        removed_relations = list(
            getattr(relation_ablation_cfg, "removed_relations", []) or []
        )
        base_data = remove_relation_types(base_data, removed_relations)
    else:
        removed_relations = []

    base_data.removed_relations = removed_relations
    base_data.held_out_motif_edge_ids = list(held_out_motif_edge_ids)
    base_data.graph_fingerprint = graph_fingerprint(base_data)
    print(
        "\nGRAPH FINGERPRINT"
        f"\n  removed_relations: {removed_relations}"
        f"\n  held_out_motif_edges: {len(held_out_motif_edge_ids)}"
        f"\n  graph_fingerprint: {base_data.graph_fingerprint}"
    )

    # Shared Task A layout: loader and model must use the same function.
    _, node_to_idx = global_layout(base_data)

    # Deep copies so later edits to one split cannot mutate another.
    train_data = _attach_candidates(
        copy.deepcopy(base_data),
        split_df,
        "train",
        node_to_idx,
    )

    valid_data = _attach_candidates(
        copy.deepcopy(base_data),
        split_df,
        "validation",
        node_to_idx,
    )

    test_data = _attach_candidates(
        copy.deepcopy(base_data),
        split_df,
        "test",
        node_to_idx,
    )

    train_signature = layout_signature(train_data)
    valid_signature = layout_signature(valid_data)
    test_signature = layout_signature(test_data)

    if train_signature != valid_signature:
        raise AssertionError(
            "Train and validation use different global node layouts.\n"
            f"train={train_signature}\n"
            f"valid={valid_signature}"
        )

    if train_signature != test_signature:
        raise AssertionError(
            "Train and test use different global node layouts.\n"
            f"train={train_signature}\n"
            f"test={test_signature}"
        )

    return (
        train_data,
        valid_data,
        test_data,
        node_to_idx,
    )
