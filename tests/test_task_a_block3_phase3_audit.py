from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _assert_contains_sections(doc: str | None, *sections: str) -> None:
    assert doc is not None and doc.strip()
    for section in sections:
        assert section in doc


def test_block3_phase3_modules_drop_demo_wording_and_keep_audit_docstrings() -> None:
    import tasks.task_A.block3 as block3_package
    import tasks.task_A.block3.bundle as bundle_module
    import tasks.task_A.block3.contracts as contracts_module
    import tasks.task_A.block3.execution as execution_module
    import tasks.task_A.block3.registry as registry_module
    import tasks.task_A.block3.review as review_module
    import stride.workflows.fit_stride as fit_stride_module

    module_expectations = {
        block3_package: ("Role:", "Authority anchors:", "Local boundary:", "Why this module exists:"),
        bundle_module: ("Role:", "Authority anchors:", "Local boundary:", "Why this module exists:"),
        contracts_module: ("Role:", "Authority anchors:", "Local boundary:", "Why this module exists:"),
        execution_module: (
            "Role:",
            "Authority anchors:",
            "Local boundary:",
            "Core logic flow:",
            "Why this module exists:",
        ),
        registry_module: ("Role:", "Authority anchors:", "Local boundary:", "Why this module exists:"),
        review_module: ("Role:", "Authority anchors:", "Local boundary:", "Why this module exists:"),
        fit_stride_module: ("Role:", "Local boundary:", "Primary contents:", "Why this module exists:"),
    }
    for module, sections in module_expectations.items():
        _assert_contains_sections(inspect.getdoc(module), *sections)

    forbidden_tokens = (
        "block3_internal_demo",
        "internal demo execution",
        "internal demo implementation surface",
        "_DEMO_",
        "method_penalty",
        "patient_offset",
        "rerun_offset",
    )
    production_files = (
        ROOT / "tasks/task_A/block3/__init__.py",
        ROOT / "tasks/task_A/block3/contracts.py",
        ROOT / "tasks/task_A/block3/registry.py",
        ROOT / "tasks/task_A/block3/bundle.py",
        ROOT / "tasks/task_A/block3/review.py",
        ROOT / "tasks/task_A/block3/execution.py",
        ROOT / "src/stride/workflows/fit_stride.py",
    )
    for path in production_files:
        text = _read(path)
        for token in forbidden_tokens:
            assert token not in text, f"{token!r} should not remain in {path}"


def test_block3_phase3_audit_critical_functions_have_interface_docstrings() -> None:
    from tasks.task_A.block3.execution import (
        _build_3a_rows,
        _build_3b1_rows,
        _build_3b2_rows,
        _build_3c1_rows,
        _build_3c2_rows,
        _build_generator_reruns,
        _build_open_metric_rows,
        _estimate_open_calibration,
        _load_block3_cohort_inputs,
        _run_balanced_ot_baseline,
        _run_diagonal_transport_baseline,
        _run_partial_ot_baseline,
        _run_uot_baseline,
        _run_stride_method,
        build_phase2_execution_plan,
        execute_internal_block3_subexperiment,
        resolve_upstream_inputs,
    )
    from stride.workflows.fit_stride import (
        STRIDEFitConfig,
        _apply_benchmark_mode_to_relation,
        _canonicalize_proxy_patient_result,
        _resolve_benchmark_controls,
        build_patient_bridge_inputs,
        run_stride_fit,
        run_stride_proxy_fit,
    )

    structured_doc_targets = (
        resolve_upstream_inputs,
        build_phase2_execution_plan,
        _load_block3_cohort_inputs,
        _build_generator_reruns,
        execute_internal_block3_subexperiment,
        _apply_benchmark_mode_to_relation,
        _canonicalize_proxy_patient_result,
        build_patient_bridge_inputs,
        run_stride_fit,
    )
    for target in structured_doc_targets:
        _assert_contains_sections(inspect.getdoc(target), "Purpose:", "Core flow:")

    simple_doc_targets = (
        _estimate_open_calibration,
        _run_stride_method,
        _run_balanced_ot_baseline,
        _run_diagonal_transport_baseline,
        _run_partial_ot_baseline,
        _run_uot_baseline,
        _build_3a_rows,
        _build_3b1_rows,
        _build_3b2_rows,
        _build_open_metric_rows,
        _build_3c1_rows,
        _build_3c2_rows,
        _resolve_benchmark_controls,
        run_stride_proxy_fit,
    )
    for target in simple_doc_targets:
        assert inspect.getdoc(target)

    config_doc = inspect.getdoc(STRIDEFitConfig)
    _assert_contains_sections(config_doc, "Fields:", "benchmark_mode", "cohort_shrinkage_weight")


def test_block3_phase3_audit_docs_exist_and_expose_review_structure() -> None:
    memo_path = ROOT / "tasks/task_A/contracts/block3_phase3_audit_memo.md"
    matrix_path = ROOT / "tasks/task_A/contracts/block3_phase3_traceability_matrix.md"

    memo_text = _read(memo_path)
    matrix_text = _read(matrix_path)

    for section in (
        "Status:",
        "Purpose:",
        "Reviewed sources:",
        "Reviewed code/tests:",
        "Traceability findings:",
        "Anti-fabrication findings:",
        "Annotation coverage findings:",
        "Targeted verification:",
        "Decision:",
    ):
        assert section in memo_text

    for snippet in (
        "24 train / 8 test",
        "relation_*",
        "method_native_output_store",
        "benchmark_mode",
        "tests/test_task_a_block3_phase3_audit.py",
    ):
        assert snippet in matrix_text
