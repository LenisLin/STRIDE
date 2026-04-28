from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def test_stride_adapter_imports_succeed_in_clean_interpreter() -> None:
    command = [
        sys.executable,
        "-c",
        (
            "import importlib, sys; "
            f"sys.path.insert(0, {str(SRC)!r}); "
            "importlib.import_module('stride.adapters'); "
            "importlib.import_module('stride.adapters.ot_sinkhorn'); "
            "observation = importlib.import_module('stride.observation'); "
            "fit_module = importlib.import_module('stride.api.fit'); "
            "proxy_module = importlib.import_module('stride.api.proxy'); "
            "workflow_module = importlib.import_module('stride.workflows.fit_stride'); "
            "proxy_workflow_module = importlib.import_module('stride.workflows.fit_proxy'); "
            "fit_result_module = importlib.import_module('stride.outputs.fit_result'); "
            "uncertainty_module = importlib.import_module('stride.outputs.uncertainty'); "
            "assert hasattr(observation, 'build_observation_kernels'); "
            "assert hasattr(observation, 'match_observation_clouds'); "
            "assert hasattr(fit_module, 'fit_stride'); "
            "assert hasattr(fit_module, 'fit_stride_proxy'); "
            "assert hasattr(proxy_module, 'fit_stride_proxy'); "
            "assert hasattr(workflow_module, 'build_patient_bridge_inputs'); "
            "assert hasattr(workflow_module, 'run_stride_proxy_fit'); "
            "assert hasattr(proxy_workflow_module, 'run_stride_proxy_fit'); "
            "assert hasattr(fit_result_module, 'PatientBridgeResult'); "
            "assert hasattr(fit_result_module, 'STRIDEFitResult'); "
            "assert hasattr(uncertainty_module, 'PatientBootstrapConfig'); "
            "assert hasattr(uncertainty_module, 'STRIDEBootstrapUncertaintyResult')"
        ),
    ]
    result = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_stride_package_roots_expose_only_stable_first_pass_surfaces() -> None:
    stride_module = importlib.import_module("stride")
    api_module = importlib.import_module("stride.api")
    outputs_module = importlib.import_module("stride.outputs")
    fit_module = importlib.import_module("stride.api.fit")
    proxy_module = importlib.import_module("stride.api.proxy")
    uncertainty_module = importlib.import_module("stride.outputs.uncertainty")

    assert set(stride_module.__all__) == {
        "BasisSpec",
        "ContractError",
        "DataContractError",
        "DatasetHandle",
        "build_patient_relation",
        "fit_stride",
        "summarize_fit",
    }
    assert hasattr(stride_module, "fit_stride")
    assert hasattr(stride_module, "build_patient_relation")
    assert not hasattr(stride_module, "STRIDEModel")
    assert not hasattr(stride_module, "StateBasis")

    assert set(api_module.__all__) == {
        "BasisSpec",
        "BridgeConfig",
        "DatasetHandle",
        "STRIDEFitConfig",
        "build_patient_relation",
        "fit_stride",
        "summarize_fit",
    }
    assert hasattr(api_module, "fit_stride")
    assert hasattr(api_module, "build_patient_relation")
    assert not hasattr(api_module, "bridge_observation_matches")
    assert not hasattr(api_module, "fit_patient_relation")
    assert not hasattr(api_module, "PatientBridgeResult")
    assert not hasattr(api_module, "STRIDEModel")
    assert hasattr(fit_module, "bridge_observation_matches")
    assert hasattr(fit_module, "fit_stride_proxy")
    assert set(proxy_module.__all__) == {"ProxySTRIDEFitConfig", "fit_stride_proxy"}
    assert hasattr(proxy_module, "fit_stride_proxy")

    assert set(outputs_module.__all__) == {
        "BootstrapArraySummary",
        "BootstrapConfig",
        "CohortBootstrapUncertaintySummary",
        "EVENTS_FILENAME",
        "META_FILENAME",
        "METRICS_FILENAME",
        "PatientBridgeResult",
        "PatientBootstrapConfig",
        "PatientBootstrapUncertaintyResult",
        "PatientRelationFitResult",
        "STRIDEBootstrapUncertaintyResult",
        "STRIDEFitResult",
        "validate_patient_bridge_result",
        "validate_stride_fit_result",
        "write_r_handover",
    }
    assert hasattr(outputs_module, "PatientBridgeResult")
    assert hasattr(outputs_module, "STRIDEBootstrapUncertaintyResult")
    assert not hasattr(outputs_module, "bootstrap_observation_unit")
    assert not hasattr(outputs_module, "estimate_log_measurement_error")
    assert not hasattr(outputs_module, "summarize_patient_relation")
    assert hasattr(uncertainty_module, "bootstrap_observation_unit")
