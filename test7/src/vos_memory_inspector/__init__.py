"""CMMT state inspection and component-wise translator baselines.

Heavy torch-backed symbols are loaded lazily so repository-management commands can
run on machines that do not have the GPU research environment installed.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "ProbeConfig",
    "CanonicalState",
    "DirectCopyTranslator",
    "LinearStateTranslator",
    "ResidualMLPStateTranslator",
    "RidgeStateTranslator",
    "StateSpec",
    "SUPPORTED_SAM2_COMMIT",
    "StateProbe",
    "compare_manifests",
]


_EXPORTS = {
    "compare_manifests": (".compatibility", "compare_manifests"),
    "CanonicalState": (".state_schema", "CanonicalState"),
    "StateSpec": (".state_schema", "StateSpec"),
    "DirectCopyTranslator": (".translators", "DirectCopyTranslator"),
    "LinearStateTranslator": (".translators", "LinearStateTranslator"),
    "ResidualMLPStateTranslator": (".translators", "ResidualMLPStateTranslator"),
    "RidgeStateTranslator": (".translators", "RidgeStateTranslator"),
    "ProbeConfig": (".probe", "ProbeConfig"),
    "StateProbe": (".probe", "StateProbe"),
    "SUPPORTED_SAM2_COMMIT": (".upstream", "SUPPORTED_SAM2_COMMIT"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value

