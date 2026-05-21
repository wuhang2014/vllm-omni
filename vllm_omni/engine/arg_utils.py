import argparse
import dataclasses
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field, fields
from typing import Any

import yaml
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.config import OmniModelConfig
from vllm_omni.engine.output_modality import OutputModality
from vllm_omni.platforms import current_omni_platform
from vllm_omni.plugins import load_omni_general_plugins

logger = init_logger(__name__)

# Maps model architecture names to their HuggingFace model_type values.
# Used when auto-injecting hf_overrides for models with missing config.json.
_ARCH_TO_MODEL_TYPE: dict[str, str] = {
    "CosyVoice3Model": "cosyvoice3",
    "OmniVoiceModel": "omnivoice",
    "VoxCPM2TalkerForConditionalGeneration": "voxcpm2",
}

# Maps model architecture names to tokenizer subfolder paths within HF repos.
_TOKENIZER_SUBFOLDER_MAP: dict[str, str] = {
    "CosyVoice3Model": "CosyVoice-BlankEN",
}


def _register_omni_hf_configs() -> None:
    try:
        from transformers import AutoConfig

        from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
            Qwen3TTSConfig,
        )
    except Exception as exc:  # pragma: no cover - best-effort optional registration
        logger.warning("Skipping omni HF config registration due to import error: %s", exc)
        return

    # Register with both transformers AutoConfig and vLLM's config registry
    # so models with empty/missing config.json (e.g. CosyVoice3) can be
    # resolved when model_type is injected via hf_overrides.
    try:
        from vllm.transformers_utils.config import _CONFIG_REGISTRY
    except ImportError:
        _CONFIG_REGISTRY = None

    for model_type, config_cls in [
        ("qwen3_tts", Qwen3TTSConfig),
    ]:
        try:
            AutoConfig.register(model_type, config_cls)
        except ValueError:
            # Already registered elsewhere; ignore.
            pass
        if _CONFIG_REGISTRY is not None and model_type not in _CONFIG_REGISTRY:
            _CONFIG_REGISTRY[model_type] = config_cls


def register_omni_models_to_vllm():
    from vllm.model_executor.models import ModelRegistry

    from vllm_omni.model_executor.models.registry import _OMNI_MODELS

    _register_omni_hf_configs()

    supported_archs = ModelRegistry.get_supported_archs()
    for arch, (mod_folder, mod_relname, cls_name) in _OMNI_MODELS.items():
        if arch not in supported_archs:
            ModelRegistry.register_model(arch, f"vllm_omni.model_executor.models.{mod_folder}.{mod_relname}:{cls_name}")


@dataclass
class OmniEngineArgs(EngineArgs):
    """Engine arguments for omni models, extending base EngineArgs.
    Adds omni-specific configuration fields for multi-stage pipeline
    processing and output type specification.
    Args:
        stage_id: Identifier for the stage in a multi-stage pipeline.
            Defaults to 0 for per-stage engine construction. The CLI-level
            single-stage selector remains optional on the parsed argparse
            namespace and should not be forwarded as a nullable per-stage
            engine argument.
        model_stage: Stage type identifier, e.g., "thinker" or "talker"
            (default: "thinker")
        model_arch: Model architecture name
            (default: "Qwen2_5OmniForConditionalGeneration")
        engine_output_type: Optional output type specification for the engine.
            Used to route outputs to appropriate processors (e.g., "image",
            "audio", "latents"). If None, output type is inferred.
        hf_config_name: Optional key for HF config subkey to be extracted
            for this stage, e.g., talker_config; If None, the default
            HF config will be used.
        custom_process_next_stage_input_func: Optional path to a custom function for processing
            inputs from previous stages
            If None, default processing is used.
        stage_connector_spec: Extra configuration for stage connector
        async_chunk: If set to True, perform async chunk
        worker_type: Model Type, e.g., "ar" or "generation"
        task_type: Default task type for TTS models (CustomVoice, VoiceDesign, or Base).
            If not specified, will be inferred from model path.
        omni_master_address: TCP address that the OmniMasterServer (running
            inside AsyncOmniEngine) listens on for engine core registrations.
            Required when single-stage mode is active.
        omni_master_port: TCP port for the OmniMasterServer registration
            socket.  Required when single-stage mode is active.
        stage_configs_path: Optional path to a JSON/YAML file containing
            stage configurations for the multi-stage pipeline. If None,
            stage configs are resolved from the model's default configuration.
        output_modalities: Optional list of output modality names to enable
            (e.g. ["text", "audio"]). If None, all modalities supported by
            the model are used.
        log_stats: Whether to log engine statistics. Defaults to False.
        custom_pipeline_args: Dictionary of arguments for custom pipeline
            initialization (e.g., ``{"pipeline_class": "my.Module"}``).
            Passed through to the diffusion stage engine.
    """

    stage_id: int = 0
    model_stage: str = "thinker"
    model_arch: str | None = None
    engine_output_type: str | None = None
    hf_config_name: str | None = None
    custom_process_next_stage_input_func: str | None = None
    stage_connector_spec: dict[str, Any] = field(default_factory=dict)
    subtalker_sampling_params: dict[str, Any] | None = None
    async_chunk: bool = False
    omni_kv_config: dict | None = None
    quantization_config: Any | None = None
    force_cutlass_fp8: bool | None = None
    worker_type: str | None = None
    task_type: str | None = None
    worker_cls: str = None
    enable_sleep_mode: bool = False
    omni: bool = False

    @classmethod
    def _add_omni_specific_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        try:
            parser.add_argument("--omni", action="store_true", default=False, help="Enable Omni engine features.")
        except argparse.ArgumentError:
            pass
        try:
            parser.add_argument(
                "--enable-sleep-mode", action="store_true", default=False, help="Enable GPU memory pool for sleep mode."
            )
        except argparse.ArgumentError:
            pass
        return parser

    omni_master_address: str | None = None
    omni_master_port: int | None = None
    # OmniCoordinator integration knobs (process-local).
    omni_dp_size_local: int = 1
    omni_lb_policy: str = "random"
    omni_heartbeat_timeout: float = 30.0
    stage_configs_path: str | None = None
    output_modalities: list[str] | None = None
    log_stats: bool = False
    custom_pipeline_args: dict[str, Any] | None = None
    has_sampling_extra_args: bool = False

    def __post_init__(self) -> None:
        if self.worker_cls is None:
            if self.worker_type == "ar":
                self.worker_cls = current_omni_platform.get_omni_ar_worker_cls()
            elif self.worker_type == "generation":
                self.worker_cls = current_omni_platform.get_omni_generation_worker_cls()
        load_omni_general_plugins()
        super().__post_init__()

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "OmniEngineArgs":
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        engine_args = cls(**{attr: getattr(args, attr) for attr in attrs if hasattr(args, attr)})
        engine_args._explicit_fields = frozenset(
            attr for attr in attrs if hasattr(args, attr) and getattr(args, attr) is not None
        )
        return engine_args

    @classmethod
    def create(cls, **explicit: Any) -> "OmniEngineArgs":
        """Tracks caller-set fields for ``Omni(..., engine_args=ea)``."""
        ea = cls(**explicit)
        ea._explicit_fields = frozenset(explicit.keys())
        return ea

    def explicit_kwargs(self) -> dict[str, Any]:
        explicit = getattr(self, "_explicit_fields", None)
        if explicit is None:
            return {
                f.name: getattr(self, f.name) for f in dataclasses.fields(self) if getattr(self, f.name) is not None
            }
        return {k: getattr(self, k) for k in explicit}

    def _ensure_omni_models_registered(self):
        if hasattr(self, "_omni_models_registered"):
            return True
        register_omni_models_to_vllm()
        self._omni_models_registered = True
        return True

    def _patch_empty_hf_config(self, model_type: str) -> None:
        """For models with empty config.json (e.g. CosyVoice3), create a
        patched config in a temp directory with model_type set so that
        transformers AutoConfig.from_pretrained can resolve the config class.
        Sets self.hf_config_path to point to the patched directory."""
        try:
            from transformers import PretrainedConfig

            config_dict, _ = PretrainedConfig.get_config_dict(self.model)
            if config_dict.get("model_type"):
                return  # config.json already has model_type, no patching needed
        except Exception:
            return  # can't load config, let vLLM handle the error

        # Create a temp dir with a patched config.json
        temp_dir = tempfile.mkdtemp(prefix="omni_hf_config_")
        config_dict["model_type"] = model_type
        config_dict.setdefault("architectures", [self.model_arch])
        with open(os.path.join(temp_dir, "config.json"), "w") as f:
            json.dump(config_dict, f)
        self.hf_config_path = temp_dir
        self._temp_config_dir = temp_dir
        logger.info("Patched empty HF config with model_type=%s at %s", model_type, temp_dir)

    def create_model_config(self) -> OmniModelConfig:
        """Create an OmniModelConfig from these engine arguments.
        Returns:
            OmniModelConfig instance with all configuration fields set
        """
        # register omni models to avoid model not found error
        self._ensure_omni_models_registered()

        # Build stage_connector_config from stage_connector_spec
        stage_connector_config = {
            "name": self.stage_connector_spec.get("name", "SharedMemoryConnector"),
            "extra": self.stage_connector_spec.get("extra", {}).copy(),
        }
        stage_connector_config["extra"]["stage_id"] = self.stage_id

        # If model_arch is specified, inject it into hf_overrides so vLLM can
        # resolve the architecture even when config.json lacks 'architectures'.
        # Also inject model_type so AutoConfig can resolve the correct config
        # class for models with empty or missing config.json (e.g. CosyVoice3).
        if self.model_arch:
            if self.hf_overrides is None:
                self.hf_overrides = {}
            if isinstance(self.hf_overrides, dict):
                self.hf_overrides.setdefault("architectures", [self.model_arch])
                if "model_type" not in self.hf_overrides:
                    model_type = _ARCH_TO_MODEL_TYPE.get(self.model_arch)
                    if model_type is not None:
                        self.hf_overrides.setdefault("model_type", model_type)

                # Stage wrappers (e.g. Code2Wav) may need max_model_len larger
                # than the base checkpoint's text max_position_embeddings.
                if self.model_arch == "Qwen3TTSCode2Wav" and self.max_model_len is not None:
                    self.hf_overrides.setdefault("talker_config", {}).setdefault(
                        "max_position_embeddings", int(self.max_model_len)
                    )

            # For models whose HF config.json is empty or lacks model_type
            # (e.g. CosyVoice3), AutoConfig.from_pretrained fails because it
            # cannot determine which config class to use from the empty dict.
            # hf_overrides alone is not enough since transformers reads
            # model_type from config_dict before applying overrides.
            # Workaround: create a patched config.json in a temp directory
            # and point hf_config_path to it so vLLM reads model_type from it.
            if not self.hf_config_path:
                model_type = _ARCH_TO_MODEL_TYPE.get(self.model_arch)
                if model_type is not None:
                    self._patch_empty_hf_config(model_type)

        # Auto-detect tokenizer for models that store it in a subdirectory
        # rather than the root (e.g. CosyVoice3 uses CosyVoice-BlankEN/).
        if not self.tokenizer and self.model:
            model_path = self.model
            if os.path.isdir(model_path) and not os.path.isfile(os.path.join(model_path, "tokenizer_config.json")):
                for subfolder in sorted(os.listdir(model_path)):
                    candidate = os.path.join(model_path, subfolder)
                    if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "tokenizer_config.json")):
                        self.tokenizer = candidate
                        logger.info("Auto-detected tokenizer at %s", candidate)
                        break
            elif not os.path.isdir(model_path):
                subfolder = _TOKENIZER_SUBFOLDER_MAP.get(self.model_arch)
                if subfolder:
                    # Download just the tokenizer files from the subfolder
                    try:
                        from huggingface_hub import snapshot_download

                        local_dir = snapshot_download(
                            model_path,
                            allow_patterns=[
                                f"{subfolder}/tokenizer*",
                                f"{subfolder}/special_tokens*",
                                f"{subfolder}/vocab*",
                                f"{subfolder}/merges*",
                                f"{subfolder}/added_tokens*",
                            ],
                        )
                        candidate = os.path.join(local_dir, subfolder)
                        if os.path.isdir(candidate):
                            self.tokenizer = candidate
                            logger.info("Downloaded tokenizer from %s/%s", model_path, subfolder)
                    except Exception as e:
                        logger.warning("Failed to download tokenizer subfolder: %s", e)

        # Build the vLLM config first, then use it to create the Omni config.
        try:
            model_config = super().create_model_config()
        finally:
            # Clean up temp config dir if we created one
            if hasattr(self, "_temp_config_dir"):
                import shutil

                shutil.rmtree(self._temp_config_dir, ignore_errors=True)
                del self._temp_config_dir

        omni_config = OmniModelConfig.from_vllm_model_config(
            model_config=model_config,
            # All kwargs below are Omni specific
            stage_id=self.stage_id,
            async_chunk=self.async_chunk,
            model_stage=self.model_stage,
            model_arch=self.model_arch,
            worker_type=self.worker_type,
            engine_output_type=self.engine_output_type,
            hf_config_name=self.hf_config_name,
            custom_process_next_stage_input_func=self.custom_process_next_stage_input_func,
            stage_connector_config=stage_connector_config,
            subtalker_sampling_params=self.subtalker_sampling_params,
            omni_kv_config=self.omni_kv_config,
            task_type=self.task_type,
            has_sampling_extra_args=self.has_sampling_extra_args,
        )
        return omni_config


@dataclass
class OmniAsyncEngineArgs(AsyncEngineArgs, OmniEngineArgs):
    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = AsyncEngineArgs.add_cli_args(parser)
        parser = OmniEngineArgs._add_omni_specific_args(parser)
        return parser

    @property
    def output_modality(self) -> OutputModality:
        """Parse engine_output_type into a type-safe OutputModality flag."""
        return OutputModality.from_string(self.engine_output_type)


# ============================================================================
# CLI argument routing
# ============================================================================
#
# vLLM-Omni's CLI flags live in three buckets:
#
#     ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
#     │ OrchestratorArgs │    │  OmniEngineArgs  │    │  (upstream vllm) │
#     │                  │    │                  │    │    server/api    │
#     │  stage_timeout   │    │  max_num_seqs    │    │  host, port      │
#     │  worker_backend  │    │  gpu_mem_util    │    │  ssl_keyfile     │
#     │  deploy_config   │    │  dtype, quant    │    │  api_key         │
#     │     ...          │    │     ...          │    │     ...          │
#     └──────────────────┘    └──────────────────┘    └──────────────────┘
#             │                        │                        │
#             ▼                        ▼                        ▼
#        orchestrator              each stage               uvicorn /
#        consumes                  engine                   FastAPI
#
# Fields in ``SHARED_FIELDS`` (e.g. ``model``, ``log_stats``) flow to BOTH
# orchestrator and engine by design.
#
# Invariants enforced by ``tests/test_arg_utils.py``:
#
#   1. ``OrchestratorArgs`` ∩ ``OmniEngineArgs`` ⊆ ``SHARED_FIELDS``
#   2. Every CLI flag is classifiable into one of the three buckets
#   3. User-typed flags that match none of the above are logged as dropped
#
# Adding a new orchestrator-only flag → add a field to ``OrchestratorArgs``.
# Everything else is automatic.


@dataclass(frozen=True)
class OrchestratorArgs:
    """CLI flags consumed by the orchestrator.

    Contract: every field here is either
      (a) orchestrator-only (never needed by a stage engine), OR
      (b) orchestrator-read-then-redistributed (e.g. ``async_chunk`` is read
          from CLI, written to ``DeployConfig``, then propagated to every
          stage via ``merge_pipeline_deploy`` — not via direct kwargs
          forwarding).

    Fields that BOTH orchestrator and engine genuinely need (e.g. ``model``,
    ``log_stats``) should be listed in ``SHARED_FIELDS`` below; ``split_kwargs``
    will copy them to both buckets.
    """

    # === Lifecycle ===
    stage_init_timeout: int = 300
    init_timeout: int = 600

    # === Cross-stage Communication ===
    shm_threshold_bytes: int = 65536
    batch_timeout: int = 10

    # === Cluster / Backend ===
    worker_backend: str = "multi_process"
    ray_address: str | None = None

    # === Config Files ===
    stage_configs_path: str | None = None
    deploy_config: str | None = None
    stage_overrides: str | None = None  # raw JSON string; parsed downstream

    # === Mode Switches (orchestrator reads, DeployConfig redistributes) ===
    async_chunk: bool | None = None

    # === Observability ===
    log_stats: bool = False

    # === Headless Mode (also forwarded to engine — see SHARED_FIELDS) ===
    stage_id: int | None = None

    # === Pre-built Objects ===
    parallel_config: Any = None

    # === Multi-stage guards ===
    # --tokenizer is captured by the orchestrator and forwarded to stages
    # only when the stage does not define tokenizer/tokenizer_subdir itself.
    # Users wanting a per-stage tokenizer should set it in the deploy YAML.
    tokenizer: str | None = None


# Fields that live in BOTH OrchestratorArgs and OmniEngineArgs by design.
# Changes to this set are a review red flag — revisit the contract.
SHARED_FIELDS: frozenset[str] = frozenset(
    {
        "model",  # orch: detect model_type; engine: load weights
        "stage_id",  # orch: route (headless); engine: identity
        "log_stats",  # both want the flag
        "stage_configs_path",  # orch: load legacy YAML; engine: may reference for validation
    }
)


def orchestrator_field_names() -> frozenset[str]:
    """Return the names of every field on OrchestratorArgs."""
    return frozenset(f.name for f in fields(OrchestratorArgs))


def internal_blacklist_keys() -> frozenset[str]:
    """Return the set of CLI keys that must never be forwarded as per-stage
    engine overrides.

    Derived from ``OrchestratorArgs`` fields minus ``SHARED_FIELDS``, so
    adding a new orchestrator-owned flag is a one-line change to the
    dataclass — this function updates automatically.
    """
    return orchestrator_field_names() - SHARED_FIELDS


def split_kwargs(
    kwargs: dict[str, Any],
    *,
    engine_cls: type | None = None,
    user_typed: set[str] | None = None,
    strict: bool = False,
) -> tuple[OrchestratorArgs, dict[str, Any]]:
    """Partition CLI kwargs into (orchestrator, engine) buckets.

    Args:
        kwargs: Raw dict, typically ``vars(args)``.
        engine_cls: Engine dataclass used to whitelist-filter the engine
            bucket. Defaults to ``OmniEngineArgs``. Pass a custom class
            for testing.
        user_typed: Keys the user actually typed on the command line. Used
            to warn when a user-typed flag is unclassifiable.
        strict: If True, raise ``ValueError`` on ambiguous (double-classified
            but not in ``SHARED_FIELDS``) fields. Default False to keep the
            rollout non-breaking; flip to True in tests and CI.

    Returns:
        ``(orchestrator_args, engine_kwargs)``. ``engine_kwargs`` has already
        been whitelist-filtered against ``engine_cls`` — safe to pass directly
        to ``engine_cls(**engine_kwargs)``.
    """
    if engine_cls is None:
        engine_cls = OmniEngineArgs

    orch_fields = orchestrator_field_names()
    engine_fields = {f.name for f in fields(engine_cls)}

    orch_kwargs: dict[str, Any] = {}
    engine_candidate: dict[str, Any] = {}
    shared_values: dict[str, Any] = {}
    unclassified: dict[str, Any] = {}

    for key, value in kwargs.items():
        in_orch = key in orch_fields
        in_engine = key in engine_fields
        is_shared = key in SHARED_FIELDS

        if is_shared:
            shared_values[key] = value
        elif in_orch and in_engine:
            # Declared in both but not marked shared → ambiguous.
            msg = (
                f"Field {key!r} is defined on both OrchestratorArgs and "
                f"{engine_cls.__name__} but is not in SHARED_FIELDS. "
                f"This causes double-routing. Either remove the duplicate or "
                f"add {key!r} to SHARED_FIELDS if the sharing is intentional."
            )
            if strict:
                raise ValueError(msg)
            logger.error(msg)
            # Default: treat as orchestrator-only to preserve existing behavior.
            orch_kwargs[key] = value
        elif in_orch:
            orch_kwargs[key] = value
        elif in_engine:
            engine_candidate[key] = value
        else:
            unclassified[key] = value

    # Warn on user-typed but unclassifiable flags so we don't silently drop
    # something the user cared about (fixes the class of bug that spawned #873).
    if unclassified and user_typed:
        user_typed_unknown = sorted(k for k in unclassified if k in user_typed)
        if user_typed_unknown:
            logger.warning(
                "CLI flags not consumed by vllm-omni and dropped before "
                "per-stage engine construction: %s. If these are vllm "
                "frontend/uvicorn flags (host, port, ssl_*, api_key, …) this "
                "is expected; otherwise check your spelling.",
                user_typed_unknown,
            )

    # Engine bucket: shared + engine-only. We do NOT pass through unclassified
    # fields — that's exactly the server/uvicorn noise we want to shed.
    engine_kwargs = {**shared_values, **engine_candidate}

    # Construct the orchestrator dataclass. Shared fields that OrchestratorArgs
    # also declares get copied into its constructor.
    orch_init: dict[str, Any] = dict(orch_kwargs)
    for key, value in shared_values.items():
        if key in orch_fields:
            orch_init[key] = value
    orch_args = OrchestratorArgs(**orch_init)

    return orch_args, engine_kwargs


def derive_server_dests_from_vllm_parser() -> frozenset[str]:
    """Derive the set of argparse dests that belong to vllm's frontend/server.

    Returns every dest registered by ``make_arg_parser`` that is NOT a field
    of ``OmniEngineArgs`` and NOT a field of ``OrchestratorArgs``. Useful for
    CI tests to assert all CLI flags are classifiable without maintaining
    a hardcoded server list.

    Returns empty frozenset if vllm's parser cannot be built (e.g. in a
    minimal test environment).
    """
    try:
        from vllm.entrypoints.openai.cli_args import make_arg_parser
        from vllm.utils.argparse_utils import FlexibleArgumentParser
    except ImportError:
        logger.debug("Cannot import vllm parser — server-dest derivation skipped")
        return frozenset()

    try:
        parser = make_arg_parser(FlexibleArgumentParser())
        all_dests = {a.dest for a in parser._actions if a.dest and a.dest != "help"}
    except Exception as exc:
        logger.debug("Failed to build vllm parser: %s", exc)
        return frozenset()

    engine_fields = {f.name for f in fields(OmniEngineArgs)}
    orch_fields = orchestrator_field_names()

    return frozenset(all_dests - engine_fields - orch_fields - SHARED_FIELDS)


def orchestrator_args_from_argparse(args: Any) -> OrchestratorArgs:
    """Build an ``OrchestratorArgs`` from an ``argparse.Namespace``.

    Only copies attributes that exist on the namespace — missing fields fall
    back to the dataclass default. Useful when the full parser is already
    built and ``vars(args)`` would include noise.
    """
    kwargs: dict[str, Any] = {}
    for f in fields(OrchestratorArgs):
        if hasattr(args, f.name):
            value = getattr(args, f.name)
            if value is not None or f.default is None:
                kwargs[f.name] = value
    return OrchestratorArgs(**kwargs)


# ============================================================================
# OmniArgumentParser — model-aware default injection
# ============================================================================


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*; override keys win.

    Keys are normalised to strings to avoid duplicate keys when
    callers provide integer keys alongside string keys."""
    result = dict(base)
    for key, val in override.items():
        key_str = str(key)
        if key_str in result and isinstance(result[key_str], dict) and isinstance(val, dict):
            result[key_str] = _deep_merge(result[key_str], val)
        else:
            result[key_str] = val
    return result


class OmniArgumentParser(FlexibleArgumentParser):
    """FlexibleArgumentParser that injects model-specific defaults from deploy YAML.

    On ``parse_args()``, peeks the model from raw ``sys.argv``, detects its
    ``model_type`` via HuggingFace config, loads the corresponding
    ``deploy/<model_type>.yaml``, and sets YAML values as ``action.default`` on
    the serve subparser.  Argparse then uses these as fallback values — any
    explicit CLI flag typed by the user naturally takes precedence.

    Per-stage fields in normal-serve mode are collapsed into a JSON default on
    ``--stage-overrides`` and deep-merged post-parse with any user-supplied
    ``--stage-overrides`` value.
    """

    def parse_args(self, args: list[str] | None = None, namespace: argparse.Namespace | None = None):
        if args is None:
            args = sys.argv[1:]

        # Clear stale state from a previous parse_args() call (fixes P4).
        self._yaml_stage_overrides = None

        # Gate: only inject defaults for "serve" subcommand or flat-parser
        # mode (api_server direct launch).  Avoids unnecessary HF config
        # fetches for unrelated subcommands like "bench" (fixes P3).
        if self._is_serve_or_flat(args) and not self._is_help_or_version(args):
            # Use flat-parser peek when there are no subparsers (api_server),
            # otherwise use the standard serve-subcommand peek.
            if self._subparsers is None:
                model = self._peek_model_flat(args)
            else:
                model = self._peek_model(args)
            if model:
                stage_id = self._peek_stage_id(args)
                deploy_config_path = self._peek_deploy_config(args)
                self._inject_model_defaults(model, stage_id, deploy_config_path=deploy_config_path)

        result = super().parse_args(args, namespace)

        # Post-parse: deep-merge stashed YAML stage_overrides with user value.
        # Only apply when stage_overrides is present in the parsed namespace
        # and we actually stashed yaml defaults (fixes P4).
        yaml_overrides = getattr(self, "_yaml_stage_overrides", None)
        if yaml_overrides is not None and hasattr(result, "stage_overrides"):
            user_raw = getattr(result, "stage_overrides", None)
            if user_raw:
                user_overrides = json.loads(user_raw) if isinstance(user_raw, str) else user_raw
                merged = _deep_merge(yaml_overrides, user_overrides)
            else:
                merged = yaml_overrides
            setattr(result, "stage_overrides", json.dumps(merged, sort_keys=True))

        return result

    # ── Peek helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _is_help_or_version(args: list[str]) -> bool:
        for a in args:
            if a in ("--help", "-h", "--version", "-v") or a.startswith("--help="):
                return True
        return False

    def _is_serve_or_flat(self, args: list[str]) -> bool:
        """True when injection should proceed: ``serve`` subcommand or flat-parser (api_server)."""
        if self._subparsers is None:
            return True
        return bool(args) and args[0] == "serve"

    @classmethod
    def _peek_model(cls, args: list[str]) -> str | None:
        """Extract model from raw CLI args before full preprocessing.

        Handles the cases FlexibleArgumentParser handles:
        1. --config config.yaml / --config=config.yaml → reads yaml['model']
        2. --model foo/bar → promoted to positional by FlexibleArgumentParser
        3. vllm serve foo/bar → first positional after subcommand
        4. python -m ...api_server foo/bar → first positional (flat parser)
        """
        current = list(args)

        # Case 1: --config YAML (both --config <path> and --config=<path>)
        for i, arg in enumerate(current):
            path: str | None = None
            if arg == "--config" and i + 1 < len(current):
                path = current[i + 1]
            elif arg.startswith("--config="):
                path = arg.split("=", 1)[1]
            if path:
                try:
                    with open(path) as fh:
                        cfg = yaml.safe_load(fh)
                    if isinstance(cfg, dict) and cfg.get("model"):
                        return cfg["model"]
                except Exception:
                    pass

        # Case 2: --model flag
        for i, arg in enumerate(current):
            if arg == "--model" and i + 1 < len(current):
                return current[i + 1]
            if arg.startswith("--model="):
                return arg.split("=", 1)[1]

        # Case 3: positional model for "serve" subcommand
        if current and current[0] == "serve" and len(current) > 1:
            if not current[1].startswith("-"):
                return current[1]

        return None

    def _peek_model_flat(self, args: list[str]) -> str | None:
        """Peek model for flat-parser mode (no subparsers — api_server direct launch)."""
        current = list(args)
        if current and not current[0].startswith("-"):
            return current[0]
        return None

    @staticmethod
    def _peek_stage_id(args: list[str]) -> int | None:
        for i, arg in enumerate(args):
            if arg == "--stage-id" and i + 1 < len(args):
                try:
                    return int(args[i + 1])
                except (ValueError, IndexError):
                    pass
            if arg.startswith("--stage-id="):
                try:
                    return int(arg.split("=", 1)[1])
                except ValueError:
                    pass
        return None

    @staticmethod
    def _peek_deploy_config(args: list[str]) -> str | None:
        """Peek --deploy-config from raw argv (supports both --deploy-config <p> and --deploy-config=<p>)."""
        for i, arg in enumerate(args):
            if arg == "--deploy-config" and i + 1 < len(args):
                return args[i + 1]
            if arg.startswith("--deploy-config="):
                return arg.split("=", 1)[1]
        return None

    # ── Default injection ───────────────────────────────────────────────────

    def _inject_model_defaults(
        self, model: str, stage_id: int | None, *, deploy_config_path: str | None = None
    ) -> None:
        """Load deploy YAML and set ``action.default`` on the serve subparser.

        If *deploy_config_path* is given (from ``--deploy-config``), that file
        is used directly.  Otherwise the per-model ``deploy/<model_type>.yaml``
        is resolved from the HuggingFace config.
        """
        from dataclasses import fields as dc_fields
        from pathlib import Path

        from vllm_omni.config.stage_config import (
            _DEPLOY_DIR,
            DeployConfig,
            StageConfigFactory,
            StageDeployConfig,
            load_deploy_config,
        )

        if deploy_config_path:
            deploy_path = Path(deploy_config_path)
            if not deploy_path.exists():
                return
        else:
            model_type, _hf_config = StageConfigFactory._auto_detect_model_type(model)
            if not model_type:
                return
            deploy_path = _DEPLOY_DIR / f"{model_type}.yaml"
            if not deploy_path.exists():
                return

        deploy = load_deploy_config(deploy_path)
        serve_parser = self._get_serve_subparser()
        if serve_parser is None:
            if self._subparsers is None:
                serve_parser = self  # flat-parser mode (api_server direct invocation)
            else:
                return

        # ── Pipeline-wide: DeployConfig fields → action.default ──
        for f in dc_fields(DeployConfig):
            val = getattr(deploy, f.name)
            if val is not None:
                self._set_action_default(serve_parser, f.name, val)

        # ── Per-stage ──
        if stage_id is not None:
            # Headless: set per-stage action.defaults directly
            stage_dep = next((s for s in deploy.stages if s.stage_id == stage_id), None)
            if stage_dep:
                for f in dc_fields(StageDeployConfig):
                    if f.name == "engine_extras":
                        continue
                    val = getattr(stage_dep, f.name)
                    if val is not None:
                        self._set_action_default(serve_parser, f.name, val)
        else:
            # Normal serve: collapse all stages into stage_overrides JSON
            stage_overrides: dict[str, dict[str, Any]] = {}
            for stage in deploy.stages:
                d: dict[str, Any] = {}
                for f in dc_fields(StageDeployConfig):
                    if f.name == "engine_extras":
                        continue
                    val = getattr(stage, f.name)
                    if val is not None:
                        d[f.name] = val
                if d:
                    stage_overrides[str(stage.stage_id)] = d

            if stage_overrides:
                yaml_json = json.dumps(stage_overrides, sort_keys=True)
                self._set_action_default(serve_parser, "stage_overrides", yaml_json)
                self._yaml_stage_overrides = stage_overrides

    @staticmethod
    def _set_action_default(parser: argparse.ArgumentParser, dest: str, value: Any) -> None:
        for action in parser._actions:
            if action.dest == dest:
                action.default = value
                break

    def _get_serve_subparser(self) -> argparse.ArgumentParser | None:
        if self._subparsers is None:
            return None
        subparsers_action = self._subparsers._group_actions[0]
        return subparsers_action.choices.get("serve")


# ============================================================================
# End OmniArgumentParser
# ============================================================================


def nullify_stage_engine_defaults(parser: argparse.ArgumentParser) -> None:
    """Reset stage-level engine flag defaults to ``None``; preserve real
    default in help text. Only deploy-YAML override fields are touched.
    Idempotent.

    .. deprecated::
        Use :class:`OmniArgumentParser` instead — it injects model-specific
        defaults from deploy YAML directly as argparse defaults, eliminating
        the need for nullify-then-refill.
    """
    import warnings

    warnings.warn(
        "nullify_stage_engine_defaults is deprecated. "
        "Use OmniArgumentParser instead, which injects model-specific "
        "defaults from deploy YAML directly.",
        DeprecationWarning,
        stacklevel=2,
    )
    from vllm_omni.config.stage_config import deploy_override_field_names

    override_dests = deploy_override_field_names()

    for action in parser._actions:
        if action.dest in ("help", "version") or not action.option_strings:
            continue
        if action.dest not in override_dests:
            continue
        if action.default is None or action.default is argparse.SUPPRESS:
            continue
        if action.help and "(default:" not in action.help and "%(default)" not in action.help:
            action.help = f"{action.help} (default: {action.default})"
        action.default = None

    parser._omni_nullified = True  # type: ignore[attr-defined]
