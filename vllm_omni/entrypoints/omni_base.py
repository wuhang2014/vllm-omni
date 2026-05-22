from __future__ import annotations

import os
import time
import weakref
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import huggingface_hub
from vllm.logger import init_logger
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError

from vllm_omni.config import VllmOmniConfig
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.engine.messages import (
    EngineQueueMessage,
    ErrorMessage,
    OutputMessage,
    StageMetricsMessage,
)
from vllm_omni.entrypoints.client_request_state import ClientRequestState
from vllm_omni.entrypoints.pd_utils import PDDisaggregationMixin
from vllm_omni.entrypoints.utils import coerce_param_message_types, get_final_stage_id_for_e2e
from vllm_omni.metrics.stats import OrchestratorAggregator as OrchestratorMetrics
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific
from vllm_omni.outputs import OmniRequestOutput

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)


class OmniEngineDeadError(EngineDeadError):
    _DEFAULT_MESSAGE = EngineDeadError().args[0]
    error_stage_id: int | None

    def __init__(
        self,
        message: str | None = None,
        *,
        error_stage_id: int | None = None,
        suppress_context: bool = False,
    ) -> None:
        resolved_message = message or self._DEFAULT_MESSAGE
        Exception.__init__(self, resolved_message)
        self.__suppress_context__ = suppress_context
        self.error_stage_id = error_stage_id


def _weak_shutdown_engine(engine: AsyncOmniEngine) -> None:
    """Best-effort engine cleanup for GC finalization."""
    try:
        engine.shutdown()
    except Exception:
        pass


def omni_snapshot_download(model_id: str) -> str:
    if os.path.exists(model_id):
        return model_id

    # TODO: this is just a workaround for quickly use modelscope, we should support
    # modelscope in weight loading feature instead of using `snapshot_download`
    if os.environ.get("VLLM_USE_MODELSCOPE", False):
        from modelscope.hub.snapshot_download import snapshot_download

        return snapshot_download(model_id)

    try:
        download_weights_from_hf_specific(
            model_name_or_path=model_id,
            cache_dir=None,
            allow_patterns=["*"],
            require_all=True,
        )
    except huggingface_hub.errors.GatedRepoError:
        raise ValueError(
            f"Access to model '{model_id}' is restricted. "
            f"Visit https://huggingface.co/{model_id} to accept "
            f"the license and request access."
        )
    except huggingface_hub.errors.RepositoryNotFoundError:
        raise ValueError(f"Repository not found for '{model_id}'. Please check the model name or path.")
    except PermissionError:
        logger.warning(
            "Permission denied when downloading '%s'. Assuming the model is already cached locally.",
            model_id,
        )

    return model_id


OutputMessageHandleResult = tuple[Literal[True], None, None, None] | tuple[Literal[False], str, int, ClientRequestState]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*; override keys win.

    Stage IDs are normalised to strings to avoid duplicate keys when
    callers provide integer keys alongside YAML's string keys.
    """
    result = dict(base)
    for key, val in override.items():
        key_str = str(key)
        if key_str in result and isinstance(result[key_str], dict) and isinstance(val, dict):
            result[key_str] = _deep_merge(result[key_str], val)
        else:
            result[key_str] = val
    return result


def _inject_deploy_defaults(model: str, kwargs: dict[str, Any]) -> None:
    """Inject model-specific defaults from deploy YAML into *kwargs* in place.

    Pipeline-wide ``DeployConfig`` fields and (for headless) per-stage
    ``StageDeployConfig`` fields are set via ``kwargs.setdefault`` so explicit
    user-supplied values always win.  For normal serve, per-stage defaults
    are deep-merged into ``stage_overrides``.

    Also loads ``PipelineConfig`` from registry and serialises its per-stage
    topology into ``stage_overrides`` alongside deploy YAML values.

    If *kwargs* contains ``deploy_config`` (from ``--deploy-config``), that
    file is used directly.  Otherwise ``deploy/<model_type>.yaml`` is
    resolved from the HuggingFace config.

    This is the **offline-path** counterpart to ``OmniArgumentParser``.
    """
    import json as _json
    from dataclasses import fields as dc_fields
    from pathlib import Path

    from vllm_omni.config.stage_config import (
        _DEPLOY_DIR,
        _PIPELINE_REGISTRY,
        DeployConfig,
        StageConfigFactory,
        StageDeployConfig,
        load_deploy_config,
    )
    from vllm_omni.engine.arg_utils import (
        _build_unified_stage_overrides,
    )

    deploy_config_path = kwargs.get("deploy_config")
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

    try:
        deploy = load_deploy_config(deploy_path)
    except Exception:
        import logging

        _logger = logging.getLogger("vllm_omni.entrypoints.omni_base")
        _logger.warning("Failed to load deploy config from %s", deploy_path)
        return

    # Pipeline-wide: fill missing from DeployConfig.
    for f in dc_fields(DeployConfig):
        val = getattr(deploy, f.name)
        if val is not None:
            kwargs.setdefault(f.name, val)

    # Pipeline topology from registry.
    pipeline_key = deploy.pipeline or (model_type if "model_type" in dir() else None)
    # model_type may not be defined if deploy_config_path was used.
    try:
        pipeline = _PIPELINE_REGISTRY.get(pipeline_key) if pipeline_key else None
    except Exception:
        pipeline = None

    # Per-stage.
    stage_id = kwargs.get("stage_id")
    if stage_id is not None:
        # Headless: fill per-stage kwargs directly.
        stage_dep = next((s for s in deploy.stages if s.stage_id == stage_id), None)
        if stage_dep:
            for f in dc_fields(StageDeployConfig):
                if f.name == "engine_extras":
                    continue
                val = getattr(stage_dep, f.name)
                if val is not None:
                    kwargs.setdefault(f.name, val)
    else:
        # Normal serve: build unified stage_overrides from pipeline + deploy.
        overrides = _build_unified_stage_overrides(deploy, pipeline)
        if overrides:
            existing = kwargs.get("stage_overrides")
            if existing:
                if isinstance(existing, str):
                    existing = _json.loads(existing)
                kwargs["stage_overrides"] = _json.dumps(_deep_merge(overrides, existing), sort_keys=True)
            else:
                kwargs["stage_overrides"] = _json.dumps(overrides, sort_keys=True)


class OmniBase(PDDisaggregationMixin):
    """Shared runtime foundation for AsyncOmni and Omni."""

    def __init__(
        self,
        model: str,
        *,
        omni_config: VllmOmniConfig | None = None,
        **kwargs: Any,
    ) -> None:
        if "log_requests" in kwargs:
            raise TypeError("`log_requests` has been removed in Omni/AsyncOmni. Use `log_stats`.")

        output_modalities = kwargs.pop("output_modalities", None)
        self._enable_ar_profiler = kwargs.pop("enable_ar_profiler", False)
        self.tts_batch_max_items: int = kwargs.pop("tts_batch_max_items", 32)

        # Build or receive the unified config.
        if omni_config is not None:
            pass  # already resolved
        else:
            _inject_deploy_defaults(model, kwargs)
            from vllm_omni.engine.arg_utils import OmniEngineArgs as _EA  # noqa: N814

            engine_args = _EA.from_kwargs(model, **kwargs)
            omni_config = engine_args.create_omni_config(model)

        model = omni_snapshot_download(model)
        self.__dict__["_name"] = self.__class__.__name__
        self.model = model
        self.omni_config = omni_config
        self.async_chunk = omni_config.async_chunk
        self.log_stats = omni_config.log_stats
        self.output_modalities = output_modalities or []

        logger.info("[%s] Initializing with model %s", self.__class__.__name__, model)
        st = time.time()
        self.engine = AsyncOmniEngine(model=model, omni_config=omni_config)
        self._shutdown_called = False
        self._weak_finalizer = weakref.finalize(self, _weak_shutdown_engine, self.engine)
        et = time.time()
        logger.info("[%s] AsyncOmniEngine initialized in %.2f seconds", self.__class__.__name__, et - st)
        self.async_chunk = self.omni_config.async_chunk

        self.request_states: dict[str, ClientRequestState] = {}

        self.default_sampling_params_list = self.engine.default_sampling_params_list
        if not self.output_modalities:
            self.output_modalities = [
                self.engine.get_stage_metadata(i).final_output_type for i in range(self.engine.num_stages)
            ]

        self._stage_meta_list = [self.engine.get_stage_metadata(i) for i in range(self.engine.num_stages)]

        logger.info(
            "[%s] Initialized with %s stages for model %s",
            self.__class__.__name__,
            self.engine.num_stages,
            model,
        )

        # PD disaggregation state (detects if a prefill/decode stage pair is configured)
        self._init_pd_state()

    @property
    def num_stages(self) -> int:
        return self.engine.num_stages

    @property
    def stage_configs(self) -> list:
        """Expose engine stage configs for PD disaggregation detection and validation."""
        return self.engine.stage_configs

    def _has_dead_stage(self) -> bool:
        for stage_client in self.engine.stage_clients:
            if getattr(stage_client, "_engine_dead", False):
                return True
            resources = getattr(stage_client, "resources", None)
            if resources is not None and getattr(resources, "engine_dead", False):
                return True
        return False

    @property
    def is_running(self) -> bool:
        return self.engine.is_alive() and not self._has_dead_stage()

    @property
    def errored(self) -> bool:
        """Whether the engine is in a non-recoverable error state.

        True when the orchestrator thread is dead **or** any stage client
        has been marked dead (e.g. diffusion worker OOM / process death).

        Checks both ``_engine_dead`` (StageDiffusionClient) and
        ``resources.engine_dead`` (StageEngineCoreClient / AsyncMPClient)
        since the two client types store the flag differently.
        """
        return not self.engine.is_alive() or self._has_dead_stage()

    def check_health(self) -> None:
        if not self.engine.is_alive():
            raise EngineDeadError("Orchestrator process is not alive")
        for stage_client in self.engine.stage_clients:
            if hasattr(stage_client, "check_health"):
                stage_client.check_health()

    def resolve_sampling_params_list(
        self,
        sampling_params_list: Sequence[Any] | Any | None,
        allow_delta_coercion: bool = False,
    ) -> Sequence[Any]:
        if sampling_params_list is None:
            normalized = self.default_sampling_params_list
            # Set the output kind to delta since no params were specified
            if allow_delta_coercion:
                normalized = coerce_param_message_types(list(normalized), is_streaming=True)

        elif isinstance(sampling_params_list, Sequence) and not isinstance(sampling_params_list, (str, bytes)):
            normalized = sampling_params_list
        elif self.num_stages == 1:
            normalized = [sampling_params_list]
        else:
            raise ValueError(f"Expected {self.num_stages} sampling params, got a single sampling params object")
        if len(normalized) != self.num_stages:
            raise ValueError(f"Expected {self.num_stages} sampling params, got {len(normalized)}")
        return normalized

    def _log_summary_and_cleanup(self, request_id: str) -> None:
        req_state = self.request_states.get(request_id)
        try:
            if req_state is None or req_state.metrics is None:
                return
        except Exception:
            logger.exception(
                "[%s] Failed to build/log summary for req=%s",
                self.__class__.__name__,
                request_id,
            )
        finally:
            self.request_states.pop(request_id, None)

    def _compute_final_stage_id(self, output_modalities: list[str] | None) -> int:
        return get_final_stage_id_for_e2e(
            output_modalities,
            self.output_modalities,
            self._stage_meta_list,
        )

    def _process_stage_metrics_message(self, msg: StageMetricsMessage) -> None:
        req_id = msg.request_id
        req_state = self.request_states.get(req_id)
        if req_state is None or req_state.metrics is None:
            return
        _m = msg.metrics
        stage_id = msg.stage_id
        req_state.metrics.on_stage_metrics(stage_id, req_id, _m)
        submit_ts = msg.stage_submit_ts
        now = time.time()
        if req_state.metrics.stage_first_ts[stage_id] is None:
            req_state.metrics.stage_first_ts[stage_id] = submit_ts if submit_ts is not None else now
        req_state.metrics.stage_last_ts[stage_id] = max(req_state.metrics.stage_last_ts[stage_id] or 0.0, now)

    def _handle_output_message(
        self,
        msg: EngineQueueMessage | None,
    ) -> OutputMessageHandleResult:
        """Handle one Orchestrator output-queue message."""
        if msg is None:
            return True, None, None, None

        if isinstance(msg, StageMetricsMessage):
            self._process_stage_metrics_message(msg)
            return True, None, None, None

        if isinstance(msg, ErrorMessage):
            if msg.fatal:
                raise OmniEngineDeadError(
                    msg.error,
                    error_stage_id=msg.stage_id,
                )
            raise RuntimeError(msg.error)

        if not isinstance(msg, OutputMessage):
            logger.warning("[%s] got unexpected msg type: %s", self.__class__.__name__, msg.type)
            return True, None, None, None

        req_id = msg.request_id
        stage_id = msg.stage_id

        req_state = self.request_states.get(req_id)
        if req_state is None:
            logger.debug(
                "[%s] dropping output for unknown req %s",
                self.__class__.__name__,
                req_id,
            )
            return True, None, None, None

        req_state.stage_id = stage_id

        return False, req_id, stage_id, req_state

    def _check_engine_output_error(
        self,
        result: OutputMessage,
        request_id: str,
        stage_id: int,
    ) -> None:
        """Raise if ``engine_outputs`` carries an error field.

        Raises :class:`EngineDeadError` when ``self.errored`` indicates the
        engine is unrecoverable, otherwise raises :class:`EngineGenerateError`
        (recoverable, single-request failure).
        """
        engine_outputs = result.engine_outputs
        error_text = getattr(engine_outputs, "error", None)
        if error_text is None:
            return
        logger.error(
            "[%s] Stage error for req=%s stage-%s: %s",
            self.__class__.__name__,
            request_id,
            stage_id,
            error_text,
        )
        # NOTE: O(n_stages) check for every error.
        if self.errored:
            raise OmniEngineDeadError(
                error_text,
                error_stage_id=stage_id,
            )
        raise EngineGenerateError(error_text)

    def _process_single_result(
        self,
        result: OutputMessage,
        stage_id: int,
        metrics: OrchestratorMetrics,
        req_start_ts: dict[str, float],
        wall_start_ts: float,
        final_stage_id_for_e2e: int,
    ) -> OmniRequestOutput | None:
        req_id = result.request_id
        engine_outputs = result.engine_outputs
        stage_durations = getattr(engine_outputs, "stage_durations", {})
        peak_memory_mb = getattr(engine_outputs, "peak_memory_mb", 0.0)

        # Merge AR stage timing from OrchestratorAggregator.stage_events
        if self._enable_ar_profiler:
            ar_events = metrics.stage_events.get(str(req_id), [])
            for evt in ar_events:
                if evt.stage_id != stage_id:
                    stage_durations[f"ar_stage_{evt.stage_id}"] = evt.stage_gen_time_ms / 1000.0

        # Merge pipeline timings from Orchestrator into stage_durations
        _m = result.metrics
        if _m is not None and hasattr(_m, "pipeline_timings") and _m.pipeline_timings:
            for key, value in _m.pipeline_timings.items():
                if key not in stage_durations:
                    stage_durations[key] = value

        # Merge per-stage gen times into stage_durations
        for evt in metrics.stage_events.get(str(req_id), []):
            key = f"stage_{evt.stage_id}_gen_ms"
            if key not in stage_durations:
                stage_durations[key] = evt.stage_gen_time_ms
        # Current stage gen time (not yet in stage_events at this point)
        if _m is not None:
            stage_durations.setdefault(f"stage_{stage_id}_gen_ms", _m.stage_gen_time_ms)

        finished = engine_outputs.finished

        submit_ts = result.stage_submit_ts
        now = time.time()
        if metrics.stage_first_ts[stage_id] is None:
            metrics.stage_first_ts[stage_id] = submit_ts if submit_ts is not None else now
        metrics.stage_last_ts[stage_id] = max(metrics.stage_last_ts[stage_id] or 0.0, now)

        _m = result.metrics
        if finished and _m is not None:
            metrics.on_stage_metrics(stage_id, req_id, _m)

        stage_meta = self.engine.get_stage_metadata(stage_id)
        if not stage_meta.final_output:
            return None

        try:
            rid_key = str(req_id)
            if stage_id == final_stage_id_for_e2e and rid_key not in metrics.e2e_done and finished:
                metrics.on_finalize_request(
                    stage_id,
                    req_id,
                    req_start_ts.get(req_id, wall_start_ts),
                )
        except Exception:
            logger.exception("[%s] Finalize request handling error", self.__class__.__name__)

        output_type = getattr(engine_outputs, "final_output_type", stage_meta.final_output_type)
        images = getattr(engine_outputs, "images", []) if output_type == "image" else []
        return OmniRequestOutput(
            request_id=req_id or "",
            stage_id=stage_id,
            final_output_type=output_type,
            request_output=engine_outputs,
            images=images,
            trajectory_latents=getattr(engine_outputs, "trajectory_latents", None),
            trajectory_timesteps=getattr(engine_outputs, "trajectory_timesteps", None),
            trajectory_log_probs=getattr(engine_outputs, "trajectory_log_probs", None),
            trajectory_decoded=getattr(engine_outputs, "trajectory_decoded", None),
            _custom_output=getattr(engine_outputs, "_custom_output", {}),
            stage_durations=stage_durations,
            peak_memory_mb=peak_memory_mb,
        )

    def shutdown(self) -> None:
        logger.info("[%s] Shutting down", self.__class__.__name__)
        self._shutdown_base()

    def close(self) -> None:
        self.shutdown()

    def start_profile(
        self,
        profile_prefix: str | None = None,
        stages: list[int] | None = None,
    ) -> list[Any]:
        """Start profiling specified stages.

        Uses vLLM-compatible profile(is_start=True, profile_prefix) interface.

        Args:
            profile_prefix: Optional prefix for the trace file names.
            stages: List of stage IDs to profile. If None, profiles all stages.

        Returns:
            List of results from each stage.
        """
        return self.engine.collective_rpc(method="profile", args=(True, profile_prefix), stage_ids=stages)

    def stop_profile(self, stages: list[int] | None = None) -> list[Any]:
        """Stop profiling specified stages.

        Uses vLLM-compatible profile(is_start=False) interface.

        Args:
            stages: List of stage IDs to profile. If None, stops all stages.

        Returns:
            List of results from each stage.
        """
        return self.engine.collective_rpc(method="profile", args=(False, None), stage_ids=stages)

    def _shutdown_base(self) -> None:
        if getattr(self, "_shutdown_called", False):
            return
        self._shutdown_called = True
        finalizer = getattr(self, "_weak_finalizer", None)
        if finalizer is not None and finalizer.alive:
            finalizer.detach()
        self.engine.shutdown()
