"""
Omni serve command for vLLM-Omni.

Supports both multi-stage LLM models (e.g., Qwen2.5-Omni) and
diffusion models (e.g., Qwen-Image) through the same CLI interface.
"""

import argparse
import os
import signal
import threading
from multiprocessing import connection
from types import FrameType
from typing import Any

import uvloop
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG
from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.engine.arg_utils import OmniArgumentParser
from vllm_omni.entrypoints.cli.logo import log_logo
from vllm_omni.entrypoints.openai.api_server import omni_run_server

logger = init_logger(__name__)

DESCRIPTION = """Launch a local OpenAI-compatible API server to serve Omni models
via HTTP. Supports both multi-stage LLM models and diffusion models.

The server automatically detects the model type:
- LLM models: Served via /v1/chat/completions endpoint
- Diffusion models: Served via /v1/images/generations endpoint

Examples:
  # Start an Omni LLM server
  vllm serve Qwen/Qwen2.5-Omni-7B --omni --port 8091

  # Start a diffusion model server
  vllm serve Qwen/Qwen-Image --omni --port 8091

Search by using: `--help=<ConfigGroup>` to explore options by section (e.g.,
--help=OmniConfig)
  Use `--help=all` to show all available flags at once.
"""


def _ensure_vllm_platform():
    """Ensure vLLM's current_platform is valid before arg parsing.

    Upstream vLLM's argument parser now instantiates DeviceConfig during
    ``make_arg_parser``, which requires a resolved platform with a non-empty
    ``device_type``.  In some environments (e.g. editable installs with
    broken package metadata), vLLM's own platform auto-detection may fail
    and fall back to ``UnspecifiedPlatform``.  When that happens, use the
    Omni platform (which has its own detection logic) as a drop-in
    replacement so that argument parsing succeeds.
    """
    from vllm import platforms as vllm_platforms

    if vllm_platforms.current_platform.is_unspecified():
        from vllm_omni.platforms import current_omni_platform

        if not current_omni_platform.is_unspecified():
            vllm_platforms.current_platform = current_omni_platform
            logger.debug(
                "Replaced vLLM UnspecifiedPlatform with omni platform %s",
                type(current_omni_platform).__name__,
            )
        else:
            from vllm.platforms.cpu import CpuPlatform

            vllm_platforms.current_platform = CpuPlatform()
            logger.debug(
                "Both vLLM and omni platforms are unspecified, falling back to CpuPlatform for arg parsing",
            )


class OmniServeCommand(CLISubcommand):
    """The `serve` subcommand for the vLLM CLI."""

    name = "serve"
    # Parser stashed at subparser_init so ``cmd`` can resolve each user-typed
    # flag to its real ``dest`` via the parser's action table.
    _parser: FlexibleArgumentParser | None = None

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        if not os.environ.get("VLLM_DISABLE_LOG_LOGO"):
            os.environ["VLLM_DISABLE_LOG_LOGO"] = "1"
            log_logo()

        # If model is specified in CLI (as positional arg), it takes precedence
        if hasattr(args, "model_tag") and args.model_tag is not None:
            args.model = args.model_tag

        if args.headless:
            run_headless(args)
        else:
            uvloop.run(omni_run_server(args))

    def validate(self, args: argparse.Namespace) -> None:
        if args.stage_id is not None and (args.omni_master_address is None or args.omni_master_port is None):
            raise ValueError("--stage-id requires both --omni-master-address and --omni-master-port to be set")

        # --omni-replica-address is only consulted in run_headless(); reject it
        # on the head so a misconfigured launch fails loudly instead of being
        # silently ignored.
        if getattr(args, "omni_replica_address", None) is not None and not args.headless:
            raise ValueError("--omni-replica-address requires --headless to be set")

        # --omni-dp-size-local is process-local. A value other than 1 only
        # makes sense when this process owns a stage (head or headless).
        omni_dp_size_local = getattr(args, "omni_dp_size_local", None)
        if omni_dp_size_local is not None:
            if omni_dp_size_local < 1:
                raise ValueError(f"--omni-dp-size-local must be >= 1, got {omni_dp_size_local}")
            if omni_dp_size_local != 1 and args.stage_id is None:
                raise ValueError("--omni-dp-size-local != 1 requires --stage-id to be set")

        # vLLM CLI args that omni does not honor: parallelism comes from the
        # per-stage YAML (parallel_config:, enable_expert_parallel:) and the
        # process-local replica count from --omni-dp-size-local. Passing the
        # vLLM equivalents on the command line would silently disagree with
        # those sources of truth, so reject them at parse time.
        if getattr(args, "omni", False):
            explicit_cli_keys: set[str] = getattr(args, "_cli_explicit_keys", set()) or set()
            prohibited_with_omni: dict[str, str] = {
                "data_parallel_size": "--data-parallel-size",
                "data_parallel_size_local": "--data-parallel-size-local",
                "data_parallel_address": "--data-parallel-address",
                "data_parallel_rpc_port": "--data-parallel-rpc-port",
                "data_parallel_start_rank": "--data-parallel-start-rank",
                "data_parallel_backend": "--data-parallel-backend",
                "api_server_count": "--api-server-count",
                "enable_expert_parallel": "--enable-expert-parallel",
            }
            offenders = sorted(flag for dest, flag in prohibited_with_omni.items() if dest in explicit_cli_keys)
            if offenders:
                raise ValueError(
                    "The following CLI args are not supported under --omni: "
                    f"{', '.join(offenders)}. Configure parallelism through the "
                    "per-stage YAML (`--deploy-config` / `--stage-configs-path`) "
                    "and replica count via `--omni-dp-size-local`."
                )

        # --omni-lb-policy is validated against the LoadBalancingPolicy enum.
        omni_lb_policy = getattr(args, "omni_lb_policy", None)
        if omni_lb_policy is not None:
            from vllm_omni.distributed.omni_coordinator import LoadBalancingPolicy

            try:
                LoadBalancingPolicy(omni_lb_policy)
            except ValueError as exc:
                valid = ", ".join(p.value for p in LoadBalancingPolicy)
                raise ValueError(f"--omni-lb-policy={omni_lb_policy!r} is not one of: {valid}") from exc

        omni_heartbeat_timeout = getattr(args, "omni_heartbeat_timeout", None)
        if omni_heartbeat_timeout is not None and omni_heartbeat_timeout <= 0:
            raise ValueError(f"--omni-heartbeat-timeout must be > 0, got {omni_heartbeat_timeout}")

        # Skip validation for diffusion models as they have different requirements
        from vllm_omni.diffusion.utils.hf_utils import is_diffusion_model

        model = getattr(args, "model_tag", None) or getattr(args, "model", None)
        if model and is_diffusion_model(model):
            logger.info("Detected diffusion model: %s", model)
            return
        validate_parsed_serve_args(args)

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> OmniArgumentParser:
        serve_parser = subparsers.add_parser(
            self.name,
            description=DESCRIPTION,
            usage="vllm serve [model_tag] --omni [options]",
        )

        _ensure_vllm_platform()
        serve_parser = make_arg_parser(serve_parser)
        serve_parser.epilog = VLLM_SUBCMD_PARSER_EPILOG.format(subcmd=self.name)

        # Register omni-specific CLI flags (upstream vLLM flags are already
        # registered by make_arg_parser above). OmniArgumentParser will
        # inject deploy YAML defaults as action.default before parse.
        from vllm_omni.engine.arg_utils import OmniEngineArgs

        OmniEngineArgs.add_cli_args(serve_parser, omni_args_only=True)

        type(self)._parser = serve_parser
        return serve_parser


class _StageCompatWrapper:
    """Lightweight compatibility wrapper around StageResolvedConfig that
    provides the attribute interface expected by run_headless's existing
    code paths (engine_args, runtime, default_sampling_params, etc.).
    """

    __slots__ = ("_stage",)

    def __init__(self, stage: Any) -> None:  # StageResolvedConfig
        self._stage = stage

    @property
    def stage_id(self) -> int:
        return self._stage.stage_id

    @property
    def stage_type(self) -> str:
        return self._stage.stage_type

    @property
    def engine_args(self) -> dict[str, Any]:
        """Provide dict-like engine args for compat with build_engine_args_dict,
        get_stage_devices_per_replica, inject_omni_kv_config, etc."""
        stage = self._stage
        if stage.vllm_config is not None:
            # LLM stage: VllmConfig has all the fields.
            return stage.vllm_config
        if stage.diffusion_config is not None:
            # Diffusion: provide parallel_config for get_stage_devices_per_replica.
            return {"parallel_config": stage.diffusion_config.parallel_config}
        return {}

    @property
    def runtime(self) -> dict[str, Any]:
        return self._stage.runtime or {}

    @property
    def default_sampling_params(self) -> dict[str, Any]:
        if self._stage.metadata and self._stage.metadata.default_sampling_params:
            from dataclasses import asdict as _asdict

            return _asdict(self._stage.metadata.default_sampling_params)
        return {}

    @property
    def engine_input_source(self) -> list[int]:
        return self._stage.engine_input_source

    @property
    def is_prefill_only(self) -> bool:
        return self._stage.is_prefill_only

    @property
    def is_decode_only(self) -> bool:
        return self._stage.is_decode_only

    @property
    def is_comprehension(self) -> bool:
        if self._stage.metadata:
            return self._stage.metadata.is_comprehension
        return False

    @property
    def final_output(self) -> bool:
        if self._stage.metadata:
            return self._stage.metadata.final_output
        return False

    @property
    def final_output_type(self) -> str | None:
        if self._stage.metadata:
            return self._stage.metadata.final_output_type
        return None

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def run_headless(args: argparse.Namespace) -> None:
    """Run a single stage in headless mode.

    Honors ``--omni-dp-size-local``: launches that many replicas locally for
    ``--stage-id``. Each replica registers with the head's OmniMasterServer
    (auto-assigned replica id when ``--omni-dp-size-local > 1`` so multiple
    headless invocations can coexist) and reports heartbeats to the head's
    OmniCoordinator.
    """
    from vllm.v1.engine.coordinator import DPCoordinator
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor
    from vllm.version import __version__ as VLLM_VERSION

    from vllm_omni.diffusion.stage_diffusion_proc import (
        complete_diffusion_handshake,
        spawn_diffusion_proc,
    )
    from vllm_omni.engine.omni_core_engine_proc_manager import OmniCoreEngineProcManager
    from vllm_omni.engine.stage_engine_startup import register_stage_with_omni_master
    from vllm_omni.engine.stage_init_utils import (
        get_stage_connector_spec,
        get_stage_devices_per_replica,
        prepare_engine_environment,
        setup_stage_devices,
        split_devices_for_replicas,
        terminate_alive_proc,
    )
    from vllm_omni.platforms import current_omni_platform

    model = args.model
    stage_id: int | None = args.stage_id
    omni_master_address: str | None = args.omni_master_address
    omni_master_port: int | None = args.omni_master_port
    omni_replica_address: str | None = getattr(args, "omni_replica_address", None)
    omni_dp_size_local: int = max(1, int(getattr(args, "omni_dp_size_local", 1) or 1))

    if stage_id is None:
        raise ValueError("--stage-id is required in headless mode")

    # ``--replica-id`` is deprecated and ignored — replica ids are
    # auto-assigned by ``OmniMasterServer`` so headless processes carry
    # no knowledge of their per-replica id at launch time. Warn (don't
    # error) when the operator still supplies it so existing launchers
    # keep working with a single log line.
    explicit_cli_keys: set[str] = getattr(args, "_cli_explicit_keys", set()) or set()
    if "replica_id" in explicit_cli_keys:
        logger.warning(
            "[Headless] --replica-id is deprecated and ignored "
            "(supplied value: %s). Replica ids are auto-assigned by the "
            "master server.",
            args.replica_id,
        )
    if omni_master_address is None or omni_master_port is None:
        raise ValueError("--omni-master-address and --omni-master-port are required in headless mode")
    api_server_count = args.api_server_count or 0
    if api_server_count > 1:
        raise ValueError("api_server_count can't be set in headless mode")
    if args.worker_backend != "multi_process":
        raise ValueError("headless mode requires worker_backend=multi_process")

    # Build resolved omni config from CLI args.
    from vllm_omni.engine.arg_utils import OmniEngineArgs

    engine_args = OmniEngineArgs.from_cli_args(args)
    omni_config = engine_args.create_omni_config(model)

    # Locate the stage config that matches stage_id.
    stage = None
    for s in omni_config.stages:
        if s.stage_id == stage_id:
            stage = s
            break
    if stage is None:
        raise ValueError(
            f"No stage config found for stage_id={stage_id}. "
            f"Available stage ids: {[s.stage_id for s in omni_config.stages]}"
        )

    # Wrap for compat with existing code paths.
    stage_cfg = _StageCompatWrapper(stage)

    prepare_engine_environment()
    omni_transfer_config = omni_config.omni_transfer_config

    # When ``--omni-dp-size-local > 1``, slice the YAML's ``devices:`` field
    # into per-replica subsets so each subprocess we spawn below sees a
    # narrowed ``CUDA_VISIBLE_DEVICES`` and doesn't stack on cuda:0. Mirrors
    # the head-side per-replica device application at
    # ``async_omni_engine.py`` (setup_stage_devices around each launch).
    runtime_cfg = getattr(stage_cfg, "runtime", None)
    devices_str: str | None = None
    if runtime_cfg is not None:
        devices_str = (
            runtime_cfg.get("devices") if hasattr(runtime_cfg, "get") else getattr(runtime_cfg, "devices", None)
        )
    devices_per_replica = get_stage_devices_per_replica(stage_cfg)
    if devices_str:
        # Always remap YAML's logical devices through setup_stage_devices,
        # even for omni_dp_size_local==1. The launcher's CUDA_VISIBLE_DEVICES
        # is dropped from the engine-subprocess env between vllm-serve and
        # OmniCoreEngineProcManager.Process, so the worker would otherwise
        # default cuda:0 to physical GPU 0 and collide with a co-located
        # head on the same host (see hyi3_multi_host_1 reproducer).
        per_replica_devices: list[str | None] = split_devices_for_replicas(
            devices_str, omni_dp_size_local, devices_per_replica, stage_id
        )
        logger.info(
            "[Headless] Stage %d: %d local replicas, devices_per_replica=%d, per-replica devices: %s",
            stage_id,
            omni_dp_size_local,
            devices_per_replica,
            per_replica_devices,
        )
    else:
        per_replica_devices = [None] * omni_dp_size_local
    device_control_env = current_omni_platform.device_control_env_var

    if stage_cfg.stage_type == "diffusion":
        od_config = stage.diffusion_config

        logger.info(
            "[Headless] Launching %d diffusion replica(s) for stage %d via OmniMasterServer at %s:%d",
            omni_dp_size_local,
            stage_id,
            omni_master_address,
            omni_master_port,
        )

        procs: list[Any] = []
        try:
            for _rep_idx in range(omni_dp_size_local):
                # Always auto-assign: headless processes carry no knowledge
                # of their per-replica id and the master server is the sole
                # authority on the per-stage id namespace.
                response = register_stage_with_omni_master(
                    omni_master_address=omni_master_address,
                    omni_master_port=omni_master_port,
                    omni_stage_id=stage_id,
                    omni_stage_config=stage_cfg,
                    replica_id=None,
                    return_full_response=True,
                    replica_bind_address=omni_replica_address,
                )
                # Apply this replica's CUDA_VISIBLE_DEVICES (only when
                # ``--omni-dp-size-local > 1`` and the YAML's stage devices
                # field is set). The spawned subprocess inherits the env at
                # spawn time; we restore the parent env afterwards so the
                # next replica's setup sees the same baseline.
                previous_visible_devices = os.environ.get(device_control_env)
                try:
                    if per_replica_devices[_rep_idx] is not None:
                        setup_stage_devices(stage_id, {"devices": per_replica_devices[_rep_idx]})
                    # Each StageDiffusionProc starts its own
                    # torch.distributed group bound to
                    # ``od_config.master_port``. Without an explicit
                    # per-replica override all spawned subprocesses
                    # share the value ``OmniDiffusionConfig.__post_init__``
                    # picked once (and the second binder hits EADDRINUSE
                    # on ``init_process_group``). We can't use
                    # kernel-ephemeral allocation either, because the
                    # master server's pre-allocated ZMQ ports (returned
                    # by ``register_stage_with_omni_master``) also live
                    # in the ephemeral range and are not actually bound
                    # until the headless ``_perform_diffusion_handshake``
                    # runs — so picking an ephemeral port here can steal
                    # a port the master server already promised to a
                    # sibling headless. Use ``settle_port`` from a base
                    # above the Linux default ephemeral range
                    # (32768-60999) so torch.distributed master ports
                    # never overlap with ZMQ allocations.
                    if omni_dp_size_local > 1:
                        od_config.master_port = od_config.settle_port(
                            61000 + _rep_idx * 100,
                            port_inc=37,
                        )
                    proc, _, _, _ = spawn_diffusion_proc(
                        model,
                        od_config,
                        handshake_address=response.handshake_address,
                        request_address=response.input_address,
                        response_address=response.output_address,
                        omni_coordinator_address=response.coordinator_router_address,
                        omni_stage_id=stage_id,
                        omni_replica_id=response.replica_id,
                    )
                finally:
                    if previous_visible_devices is None:
                        current_omni_platform.unset_device_control_env_var()
                    else:
                        current_omni_platform.set_device_control_env_var(previous_visible_devices)
                complete_diffusion_handshake(proc, response.handshake_address, args.stage_init_timeout)
                procs.append(proc)
                logger.info(
                    "[Headless] Diffusion replica id=%d for stage %d is up (coord=%s)",
                    response.replica_id,
                    stage_id,
                    response.coordinator_router_address,
                )

            # Block on the sentinel set so any replica crash is detected
            # immediately (the previous per-proc join loop only noticed
            # crashes in registration order). Any exit triggers fleet
            # shutdown via the finally block; non-zero exits propagate.
            sentinel_to_proc = {p.sentinel: p for p in procs}
            died = connection.wait(list(sentinel_to_proc.keys()))
            first = sentinel_to_proc[died[0]]
            logger.info(
                "[Headless] Diffusion replica %s exited (code=%s); shutting down stage %d.",
                first.name,
                first.exitcode,
                stage_id,
            )
            if first.exitcode not in (None, 0):
                raise RuntimeError(
                    f"Diffusion stage {stage_id} replica {first.name!r} exited with code {first.exitcode}"
                )
            return
        finally:
            logger.info("[Headless] Shutting down %d diffusion replica(s) for stage %d.", len(procs), stage_id)
            for p in procs:
                if p.is_alive():
                    terminate_alive_proc(p)

    vllm_config = stage.vllm_config
    executor_class = stage.executor_class

    if vllm_config is None:
        raise ValueError(f"Stage {stage_id} is LLM type but has no vllm_config")

    parallel_config = vllm_config.parallel_config
    local_engine_count = parallel_config.data_parallel_size_local

    if local_engine_count <= 0:
        raise ValueError("data_parallel_size_local must be > 0 in headless mode")

    shutdown_requested = False

    def signal_handler(signum: int, frame: FrameType | None) -> None:
        nonlocal shutdown_requested
        logger.debug("Received %d signal.", signum)
        if not shutdown_requested:
            shutdown_requested = True
            raise SystemExit

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    if parallel_config.node_rank_within_dp > 0:
        head_node_address = f"{parallel_config.master_addr}:{parallel_config.master_port}"
        logger.info(
            "Launching vLLM-Omni (v%s) headless multiproc executor, "
            "with head node address %s for torch.distributed process group.",
            VLLM_VERSION,
            head_node_address,
        )

        executor = MultiprocExecutor(vllm_config, monitor_workers=False)
        executor.start_worker_monitor(inline=True)
        return

    dp_rank = parallel_config.data_parallel_rank if parallel_config.data_parallel_rank is not None else 0
    coordinator = None
    if vllm_config.needs_dp_coordinator and dp_rank == 0:
        coordinator = DPCoordinator(
            parallel_config,
            enable_wave_coordination=vllm_config.model_config.is_moe,
        )
        logger.info(
            "[Headless] Started DP Coordinator process for stage %d (PID: %d)",
            stage_id,
            coordinator.proc.pid,
        )

    logger.info(
        "[Headless] Launching %d omni replica(s) (vLLM dp_size_local=%d each) for stage %d "
        "via OmniMasterServer at %s:%d",
        omni_dp_size_local,
        local_engine_count,
        stage_id,
        omni_master_address,
        omni_master_port,
    )

    # One OmniMasterServer registration per omni replica; each registration
    # yields its own (handshake, input, output) allocation and the head's
    # OmniCoordinator ROUTER address. We then spawn one
    # OmniCoreEngineProcManager per replica so its subprocess gets the
    # right replica id wired into its OmniCoordClientForStage.
    log_stats = bool(args.log_stats)
    if args.disable_log_stats:
        log_stats = False

    engine_managers: list[Any] = []
    monitor_threads: list[threading.Thread] = []

    def _monitor_target(mgr: Any) -> None:
        try:
            mgr.monitor_engine_liveness()
        except Exception:
            logger.exception("[Headless] monitor_engine_liveness raised")

    try:
        for _rep_idx in range(omni_dp_size_local):
            # Always auto-assign: see the diffusion branch comment above
            # for the rationale (headless owns no replica-id namespace).
            response = register_stage_with_omni_master(
                omni_master_address=omni_master_address,
                omni_master_port=omni_master_port,
                omni_stage_id=stage_id,
                omni_stage_config=stage_cfg,
                coordinator=coordinator,
                replica_id=None,
                return_full_response=True,
                replica_bind_address=omni_replica_address,
                # LLM headless: the head binds *all* three sockets —
                # handshake ROUTER (``connect_remote_engine_cores``),
                # input ROUTER and output PULL (``CoreClient`` —
                # ``make_zmq_socket`` defaults bind=True for PULL).
                # The remote LLM worker is purely a connector: it
                # opens 3 outbound TCP connections to the master's
                # host. So the master must keep every address on
                # its own host; rewriting any of them to this
                # replica's NIC makes the head's ``bind`` go
                # EADDRNOTAVAIL on a cross-host launch.
                replica_binds_sockets=False,
            )
            # Per-replica CUDA_VISIBLE_DEVICES, same pattern as the diffusion
            # branch above. OmniCoreEngineProcManager.__init__ spawns its
            # subprocesses via context.Process inside the constructor, so we
            # must set the env *before* instantiation and restore after.
            previous_visible_devices = os.environ.get(device_control_env)
            try:
                if per_replica_devices[_rep_idx] is not None:
                    setup_stage_devices(stage_id, {"devices": per_replica_devices[_rep_idx]})
                mgr = OmniCoreEngineProcManager(
                    local_engine_count=local_engine_count,
                    start_index=dp_rank,
                    local_start_index=0,
                    vllm_config=vllm_config,
                    local_client=False,
                    handshake_address=response.handshake_address,
                    executor_class=executor_class,
                    log_stats=log_stats,
                    omni_stage_id=stage_id,
                    omni_coordinator_address=response.coordinator_router_address,
                    omni_replica_base_id=response.replica_id,
                )
            finally:
                if previous_visible_devices is None:
                    current_omni_platform.unset_device_control_env_var()
                else:
                    current_omni_platform.set_device_control_env_var(previous_visible_devices)
            engine_managers.append(mgr)
            logger.info(
                "[Headless] Stage %d replica id=%d up (coord=%s)",
                stage_id,
                response.replica_id,
                response.coordinator_router_address,
            )

        # Run all managers' liveness monitors in parallel. Each blocks
        # until its own subprocesses exit (or fail).
        if len(engine_managers) == 1:
            engine_managers[0].monitor_engine_liveness()
        else:
            for mgr in engine_managers:
                t = threading.Thread(target=_monitor_target, args=(mgr,), name=f"omni-replica-monitor-{id(mgr):x}")
                t.start()
                monitor_threads.append(t)
            for t in monitor_threads:
                t.join()
    finally:
        logger.info("[Headless] Shutting down stage %d (%d managers).", stage_id, len(engine_managers))
        for mgr in engine_managers:
            try:
                mgr.shutdown()
            except Exception:
                logger.exception("[Headless] engine manager shutdown failed")
        if coordinator is not None:
            coordinator.shutdown()


def cmd_init() -> list[CLISubcommand]:
    return [OmniServeCommand()]
