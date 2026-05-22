"""
Omni serve command for vLLM-Omni.

Supports both multi-stage LLM models (e.g., Qwen2.5-Omni) and
diffusion models (e.g., Qwen-Image) through the same CLI interface.
"""

import argparse
import os

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




def run_headless(args: argparse.Namespace) -> None:
    """Run a single stage in headless mode.

    Honors ``--omni-dp-size-local``: launches that many replicas locally for
    ``--stage-id``. Each replica registers with the head's OmniMasterServer
    (auto-assigned replica id when ``--omni-dp-size-local > 1`` so multiple
    headless invocations can coexist) and reports heartbeats to the head's
    OmniCoordinator.
    """
    from vllm_omni.engine.arg_utils import OmniEngineArgs
    from vllm_omni.engine.stage_init_utils import (
        compute_per_replica_devices,
        launch_headless_diffusion_replicas,
        launch_headless_llm_replicas,
        prepare_engine_environment,
        terminate_alive_proc,
    )

    model = args.model
    stage_id: int | None = args.stage_id
    omni_master_address: str | None = args.omni_master_address
    omni_master_port: int | None = args.omni_master_port
    omni_replica_address: str | None = getattr(args, "omni_replica_address", None)
    omni_dp_size_local: int = max(1, int(getattr(args, "omni_dp_size_local", 1) or 1))

    if stage_id is None:
        raise ValueError("--stage-id is required in headless mode")

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
    engine_args = OmniEngineArgs.from_cli_args(args)
    omni_config = engine_args.create_omni_config(model)

    # Locate the stage in the resolved config.
    stage = None
    for s in omni_config.stages:
        if s.stage_id == stage_id:
            stage = s
            break
    if stage is None:
        available = [s.stage_id for s in omni_config.stages]
        raise ValueError(
            f"No stage config found for stage_id={stage_id}. "
            f"Available stage ids: {available}"
        )

    # Common prelude.
    prepare_engine_environment()
    per_replica_devices = compute_per_replica_devices(stage, omni_dp_size_local, stage_id)

    # Dispatch by stage type.
    if stage.stage_type == "diffusion":
        launch_headless_diffusion_replicas(
            model=model,
            stage=stage,
            stage_id=stage_id,
            omni_dp_size_local=omni_dp_size_local,
            omni_master_address=omni_master_address,
            omni_master_port=omni_master_port,
            omni_replica_address=omni_replica_address,
            stage_init_timeout=args.stage_init_timeout,
            per_replica_devices=per_replica_devices,
        )
    else:
        launch_headless_llm_replicas(
            stage=stage,
            stage_id=stage_id,
            omni_dp_size_local=omni_dp_size_local,
            omni_master_address=omni_master_address,
            omni_master_port=omni_master_port,
            omni_replica_address=omni_replica_address,
            log_stats=bool(args.log_stats),
            disable_log_stats=args.disable_log_stats,
            per_replica_devices=per_replica_devices,
        )


def cmd_init() -> list[CLISubcommand]:
    return [OmniServeCommand()]
