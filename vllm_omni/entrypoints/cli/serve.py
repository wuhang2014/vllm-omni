"""
Omni serve command for vLLM-Omni.

Supports both multi-stage LLM models (e.g., Qwen2.5-Omni) and
diffusion models (e.g., Qwen-Image) through the same CLI interface.
"""

import argparse
import os
import signal
from typing import Any

import uvloop
import zmq
from omegaconf import OmegaConf
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG
from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.distributed.omni_connectors import (
    get_connectors_config_for_stage,
    load_omni_transfer_config,
)
from vllm_omni.entrypoints.omni import OmniBase, omni_snapshot_download
from vllm_omni.entrypoints.omni_stage import OmniStage
from vllm_omni.entrypoints.openai.api_server import omni_run_server
from vllm_omni.entrypoints.utils import (
    load_stage_configs_from_model,
    load_stage_configs_from_yaml,
    resolve_model_config_path,
)
from vllm_omni.entrypoints.zmq_utils import ZmqQueueSpec

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


class OmniServeCommand(CLISubcommand):
    """The `serve` subcommand for the vLLM CLI."""

    name = "serve"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        # If model is specified in CLI (as positional arg), it takes precedence
        if hasattr(args, "model_tag") and args.model_tag is not None:
            args.model = args.model_tag

        if args.headless or args.api_server_count < 1:
            run_headless(args)
        else:
            uvloop.run(omni_run_server(args))

    def validate(self, args: argparse.Namespace) -> None:
        # Skip validation for diffusion models as they have different requirements
        from vllm_omni.diffusion.utils.hf_utils import is_diffusion_model

        model = getattr(args, "model_tag", None) or getattr(args, "model", None)
        if model and is_diffusion_model(model):
            logger.info("Detected diffusion model: %s", model)
            return
        validate_parsed_serve_args(args)

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        serve_parser = subparsers.add_parser(
            self.name,
            description=DESCRIPTION,
            usage="vllm serve [model_tag] --omni [options]",
        )

        serve_parser = make_arg_parser(serve_parser)
        serve_parser.epilog = VLLM_SUBCMD_PARSER_EPILOG.format(subcmd=self.name)

        # Create OmniConfig argument group for omni-related parameters
        # This ensures the parameters appear in --help output
        omni_config_group = serve_parser.add_argument_group(
            title="OmniConfig", description="Configuration for vLLM-Omni multi-stage and diffusion models."
        )

        omni_config_group.add_argument(
            "--omni",
            action="store_true",
            help="Enable vLLM-Omni mode for multi-modal and diffusion models",
        )
        omni_config_group.add_argument(
            "--stage-configs-path",
            type=str,
            default=None,
            help="Path to the stage configs file. If not specified, the stage configs will be loaded from the model.",
        )
        omni_config_group.add_argument(
            "--stage-id",
            type=int,
            default=None,
            help="Select and launch a single stage by stage_id.",
        )
        omni_config_group.add_argument(
            "--stage-init-timeout",
            type=int,
            default=300,
            help="The timeout for initializing a single stage in seconds (default: 300)",
        )
        omni_config_group.add_argument(
            "--init-timeout",
            type=int,
            default=600,
            help="The timeout for initializing the stages.",
        )
        omni_config_group.add_argument(
            "--shm-threshold-bytes",
            type=int,
            default=65536,
            help="The threshold for the shared memory size.",
        )
        omni_config_group.add_argument(
            "--log-stats",
            action="store_true",
            help="Enable logging the stats.",
        )
        omni_config_group.add_argument(
            "--log-file",
            type=str,
            default=None,
            help="The path to the log file.",
        )
        omni_config_group.add_argument(
            "--batch-timeout",
            type=int,
            default=10,
            help="The timeout for the batch.",
        )
        omni_config_group.add_argument(
            "--worker-backend",
            type=str,
            default="multi_process",
            choices=["multi_process", "ray"],
            help="The backend to use for stage workers.",
        )
        omni_config_group.add_argument(
            "--ray-address",
            type=str,
            default=None,
            help="The address of the Ray cluster to connect to.",
        )
        omni_config_group.add_argument(
            "--omni-master-address",
            type=str,
            default="127.0.0.1",
            help="Master address for Omni ZMQ IPC (orchestrator bind address).",
        )
        omni_config_group.add_argument(
            "--omni-master-port",
            type=int,
            default=5555,
            help="Base port for Omni ZMQ IPC (two ports per stage).",
        )

        # Diffusion model specific arguments
        omni_config_group.add_argument(
            "--num-gpus",
            type=int,
            default=None,
            help="Number of GPUs to use for diffusion model inference.",
        )
        omni_config_group.add_argument(
            "--usp",
            "--ulysses-degree",
            dest="ulysses_degree",
            type=int,
            default=None,
            help="Ulysses Sequence Parallelism degree for diffusion models. "
            "Equivalent to setting DiffusionParallelConfig.ulysses_degree.",
        )
        omni_config_group.add_argument(
            "--ring",
            dest="ring_degree",
            type=int,
            default=None,
            help="Ring Sequence Parallelism degree for diffusion models. "
            "Equivalent to setting DiffusionParallelConfig.ring_degree.",
        )

        # Cache optimization parameters
        omni_config_group.add_argument(
            "--cache-backend",
            type=str,
            default="none",
            help="Cache backend for diffusion models, options: 'tea_cache', 'cache_dit'",
        )
        omni_config_group.add_argument(
            "--cache-config",
            type=str,
            default=None,
            help="JSON string of cache configuration (e.g., '{\"rel_l1_thresh\": 0.2}').",
        )
        omni_config_group.add_argument(
            "--enable-cache-dit-summary",
            action="store_true",
            help="Enable cache-dit summary logging after diffusion forward passes.",
        )

        # VAE memory optimization parameters
        omni_config_group.add_argument(
            "--vae-use-slicing",
            action="store_true",
            help="Enable VAE slicing for memory optimization (useful for mitigating OOM issues).",
        )
        omni_config_group.add_argument(
            "--vae-use-tiling",
            action="store_true",
            help="Enable VAE tiling for memory optimization (useful for mitigating OOM issues).",
        )

        # diffusion model offload parameters
        serve_parser.add_argument(
            "--enable-cpu-offload",
            action="store_true",
            help="Enable CPU offloading for diffusion models.",
        )
        serve_parser.add_argument(
            "--enable-layerwise-offload",
            action="store_true",
            help="Enable layerwise (blockwise) offloading on DiT modules.",
        )
        serve_parser.add_argument(
            "--layerwise-num-gpu-layers",
            type=int,
            default=1,
            help="Number of layers (blocks) to keep on GPU during generation.",
        )

        # Video model parameters (e.g., Wan2.2) - engine-level
        omni_config_group.add_argument(
            "--boundary-ratio",
            type=float,
            default=None,
            help="Boundary split ratio for low/high DiT in video models (e.g., 0.875 for Wan2.2).",
        )
        omni_config_group.add_argument(
            "--flow-shift",
            type=float,
            default=None,
            help="Scheduler flow_shift for video models (e.g., 5.0 for 720p, 12.0 for 480p).",
        )
        omni_config_group.add_argument(
            "--cfg-parallel-size",
            type=int,
            default=1,
            choices=[1, 2],
            help="Number of devices for CFG parallel computation for diffusion models. "
            "Equivalent to setting DiffusionParallelConfig.cfg_parallel_size.",
        )

        # Default sampling parameters
        omni_config_group.add_argument(
            "--default-sampling-params",
            type=str,
            help="Json str for Default sampling parameters, \n"
            'Structure: {"<stage_id>": {<sampling_param>: value, ...}, ...}\n'
            'e.g., \'{"0": {"num_inference_steps":50, "guidance_scale":1}}\'. '
            "Currently only supports diffusion models.",
        )
        # Diffusion model mixed precision
        omni_config_group.add_argument(
            "--max-generated-image-size",
            type=float,
            help="The max size of generate image (height * width).",
        )
        return serve_parser


def _create_default_diffusion_stage_cfg(args: argparse.Namespace) -> list[dict[str, Any]]:
    omni_base = OmniBase.__new__(OmniBase)
    return omni_base._create_default_diffusion_stage_cfg(vars(args))


def run_headless(args: argparse.Namespace) -> None:
    if args.api_server_count > 1:
        raise ValueError("api_server_count can't be set in headless mode")
    if getattr(args, "worker_backend", "multi_process") != "multi_process":
        raise ValueError("headless mode requires worker_backend=multi_process")

    model = getattr(args, "model", None)
    if not model:
        raise ValueError("model must be specified in headless mode")
    model = omni_snapshot_download(model)

    tokenizer = getattr(args, "tokenizer", None)
    base_engine_args = {"tokenizer": tokenizer} if tokenizer is not None else None

    parallel_keys = [
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "data_parallel_size_local",
        "data_parallel_backend",
        "distributed_executor_backend",
    ]
    parallel_overrides = {
        k: getattr(args, k) for k in parallel_keys if hasattr(args, k) and getattr(args, k) is not None
    }
    if parallel_overrides:
        base_engine_args = base_engine_args or {}
        base_engine_args.update(parallel_overrides)

    stage_configs_path = getattr(args, "stage_configs_path", None)
    if stage_configs_path is None:
        config_path = resolve_model_config_path(model)
        stage_configs = load_stage_configs_from_model(model, base_engine_args=base_engine_args)
        if not stage_configs:
            default_stage_cfg = _create_default_diffusion_stage_cfg(args)
            stage_configs = OmegaConf.create(default_stage_cfg)
    else:
        config_path = stage_configs_path
        stage_configs = load_stage_configs_from_yaml(stage_configs_path, base_engine_args=base_engine_args)

    if not stage_configs:
        raise ValueError("No stage configs found; provide --stage-configs-path or a supported model.")

    single_stage_id = getattr(args, "stage_id", None)
    if single_stage_id is None:
        if len(stage_configs) != 1:
            raise ValueError("--stage-id is required in headless mode for multi-stage configs")
        single_stage_id = getattr(stage_configs[0], "stage_id", 0)

    stage_config = None
    for cfg in stage_configs:
        if getattr(cfg, "stage_id", None) == single_stage_id:
            stage_config = cfg
            break
    if stage_config is None:
        raise ValueError(f"No stage matches stage_id={single_stage_id}.")

    transfer_config = load_omni_transfer_config(config_path, default_shm_threshold=args.shm_threshold_bytes)
    connectors_config = get_connectors_config_for_stage(transfer_config, single_stage_id)

    omni_master_address = getattr(args, "omni_master_address", None) or "127.0.0.1"
    omni_master_port = int(getattr(args, "omni_master_port", 5555) or 5555)
    base_port = omni_master_port + 1
    in_endpoint = f"tcp://{omni_master_address}:{base_port + single_stage_id * 2}"
    out_endpoint = f"tcp://{omni_master_address}:{base_port + single_stage_id * 2 + 1}"

    in_q_spec = ZmqQueueSpec(endpoint=in_endpoint, socket_type=zmq.PULL, bind=False)
    out_q_spec = ZmqQueueSpec(endpoint=out_endpoint, socket_type=zmq.PUSH, bind=False)
    zmq_ctx = zmq.Context()
    in_q = None
    out_q = None

    shutdown_requested = False

    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            return
        shutdown_requested = True
        raise SystemExit

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    stage = OmniStage(stage_config, stage_init_timeout=int(getattr(args, "stage_init_timeout", 300)))
    stage.set_zmq_master(omni_master_address, omni_master_port)
    stage.attach_queues(in_q, out_q, in_q_spec=in_q_spec, out_q_spec=out_q_spec)

    old_env = os.environ.get("VLLM_LOGGING_PREFIX")
    os.environ["VLLM_LOGGING_PREFIX"] = f"[Stage-{single_stage_id}] {'' if old_env is None else old_env}"
    try:
        stage.init_stage_worker(
            model,
            is_async=True,
            shm_threshold_bytes=int(getattr(args, "shm_threshold_bytes", 65536)),
            batch_timeout=int(getattr(args, "batch_timeout", 10)),
            connectors_config=connectors_config,
            worker_backend="multi_process",
        )
        if stage._proc is not None:
            stage._proc.join()
    finally:
        stage.stop_stage_worker()
        try:
            zmq_ctx.term()
        except Exception:
            pass
        if old_env is None:
            os.environ.pop("VLLM_LOGGING_PREFIX", None)
        else:
            os.environ["VLLM_LOGGING_PREFIX"] = old_env


def cmd_init() -> list[CLISubcommand]:
    return [OmniServeCommand()]
