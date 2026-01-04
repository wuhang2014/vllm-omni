"""Unit tests for vllm_omni/entrypoints/log_utils.py"""
import time
from unittest.mock import MagicMock, patch

import pytest

from vllm_omni.entrypoints.log_utils import (
    OrchestratorMetrics,
    StageRequestMetrics,
    StageStats,
    aggregate_rx_and_maybe_total,
    build_stage_summary,
    build_transfer_summary,
    compute_and_log_stage_request_stats,
    count_tokens_from_outputs,
    log_stage_request_stats,
    log_transfer_rx,
    log_transfer_total,
    log_transfer_tx,
    record_sender_transfer_agg,
    record_stage_metrics,
)


class TestStageStats:
    """Tests for StageStats dataclass"""

    def test_stage_stats_creation(self):
        """Test creating StageStats instance"""
        stats = StageStats(total_token=100, total_gen_time=500.0)

        assert stats.total_token == 100
        assert stats.total_gen_time == 500.0


class TestStageRequestMetrics:
    """Tests for StageRequestMetrics dataclass"""

    def test_stage_request_metrics_creation(self):
        """Test creating StageRequestMetrics instance"""
        stage_stats = StageStats(total_token=100, total_gen_time=500.0)
        metrics = StageRequestMetrics(
            num_tokens_in=10,
            num_tokens_out=20,
            stage_gen_time_ms=100.0,
            batch_id=1,
            batch_size=2,
            rx_decode_time_ms=5.0,
            rx_transfer_bytes=1024,
            rx_in_flight_time_ms=10.0,
            stage_stats=stage_stats,
        )

        assert metrics.num_tokens_in == 10
        assert metrics.num_tokens_out == 20
        assert metrics.stage_gen_time_ms == 100.0
        assert metrics.batch_id == 1
        assert metrics.batch_size == 2
        assert metrics.rx_decode_time_ms == 5.0
        assert metrics.rx_transfer_bytes == 1024
        assert metrics.rx_in_flight_time_ms == 10.0
        assert metrics.stage_stats == stage_stats


class TestLoggingFunctions:
    """Tests for logging functions"""

    @patch("vllm_omni.entrypoints.log_utils.logger")
    def test_log_transfer_tx(self, mock_logger):
        """Test log_transfer_tx function"""
        log_transfer_tx(from_stage=0, to_stage=1, request_id="req-1", size_bytes=1024, tx_time_ms=10.0, used_shm=True)

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "transfer_stats" in call_args
        assert "1024" in call_args
        assert "10.0" in call_args

    @patch("vllm_omni.entrypoints.log_utils.logger")
    def test_log_transfer_rx(self, mock_logger):
        """Test log_transfer_rx function"""
        log_transfer_rx(
            from_stage=0, to_stage=1, request_id="req-1", rx_bytes=2048, rx_decode_time_ms=5.0, in_flight_time_ms=15.0
        )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "transfer_rx_stats" in call_args

    @patch("vllm_omni.entrypoints.log_utils.logger")
    def test_log_transfer_total(self, mock_logger):
        """Test log_transfer_total function"""
        log_transfer_total(
            from_stage=0,
            to_stage=1,
            request_id="req-1",
            size_bytes=1024,
            tx_time_ms=10.0,
            in_flight_time_ms=15.0,
            rx_decode_time_ms=5.0,
            total_time_ms=30.0,
        )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "transfer_total_stats" in call_args

    @patch("vllm_omni.entrypoints.log_utils.logger")
    def test_log_stage_request_stats(self, mock_logger):
        """Test log_stage_request_stats function"""
        log_stage_request_stats(
            stage_id=0,
            request_id="req-1",
            batch_size=2,
            num_tokens_out=50,
            stage_gen_time_ms=100.0,
            tokens_per_s=500.0,
            rx_transfer_bytes=1024,
            rx_decode_time_ms=5.0,
            rx_mbps=1.6,
        )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "Request_stage_stats" in call_args

    @patch("vllm_omni.entrypoints.log_utils.log_stage_request_stats")
    def test_compute_and_log_stage_request_stats(self, mock_log_stats):
        """Test compute_and_log_stage_request_stats function"""
        compute_and_log_stage_request_stats(
            stage_id=0,
            request_id="req-1",
            batch_size=2,
            num_engine_outputs=50,
            stage_gen_time_ms=100.0,
            rx_transfer_bytes=1024,
            rx_decode_time_ms=5.0,
        )

        mock_log_stats.assert_called_once()
        call_args = mock_log_stats.call_args
        # Verify tokens_per_s is calculated correctly
        assert call_args[1]["tokens_per_s"] == 500.0  # 50 * 1000.0 / 100.0


class TestRecordStagMetrics:
    """Tests for record_stage_metrics function"""

    def test_record_stage_metrics_basic(self):
        """Test recording basic stage metrics"""
        per_request = {}
        stage_req_counts = [0, 0]
        stage_total_time_ms = [0.0, 0.0]
        stage_total_tokens = [0, 0]

        metrics = {"num_tokens_out": 50, "stage_gen_time_ms": 100.0}

        record_stage_metrics(per_request, stage_req_counts, stage_total_time_ms, stage_total_tokens, 0, "req-1", metrics)

        assert stage_req_counts[0] == 1
        assert stage_total_tokens[0] == 50
        assert "req-1" in per_request
        assert per_request["req-1"]["stages"][0]["num_tokens_out"] == 50

    def test_record_stage_metrics_with_input_tokens_stage_0(self):
        """Test recording metrics with input tokens for stage 0"""
        per_request = {}
        stage_req_counts = [0, 0]
        stage_total_time_ms = [0.0, 0.0]
        stage_total_tokens = [0, 0]

        metrics = {"num_tokens_in": 10, "num_tokens_out": 50, "stage_gen_time_ms": 100.0}

        record_stage_metrics(per_request, stage_req_counts, stage_total_time_ms, stage_total_tokens, 0, "req-1", metrics)

        # Stage 0 should include num_tokens_in
        assert per_request["req-1"]["stages"][0]["num_tokens_in"] == 10

    def test_record_stage_metrics_excludes_input_tokens_non_zero_stage(self):
        """Test that input tokens are not recorded for non-zero stages"""
        per_request = {}
        stage_req_counts = [0, 0]
        stage_total_time_ms = [0.0, 0.0]
        stage_total_tokens = [0, 0]

        metrics = {"num_tokens_in": 10, "num_tokens_out": 50, "stage_gen_time_ms": 100.0}

        record_stage_metrics(per_request, stage_req_counts, stage_total_time_ms, stage_total_tokens, 1, "req-1", metrics)

        # Stage 1 should NOT include num_tokens_in
        assert "num_tokens_in" not in per_request["req-1"]["stages"][1]

    def test_record_stage_metrics_multiple_requests(self):
        """Test recording metrics for multiple requests"""
        per_request = {}
        stage_req_counts = [0]
        stage_total_time_ms = [0.0]
        stage_total_tokens = [0]

        metrics1 = {"num_tokens_out": 50, "stage_gen_time_ms": 100.0}
        metrics2 = {"num_tokens_out": 30, "stage_gen_time_ms": 75.0}

        record_stage_metrics(per_request, stage_req_counts, stage_total_time_ms, stage_total_tokens, 0, "req-1", metrics1)
        record_stage_metrics(per_request, stage_req_counts, stage_total_time_ms, stage_total_tokens, 0, "req-2", metrics2)

        assert stage_req_counts[0] == 2
        assert stage_total_tokens[0] == 80
        assert "req-1" in per_request
        assert "req-2" in per_request


class TestAggregateRxAndMaybeTotal:
    """Tests for aggregate_rx_and_maybe_total function"""

    def test_aggregate_rx_for_stage_0_returns_none(self):
        """Test that stage 0 (no previous stage) returns None"""
        transfer_edge_req = {}
        transfer_agg = {}
        per_request = {}

        result = aggregate_rx_and_maybe_total(transfer_edge_req, transfer_agg, per_request, 0, "req-1", 1024.0, 5.0, 10.0)

        assert result is None

    def test_aggregate_rx_creates_new_entry(self):
        """Test creating new aggregate entry"""
        transfer_edge_req = {}
        transfer_agg = {}
        per_request = {}

        aggregate_rx_and_maybe_total(transfer_edge_req, transfer_agg, per_request, 1, "req-1", 1024.0, 5.0, 10.0)

        key = (0, 1)
        assert key in transfer_agg
        assert transfer_agg[key]["sum_rx_bytes"] == 1024.0
        assert transfer_agg[key]["sum_rx_ms"] == 5.0
        assert transfer_agg[key]["rx_count"] == 1.0

    def test_aggregate_rx_without_sender_data_returns_none(self):
        """Test returns None when sender data not present"""
        transfer_edge_req = {}
        transfer_agg = {}
        per_request = {}

        result = aggregate_rx_and_maybe_total(transfer_edge_req, transfer_agg, per_request, 1, "req-1", 1024.0, 5.0, 10.0)

        # Should return None because sender data doesn't exist
        assert result is None

    def test_aggregate_rx_with_sender_data(self):
        """Test aggregating with sender data present"""
        transfer_edge_req = {(0, 1, "req-1"): {"tx_ms": 10.0, "size_bytes": 1024.0}}
        transfer_agg = {}
        per_request = {}

        result = aggregate_rx_and_maybe_total(transfer_edge_req, transfer_agg, per_request, 1, "req-1", 1024.0, 5.0, 10.0)

        assert result is not None
        size_b, tx_ms, total_ms = result
        assert size_b == 1024
        assert tx_ms == 10.0
        assert total_ms == 10.0 + 10.0 + 5.0  # tx + in_flight + rx


class TestRecordSenderTransferAgg:
    """Tests for record_sender_transfer_agg function"""

    def test_record_sender_creates_new_entry(self):
        """Test creating new sender transfer entry"""
        transfer_agg = {}
        transfer_edge_req = {}

        record_sender_transfer_agg(transfer_agg, transfer_edge_req, 0, 1, "req-1", 1024, 10.0)

        key = (0, 1)
        assert key in transfer_agg
        assert transfer_agg[key]["sum_bytes"] == 1024.0
        assert transfer_agg[key]["sum_ms"] == 10.0
        assert transfer_agg[key]["count"] == 1.0

        # Check edge request storage
        edge_key = (0, 1, "req-1")
        assert edge_key in transfer_edge_req
        assert transfer_edge_req[edge_key]["tx_ms"] == 10.0

    def test_record_sender_accumulates(self):
        """Test accumulating multiple sender records"""
        transfer_agg = {}
        transfer_edge_req = {}

        record_sender_transfer_agg(transfer_agg, transfer_edge_req, 0, 1, "req-1", 1024, 10.0)
        record_sender_transfer_agg(transfer_agg, transfer_edge_req, 0, 1, "req-2", 2048, 20.0)

        key = (0, 1)
        assert transfer_agg[key]["sum_bytes"] == 3072.0  # 1024 + 2048
        assert transfer_agg[key]["sum_ms"] == 30.0  # 10 + 20
        assert transfer_agg[key]["count"] == 2.0


class TestCountTokensFromOutputs:
    """Tests for count_tokens_from_outputs function"""

    def test_count_tokens_empty_list(self):
        """Test counting tokens from empty list"""
        result = count_tokens_from_outputs([])
        assert result == 0

    def test_count_tokens_single_output(self):
        """Test counting tokens from single output"""
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(token_ids=[1, 2, 3, 4, 5])]

        result = count_tokens_from_outputs([mock_output])
        assert result == 5

    def test_count_tokens_multiple_outputs(self):
        """Test counting tokens from multiple outputs"""
        mock_output1 = MagicMock()
        mock_output1.outputs = [MagicMock(token_ids=[1, 2, 3])]

        mock_output2 = MagicMock()
        mock_output2.outputs = [MagicMock(token_ids=[4, 5, 6, 7])]

        result = count_tokens_from_outputs([mock_output1, mock_output2])
        assert result == 7

    def test_count_tokens_handles_missing_outputs(self):
        """Test handles outputs with no tokens gracefully"""
        mock_output = MagicMock()
        mock_output.outputs = None

        result = count_tokens_from_outputs([mock_output])
        assert result == 0


class TestBuildStageSummary:
    """Tests for build_stage_summary function"""

    def test_build_stage_summary_basic(self):
        """Test building basic stage summary"""
        stage_req_counts = [2, 3]
        stage_total_tokens = [100, 150]
        stage_total_time_ms = [1000.0, 1500.0]

        result = build_stage_summary(stage_req_counts, stage_total_tokens, stage_total_time_ms)

        assert len(result) == 2
        assert result[0]["stage_id"] == 0
        assert result[0]["requests"] == 2
        assert result[0]["tokens"] == 100
        assert result[0]["total_time_ms"] == 1000.0

    def test_build_stage_summary_calculates_averages(self):
        """Test that averages are calculated correctly"""
        stage_req_counts = [2]
        stage_total_tokens = [100]
        stage_total_time_ms = [1000.0]

        result = build_stage_summary(stage_req_counts, stage_total_tokens, stage_total_time_ms)

        assert result[0]["avg_time_per_request_ms"] == 500.0  # 1000 / 2
        assert result[0]["avg_tokens_per_s"] == 100.0  # 100 * 1000 / 1000


class TestBuildTransferSummary:
    """Tests for build_transfer_summary function"""

    def test_build_transfer_summary_basic(self):
        """Test building basic transfer summary"""
        transfer_agg = {
            (0, 1): {
                "sum_bytes": 1024.0,
                "sum_ms": 10.0,
                "count": 1.0,
                "sum_rx_bytes": 1024.0,
                "sum_rx_ms": 5.0,
                "rx_count": 1.0,
                "sum_total_ms": 15.0,
                "total_count": 1.0,
            }
        }

        result = build_transfer_summary(transfer_agg)

        assert len(result) == 1
        assert result[0]["from_stage"] == 0
        assert result[0]["to_stage"] == 1
        assert result[0]["samples"] == 1
        assert result[0]["total_bytes"] == 1024

    def test_build_transfer_summary_calculates_mbps(self):
        """Test that Mbps calculations are correct"""
        transfer_agg = {
            (0, 1): {
                "sum_bytes": 1000000.0,  # 1 MB
                "sum_ms": 1000.0,  # 1 second
                "count": 1.0,
                "sum_rx_bytes": 1000000.0,
                "sum_rx_ms": 500.0,
                "rx_count": 1.0,
                "sum_total_ms": 1500.0,
                "total_count": 1.0,
            }
        }

        result = build_transfer_summary(transfer_agg)

        # 1000000 bytes * 8 bits/byte / (1000 ms * 1000 ms/s) = 8 Mbps
        assert result[0]["tx_mbps"] == pytest.approx(8.0, rel=0.01)


class TestOrchestratorMetrics:
    """Tests for OrchestratorMetrics class"""

    def test_initialization(self):
        """Test OrchestratorMetrics initialization"""
        metrics = OrchestratorMetrics(num_stages=3, enable_stats=True, wall_start_ts=time.time())

        assert metrics.num_stages == 3
        assert metrics.enable_stats is True
        assert len(metrics.stage_total_time_ms) == 3
        assert len(metrics.stage_total_tokens) == 3
        assert len(metrics.stage_req_counts) == 3

    @patch("vllm_omni.entrypoints.log_utils.compute_and_log_stage_request_stats")
    def test_on_stage_metrics(self, mock_log):
        """Test on_stage_metrics method"""
        metrics = OrchestratorMetrics(num_stages=2, enable_stats=True, wall_start_ts=time.time())

        stage_metrics = {
            "num_tokens_out": 50,
            "stage_gen_time_ms": 100.0,
            "batch_size": 2,
            "rx_decode_time_ms": 5.0,
            "rx_transfer_bytes": 1024,
        }

        metrics.on_stage_metrics(0, "req-1", stage_metrics)

        assert metrics.stage_req_counts[0] == 1
        assert metrics.stage_total_tokens[0] == 50
        assert mock_log.called

    def test_on_forward(self):
        """Test on_forward method"""
        metrics = OrchestratorMetrics(num_stages=2, enable_stats=False, wall_start_ts=time.time())

        metrics.on_forward(0, 1, "req-1", 1024, 10.0, True)

        # Check that stage first timestamp is set
        assert metrics.stage_first_ts[1] is not None

    def test_on_finalize_request(self):
        """Test on_finalize_request method"""
        wall_start = time.time()
        metrics = OrchestratorMetrics(num_stages=2, enable_stats=False, wall_start_ts=wall_start)

        # Add some metrics first
        stage_metrics = {"num_tokens_in": 10, "num_tokens_out": 50, "stage_gen_time_ms": 100.0}
        metrics.on_stage_metrics(0, "req-1", stage_metrics)

        req_start = time.time()
        time.sleep(0.01)  # Small delay to ensure time difference

        metrics.on_finalize_request(0, "req-1", req_start)

        assert "req-1" in metrics.e2e_done
        assert metrics.e2e_count == 1
        assert metrics.e2e_total_ms > 0

    def test_build_and_log_summary(self):
        """Test build_and_log_summary method"""
        wall_start = time.time()
        metrics = OrchestratorMetrics(num_stages=2, enable_stats=False, wall_start_ts=wall_start)

        # Add some data
        metrics.stage_first_ts[0] = wall_start
        metrics.stage_last_ts[0] = wall_start + 1.0
        metrics.stage_req_counts[0] = 5
        metrics.stage_total_tokens[0] = 100

        final_stage_map = {"req-1": 0}
        summary = metrics.build_and_log_summary(final_stage_map)

        assert "stages" in summary
        assert "transfers" in summary
        assert "e2e_requests" in summary
        assert summary["e2e_requests"] == 0  # No finalized requests yet
