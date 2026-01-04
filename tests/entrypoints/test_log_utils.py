import pytest

from vllm_omni.entrypoints import log_utils


def test_record_and_aggregate_transfer_stats():
    transfer_agg: dict = {}
    transfer_edge_req: dict = {}
    per_request: dict = {}

    log_utils.record_sender_transfer_agg(
        transfer_agg,
        transfer_edge_req,
        from_stage=0,
        to_stage=1,
        req_id="req-1",
        size_bytes=1024,
        tx_ms=2.5,
    )

    combined = log_utils.aggregate_rx_and_maybe_total(
        transfer_edge_req,
        transfer_agg,
        per_request,
        stage_id=1,
        req_id="req-1",
        rx_bytes=512,
        rx_ms=1.0,
        in_flight_ms=0.5,
    )

    assert combined is not None
    size_b, tx_ms, total_ms = combined
    assert size_b == 1024
    assert tx_ms == pytest.approx(2.5)
    assert total_ms == pytest.approx(4.0)

    agg = transfer_agg[(0, 1)]
    assert agg["sum_rx_bytes"] == pytest.approx(512)
    assert agg["sum_total_ms"] == pytest.approx(4.0)

    request_metrics = per_request["req-1"]
    assert request_metrics["transfers_bytes"] == 512
    assert request_metrics["transfers_ms"] == pytest.approx(4.0)


def test_build_transfer_summary_computes_rates():
    transfer_agg = {
        (0, 1): {
            "sum_bytes": 2048.0,
            "sum_ms": 4.0,
            "count": 2.0,
            "sum_rx_bytes": 1024.0,
            "sum_rx_ms": 2.0,
            "rx_count": 2.0,
            "sum_total_ms": 6.0,
            "total_count": 2.0,
        }
    }

    summary = log_utils.build_transfer_summary(transfer_agg)

    assert len(summary) == 1
    row = summary[0]
    assert row["tx_mbps"] == pytest.approx(4.096)
    assert row["rx_mbps"] == pytest.approx(4.096)
    assert row["total_mbps"] == pytest.approx(2.730666, rel=1e-6)
    assert row["samples"] == 2
    assert row["rx_samples"] == 2
    assert row["total_samples"] == 2


def test_count_tokens_from_outputs_handles_missing():
    class _TokenObj:
        def __init__(self, tokens):
            self.token_ids = tokens

    class _Result:
        def __init__(self, tokens):
            self.outputs = [_TokenObj(tokens)]

    outputs = [_Result([1, 2, 3, 4]), _Result([5])]
    total = log_utils.count_tokens_from_outputs(outputs)

    assert total == 5


def test_build_stage_summary_uses_totals():
    summary = log_utils.build_stage_summary(
        stage_req_counts=[2, 0],
        stage_total_tokens=[30, 0],
        stage_total_time_ms=[60.0, 0.0],
    )

    assert summary[0]["avg_time_per_request_ms"] == pytest.approx(30.0)
    assert summary[0]["avg_tokens_per_s"] == pytest.approx(500.0)
    assert summary[0]["requests"] == 2
    assert summary[0]["tokens"] == 30
