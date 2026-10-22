"""Tests for the workbench static UI and the SSE `result` / `error` events."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

import app.api.server as server
from app.api.server import app

_FULL_QUERY = "帮我针对首贷客群全流程构建一张A卡评分模型，要求 KS >= 0.35，PSI <= 0.10，目标通过率 75%，并输出合规报告"


def _parse_sse(text: str):
    events = []
    for block in text.strip().split("\n\n"):
        name, data = "message", None
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())
        events.append((name, data))
    return events


def test_workbench_static_files_and_api_not_shadowed():
    client = TestClient(app)
    index = client.get("/")
    assert index.status_code == 200 and 'lang="zh-CN"' in index.text
    assert client.get("/app.js").status_code == 200
    assert client.get("/style.css").status_code == 200
    assert client.get("/health").json()["status"] == "healthy"
    assert client.get("/api/v1/datasets").status_code == 200


def test_stream_emits_node_events_then_result():
    client = TestClient(app)
    resp = client.post("/api/v1/modeling/stream", json={"query": _FULL_QUERY, "session_id": "sse-full"})
    assert resp.status_code == 200
    events = _parse_sse(resp.text)

    names = [n for n, _ in events]
    assert names[-1] == "result" and names.count("result") == 1 and "error" not in names
    nodes = [d["node"] for n, d in events if n == "message"]
    assert nodes[0] == "orchestrator" and nodes[-1] == "compliance_reporter"
    assert nodes.count("feature_engineer") == 2  # one self-reflection rollback

    result = events[-1][1]
    for key in (
        "card_type", "validation_passed", "validation_metrics", "oot_score_bands",
        "selected_feature_metrics", "challenger_result", "reflection_history", "final_report_markdown",
    ):
        assert key in result
    assert result["validation_passed"] is True and result["card_type"] == "A"
    assert len(result["oot_score_bands"]) == 5 and result["challenger_result"]["challenger_metrics"]


def test_stream_result_for_non_modeling_intent_has_no_metrics():
    client = TestClient(app)
    resp = client.post("/api/v1/modeling/stream", json={"query": "什么是数据穿越？", "session_id": "sse-qa"})
    result = _parse_sse(resp.text)[-1]
    assert result[0] == "result"
    assert result[1]["intent_type"] == "knowledge_qa" and result[1]["oot_score_bands"] == []


def test_stream_emits_error_event_when_workflow_fails(monkeypatch):
    def boom(**_):
        raise RuntimeError("graph exploded")

    monkeypatch.setattr(server, "build_credit_modeling_graph", boom)
    resp = TestClient(app).post("/api/v1/modeling/stream", json={"query": "x", "session_id": "sse-err"})
    events = _parse_sse(resp.text)
    assert events == [("error", {"message": "RuntimeError: graph exploded"})]


def test_run_endpoint_keeps_existing_fields_and_adds_chart_data():
    resp = TestClient(app).post("/api/v1/modeling/run", json={"query": _FULL_QUERY, "session_id": "run-ui"})
    body = resp.json()
    assert body["session_id"] == "run-ui" and body["validation_metrics"]["oot_ks"] >= 0.35
    assert body["oot_score_bands"] and body["selected_feature_metrics"]
