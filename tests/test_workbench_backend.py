"""Tests for dataset upload, the inspect endpoint, session history persistence and the new workbench pages."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

import app.api.server as server
from app.api.server import app
from app.tools import session_store
from app.tools.dataset_files import UploadTooLarge, save_upload

_QUERY = "帮我针对首贷客群全流程构建一张A卡评分模型，要求 KS >= 0.35，PSI <= 0.10，目标通过率 75%，并输出合规报告"
CSV = b"a,b,y\n1,2,0\n3,4,1\n"


def _put(client, name, body=CSV):
    return client.put(f"/api/v1/datasets/files/{name}", content=body, headers={"Content-Type": "application/octet-stream"})


def _leftovers(tmp_path):
    return [p.name for p in tmp_path.iterdir() if p.name.startswith(".upload-")]


# ---- upload ---------------------------------------------------------------------------------------
def test_upload_success_then_register(tmp_path):
    client = TestClient(app)
    resp = _put(client, "up_train.csv")
    assert resp.status_code == 201 and resp.json() == {"filename": "up_train.csv", "size_bytes": len(CSV)}
    assert (tmp_path / "up_train.csv").read_bytes() == CSV and _leftovers(tmp_path) == []

    reg = client.post(
        "/api/v1/datasets/register",
        json={"uri": "ds://up_train", "path": "up_train.csv", "target_col": "y",
              "obs_start_date": "2025-01-01", "obs_end_date": "2025-06-30"},
    )
    assert reg.status_code == 200 and reg.json()["row_count"] == 2


def test_upload_chinese_filename(tmp_path):
    client = TestClient(app)
    name = "风险用户与非风险用户.csv"
    resp = _put(client, name)
    assert resp.status_code == 201 and resp.json() == {"filename": name, "size_bytes": len(CSV)}
    assert (tmp_path / name).read_bytes() == CSV


def test_upload_never_overwrites(tmp_path):
    client = TestClient(app)
    assert _put(client, "dup.csv").status_code == 201
    again = _put(client, "dup.csv", b"x,y\n9,9\n")
    assert again.status_code == 409 and (tmp_path / "dup.csv").read_bytes() == CSV
    assert _leftovers(tmp_path) == []


@pytest.mark.parametrize("name", ["data.txt", "a b.csv", ".hidden.csv", "a..b.csv", "evil.csv.exe", "x" * 120 + ".csv"])
def test_upload_rejects_bad_names(tmp_path, name):
    resp = _put(TestClient(app), name)
    assert resp.status_code == 400
    assert [p.name for p in tmp_path.iterdir()] == []


def test_upload_rejects_path_traversal(tmp_path):
    # The normalised path no longer matches the upload route (400/404/405 depending on routing); any rejection
    # is fine as long as nothing is written inside or outside the data directory.
    resp = _put(TestClient(app), "%2e%2e%2fescape.csv")
    assert resp.status_code in (400, 404, 405)
    assert not (tmp_path.parent / "escape.csv").exists()
    assert list(tmp_path.iterdir()) == []


def test_upload_rejects_empty_body(tmp_path):
    assert _put(TestClient(app), "empty.csv", b"").status_code == 400
    assert [p.name for p in tmp_path.iterdir()] == []


def test_upload_size_limit_streams_and_cleans_up(tmp_path):
    async def chunks():
        for _ in range(5):
            yield b"x" * 10

    with pytest.raises(UploadTooLarge):
        asyncio.run(save_upload("big.csv", chunks(), max_bytes=25))
    assert [p.name for p in tmp_path.iterdir()] == []


def test_upload_endpoint_returns_413_when_too_large(monkeypatch, tmp_path):
    real = server.save_upload
    monkeypatch.setattr(server, "save_upload", lambda f, c: real(f, c, max_bytes=10))
    assert _put(TestClient(app), "big.csv", b"x" * 50).status_code == 413
    assert [p.name for p in tmp_path.iterdir()] == []


# ---- inspect --------------------------------------------------------------------------------------
def test_inspect_endpoint():
    client = TestClient(app)
    body = {"train_uri": "ds://credit_a_card_train", "oot_uri": "ds://credit_a_card_oot"}
    resp = client.post("/api/v1/datasets/inspect", json=body)
    gov = resp.json()["governance_audit"]
    assert resp.status_code == 200 and gov["quarantined_leakage_features"] == ["post_loan_repay_amt_m1"]
    assert {"exemptable_features", "exempted_features", "detection_reasons"} <= set(gov)
    assert client.post("/api/v1/datasets/inspect", json={**body, "train_uri": "ds://nope"}).status_code == 404
    assert client.post("/api/v1/datasets/inspect", json={**body, "card_type": "Z"}).status_code == 422


# ---- sessions -------------------------------------------------------------------------------------
def test_run_and_stream_persist_sessions_and_api_returns_them():
    client = TestClient(app)
    client.post("/api/v1/modeling/run", json={"query": _QUERY, "session_id": "hist-run"})
    client.post("/api/v1/modeling/stream", json={"query": "什么是数据穿越？", "session_id": "hist-stream"})

    listing = client.get("/api/v1/sessions").json()["sessions"]
    by_id = {s["session_id"]: s for s in listing}
    assert {"hist-run", "hist-stream"} <= set(by_id)
    assert by_id["hist-run"]["validation_passed"] is True and by_id["hist-run"]["card_type"] == "A"
    assert listing[0]["session_id"] == "hist-stream"  # newest first

    stored = client.get("/api/v1/sessions/hist-run").json()
    assert stored["query"] == _QUERY and stored["result"]["validation_metrics"]["oot_ks"] >= 0.35
    assert client.get("/api/v1/sessions/hist-stream").json()["result"]["intent_type"] == "knowledge_qa"


@pytest.mark.parametrize("sid", ["missing", "..%2F..%2Fetc%2Fpasswd", "a.b", "x" * 70])
def test_session_lookup_rejects_unknown_and_malicious_ids(sid):
    assert TestClient(app).get(f"/api/v1/sessions/{sid}").status_code == 404


def test_session_ids_are_sanitised_on_save(tmp_path):
    stored = session_store.save_session("a:b/c d", "q", {"card_type": "A"})
    assert stored == "a_b_c_d" and (tmp_path / "sessions" / "a_b_c_d.json").is_file()
    assert session_store.load_session("a_b_c_d")["query"] == "q"


def test_session_cap_prunes_oldest(monkeypatch, tmp_path):
    monkeypatch.setattr(session_store, "MAX_SESSIONS", 3)
    for i in range(5):
        session_store.save_session(f"s{i}", f"q{i}", {})
    assert [s["session_id"] for s in session_store.list_sessions()] == ["s4", "s3", "s2"]
    assert sorted(p.name for p in (tmp_path / "sessions").iterdir()) == ["s2.json", "s3.json", "s4.json"]


def test_corrupt_session_file_is_skipped(tmp_path):
    session_store.save_session("ok", "q", {})
    (tmp_path / "sessions" / "bad.json").write_text("{nope", encoding="utf-8")
    assert [s["session_id"] for s in session_store.list_sessions()] == ["ok"]
    assert session_store.load_session("bad") is None


def test_storage_failure_does_not_break_modeling_response(monkeypatch):
    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(server, "save_session", boom)
    resp = TestClient(app).post("/api/v1/modeling/run", json={"query": "什么是数据穿越？", "session_id": "no-store"})
    assert resp.status_code == 200 and resp.json()["intent_type"] == "knowledge_qa"


# ---- pages ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("path", ["/data.html", "/history.html", "/data.js", "/history.js", "/common.js"])
def test_new_workbench_assets_are_served(path):
    assert TestClient(app).get(path).status_code == 200
