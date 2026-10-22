"""Tests for external dataset mounting and memory/knowledge persistence."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.api.server import app
from app.graph.workflow import run_credit_modeling_workflow
from app.memory.memory_manager import MemoryManager
from app.tools.data_registry import DataPointerRegistry
from app.tools.rag_retriever import HybridCreditRAG

_REG = dict(obs_start_date="2025-01-01", obs_end_date="2025-06-30")


@pytest.fixture
def ext_files(tmp_path):
    """Dumps the synthetic warehouse into CSV/Parquet files inside the (temp) data dir."""
    reg = DataPointerRegistry.get_instance()
    train = reg.get_df("ds://credit_a_card_train")
    oot = reg.get_df("ds://credit_a_card_oot")
    train.to_csv(tmp_path / "ext_train.csv", index=False)
    oot.to_csv(tmp_path / "ext_oot.csv", index=False)
    return tmp_path


def test_register_csv_and_reject_invalid_inputs(ext_files):
    reg = DataPointerRegistry.get_instance()
    meta = reg.register_from_file("ds://t_ext", "ext_train.csv", "target_mob3_dpd30", **_REG)
    assert meta.row_count == 4000 and meta.tags["source_file"] == "ext_train.csv"

    with pytest.raises(ValueError, match="outside the data directory"):
        reg.register_from_file("ds://bad", "../secret.csv", "target_mob3_dpd30", **_REG)
    with pytest.raises(ValueError, match="not found in file"):
        reg.register_from_file("ds://bad", "ext_train.csv", "no_such_col", **_REG)
    with pytest.raises(ValueError, match="only 0/1"):
        reg.register_from_file("ds://bad", "ext_train.csv", "credit_inq_cnt_3m", **_REG)
    with pytest.raises(ValueError, match="ds://"):
        reg.register_from_file("bad_uri", "ext_train.csv", "target_mob3_dpd30", **_REG)
    with pytest.raises(FileNotFoundError):
        reg.register_from_file("ds://bad", "missing.csv", "target_mob3_dpd30", **_REG)
    (ext_files / "x.txt").write_text("a")
    with pytest.raises(ValueError, match="csv and .parquet"):
        reg.register_from_file("ds://bad", "x.txt", "target_mob3_dpd30", **_REG)


def test_end_to_end_modeling_on_external_dataset(ext_files):
    reg = DataPointerRegistry.get_instance()
    reg.register_from_file("ds://ext_train", "ext_train.csv", "target_mob3_dpd30", **_REG)
    reg.register_from_file(
        "ds://ext_oot", "ext_oot.csv", "target_mob3_dpd30", obs_start_date="2025-07-01", obs_end_date="2025-09-30"
    )
    state = run_credit_modeling_workflow(
        user_query="帮我全流程构建A卡评分模型，KS >= 0.30，PSI <= 0.10",
        session_id="pytest-ext-001",
        train_uri="ds://ext_train",
        oot_uri="ds://ext_oot",
    )
    assert state["dataset_refs"]["train_uri"] == "ds://ext_train"
    assert state["dataset_refs"]["train_woe_uri"] == "ds://ext_train_woe"
    assert "post_loan_repay_amt_m1" in state["quarantined_leakage_features"]
    assert state["validation_result"]["validation_passed"] is True
    assert state["compliance_passed"] is True


def test_register_endpoint_status_codes(ext_files):
    client = TestClient(app)
    body = dict(uri="ds://api_ext", path="ext_train.csv", target_col="target_mob3_dpd30", **_REG)
    ok = client.post("/api/v1/datasets/register", json=body)
    assert ok.status_code == 200 and ok.json()["row_count"] == 4000
    assert client.post("/api/v1/datasets/register", json={**body, "path": "../x.csv"}).status_code == 400
    assert client.post("/api/v1/datasets/register", json={**body, "path": "nope.csv"}).status_code == 404


def test_episodic_memory_persists_across_instances(tmp_path):
    mm = MemoryManager()
    builtin = len(mm._episodic_store)
    mm.record_reflection_experience("场景A", "教训A", ["f1"], "动作A")

    reloaded = MemoryManager()
    assert len(reloaded._episodic_store) == builtin + 1
    assert reloaded._episodic_store[-1].lesson_learned == "教训A"
    assert not (tmp_path / "memory_episodic.json.tmp").exists()


def test_corrupt_memory_file_falls_back_to_builtin(tmp_path):
    (tmp_path / "memory_episodic.json").write_text("{not json", encoding="utf-8")
    mm = MemoryManager()
    assert [r.memory_id for r in mm._episodic_store] == ["MEM-001", "MEM-002"]


def test_extra_knowledge_loaded_and_invalid_entries_skipped(tmp_path):
    extra = [
        {"doc_id": "EXT-001", "category": "compliance_sop", "title": "自定义制度", "content": "额外制度条款XYZ专属内容"},
        {"doc_id": "CAT-001", "category": "x", "title": "重复ID", "content": "应被忽略"},
        {"doc_id": "EXT-BAD", "title": "缺字段"},
    ]
    (tmp_path / "knowledge_extra.json").write_text(json.dumps(extra, ensure_ascii=False), encoding="utf-8")
    ids = {c.doc_id for c in HybridCreditRAG().corpus}
    assert "EXT-001" in ids and "EXT-BAD" not in ids
    assert sum(1 for c in HybridCreditRAG().corpus if c.doc_id == "CAT-001") == 1


# ---- optional observation window ------------------------------------------------------------------
def _write_csv(tmp_path, name, text):
    (tmp_path / name).write_text(text, encoding="utf-8")


def test_window_inferred_from_date_col(tmp_path):
    from app.tools.data_registry import DataPointerRegistry

    _write_csv(tmp_path, "dc.csv", "apply_date,x,y\n2025-03-05,1,0\n2025-01-10,2,1\n2025-02-01,3,0\n")
    meta = DataPointerRegistry.get_instance().register_from_file("ds://dc", "dc.csv", "y", date_col="apply_date")
    assert (meta.obs_start_date, meta.obs_end_date) == ("2025-01-10", "2025-03-05") and meta.tags["date_col"] == "apply_date"


def test_explicit_dates_override_date_col(tmp_path):
    from app.tools.data_registry import DataPointerRegistry

    _write_csv(tmp_path, "dc2.csv", "d,y\n2025-03-05,0\n2025-01-10,1\n")
    meta = DataPointerRegistry.get_instance().register_from_file(
        "ds://dc2", "dc2.csv", "y", obs_start_date="2024-01-01", obs_end_date="2024-12-31", date_col="d"
    )
    assert (meta.obs_start_date, meta.obs_end_date) == ("2024-01-01", "2024-12-31")


def test_bad_date_col_rejected(tmp_path):
    from app.tools.data_registry import DataPointerRegistry

    _write_csv(tmp_path, "dc3.csv", "d,y\nnot-a-date,0\n2025-01-10,1\n")
    registry = DataPointerRegistry.get_instance()
    with pytest.raises(ValueError, match="cannot be parsed"):
        registry.register_from_file("ds://dc3", "dc3.csv", "y", date_col="d")
    with pytest.raises(ValueError, match="not found"):
        registry.register_from_file("ds://dc3", "dc3.csv", "y", date_col="missing")


def test_no_window_skips_chronological_check_and_excludes_date_col(tmp_path):
    from app.tools.credit_metrics import DataInspectionInput, tool_inspect_dataset_and_check_leakage
    from app.tools.data_registry import DataPointerRegistry

    registry = DataPointerRegistry.get_instance()
    rows = "\n".join(f"2025-01-{(i % 28) + 1:02d},{i % 7},{i % 2}" for i in range(60))
    _write_csv(tmp_path, "nw_train.csv", "apply_date,x,y\n" + rows + "\n")
    _write_csv(tmp_path, "nw_oot.csv", "x,y\n" + "\n".join(f"{i % 7},{i % 2}" for i in range(30)) + "\n")
    registry.register_from_file("ds://nw_train", "nw_train.csv", "y", date_col="apply_date")
    registry.register_from_file("ds://nw_oot", "nw_oot.csv", "y")  # no dates at all

    result = tool_inspect_dataset_and_check_leakage(DataInspectionInput(train_uri="ds://nw_train", oot_uri="ds://nw_oot"))
    assert result["window_validation"]["is_chronological_valid"] is None
    assert result["window_validation"]["oot_window"] == "未提供"
    assert "apply_date" not in result["governance_audit"]["clean_candidate_features"]
    assert "apply_date" not in result["governance_audit"]["quarantined_leakage_features"]


def test_register_endpoint_without_dates(tmp_path):
    from fastapi.testclient import TestClient
    from app.api.server import app

    _write_csv(tmp_path, "ep.csv", "x,y\n1,0\n2,1\n")
    resp = TestClient(app).post("/api/v1/datasets/register", json={"uri": "ds://ep", "path": "ep.csv", "target_col": "y"})
    assert resp.status_code == 200
    listed = {d["uri"]: d for d in TestClient(app).get("/api/v1/datasets").json()["datasets"]}
    assert listed["ds://ep"]["obs_window"] == "未提供"


# ---- row split when no OOT dataset is selected ----------------------------------------------------
def _split_csv(tmp_path, name="rs.csv", n=100, date_col=True):
    head = "apply_date,x,y" if date_col else "x,y"
    rows = [
        (f"2025-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}," if date_col else "") + f"{i % 9},{i % 2}"
        for i in range(n)
    ]
    (tmp_path / name).write_text(head + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def test_split_by_rows_70_30(tmp_path):
    _split_csv(tmp_path)
    registry = DataPointerRegistry.get_instance()
    registry.register_from_file("ds://rs", "rs.csv", "y", date_col="apply_date")
    dev, oot = registry.split_by_rows("ds://rs")
    assert (dev, oot) == ("ds://rs#dev", "ds://rs#oot")
    assert registry.get_meta(dev).row_count == 70 and registry.get_meta(oot).row_count == 30
    assert registry.get_meta(oot).tags["row_split_of"] == "ds://rs"
    # row order preserved: dev holds the first 70 rows, OOT the last 30
    assert registry.get_df(dev)["apply_date"].iloc[0] == registry.get_df("ds://rs")["apply_date"].iloc[0]
    assert registry.get_meta(oot).obs_start_date > registry.get_meta(dev).obs_end_date  # sorted dates -> truly later


def test_split_rejects_tiny_or_single_class(tmp_path):
    _split_csv(tmp_path, "tiny.csv", n=20, date_col=False)
    (tmp_path / "one.csv").write_text("x,y\n" + "\n".join(f"{i},{0 if i < 70 else 1}" for i in range(100)) + "\n")
    registry = DataPointerRegistry.get_instance()
    registry.register_from_file("ds://tiny", "tiny.csv", "y")
    registry.register_from_file("ds://one", "one.csv", "y")  # dev part has only label 0
    for uri in ("ds://tiny", "ds://one"):
        with pytest.raises(ValueError, match="row split"):
            registry.split_by_rows(uri)


def test_resolve_modeling_uris_rules(tmp_path):
    _split_csv(tmp_path, "rr.csv", date_col=False)
    registry = DataPointerRegistry.get_instance()
    registry.register_from_file("ds://rr", "rr.csv", "y")
    assert registry.resolve_modeling_uris(None, None) == (None, None)
    assert registry.resolve_modeling_uris("a", "b") == ("a", "b")
    assert registry.resolve_modeling_uris("ds://rr", None) == ("ds://rr#dev", "ds://rr#oot")
    with pytest.raises(ValueError):
        registry.resolve_modeling_uris(None, "ds://rr")


def test_modeling_run_with_train_only_flags_row_split(tmp_path):
    rows = ["x1,x2,y"]
    import random

    rng = random.Random(7)
    for i in range(600):
        x1, x2 = rng.gauss(0, 1), rng.gauss(0, 1)
        rows.append(f"{x1:.4f},{x2:.4f},{1 if x1 + rng.gauss(0, 1) > 0.5 else 0}")
    (tmp_path / "only_train.csv").write_text("\n".join(rows) + "\n")
    client = TestClient(app)
    reg = client.post("/api/v1/datasets/register", json={"uri": "ds://only_train", "path": "only_train.csv", "target_col": "y"})
    assert reg.status_code == 200
    resp = client.post("/api/v1/modeling/run", json={"query": "构建A卡评分模型", "train_uri": "ds://only_train", "max_reflections": 0})
    assert resp.status_code == 200
    report = resp.json()["final_report_markdown"]
    assert "ds://only_train#dev" in report and "ds://only_train#oot" in report and "不是严格的跨期验证" in report


def test_modeling_oot_only_rejected_and_unknown_train_404():
    client = TestClient(app)
    assert client.post("/api/v1/modeling/run", json={"oot_uri": "ds://credit_a_card_oot"}).status_code == 400
    assert client.post("/api/v1/modeling/run", json={"train_uri": "ds://nope"}).status_code == 404


def test_unregister_parent_drops_split_pointers(tmp_path):
    _split_csv(tmp_path, "up.csv", date_col=False)
    registry = DataPointerRegistry.get_instance()
    registry.register_from_file("ds://up", "up.csv", "y")
    registry.split_by_rows("ds://up")
    registry.unregister("ds://up")
    assert not [d for d in registry.list_datasets() if d["uri"].startswith("ds://up")]


def test_non_numeric_columns_excluded_from_candidates_instead_of_crashing(tmp_path):
    import random

    rng = random.Random(11)
    rng_rows = ["x1,x2,rating,pct,y"]
    for i in range(600):
        x1, x2 = rng.gauss(0, 1), rng.gauss(0, 1)
        y = 1 if x1 + 0.5 * x2 + rng.gauss(0, 1) > 0.5 else 0
        rng_rows.append(f"{x1:.4f},{x2:.4f},{'A' if i % 3 else 'B'},{int(abs(x1) * 50)}%,{y}")
    (tmp_path / "nn.csv").write_text("\n".join(rng_rows) + "\n")
    client = TestClient(app)
    assert client.post("/api/v1/datasets/register", json={"uri": "ds://nn", "path": "nn.csv", "target_col": "y"}).status_code == 200
    resp = client.post("/api/v1/modeling/run", json={"query": "构建A卡评分模型", "train_uri": "ds://nn", "max_reflections": 0})
    assert resp.status_code == 200
    body = resp.json()
    assert "rating" not in body["selected_features"] and "pct" not in body["selected_features"]
    assert "非数值字段" in client.post(
        "/api/v1/modeling/run", json={"query": "只做数据探查", "train_uri": "ds://nn"}
    ).json()["final_report_markdown"]


# ---- persistence of registrations across restarts -------------------------------------------------
def _restart():
    DataPointerRegistry._instance = None  # simulates a service restart
    return DataPointerRegistry.get_instance()


def _persist_csv(tmp_path, name="ps.csv"):
    (tmp_path / name).write_text("d,x,y\n2025-01-02,1,0\n2025-03-04,2,1\n2025-02-01,3,0\n")


def test_registration_survives_restart(tmp_path):
    _persist_csv(tmp_path)
    registry = DataPointerRegistry.get_instance()
    registry.register_from_file("ds://ps", "ps.csv", "y", date_col="d", description="persisted")
    reloaded = _restart()
    meta = reloaded.get_meta("ds://ps")
    assert meta.row_count == 3 and meta.description == "persisted"
    assert (meta.obs_start_date, meta.obs_end_date) == ("2025-01-02", "2025-03-04")
    assert any(d["uri"] == "ds://credit_a_card_train" for d in reloaded.list_datasets())  # built-ins still there


def test_unregister_and_file_delete_forget_the_record(tmp_path):
    _persist_csv(tmp_path)
    _persist_csv(tmp_path, "ps2.csv")
    registry = DataPointerRegistry.get_instance()
    registry.register_from_file("ds://ps", "ps.csv", "y")
    registry.register_from_file("ds://ps2", "ps2.csv", "y")
    registry.unregister("ds://ps")
    client = TestClient(app)
    assert client.delete("/api/v1/datasets/files/ps2.csv").status_code == 200
    uris = {d["uri"] for d in _restart().list_datasets()}
    assert "ds://ps" not in uris and "ds://ps2" not in uris


def test_missing_file_is_skipped_but_record_kept(tmp_path):
    _persist_csv(tmp_path)
    DataPointerRegistry.get_instance().register_from_file("ds://ps", "ps.csv", "y")
    (tmp_path / "ps.csv").rename(tmp_path / "moved.csv")
    assert "ds://ps" not in {d["uri"] for d in _restart().list_datasets()}  # start-up does not fail
    (tmp_path / "moved.csv").rename(tmp_path / "ps.csv")
    assert "ds://ps" in {d["uri"] for d in _restart().list_datasets()}  # record survived, file is back


def test_corrupt_store_ignored_and_moved_aside(tmp_path):
    _persist_csv(tmp_path)
    (tmp_path / "dataset_registry.json").write_text("{not json")
    registry = _restart()  # does not raise
    assert not [d for d in registry.list_datasets() if d["uri"] == "ds://ps"]
    registry.register_from_file("ds://ps", "ps.csv", "y")
    assert (tmp_path / "dataset_registry.json.corrupt").read_text() == "{not json"
    assert "ds://ps" in {d["uri"] for d in _restart().list_datasets()}
