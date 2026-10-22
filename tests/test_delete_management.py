"""Tests for deleting sessions, data files and dataset pointers (cascade, built-in protection, audit history kept)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.server import app
from app.tools import session_store
from app.tools.data_registry import DataPointerRegistry
from app.tools.leakage_exemptions import get_exemption_state

CSV = b"a,b,y\n1,2,0\n3,4,1\n"
_REG = {"target_col": "y", "obs_start_date": "2025-01-01", "obs_end_date": "2025-06-30"}


def _upload_and_register(client, name, *uris):
    client.put(f"/api/v1/datasets/files/{name}", content=CSV, headers={"Content-Type": "application/octet-stream"})
    for uri in uris:
        assert client.post("/api/v1/datasets/register", json={"uri": uri, "path": name, **_REG}).status_code == 200


def test_delete_file_cascades_to_pointers(tmp_path):
    client = TestClient(app)
    _upload_and_register(client, "del_a.csv", "ds://del_a1", "ds://del_a2")
    listed = {f["filename"]: f["datasets"] for f in client.get("/api/v1/datasets/files").json()["files"]}
    assert sorted(listed["del_a.csv"]) == ["ds://del_a1", "ds://del_a2"]

    resp = client.delete("/api/v1/datasets/files/del_a.csv")
    assert resp.status_code == 200 and sorted(resp.json()["unregistered"]) == ["ds://del_a1", "ds://del_a2"]
    assert not (tmp_path / "del_a.csv").exists()
    uris = {d["uri"] for d in client.get("/api/v1/datasets").json()["datasets"]}
    assert not uris & {"ds://del_a1", "ds://del_a2"} and "ds://credit_a_card_train" in uris


def test_delete_file_errors():
    client = TestClient(app)
    assert client.delete("/api/v1/datasets/files/nope.csv").status_code == 404
    assert client.delete("/api/v1/datasets/files/bad.txt").status_code == 400


def test_unregister_keeps_file_and_protects_builtin(tmp_path):
    client = TestClient(app)
    _upload_and_register(client, "del_b.csv", "ds://del_b")
    assert client.delete("/api/v1/datasets", params={"uri": "ds://del_b"}).status_code == 200
    assert (tmp_path / "del_b.csv").exists()
    assert client.delete("/api/v1/datasets", params={"uri": "ds://del_b"}).status_code == 404
    assert client.delete("/api/v1/datasets", params={"uri": "ds://credit_a_card_train"}).status_code == 400
    assert "ds://credit_a_card_train" in {d["uri"] for d in DataPointerRegistry.get_instance().list_datasets()}


def test_delete_file_keeps_exemption_history():
    client = TestClient(app)
    _upload_and_register(client, "del_c.csv", "ds://del_c")
    from app.tools import leakage_exemptions as ex

    data = ex._load()
    data["history"].append({"event": "grant", "uri": "ds://del_c", "feature": "f", "at": "t"})
    ex._save(data)
    client.delete("/api/v1/datasets/files/del_c.csv")
    assert any(h["uri"] == "ds://del_c" for h in get_exemption_state("ds://del_c")["history"])


def test_delete_session(tmp_path):
    client = TestClient(app)
    session_store.save_session("del-s1", "q", {"card_type": "A"})
    assert client.delete("/api/v1/sessions/del-s1").status_code == 200
    assert client.get("/api/v1/sessions/del-s1").status_code == 404
    assert client.delete("/api/v1/sessions/del-s1").status_code == 404
    assert client.delete("/api/v1/sessions/bad.id").status_code == 404
