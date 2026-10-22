"""FastAPI Production Microservice for the Credit Risk Modeling Multi-Agent System.

Exposes:
- `POST /api/v1/modeling/run`: Executes the multi-agent workflow and returns structured metrics + report.
- `POST /api/v1/modeling/stream`: SSE real-time streaming of each Agent's CoT, ending with a `result` (or `error`) event.
- `GET  /api/v1/datasets`: Lists registered dataset pointers in DataPointerRegistry.
- `POST /api/v1/eval/benchmark`: Triggers the Golden Benchmark evaluation suite and returns quantitative metrics.
- `GET  /`: Modeling workbench UI (static files in `app/web/`).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Union
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.core.cards import get_profile
from app.agents.db_miner import iter_db_mining, iter_project_mining
from app.graph.state import CreditModelingState
from app.graph.workflow import build_credit_modeling_graph, run_credit_modeling_workflow
from app.tools.credit_metrics import DataInspectionInput, tool_inspect_dataset_and_check_leakage
from app.tools import db_explorer, db_project
from app.tools.data_registry import DataPointerRegistry
from app.tools.dataset_files import UploadTooLarge, delete_file, list_files, save_upload, validate_filename
from app.tools.leakage_exemptions import (
    get_exemption_state,
    grant_exemption,
    revoke_exemption,
)
from app.tools.session_store import delete_session, list_sessions, load_session, save_session
from eval.evaluators import run_benchmark_suite

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Credit-Modeling Multi-Agent System API",
    version="0.1.0",
    description="Enterprise Credit Risk Scorecard Modeling MAS with LangGraph, Hybrid RAG, Self-Reflection & Agent Eval",
)


class ModelingRequest(BaseModel):
    query: str = Field(
        default="帮我针对首贷客群全流程构建一张A卡评分模型，要求 KS >= 0.35，PSI <= 0.10，目标通过率 75%，并输出合规报告",
        description="自然语言风控建模或探查指令",
    )
    session_id: str = Field(default="prod-session-001", description="会话ID (用于短期记忆状态持久化)")
    max_reflections: int = Field(default=2, description="最大允许自愈回滚次数")
    train_uri: Optional[str] = Field(default=None, description="开发集数据指针 URI (默认使用合成A卡样本)；不传 oot_uri 时由它按行切分出开发/跨期集")
    oot_uri: Optional[str] = Field(default=None, description="跨期验证集数据指针 URI (默认使用合成A卡样本)；只传 train_uri 时按行切分 train 的前70%/后30%")
    card_type: Optional[Literal["A", "B", "C"]] = Field(
        default=None, description="卡种 A/B/C，显式指定时覆盖语义识别；未指定则按问题语义识别，默认A卡"
    )


class ExemptionRequest(BaseModel):
    uri: str = Field(description="数据集 URI")
    feature: str = Field(description="被自动隔离、申请豁免的特征名")
    reason: str = Field(min_length=1, description="豁免原因（必填，写入审计记录）")
    operator: str = Field(min_length=1, description="操作人（自报标识，系统不校验身份）")


class DatasetRegisterRequest(BaseModel):
    uri: str = Field(description="数据指针 URI，须以 ds:// 开头")
    path: str = Field(description="data/ 目录内的相对文件路径 (.csv / .parquet)")
    target_col: str = Field(description="0/1 标签列名")
    obs_start_date: Optional[str] = Field(default=None, description="观察期起始日期 YYYY-MM-DD（可选，缺省时由 date_col 推断）")
    obs_end_date: Optional[str] = Field(default=None, description="观察期结束日期 YYYY-MM-DD（可选，缺省时由 date_col 推断）")
    date_col: Optional[str] = Field(default=None, description="日期列名（可选），用于推断观察期；该列不参与建模")
    perf_window_months: int = Field(default=3, description="表现期月数")
    description: str = Field(default="", description="数据集描述")


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "healthy", "service": "credit-modeling-mas"}


@app.get("/api/v1/datasets")
async def list_registered_datasets() -> Dict[str, Any]:
    registry = DataPointerRegistry.get_instance()
    return {"datasets": registry.list_datasets()}


@app.post("/api/v1/datasets/register")
async def register_dataset(req: DatasetRegisterRequest) -> Dict[str, Any]:
    registry = DataPointerRegistry.get_instance()
    try:
        meta = registry.register_from_file(**req.model_dump())
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"uri": meta.uri, "row_count": meta.row_count, "bad_rate": meta.bad_rate, "columns": meta.columns}


@app.post("/api/v1/datasets/exemptions")
async def grant_leakage_exemption(req: ExemptionRequest) -> Dict[str, Any]:
    try:
        return grant_exemption(req.uri, req.feature, req.reason, req.operator)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"dataset not found: {e.args[0]}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/datasets/exemptions")
async def get_leakage_exemptions(uri: str = Query(..., description="数据集 URI")) -> Dict[str, Any]:
    return get_exemption_state(uri)


@app.delete("/api/v1/datasets/exemptions")
async def revoke_leakage_exemption(
    uri: str = Query(...),
    feature: str = Query(...),
    reason: str = Query(..., description="撤销原因"),
    operator: str = Query(..., description="操作人(自报标识，系统不校验身份)"),
) -> Dict[str, Any]:
    try:
        return revoke_exemption(uri, feature, reason, operator)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=e.args[0])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class InspectRequest(BaseModel):
    train_uri: str = Field(description="开发集数据指针 URI")
    oot_uri: str = Field(description="跨期验证集数据指针 URI")
    card_type: Literal["A", "B", "C"] = Field(default="A", description="卡种，仅用于知识库引用检索")


@app.post("/api/v1/datasets/inspect")
async def inspect_dataset(req: InspectRequest) -> Dict[str, Any]:
    """Runs the data-audit tool (windows, leakage / compliance quarantine with reasons and exemptions)."""
    try:
        return tool_inspect_dataset_and_check_leakage(DataInspectionInput(**req.model_dump()))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"dataset not found: {e.args[0]}")


@app.put("/api/v1/datasets/files/{filename}", status_code=201)
async def upload_dataset_file(filename: str, request: Request) -> Dict[str, Any]:
    """Uploads a raw .csv/.parquet body into the data directory (no overwrite, size-capped)."""
    try:
        size = await save_upload(filename, request.stream())
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except UploadTooLarge as e:
        raise HTTPException(status_code=413, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"filename": filename, "size_bytes": size}


@app.get("/api/v1/datasets/files")
async def list_dataset_files() -> Dict[str, Any]:
    """Files in the data directory, each with the pointers mounted from it (deleting the file unregisters them)."""
    registry = DataPointerRegistry.get_instance()
    return {"files": [{**f, "datasets": registry.uris_for_file(f["filename"])} for f in list_files()]}


@app.delete("/api/v1/datasets/files/{filename}")
async def delete_dataset_file(filename: str) -> Dict[str, Any]:
    """Deletes a data file and cascades: pointers mounted from it are unregistered. Exemption history is kept."""
    try:
        validate_filename(filename)
        registry = DataPointerRegistry.get_instance()
        unregistered = registry.uris_for_file(filename)
        delete_file(filename)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    for uri in unregistered:
        registry.unregister(uri)
    return {"filename": filename, "unregistered": unregistered}


@app.delete("/api/v1/datasets")
async def unregister_dataset(uri: str = Query(..., description="要注销的数据集 URI（文件保留）")) -> Dict[str, Any]:
    try:
        DataPointerRegistry.get_instance().unregister(uri)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=e.args[0])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"uri": uri}


@app.get("/api/v1/sessions")
async def list_modeling_sessions() -> Dict[str, Any]:
    return {"sessions": list_sessions()}


@app.get("/api/v1/sessions/{session_id}")
async def get_modeling_session(session_id: str) -> Dict[str, Any]:
    stored = load_session(session_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="session not found")
    return stored


@app.delete("/api/v1/sessions/{session_id}")
async def delete_modeling_session(session_id: str) -> Dict[str, Any]:
    if not delete_session(session_id):
        raise HTTPException(status_code=404, detail="session not found")
    return {"session_id": session_id}


def _persist_session(session_id: str, query: str, result: Dict[str, Any]) -> None:
    """History is best-effort: a storage failure must never break the modeling response."""
    try:
        save_session(session_id, query, result)
    except OSError:
        logger.warning("failed to persist session %s", session_id, exc_info=True)


def _build_result(state: CreditModelingState) -> Dict[str, Any]:
    """Final workflow result shared by `/modeling/run` and the `result` SSE event."""
    val = state.get("validation_result", {})
    return {
        "intent_type": state.get("intent_type"),
        "card_type": state.get("card_type"),
        "plan_dag": state.get("plan_dag"),
        "selected_features": state.get("selected_features", []),
        "selected_feature_metrics": state.get("feature_selection_result", {}).get("selected_feature_metrics", []),
        "reflection_count": state.get("reflection_count", 0),
        "reflection_history": state.get("reflection_history", []),
        "validation_passed": val.get("validation_passed"),
        "anomalies": val.get("anomalies", []),
        "validation_metrics": val.get("metrics", {}),
        "cutoff_strategy": val.get("cutoff_strategy", {}),
        "oot_score_bands": val.get("oot_score_bands", []),
        "challenger_result": state.get("challenger_result", {}),
        "compliance_passed": state.get("compliance_passed", False),
        "memory_governance": state.get("memory_governance_stats", {}),
        "agent_logs": state.get("agent_logs", []),
        "final_report_markdown": state.get("final_report_markdown", ""),
    }


def _resolve_uris(req: ModelingRequest) -> tuple[Optional[str], Optional[str]]:
    """Only a development set selected -> the OOT set is split from it by row order (70% / 30%)."""
    try:
        return DataPointerRegistry.get_instance().resolve_modeling_uris(req.train_uri, req.oot_uri)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"dataset not found: {e.args[0]}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/v1/modeling/run")
async def execute_modeling_workflow(req: ModelingRequest) -> Dict[str, Any]:
    train_uri, oot_uri = _resolve_uris(req)
    state = run_credit_modeling_workflow(
        user_query=req.query,
        session_id=req.session_id,
        max_reflections=req.max_reflections,
        train_uri=train_uri,
        oot_uri=oot_uri,
        card_type=req.card_type,
    )
    result = _build_result(state)
    _persist_session(req.session_id, req.query, result)
    return {"session_id": req.session_id, **result}


def _sse(payload: Dict[str, Any], event: Optional[str] = None) -> str:
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


@app.post("/api/v1/modeling/stream")
async def stream_modeling_workflow(req: ModelingRequest) -> StreamingResponse:
    """SSE endpoint: one default `data` event per executed node, then a final `result` (or `error`) event."""

    train_uri, oot_uri = _resolve_uris(req)

    async def _event_generator() -> AsyncGenerator[str, None]:
        initial_state: CreditModelingState = {
            "session_id": req.session_id,
            "user_query": req.query,
            "reflection_count": 0,
            "max_reflections": req.max_reflections,
            "excluded_by_reflection": [],
            "reflection_history": [],
            "tool_calls": [],
            "agent_logs": [],
        }
        if req.card_type:
            initial_state["card_type"] = req.card_type
        if train_uri or oot_uri:
            profile = get_profile(req.card_type)
            initial_state["dataset_refs"] = {
                "train_uri": train_uri or profile.train_uri,
                "oot_uri": oot_uri or profile.oot_uri,
            }
        config = {"configurable": {"thread_id": req.session_id}}
        try:
            graph = build_credit_modeling_graph(with_checkpointer=True)
            for event in graph.stream(initial_state, config=config):
                for node_name, node_update in event.items():
                    logs = (node_update or {}).get("agent_logs", [])
                    latest_log = logs[-1] if logs else {}
                    yield _sse(
                        {
                            "node": node_name,
                            "agent": latest_log.get("agent", node_name),
                            "action": latest_log.get("action", ""),
                            "thought": latest_log.get("thought", ""),
                            "latency_ms": latest_log.get("latency_ms", 0.0),
                        }
                    )
            result = _build_result(graph.get_state(config).values)
            _persist_session(req.session_id, req.query, result)
            yield _sse(result, event="result")
        except Exception as exc:  # surface failures to the client instead of silently closing the stream
            yield _sse({"message": f"{type(exc).__name__}: {exc}"}, event="error")

    return StreamingResponse(_event_generator(), media_type="text/event-stream")


class DBMineRequest(BaseModel):
    table: str = Field(description="要挖掘的表名（须在库元数据内且在 DB_ALLOWED_TABLES 内）")
    label_col: str = Field(description="标签列")
    bad_values: List[str] = Field(min_length=1, description="标签列中代表风险事件(1)的取值")
    good_values: Optional[List[str]] = Field(default=None, description="代表非风险(0)的取值；标签列含 bad_values 之外的取值时必填")
    date_col: Optional[str] = Field(default=None, description="日期列，用于按时间切分开发集 / 跨期集")
    oot_start_date: Optional[str] = Field(default=None, description="跨期集起始日期 YYYY-MM-DD（需要 date_col）；缺省按时间排序切 70/30")
    max_rows: Optional[int] = Field(default=None, ge=100, description="取样行数上限（不超过 DB_MAX_ROWS）")


class DBSampleRequest(DBMineRequest):
    uri: str = Field(description="落盘后注册的数据集 URI，须以 ds:// 开头")
    filename: Optional[str] = Field(default=None, description="data/ 内的 .csv 文件名，缺省 db_<表名>.csv；不覆盖已有文件")


def _db_error(exc: Exception) -> HTTPException:
    if isinstance(exc, db_project.ProfileNotConfigured):
        return HTTPException(status_code=503, detail=f"项目配置未就绪：{exc}")
    if isinstance(exc, db_explorer.DBNotConfigured):
        return HTTPException(
            status_code=503,
            detail="数据库未配置：请在 .env 设置 DB_HOST、DB_PORT、DB_USER、DB_PASSWORD、DB_NAME",
        )
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc.args[0]))
    if isinstance(exc, FileExistsError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    logger.warning("database error", exc_info=True)
    return HTTPException(status_code=502, detail="数据库查询失败，详情见服务端日志")


@app.get("/api/v1/db/status")
async def db_status() -> Dict[str, Any]:
    return {"configured": db_explorer.is_configured()}


@app.get("/api/v1/db/tables")
async def db_tables() -> Dict[str, Any]:
    try:
        return {"tables": db_explorer.list_tables()}
    except Exception as e:  # noqa: BLE001
        raise _db_error(e)


@app.get("/api/v1/db/tables/{table}/columns")
async def db_table_columns(table: str) -> Dict[str, Any]:
    try:
        return {"table": table, "columns": db_explorer.describe_table(table)}
    except Exception as e:  # noqa: BLE001
        raise _db_error(e)


@app.post("/api/v1/db/mine")
async def db_mine(req: DBMineRequest) -> StreamingResponse:
    """SSE: one `data` event per stage, then `result` (or `error`). Read-only against the configured database."""
    if not db_explorer.is_configured():
        raise _db_error(db_explorer.DBNotConfigured())

    async def _gen() -> AsyncGenerator[str, None]:
        try:
            for kind, payload in iter_db_mining(
                req.table, req.label_col, req.bad_values, req.good_values, req.date_col, req.oot_start_date, req.max_rows
            ):
                yield _sse(payload, None if kind == "step" else "result")
        except Exception as e:  # noqa: BLE001
            err = _db_error(e)
            yield _sse({"message": str(err.detail)}, "error")

    return StreamingResponse(_gen(), media_type="text/event-stream")


@app.post("/api/v1/db/sample")
async def db_sample(req: DBSampleRequest) -> Dict[str, Any]:
    """Writes a cleaned sample of the table to data/ (label as 0/1) and registers it as a persisted dataset."""
    try:
        return db_explorer.sample_to_dataset(
            req.table, req.label_col, req.bad_values, req.uri, req.good_values, req.date_col, req.max_rows, req.filename
        )
    except Exception as e:  # noqa: BLE001
        raise _db_error(e)


class DBProjectMineRequest(BaseModel):
    params: Dict[str, Union[int, float, str]] = Field(
        default_factory=dict, description="覆盖项目配置 params 中的同名阈值，例如 {\"bad_overdue_days\": 15}"
    )
    oot_start_date: Optional[str] = Field(default=None, description="跨期集起始日期 YYYY-MM-DD；缺省按时间排序切 70/30")
    max_rows: Optional[int] = Field(default=None, ge=100, description="取样行数上限（不超过 DB_MAX_ROWS）")


@app.get("/api/v1/db/project")
async def db_project_info() -> Dict[str, Any]:
    """Summary of the project profile (base table, joins, label rule, tunable params); `configured: false` if absent."""
    if not db_project.is_configured():
        return {"configured": False}
    try:
        return db_project.summarize_profile(db_project.load_profile())
    except Exception as e:  # noqa: BLE001
        raise _db_error(e)


@app.post("/api/v1/db/project/mine")
async def db_project_mine(req: DBProjectMineRequest) -> StreamingResponse:
    """SSE: profile-driven multi-table mining (explore -> label -> indicators -> categories), then `result` / `error`."""
    if not db_explorer.is_configured():
        raise _db_error(db_explorer.DBNotConfigured())
    if not db_project.is_configured():
        raise _db_error(db_project.ProfileNotConfigured(str(db_project.profile_path())))

    async def _gen() -> AsyncGenerator[str, None]:
        try:
            for kind, payload in iter_project_mining(req.params, req.oot_start_date, req.max_rows):
                yield _sse(payload, None if kind == "step" else "result")
        except Exception as e:  # noqa: BLE001
            err = _db_error(e)
            yield _sse({"message": str(err.detail)}, "error")

    return StreamingResponse(_gen(), media_type="text/event-stream")


@app.post("/api/v1/eval/benchmark")
async def trigger_benchmark_evaluation() -> Dict[str, Any]:
    return run_benchmark_suite()


# Workbench UI (static, no build step). Mounted last so it never shadows the API routes above.
app.mount("/", StaticFiles(directory=Path(__file__).resolve().parents[1] / "web", html=True), name="web")
