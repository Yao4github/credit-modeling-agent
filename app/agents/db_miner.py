"""DB indicator-mining agent: schema -> in-DB profile -> single-variable indicators -> report.

All numbers come from the deterministic read-only tools in `app.tools.db_explorer`; the optional LLM only writes a short
interpretation of the finished report (never alters a figure) and is skipped when no provider is usable.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

from app.core.llm import LLMError, get_llm_client
from app.harness.guards import numbers_grounded
from app.tools import db_explorer, db_graph, db_project

_TOP_N = 20


def _fmt(v: Any) -> str:
    return "-" if v is None else str(v)


def build_report(profile: Dict[str, Any], mined: Dict[str, Any]) -> str:
    label, split = mined["label"], mined["split"]
    kinds = mined["column_kinds"]
    kind_count = {k: sum(1 for v in kinds.values() if v == k) for k in sorted(set(kinds.values()))}
    metrics = {m["feature"]: m for m in mined["feature_metrics"]}
    lines = [
        f"# 🗄️ 数据库表 `{mined['table']}` 指标挖掘报告",
        "",
        "## 1. 数据概况",
        f"- **表行数**：`{profile['row_count']}`，列数 `{len(profile['columns'])}`（库内聚合统计）",
        f"- **本次取样**：`{label['rows_used']}` 行{'（已达取样上限被截断）' if label['truncated'] else ''}",
        f"- **标签 `{label['column']}`**：风险（{', '.join(label['bad_values'])}）`{label['bad_count']}` / 非风险 `{label['good_count']}`；"
        f"灰样本（自动丢弃）`{label['dropped_unlabeled']}`",
        f"- **样本切分**：{split['mode']}（开发集 `{split['train_rows']}` / 跨期集 `{split['oot_rows']}`）",
        f"- **列类型推断**：{', '.join(f'{k} {v} 列' for k, v in kind_count.items())}"
        "（`numeric_text` 为数值存成文本，已自动转换）",
    ]
    if mined["warnings"]:
        lines += ["", "## 2. 风险提示", *[f"- ⚠️ {w}" for w in mined["warnings"]]]
    lines += ["", f"## 3. 候选特征建议（{len(mined['suggested_features'])} 个，按 IV 降序，最多展示 {_TOP_N}）"]
    if mined["suggested_features"]:
        lines += ["", "| 特征 | 注释 / 释义 | 类型 | 缺失率 | IV | KS | AUC | PSI |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for f in mined["suggested_features"][:_TOP_N]:
            m = metrics[f]
            cmt = (m.get("comment") or "-").replace("\n", " ").replace("|", "/")
            lines.append(
                f"| {f} | {cmt} | {m['kind']} | {m['missing_rate']} | {m['iv']} | {m['ks']} | {m['auc']} | {_fmt(m['psi'])} |"
            )
    else:
        lines.append("- 没有特征同时满足 IV / PSI / 缺失率 / 共线性阈值。")
    th = mined["thresholds"]
    comments = mined.get("column_comments") or {}
    lines += [
        "",
        f"> 入选阈值：IV ≥ {th['iv_min']}，PSI ≤ {th['psi_max']}，缺失率 ≤ {th['missing_max']}，与更高 IV 特征相关系数绝对值 ≤ {th['corr_max']}。",
        "",
        f"## 4. 未入选字段（{len(mined['not_suggested'])} 个）",
    ]
    lines += [
        f"- `{f}`（{comments[f]}）：{why}" if comments.get(f) else f"- `{f}`：{why}"
        for f, why in list(mined["not_suggested"].items())[:50]
    ] or ["- 无"]
    return "\n".join(lines)


def _llm_commentary(report: str, index: int = 5) -> str:
    client = get_llm_client()
    if not client.available:
        return ""
    try:
        resp = client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "你是信贷风控建模专家。请严格依据下方报告中的实测指标，用不超过150字说明数据质量与候选特征的使用建议。"
                        "严禁篡改或编造任何数值。"
                    ),
                },
                {"role": "user", "content": report},
            ],
            max_tokens=300,
        )
        if not numbers_grounded(resp.content, report)[0]:
            return ""
        return f"\n\n## {index}. 专家解读 (LLM: {resp.model})\n\n{resp.content.strip()}"
    except LLMError:
        return ""


def iter_db_mining(
    table: str,
    label_col: str,
    bad_values: Sequence[str],
    good_values: Optional[Sequence[str]] = None,
    date_col: Optional[str] = None,
    oot_start_date: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Yields ("step", log) after each stage and finally ("result", payload). Raises on invalid input."""

    def step(agent: str, t0: float, action: str) -> Tuple[str, Dict[str, Any]]:
        return "step", {"agent": agent, "action": action, "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2)}

    t0 = time.perf_counter()
    cols = db_explorer.describe_table(table)
    yield step("表结构", t0, f"读取表 `{table}` 结构：{len(cols)} 列。")

    t0 = time.perf_counter()
    profile = db_explorer.profile_table(table)
    empty_cols = [c["name"] for c in profile["columns"] if c["missing_rate"] >= 0.95]
    yield step("库内画像", t0, f"库内聚合完成：{profile['row_count']} 行，缺失率≥95% 的列 {len(empty_cols)} 个。")

    t0 = time.perf_counter()
    mined = db_explorer.mine_table(table, label_col, bad_values, good_values, date_col, oot_start_date, max_rows)
    yield step(
        "指标挖掘",
        t0,
        f"取样 {mined['label']['rows_used']} 行，计算 {len(mined['feature_metrics'])} 个特征的 IV / KS / PSI，"
        f"建议 {len(mined['suggested_features'])} 个候选特征。",
    )

    t0 = time.perf_counter()
    report = build_report(profile, mined)
    report += _llm_commentary(report)
    yield step("报告生成", t0, "已生成指标挖掘报告。")
    yield "result", {"profile": profile, "mining": mined, "report_markdown": report}


# ---- multi-table project mode ---------------------------------------------------------------------
_CANDIDATE_TABLES_SHOWN = 20


def build_project_report(profile: Dict[str, Any], mined: Dict[str, Any], candidates: Sequence[Dict[str, Any]]) -> str:
    """Single-table report plus feature categories, label rule / direction checks and candidate detail tables."""
    label = mined["label"]
    lines = [
        build_report(profile, mined),
        "",
        f"## 5. 特征分类汇总（{len(mined['category_summary'])} 类）",
        "",
        "| 类别 | 参与挖掘 | 入选 | 平均 IV | 入选特征（按 IV 降序，至多 5 个） |",
        "| --- | --- | --- | --- | --- |",
    ]
    for c in mined["category_summary"]:
        lines.append(
            f"| {c['category']} | {c['mined']} | {c['suggested']} | {c['avg_iv']} | {', '.join(c['top_features']) or '-'} |"
        )
    layers = mined.get("layer_summary") or []
    if len(layers) > 1:
        lines += [
            "",
            "### 特征工程层级分布（多源切片 / 时序统计 / 交叉衍生特征库）",
            "",
            "| 特征层级 | 参与挖掘 | 入选特征库 | 平均 IV | 入选代表特征（按 IV 降序） |",
            "| --- | --- | --- | --- | --- |",
        ]
        for lyr in layers:
            lines.append(
                f"| {lyr['layer_label']}（`{lyr['layer']}`） | {lyr['mined']} | {lyr['suggested']} | {lyr['avg_iv']} | "
                f"{', '.join(lyr['top_features']) or '-'} |"
            )
    lines += [
        "",
        "## 6. 标签口径与总体",
        f"- **坏样本（1）**：`{label.get('bad_rule', '-')}`（共 `{label['bad_count']}` 行）",
        f"- **好样本（0）**：`{label.get('good_rule', '-')}`（共 `{label['good_count']}` 行）",
        f"- **灰样本（自动丢弃）**：{label.get('gray_rule', '既不命中坏样本也不命中好样本')}（共丢弃 `{label['dropped_unlabeled']}` 行）",
        f"- **总体行数**：`{label['population_rows']}`，实际建模 `{label['rows_used']}`",
    ]
    for j in mined.get("joins", []):
        lines.append(f"- **关联** `{j['on']}`：匹配率 `{j['match_rate']}`")
    edges = mined.get("auto_edges") or []
    if edges:
        lines += ["", "### 自动关联的表（按列名推断，申请时点聚合）", "", "| 表 | 类别 | 关联 | 类型 | 覆盖率 | 特征数 |", "| --- | --- | --- | --- | --- | --- |"]
        for e in edges:
            kind = "一对一快照" if e["kind"] == "snapshot" else f"一对多聚合（事件时间 `{e['time_col']}`）"
            rel = (
                f"`{e['path']}` (`{e['via_bridge_col']}` = `{e['link_col']}`)"
                if e.get("hop", 1) > 1
                else f"`{e['sample_col']}` = `{e['link_col']}`"
            )
            lines.append(
                f"| {e['table']} | {e.get('category') or '-'} | {rel} | {kind} | "
                f"{e.get('coverage', '-')} | {e.get('features', '-')} |"
            )
    for s in mined.get("auto_skipped") or []:
        lines.append(f"- 跳过 `{s['table']}`：{s['reason']}")
    for chk in label.get("direction_checks", []):
        verdict = "通过" if chk["ok"] else "⚠️ 未通过"
        lines.append(
            f"- **方向校验** `{chk['col']}`（越大越{'好' if chk['higher_is'] == 'good' else '坏'}）："
            f"好样本均值 {_fmt(chk['good_mean'])} / 坏样本均值 {_fmt(chk['bad_mean'])} → {verdict}"
        )
    lines += ["", f"## 7. 候选明细表（共 {len(candidates)} 张含关联键；启用 auto_join 时可聚合表已自动纳入挖掘，其余未纳入；已按主题自动分类）"]
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for c in candidates:
        groups.setdefault(c.get("category") or "未分类", []).append(c)
    # usable categories first (more tables first), post-event categories last
    ordered = sorted(groups.items(), key=lambda kv: (all(c["post_event"] for c in kv[1]), -len(kv[1]), kv[0]))
    lines += ["", "| 类别 | 表数量 | 用途 | 代表表 |", "| --- | --- | --- | --- |"]
    for cat, items in ordered:
        post = all(c["post_event"] for c in items)
        lines.append(
            f"| {cat} | {len(items)} | {'⚠️ 仅可用于定义标签' if post else '可聚合为特征'} | "
            f"{', '.join(c['table'] for c in items[:3])} |"
        )
    for cat, items in ordered:
        post = all(c["post_event"] for c in items)
        lines += ["", f"### {cat}（{len(items)} 张）" + ("：直接作特征会数据穿越" if post else "：需启用 auto_join 或在项目配置中补充聚合方式后纳入"), ""]
        lines += [
            f"- `{c['table']}`（{c['comment'] or '无注释'}）：关联列 {', '.join(c['link_columns'])}"
            for c in items[:_CANDIDATE_TABLES_SHOWN]
        ]
    return "\n".join(lines)


def iter_project_mining(
    overrides: Optional[Dict[str, Any]] = None,
    oot_start_date: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Profile-driven mining across tables. Yields ("step", log) per stage and finally ("result", payload)."""

    def step(agent: str, t0: float, action: str) -> Tuple[str, Dict[str, Any]]:
        return "step", {"agent": agent, "action": action, "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2)}

    t0 = time.perf_counter()
    project = db_project.load_profile()
    base = project["sample"]["table"]
    yield step("项目配置", t0, f"加载配置 `{project.get('name') or base}`：样本表 `{base}`，关联 {len(project.get('joins') or [])} 组表。")

    t0 = time.perf_counter()
    candidates = db_project.discover_candidate_tables(project)
    usable = sum(1 for c in candidates if not c["post_event"])
    yield step("表发现", t0, f"发现 {len(candidates)} 张含关联键的候选明细表（{usable} 张可聚合为特征，其余为贷后表）。")

    t0 = time.perf_counter()
    plan = db_graph.plan_edges(project, candidates)
    if db_graph.enabled(project):
        n_detail = sum(1 for e in plan["edges"] if e["kind"] == "detail")
        yield step(
            "关系图谱",
            t0,
            f"按列名推断 {len(plan['edges'])} 条关联边（一对一快照 {len(plan['edges']) - n_detail} 张，一对多明细聚合 {n_detail} 张），跳过 {len(plan['skipped'])} 张。",
        )

    t0 = time.perf_counter()
    frame = db_project.build_frame(project, overrides, max_rows, plan["edges"])
    frame["auto_skipped"] = plan["skipped"]
    li = frame["label_info"]
    yield step(
        "取样与标签",
        t0,
        f"总体 {li['population_rows']} 行，标签：坏 {li['bad_count']} / 好 {li['good_count']}，丢弃不确定样本 {li['dropped_unlabeled']} 行；"
        f"配置排除 {len(frame['excluded'])} 个字段。",
    )

    t0 = time.perf_counter()
    mined = db_project.mine_project(project, frame, oot_start_date)
    yield step(
        "指标挖掘与分类",
        t0,
        f"计算 {len(mined['feature_metrics'])} 个特征的 IV / KS / PSI，分入 {len(mined['category_summary'])} 个类别，"
        f"建议 {len(mined['suggested_features'])} 个候选特征。",
    )

    t0 = time.perf_counter()
    profile = db_explorer.profile_table(base)
    report = build_project_report(profile, mined, candidates)
    report += _llm_commentary(report, index=8)
    yield step("报告生成", t0, "已生成多表指标挖掘报告。")
    yield "result", {"profile": profile, "mining": mined, "candidate_tables": candidates, "report_markdown": report}
