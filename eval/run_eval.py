"""CLI Entrypoint for Running the Credit Modeling Agent Evaluation Benchmark Suite."""

from __future__ import annotations

import json
from pathlib import Path
from rich.console import Console
from rich.table import Table

from eval.evaluators import run_benchmark_suite


def main() -> None:
    console = Console()
    console.print("\n[bold cyan]🚀 启动信贷风控多智能体系统 (Credit-Modeling MAS) 全链路自动化评测...[/bold cyan]\n")

    results = run_benchmark_suite()
    summary = results["summary"]

    table = Table(title="📊 Golden Benchmark 分层评测明细 (L1/L2/L3)", show_lines=True)
    table.add_column("Case ID", style="bold yellow")
    table.add_column("分层级别", style="cyan")
    table.add_column("任务完成", justify="center")
    table.add_column("Tool AST 准确率", justify="right")
    table.add_column("数值/口径幻觉率", justify="right")
    table.add_column("Reflection 次数", justify="center")
    table.add_column("耗时 (ms)", justify="right")
    table.add_column("Token 消耗", justify="right")

    for r in results["case_results"]:
        status_str = "[bold green]PASS[/bold green]" if r["task_completed"] else "[bold red]FAIL[/bold red]"
        table.add_row(
            r["case_id"],
            r["level"],
            status_str,
            f"{r['tool_calling_score']:.1%}",
            f"{r['hallucination_rate']:.1%}",
            str(r["reflection_count"]),
            f"{r['e2e_latency_ms']:.1f}",
            str(r["tokens_used"]),
        )

    console.print(table)

    kpi_table = Table(title="🏆 Agent Eval 核心量化指标汇总 (对齐岗位 JD 职责 3)", show_header=True)
    kpi_table.add_column("核心指标", style="bold magenta")
    kpi_table.add_column("实测结果", justify="right", style="bold green")
    kpi_table.add_column("生产门禁要求", justify="right")

    kpi_table.add_row("端到端任务完成率 (Task Completion Rate)", f"{summary['task_completion_rate']:.1%}", ">= 95.0%")
    kpi_table.add_row("工具调用 AST 准确率 (Tool Calling Accuracy)", f"{summary['tool_calling_ast_accuracy']:.1%}", ">= 98.0%")
    kpi_table.add_row("数值与口径幻觉率 (Hallucination Rate)", f"{summary['hallucination_rate']:.1%}", "== 0.0%")
    kpi_table.add_row("链路耗时 P50 / P95 (Latency)", f"{summary['latency_p50_ms']} / {summary['latency_p95_ms']} ms", "< 1000 ms")
    kpi_table.add_row("总 Token 消耗 / 预估成本", f"{summary['total_tokens']} tokens (${summary['estimated_cost_usd']})", "Tracked")

    console.print("\n")
    console.print(kpi_table)

    out_file = Path(__file__).parent / "latest_eval_report.json"
    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"\n[dim]📄 完整评测报告已落盘至: {out_file}[/dim]\n")


if __name__ == "__main__":
    main()
