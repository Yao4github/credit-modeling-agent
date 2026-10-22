"""End-to-End Interactive Demonstration of the Credit Risk Modeling Multi-Agent System."""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from app.graph.workflow import run_credit_modeling_workflow


def main() -> None:
    console = Console()
    query = "帮我针对首贷客群全流程构建一张A卡评分模型，要求 KS >= 0.35，PSI <= 0.10，目标通过率 75%，并输出合规报告"

    console.print(
        Panel(
            f"[bold white]{query}[/bold white]",
            title="👤 用户建模指令 (User Prompt)",
            border_style="cyan",
        )
    )

    state = run_credit_modeling_workflow(user_query=query, session_id="demo-a-card-001")

    # 1. Print Multi-Agent Execution Trajectory
    step_table = Table(title="🤖 6+1 多智能体协同与自愈轨迹 (LangGraph Execution Trace)", show_lines=True)
    step_table.add_column("Step", justify="center", style="bold yellow")
    step_table.add_column("Agent 角色", style="bold cyan")
    step_table.add_column("执行动作 (Action)", style="magenta")
    step_table.add_column("思考与决策摘要 (CoT & Reflection)", style="white")
    step_table.add_column("耗时(ms)", justify="right", style="green")

    for log in state.get("agent_logs", []):
        step_table.add_row(
            str(log["step_id"]),
            log["agent"],
            log["action"],
            log["thought"],
            f"{log['latency_ms']:.1f}",
        )

    console.print("\n")
    console.print(step_table)

    # 2. Print Memory Governance Summary
    mem_gov = state.get("memory_governance_stats", {})
    console.print(
        Panel(
            f"• 历史长上下文压缩率 (Token Saved Ratio): [bold green]{mem_gov.get('token_saved_ratio', 0.0):.1%}[/bold green]\n"
            f"• 归档步骤摘要: [dim]{mem_gov.get('compressed_history_summary', 'N/A')}[/dim]",
            title="🧠 长短期记忆与上下文治理 (Memory Governance)",
            border_style="magenta",
        )
    )

    # 3. Render Final Executive Markdown Report
    console.print("\n")
    console.print(
        Panel(
            Markdown(state.get("final_report_markdown", "")),
            title="📑 最终交付产物：信贷风控模型开发与合规验证报告",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
