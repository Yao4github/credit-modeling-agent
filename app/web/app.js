import { $, apiFetch, el, escapeHtml, fillRows, setStatus, showError } from "./common.js";

const NODE_LABELS = {
  orchestrator: "编排规划",
  data_explorer: "数据探查",
  feature_engineer: "特征工程",
  model_trainer: "算法训练",
  validator: "跨期验证",
  compliance_reporter: "合规报告",
};
const CARD_LABELS = { A: "A 卡（申请）", B: "B 卡（行为）", C: "C 卡（催收）" };
const INTENT_LABELS = {
  full_modeling: "全流程建模",
  data_audit_only: "数据探查",
  feature_selection_only: "特征筛选",
  knowledge_qa: "知识问答",
};

const form = $("run-form");
const submitBtn = $("submit-btn");

// ---- SVG bar chart (accessible: role=img + title/desc, with a data table next to it) -------------
function barChart(container, { title, description, items, max, alt = [] }) {
  const ns = "http://www.w3.org/2000/svg";
  const width = 640;
  const height = 230;
  const pad = { left: 16, right: 16, top: 24, bottom: 56 };
  const plotH = height - pad.top - pad.bottom;
  const slot = (width - pad.left - pad.right) / items.length;
  const barW = Math.min(64, slot * 0.6);

  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("class", "chart");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-labelledby", `${container.id}-t ${container.id}-d`);

  const t = document.createElementNS(ns, "title");
  t.id = `${container.id}-t`;
  t.textContent = title;
  const d = document.createElementNS(ns, "desc");
  d.id = `${container.id}-d`;
  d.textContent = description;
  svg.append(t, d);

  const axis = document.createElementNS(ns, "line");
  axis.setAttribute("class", "axis");
  axis.setAttribute("x1", pad.left);
  axis.setAttribute("x2", width - pad.right);
  axis.setAttribute("y1", pad.top + plotH);
  axis.setAttribute("y2", pad.top + plotH);
  svg.append(axis);

  items.forEach((item, i) => {
    const h = max > 0 ? Math.max(1, (item.value / max) * plotH) : 1;
    const x = pad.left + slot * i + (slot - barW) / 2;
    const rect = document.createElementNS(ns, "rect");
    rect.setAttribute("class", alt.includes(i) ? "bar alt" : "bar");
    rect.setAttribute("x", x);
    rect.setAttribute("y", pad.top + plotH - h);
    rect.setAttribute("width", barW);
    rect.setAttribute("height", h);

    const val = document.createElementNS(ns, "text");
    val.setAttribute("x", x + barW / 2);
    val.setAttribute("y", pad.top + plotH - h - 6);
    val.setAttribute("text-anchor", "middle");
    val.textContent = String(item.value);

    const lab = document.createElementNS(ns, "text");
    lab.setAttribute("x", x + barW / 2);
    lab.setAttribute("y", pad.top + plotH + 18);
    lab.setAttribute("text-anchor", "middle");
    lab.textContent = item.label;

    svg.append(rect, val, lab);
  });

  container.replaceChildren(svg);
}

// ---- Minimal Markdown renderer. All text is HTML-escaped BEFORE any markup is added. -------------
function inline(escaped) {
  return escaped.replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
}

function renderMarkdown(md) {
  const lines = md.split(/\r?\n/);
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const line = escapeHtml(lines[i]);
    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    if (heading) {
      out.push(`<h${heading[1].length}>${inline(heading[2])}</h${heading[1].length}>`);
      i += 1;
    } else if (line.startsWith("|")) {
      const rows = [];
      while (i < lines.length && lines[i].startsWith("|")) {
        rows.push(escapeHtml(lines[i]).replace(/^\||\|$/g, "").split("|").map((c) => c.trim()));
        i += 1;
      }
      const body = rows.filter((r) => !r.every((c) => /^:?-+:?$/.test(c)));
      const [head, ...rest] = body;
      out.push(
        `<div class="table-wrap"><table><thead><tr>${head.map((c) => `<th scope="col">${inline(c)}</th>`).join("")}</tr></thead><tbody>${rest
          .map((r) => `<tr>${r.map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`)
          .join("")}</tbody></table></div>`,
      );
    } else if (/^\s*-\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*-\s+/.test(lines[i])) {
        items.push(`<li>${inline(escapeHtml(lines[i]).replace(/^\s*-\s+/, ""))}</li>`);
        i += 1;
      }
      out.push(`<ul>${items.join("")}</ul>`);
    } else if (line.trim() === "") {
      i += 1;
    } else {
      out.push(`<p>${inline(line)}</p>`);
      i += 1;
    }
  }
  return out.join("\n");
}

// ---- Rendering of stream events -------------------------------------------------------------------
const nodeRounds = {};

function addTimelineItem(evt) {
  nodeRounds[evt.node] = (nodeRounds[evt.node] || 0) + 1;
  const li = document.createElement("li");
  const head = el("div");
  head.append(el("span", NODE_LABELS[evt.node] || evt.agent, "agent"));
  if (nodeRounds[evt.node] > 1) head.append(el("span", `第 ${nodeRounds[evt.node]} 轮（自愈回滚）`, "round"));
  head.append(el("span", `${evt.latency_ms} ms`, "latency"));
  li.append(head, el("p", evt.thought, "thought"));
  $("timeline").append(li);
  setStatus(`已完成：${NODE_LABELS[evt.node] || evt.agent}`);
}

function metricItem(label, value, cls) {
  const wrap = el("div");
  wrap.append(el("dt", label), el("dd", value, cls));
  return wrap;
}

function renderResult(r) {
  const m = r.validation_metrics || {};
  const items = [
    metricItem("卡种", CARD_LABELS[r.card_type] || "-"),
    metricItem("任务类型", INTENT_LABELS[r.intent_type] || r.intent_type || "-"),
  ];
  if (r.validation_passed !== null && r.validation_passed !== undefined) {
    items.push(
      metricItem("跨期门禁", r.validation_passed ? "✔ 通过" : "✖ 未通过", r.validation_passed ? "ok" : "bad"),
      metricItem("OOT KS", m.oot_ks ?? "-"),
      metricItem("OOT AUC", m.oot_auc ?? "-"),
      metricItem("Score PSI", m.score_psi ?? "-"),
      metricItem("自愈回滚次数", r.reflection_count),
    );
  }
  items.push(metricItem("合规终审", r.compliance_passed ? "✔ 通过" : "✖ 未通过", r.compliance_passed ? "ok" : "bad"));
  $("summary").replaceChildren(...items);

  const refl = r.reflection_history || [];
  $("reflection-section").hidden = refl.length === 0;
  $("reflection-list").replaceChildren(...refl.map((t) => el("li", t)));

  const feats = r.selected_feature_metrics || [];
  $("features-section").hidden = feats.length === 0;
  fillRows($("features-table").tBodies[0], feats.map((f) => [f.feature, f.iv, f.psi, f.vif]));

  const bands = r.oot_score_bands || [];
  $("bands-section").hidden = bands.length === 0;
  if (bands.length) {
    barChart($("bands-chart"), {
      title: "OOT 评分分箱坏账率",
      description: bands.map((b) => `${b.score_range} 坏账率 ${b.bad_rate}`).join("；"),
      items: bands.map((b) => ({ label: b.score_range, value: b.bad_rate })),
      max: Math.max(...bands.map((b) => b.bad_rate)),
    });
    fillRows($("bands-table").tBodies[0], bands.map((b) => [b.score_range, b.count, b.bad_rate]));
    const c = r.cutoff_strategy || {};
    $("cutoff-note").textContent =
      `在目标比例 ${c.target_approval_rate} 下，建议切点分为 ${c.recommended_cutoff_score}，` +
      `切点后坏账率 ${c.post_cutoff_bad_rate}（基准 ${c.baseline_oot_bad_rate}）。`;
  }

  const ch = r.challenger_result;
  const hasChallenger = ch && ch.challenger_metrics;
  $("challenger-section").hidden = !hasChallenger;
  if (hasChallenger) {
    const cm = ch.champion_metrics;
    const gm = ch.challenger_metrics;
    barChart($("challenger-chart"), {
      title: "冠军与挑战者 OOT 区分度",
      description: `冠军 KS ${cm.oot_ks}、AUC ${cm.oot_auc}；挑战者 KS ${gm.oot_ks}、AUC ${gm.oot_auc}`,
      items: [
        { label: "冠军 KS", value: cm.oot_ks },
        { label: "挑战者 KS", value: gm.oot_ks },
        { label: "冠军 AUC", value: cm.oot_auc },
        { label: "挑战者 AUC", value: gm.oot_auc },
      ],
      max: 1,
      alt: [1, 3],
    });
    fillRows($("challenger-table").tBodies[0], [
      ["冠军 (LR 评分卡)", cm.oot_ks, cm.oot_auc],
      [`挑战者 (${ch.challenger_model})`, gm.oot_ks, gm.oot_auc],
    ]);
    $("challenger-verdict").textContent = ch.challenger_significantly_better
      ? `挑战者 OOT KS 领先 ${ch.ks_gap}，达到显著领先阈值，需评估可解释性后再决定是否引入。`
      : `OOT KS 差值 ${ch.ks_gap}，未显著领先，保持 LR 评分卡作为上线模型。`;
  }

  $("report").innerHTML = renderMarkdown(r.final_report_markdown || "");
  $("results").hidden = false;
}

// ---- SSE over fetch (the endpoint is POST, so EventSource cannot be used) -------------------------
function handleSseBlock(block, ctx) {
  let event = "message";
  const data = [];
  for (const line of block.split(/\r?\n/)) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  }
  if (!data.length) return;
  const payload = JSON.parse(data.join("\n"));
  if (event === "result") {
    ctx.finished = true;
    renderResult(payload);
    setStatus("建模完成。");
    $("results").focus();
  } else if (event === "error") {
    ctx.finished = true;
    showError(`建模失败：${payload.message}`);
    setStatus("建模失败。");
  } else {
    addTimelineItem(payload);
  }
}

async function runModeling(body) {
  const ctx = { finished: false };
  const resp = await fetch("/api/v1/modeling/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok || !resp.body) {
    const detail = await resp.text().catch(() => "");
    throw new Error(`请求失败 (HTTP ${resp.status}) ${detail.slice(0, 200)}`);
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop();
    blocks.forEach((b) => handleSseBlock(b, ctx));
  }
  if (buffer.trim()) handleSseBlock(buffer, ctx);
  if (!ctx.finished) throw new Error("连接中断：未收到最终结果。");
}

async function loadDatasetOptions() {
  const { datasets } = await apiFetch("/api/v1/datasets");
  const options = datasets.filter((d) => !d.uri.includes("#"));
  for (const id of ["train-uri", "oot-uri"]) {
    const select = $(id);
    const first = el("option", id === "train-uri" ? "默认（所选卡种的合成样本）" : "默认（合成样本 / 仅选开发集时按行切分）");
    first.value = "";
    select.replaceChildren(
      first,
      ...options.map((d) => {
        const option = el("option", d.uri);
        option.value = d.uri;
        return option;
      }),
    );
  }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(form);
  const body = {
    query: String(fd.get("query")).trim(),
    session_id: `ui-${Date.now()}`,
    max_reflections: Number(fd.get("max_reflections")),
  };
  const card = fd.get("card_type");
  if (card) body.card_type = card;
  const trainUri = fd.get("train_uri");
  const ootUri = fd.get("oot_uri");
  if (ootUri && !trainUri) {
    showError("只选了跨期集：请同时选择开发集。");
    return;
  }
  if (trainUri) body.train_uri = trainUri;
  if (ootUri) body.oot_uri = ootUri;

  submitBtn.disabled = true;
  $("error").hidden = true;
  $("results").hidden = true;
  $("timeline").replaceChildren();
  $("timeline-section").hidden = false;
  Object.keys(nodeRounds).forEach((k) => delete nodeRounds[k]);
  setStatus("建模进行中…");
  try {
    await runModeling(body);
  } catch (err) {
    showError(err.message);
    setStatus("建模失败。");
  } finally {
    submitBtn.disabled = false;
  }
});

// ---- Replay of a stored session (index.html?session=<id>) ------------------------------------------
function addLogItem(log, rounds) {
  rounds[log.agent] = (rounds[log.agent] || 0) + 1;
  const li = document.createElement("li");
  const head = el("div");
  head.append(el("span", log.agent, "agent"));
  if (rounds[log.agent] > 1) head.append(el("span", `第 ${rounds[log.agent]} 轮（自愈回滚）`, "round"));
  head.append(el("span", `${log.latency_ms} ms`, "latency"));
  li.append(head, el("p", log.thought, "thought"));
  $("timeline").append(li);
}

async function loadStoredSession(id) {
  setStatus("正在加载历史会话…");
  try {
    const stored = await apiFetch(`/api/v1/sessions/${encodeURIComponent(id)}`);
    $("query").value = stored.query;
    $("timeline").replaceChildren();
    const rounds = {};
    (stored.result.agent_logs || []).forEach((log) => addLogItem(log, rounds));
    $("timeline-section").hidden = false;
    renderResult(stored.result);
    setStatus(`已加载历史会话 ${stored.session_id}（保存于 ${stored.saved_at}）。`);
  } catch (err) {
    showError(err.message);
    setStatus("加载历史会话失败。");
  }
}

const requestedSession = new URLSearchParams(location.search).get("session");
if (requestedSession) loadStoredSession(requestedSession);
loadDatasetOptions().catch((err) => showError(err.message));
