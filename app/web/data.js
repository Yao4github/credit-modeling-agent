import { $, apiFetch, clearError, el, fillRows, setStatus, showError } from "./common.js";

const MAX_UPLOAD_BYTES = 50 * 1024 * 1024;

const ctx = { train: null };
let pending = null; // { mode: "grant" | "revoke", feature }

function busy(button, isBusy) {
  button.disabled = isBusy;
}

// ---- datasets -------------------------------------------------------------------------------------
async function loadDatasets() {
  const { datasets } = await apiFetch("/api/v1/datasets");
  $("datasets-table").tBodies[0].replaceChildren(
    ...datasets.map((d) =>
      rowWithAction(
        [d.uri, d.description, d.obs_window, d.row_count, d.bad_rate, d.column_count],
        d.removable
          ? actionButton("注销", () => removeDataset(d.uri), `注销数据集 ${d.uri}`)
          : null,
      ),
    ),
  );
  for (const id of ["train-uri", "oot-uri"]) {
    const select = $(id);
    const keep = select.value;
    select.replaceChildren(
      ...datasets.map((d) => {
        const option = el("option", d.uri);
        option.value = d.uri;
        return option;
      }),
    );
    const fallback = id === "train-uri" ? "ds://credit_a_card_train" : "ds://credit_a_card_oot";
    const wanted = datasets.some((d) => d.uri === keep) ? keep : fallback;
    if (datasets.some((d) => d.uri === wanted)) select.value = wanted;
  }
}

async function loadFiles() {
  const { files } = await apiFetch("/api/v1/datasets/files");
  $("files-table").tBodies[0].replaceChildren(
    ...files.map((f) =>
      rowWithAction(
        [f.filename, f.size_bytes, f.datasets.join(", ") || "—"],
        actionNodes(
          actionButton("注册", () => prepareRegister(f), `注册文件 ${f.filename} 为数据集`),
          actionButton("删除", () => removeFile(f), `删除文件 ${f.filename}`),
        ),
      ),
    ),
  );
}

function actionNodes(...buttons) {
  const box = el("div", undefined, "row-actions");
  box.append(...buttons);
  return box;
}

/** Prefills the register form from a listed file; label column etc. still need the user's input. */
function prepareRegister(f) {
  $("path").value = f.filename;
  $("uri").value = `ds://${f.filename.replace(/\.[^.]+$/, "")}`;
  $("register-form").scrollIntoView({ behavior: "smooth", block: "center" });
  $("target-col").focus();
  setStatus(`已填入 ${f.filename}，请补充标签列后点击“注册数据集”。`);
}

async function refreshAll() {
  await Promise.all([loadDatasets(), loadFiles()]);
}

async function removeDataset(uri) {
  if (!confirm(`确认注销数据集 ${uri}？文件会保留，可重新注册。`)) return;
  clearError();
  try {
    await apiFetch(`/api/v1/datasets?${new URLSearchParams({ uri })}`, { method: "DELETE" });
    await refreshAll();
    setStatus(`已注销 ${uri}。`);
  } catch (err) {
    showError(err.message);
  }
}

async function removeFile(f) {
  const linked = f.datasets.length ? `\n将同时注销数据集：${f.datasets.join(", ")}` : "";
  if (!confirm(`确认删除文件 ${f.filename}？此操作不可恢复。${linked}`)) return;
  clearError();
  try {
    const r = await apiFetch(`/api/v1/datasets/files/${encodeURIComponent(f.filename)}`, { method: "DELETE" });
    await refreshAll();
    setStatus(`已删除 ${r.filename}${r.unregistered.length ? `，并注销 ${r.unregistered.join(", ")}` : ""}。`);
  } catch (err) {
    showError(err.message);
  }
}

$("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = $("file").files[0];
  if (!file) return;
  clearError();
  if (file.size > MAX_UPLOAD_BYTES) {
    showError(`文件超过 ${MAX_UPLOAD_BYTES / (1024 * 1024)} MB 上限。`);
    return;
  }
  busy($("upload-btn"), true);
  try {
    const r = await apiFetch(`/api/v1/datasets/files/${encodeURIComponent(file.name)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/octet-stream" },
      body: file,
    });
    $("path").value = r.filename;
    setStatus(`已上传 ${r.filename}（${r.size_bytes} 字节），请在下方注册为数据集。`);
    await loadFiles();
  } catch (err) {
    showError(err.message);
    setStatus("上传失败。");
  } finally {
    busy($("upload-btn"), false);
  }
});

$("register-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  clearError();
  const fd = new FormData($("register-form"));
  const body = Object.fromEntries(fd.entries());
  body.perf_window_months = Number(body.perf_window_months);
  for (const key of ["obs_start_date", "obs_end_date", "date_col"]) if (!body[key]) delete body[key];
  busy($("register-btn"), true);
  try {
    const r = await apiFetch("/api/v1/datasets/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    setStatus(`已注册 ${r.uri}（${r.row_count} 行，坏账率 ${r.bad_rate}）。`);
    await refreshAll();
  } catch (err) {
    showError(err.message);
    setStatus("注册失败。");
  } finally {
    busy($("register-btn"), false);
  }
});

// ---- audit / exemptions ---------------------------------------------------------------------------
function actionButton(label, handler, ariaLabel) {
  const button = el("button", label, "secondary small");
  button.type = "button";
  button.setAttribute("aria-label", ariaLabel);
  button.addEventListener("click", handler);
  return button;
}

function rowWithAction(cells, actionNode) {
  const tr = document.createElement("tr");
  cells.forEach((c) => tr.append(el("td", String(c))));
  const td = document.createElement("td");
  if (actionNode) td.append(actionNode);
  else td.textContent = "—";
  tr.append(td);
  return tr;
}

function renderAudit(result, exemptionState) {
  const gov = result.governance_audit;
  const win = result.window_validation;
  const samp = result.sample_summary;
  $("audit-summary").textContent =
    `开发集 ${samp.train_rows} 行（坏账率 ${samp.train_bad_rate}），跨期集 ${samp.oot_rows} 行（坏账率 ${samp.oot_bad_rate}）；` +
    `时间窗口${win.is_chronological_valid === null ? "未提供（已跳过校验）" : win.is_chronological_valid ? "无重叠" : "存在重叠"}；候选特征 ${gov.clean_candidate_features.length} 个。`;

  const exemptable = new Set(gov.exemptable_features);
  const rows = [
    ...gov.quarantined_leakage_features.map((f) => ["穿越", f]),
    ...gov.quarantined_compliance_features.map((f) => ["合规敏感", f]),
  ].map(([kind, feature]) => {
    const action = exemptable.has(feature)
      ? actionButton("申请豁免", () => openDialog("grant", feature), `对字段 ${feature} 申请豁免`)
      : null;
    return rowWithAction([feature, kind, gov.detection_reasons[feature] || "-"], action);
  });
  $("quarantine-table").tBodies[0].replaceChildren(...rows);

  $("exempt-table").tBodies[0].replaceChildren(
    ...exemptionState.active.map((rec) =>
      rowWithAction(
        [rec.feature, rec.original_reason, rec.reason, rec.operator, rec.created_at],
        actionButton("撤销", () => openDialog("revoke", rec.feature), `撤销字段 ${rec.feature} 的豁免`),
      ),
    ),
  );
  $("ignored-list").replaceChildren(
    ...gov.ignored_exemptions.map((i) => el("li", `失效豁免：${i.feature}：${i.why}`)),
  );
  fillRows(
    $("history-table").tBodies[0],
    exemptionState.history.map((h) => [h.at, h.event === "grant" ? "授予" : "撤销", h.feature, h.operator, h.reason]),
  );
  $("audit-results").hidden = false;
}

async function runInspect() {
  const fd = new FormData($("inspect-form"));
  ctx.train = fd.get("train_uri");
  const [result, exemptionState] = await Promise.all([
    apiFetch("/api/v1/datasets/inspect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        train_uri: fd.get("train_uri"),
        oot_uri: fd.get("oot_uri"),
        card_type: fd.get("card_type"),
      }),
    }),
    apiFetch(`/api/v1/datasets/exemptions?uri=${encodeURIComponent(fd.get("train_uri"))}`),
  ]);
  renderAudit(result, exemptionState);
}

$("inspect-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  clearError();
  busy($("inspect-btn"), true);
  try {
    await runInspect();
    setStatus("探查完成。");
    $("audit-results").focus();
  } catch (err) {
    showError(err.message);
    setStatus("探查失败。");
  } finally {
    busy($("inspect-btn"), false);
  }
});

// ---- exemption dialog (native <dialog>: focus handling and inert background come for free) ---------
const dialog = $("exempt-dialog");

function openDialog(mode, feature) {
  pending = { mode, feature };
  $("dialog-title").textContent = mode === "grant" ? `申请豁免：${feature}` : `撤销豁免：${feature}`;
  $("dialog-desc").textContent =
    mode === "grant"
      ? "豁免后该字段会进入候选特征池（仍需通过 IV / PSI / VIF 与跨期门禁），并在报告中醒目标注。"
      : "撤销后该字段将重新被自动隔离。";
  $("dialog-error").hidden = true;
  $("exempt-form").reset();
  dialog.showModal();
}

$("dialog-cancel").addEventListener("click", () => dialog.close());

$("exempt-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!pending) return;
  const fd = new FormData($("exempt-form"));
  const reason = String(fd.get("reason")).trim();
  const operator = String(fd.get("operator")).trim();
  const { mode, feature } = pending;
  busy($("dialog-submit"), true);
  try {
    if (mode === "grant") {
      await apiFetch("/api/v1/datasets/exemptions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ uri: ctx.train, feature, reason, operator }),
      });
    } else {
      const params = new URLSearchParams({ uri: ctx.train, feature, reason, operator });
      await apiFetch(`/api/v1/datasets/exemptions?${params}`, { method: "DELETE" });
    }
    dialog.close();
    await runInspect();
    setStatus(mode === "grant" ? `已豁免字段 ${feature}。` : `已撤销字段 ${feature} 的豁免。`);
    $("audit-results").focus();
  } catch (err) {
    $("dialog-error").textContent = err.message;
    $("dialog-error").hidden = false;
  } finally {
    busy($("dialog-submit"), false);
  }
});

refreshAll().catch((err) => showError(err.message));
