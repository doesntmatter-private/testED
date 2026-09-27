/* autodiag web UI. Vanilla JS, no build step. */
(() => {
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  let snapshot = null;        // VehicleSnapshot object
  let lastChunks = [];        // chunks from the last diagnosis, keyed by handle

  // ---------------------------------------------------------------- status
  async function refreshHealth() {
    try {
      const h = await api("/api/health");
      setPill("claude", h.backends.claude.reachable, `Claude ${h.backends.claude.reachable ? "✓" : "✗"} ${h.backends.claude.model}`);
      setPill("ollama", h.backends.ollama.reachable, `Ollama ${h.backends.ollama.reachable ? "✓" : "✗"} ${h.backends.ollama.model}`);
      setPill("kb", h.knowledge_chunks > 0, `KB ${h.knowledge_chunks} chunks`);
      setPill("queue", true, `Queue ${h.queue_pending} pending`);
      if (h.knowledge_chunks === 0) showError("Knowledge base is empty. Load the seed workflows or upload manuals below.", true);
    } catch (e) {
      showError(`Server unreachable: ${e.message}`);
    }
  }
  function setPill(k, ok, text) {
    const el = $(`.pill[data-k=${k}]`);
    el.textContent = text;
    el.className = `pill ${ok ? "ok" : "bad"}`;
  }

  // ---------------------------------------------------------------- api
  async function api(path, opts = {}) {
    const r = await fetch(path, opts);
    if (!r.ok) {
      let msg = `${r.status}`;
      try { const j = await r.json(); msg = j.detail || JSON.stringify(j); } catch { msg = await r.text(); }
      throw new Error(msg);
    }
    return r.json();
  }
  function showError(msg, soft = false) {
    const el = $("#error");
    el.textContent = msg;
    el.classList.remove("hidden");
    el.style.color = soft ? "var(--warn)" : "var(--bad)";
    el.style.borderColor = soft ? "var(--warn)" : "var(--bad)";
  }
  function clearError() { $("#error").classList.add("hidden"); }
  function busy(on, text = "Working…") {
    $("#busy").classList.toggle("hidden", !on);
    $("#busyText").textContent = text;
    ["#diagnoseBtn", "#offlineBtn", "#queueBtn", "#scanBtn"].forEach((s) => ($(s).disabled = on || (!snapshot && s !== "#scanBtn")));
  }

  // ---------------------------------------------------------------- vehicle
  async function loadFixtures() {
    const fx = await api("/api/fixtures");
    const sel = $("#fixture");
    fx.forEach((f) => {
      const o = document.createElement("option");
      o.value = f.name; o.textContent = f.label; sel.appendChild(o);
    });
  }
  $("#fixture").addEventListener("change", async (e) => {
    if (!e.target.value) return;
    clearError();
    try {
      snapshot = await api("/api/scan", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ fixture: e.target.value }) });
      renderSnapshot();
    } catch (err) { showError(err.message); }
  });
  $("#snapshotFile").addEventListener("change", async (e) => {
    const f = e.target.files[0]; if (!f) return;
    clearError();
    try { snapshot = JSON.parse(await f.text()); renderSnapshot(); }
    catch (err) { showError(`Could not read snapshot: ${err.message}`); }
  });
  $("#scanBtn").addEventListener("click", async () => {
    clearError(); busy(true, "Reading adapter…");
    try {
      snapshot = await api("/api/scan", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ port: $("#port").value || null }) });
      renderSnapshot();
    } catch (err) { showError(err.message); }
    finally { busy(false); }
  });

  function renderSnapshot() {
    const v = snapshot.vehicle || {};
    const veh = [v.year, v.make, v.model, v.engine].filter(Boolean).join(" ") || "unknown vehicle";
    const pids = (list) => list.map((p) => `<span>${esc(p.name)}</span><span>${esc(p.value)} ${esc(p.unit)}</span>`).join("");
    $("#snapshotView").innerHTML = `
      <div class="block"><b>${esc(veh)}</b>
        <div class="dim">${v.mileage ? esc(v.mileage) + " mi · " : ""}${esc(snapshot.source || "")}${v.vin ? " · VIN " + esc(v.vin) : ""}</div>
        <div style="margin-top:8px">${(snapshot.dtcs || []).map((d) => `<span class="dtc" title="${esc(d.description)}">${esc(d.code)} <small>${esc(d.status)}</small></span>`).join("") || '<span class="dim">no codes</span>'}</div>
        <div class="dim" style="margin-top:6px">${(snapshot.dtcs || []).map((d) => `${esc(d.code)}: ${esc(d.description)}`).join("<br>")}</div>
      </div>
      <div class="block"><b>Freeze frame</b><div class="pidgrid">${pids(snapshot.freeze_frame || []) || '<span class="dim">none</span>'}</div></div>
      <div class="block"><b>Live</b><div class="pidgrid">${pids(snapshot.live || [])}</div></div>
      <div class="block"><b>Readiness</b><div class="pidgrid">${Object.entries(snapshot.readiness || {}).map(([k, val]) => `<span>${esc(k)}</span><span>${esc(val)}</span>`).join("") || '<span class="dim">n/a</span>'}</div></div>`;
    $("#snapshotView").classList.remove("hidden");
    busy(false);
  }

  // ---------------------------------------------------------------- diagnose
  function formData(extra = {}) {
    const fd = new FormData();
    fd.append("snapshot", JSON.stringify(snapshot));
    fd.append("symptoms", $("#symptoms").value);
    fd.append("no_vin", $("#noVin").checked);
    const a = $("#audio").files[0]; if (a) fd.append("audio", a);
    Array.from($("#images").files).forEach((f) => fd.append("images", f));
    Object.entries(extra).forEach(([k, v]) => fd.append(k, v));
    return fd;
  }
  $("#backend").addEventListener("change", (e) => $("#localModelWrap").classList.toggle("hidden", e.target.value !== "ollama"));

  $("#diagnoseBtn").addEventListener("click", () => runDiagnose(false));
  $("#offlineBtn").addEventListener("click", () => runDiagnose(true));
  async function runDiagnose(offline) {
    clearError();
    const backend = $("#backend").value;
    busy(true, offline ? "Building offline report…" : backend === "ollama" ? "Asking local model (this can take a few minutes)…" : "Asking Claude…");
    try {
      const res = await api("/api/diagnose", {
        method: "POST",
        body: formData({ offline, backend, local_model: $("#localModel").value, strict: $("#strict").checked }),
      });
      renderResult(res);
    } catch (err) { showError(err.message); }
    finally { busy(false); refreshHealth(); }
  }
  $("#queueBtn").addEventListener("click", async () => {
    clearError(); busy(true, "Queueing…");
    try {
      const r = await api("/api/queue", { method: "POST", body: formData() });
      showError(`Queued job ${r.job}. Send it later from the Queue tab.`, true);
      loadQueue();
    } catch (err) { showError(err.message); }
    finally { busy(false); refreshHealth(); }
  });

  // ---------------------------------------------------------------- render
  function renderResult(res) {
    const body = $("#resultBody");
    $("#results").classList.remove("hidden");
    lastChunks = Object.fromEntries((res.result?.chunks || res.chunks || []).map((c) => [c.handle, c]));

    if (res.kind === "offline") {
      body.innerHTML = `${res.note ? `<div class="note">${esc(res.note)}</div>` : ""}<pre class="offline">${esc(res.report)}</pre>`;
      $("#results").scrollIntoView({ behavior: "smooth", block: "start" });
      return;
    }
    const r = res.result, d = r.diagnosis;
    const sevLabel = d.severity.replace(/_/g, " ");
    const cite = (src) => (src && lastChunks[src] ? `<button class="cite" data-h="${esc(src)}" title="${esc(lastChunks[src].title)}">${esc(src)}</button>` : src ? `<span class="code">${esc(src)}</span>` : "");

    const causes = d.causes.map((c) => `
      <div class="cause">
        <div class="cause-head"><span class="rank">#${c.rank}</span><span class="title">${esc(c.title)}</span><span class="code">${Math.round(c.probability * 100)}%</span></div>
        <div class="bar"><i style="width:${Math.round(c.probability * 100)}%"></i></div>
        <div>${esc(c.reasoning)}</div>
        ${c.evidence.map((e) => `<div class="ev"><span class="kind">${esc(e.kind)}</span> ${cite(e.source)} ${esc(e.detail)}</div>`).join("")}
        ${c.contradicting_evidence.map((t) => `<div class="ev against"><span class="kind">against</span> ${esc(t)}</div>`).join("")}
      </div>`).join("");

    const tests = d.next_tests.map((t) => `
      <div class="test">
        <div class="name">${t.order}. ${esc(t.name)} <span class="meta">~${t.est_minutes} min · ${esc(t.tools_needed.join(", ") || "no special tools")} ${t.source ? "· source " : ""}${cite(t.source)}</span></div>
        <div>${esc(t.procedure)}</div>
        <div class="kv">
          <b>Splits</b><span>${esc(t.discriminates.join(", "))}</span>
          <b>If faulty</b><span>${esc(t.expected_if_faulty)}</span>
          <b>If OK</b><span>${esc(t.expected_if_ok)}</span>
        </div>
      </div>`).join("");

    const v = r.validation;
    body.innerHTML = `
      ${res.note ? `<div class="note">${esc(res.note)}</div>` : ""}
      <div class="banner ${esc(d.severity)}"><div class="sev">${esc(sevLabel)}</div><div>${esc(d.summary)}</div></div>
      <h3>Ranked causes</h3>${causes}
      <h3>Next tests, in order</h3>${tests}
      ${d.missing_information.length ? `<h3>Missing information</h3><ul>${d.missing_information.map((m) => `<li>${esc(m)}</li>`).join("")}</ul>` : ""}
      ${d.safety_notes.length ? `<h3>Safety</h3><ul>${d.safety_notes.map((m) => `<li>${esc(m)}</li>`).join("")}</ul>` : ""}
      <div class="footer-meta">backend=${esc(r.backend)} model=${esc(r.model)}${r.fallback_model ? " (fallback)" : ""} · tokens in/out ${r.usage.input_tokens ?? "?"}/${r.usage.output_tokens ?? "?"} · cache_read ${r.usage.cache_read_input_tokens ?? 0} · invalid_citations ${v.invalid_citations}${v.downgraded.length ? " · downgraded: " + esc(v.downgraded.join("; ")) : ""} · saved as ${esc(res.report_name || "")}</div>
      ${r.backend !== "claude" ? `<div class="note">Produced by a local model; treat probabilities and citations with extra care.</div>` : ""}`;

    $$(".cite", body).forEach((b) => b.addEventListener("click", () => toggleChunk(b)));
    $("#results").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function toggleChunk(btn) {
    const host = btn.closest(".ev, .test") || btn.parentElement;
    const existing = host.nextElementSibling;
    if (existing && existing.classList.contains("chunk") && existing.dataset.h === btn.dataset.h) { existing.remove(); return; }
    const c = lastChunks[btn.dataset.h]; if (!c) return;
    const node = $("#chunkTpl").content.firstElementChild.cloneNode(true);
    node.dataset.h = c.handle;
    $(".handle", node).textContent = `[${c.handle}]`;
    $(".title", node).textContent = c.title;
    $(".loc", node).textContent = [c.section ? `section "${c.section}"` : "", c.page != null ? `p.${c.page}` : ""].filter(Boolean).join(", ");
    $(".chunk-text", node).textContent = c.text;
    host.after(node);
  }

  // ---------------------------------------------------------------- tools
  $$(".tab").forEach((t) => t.addEventListener("click", () => {
    $$(".tab").forEach((x) => x.classList.toggle("active", x === t));
    $$(".tabpane").forEach((p) => p.classList.toggle("hidden", p.id !== `tab-${t.dataset.tab}`));
    if (t.dataset.tab === "queue") loadQueue();
    if (t.dataset.tab === "reports") loadReports();
    if (t.dataset.tab === "kb") loadKb();
  }));

  async function loadKb() {
    const docs = await api("/api/knowledge");
    $("#kbTable tbody").innerHTML = docs.map((d) => `<tr><td>${esc(d.title)}</td><td>${d.n_chunks}</td><td class="dim">${esc(d.path)}</td></tr>`).join("") || `<tr><td colspan="3" class="dim">empty</td></tr>`;
  }
  $("#kbUpload").addEventListener("click", async () => {
    const files = $("#kbFiles").files; if (!files.length) return;
    const fd = new FormData(); Array.from(files).forEach((f) => fd.append("files", f));
    clearError(); busy(true, "Ingesting…");
    try { await api("/api/knowledge", { method: "POST", body: fd }); await loadKb(); await refreshHealth(); }
    catch (err) { showError(err.message); } finally { busy(false); }
  });
  $("#kbSeed").addEventListener("click", async () => {
    clearError(); busy(true, "Loading seed workflows…");
    try { await api("/api/knowledge/seed", { method: "POST" }); await loadKb(); await refreshHealth(); clearError(); }
    catch (err) { showError(err.message); } finally { busy(false); }
  });
  $("#searchBtn").addEventListener("click", doSearch);
  $("#searchQ").addEventListener("keydown", (e) => { if (e.key === "Enter") doSearch(); });
  async function doSearch() {
    const qv = $("#searchQ").value.trim(); if (!qv) return;
    const res = await api(`/api/search?q_=${encodeURIComponent(qv)}`);
    $("#searchResults").innerHTML = res.map((c) => `<div class="chunk" style="margin-left:0"><div class="chunk-head"><b>[${esc(c.handle)}]</b> ${esc(c.title)} <span class="dim">${c.section ? esc(c.section) : ""} ${c.page != null ? "p." + c.page : ""} · bm25 ${c.score.toFixed(2)}</span></div><pre class="chunk-text">${esc(c.text.slice(0, 600))}</pre></div>`).join("") || `<div class="dim">no matches</div>`;
  }

  async function loadQueue() {
    const jobs = await api("/api/queue");
    $("#queueTable tbody").innerHTML = jobs.map((j) => `<tr><td class="code">${esc(j.name)}</td><td><span class="tag">${esc(j.status)}</span> ${j.error ? `<span class="dim">${esc(j.error.slice(0, 100))}</span>` : ""}</td></tr>`).join("") || `<tr><td colspan="2" class="dim">empty</td></tr>`;
  }
  $("#drainBtn").addEventListener("click", async () => {
    clearError(); busy(true, "Sending queued jobs…");
    try {
      const r = await api("/api/queue/drain", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ backend: $("#backend").value }) });
      if (r.note) showError(r.note, true);
      await loadQueue(); await refreshHealth();
    } catch (err) { showError(err.message); } finally { busy(false); }
  });

  async function loadReports() {
    const rs = await api("/api/reports");
    $("#reportsTable tbody").innerHTML = rs.map((r) => `<tr class="click" data-name="${esc(r.name)}"><td class="code">${esc(r.name.replace("diagnosis-", ""))}</td><td>${esc(r.severity.replace(/_/g, " "))}</td><td>${esc(r.top_cause)}</td><td class="dim">${esc(r.backend)} ${esc(r.model)}</td></tr>`).join("") || `<tr><td colspan="4" class="dim">none yet</td></tr>`;
    $$("#reportsTable tr.click").forEach((tr) => tr.addEventListener("click", async () => {
      const result = await api(`/api/reports/${tr.dataset.name}`);
      renderResult({ kind: "diagnosis", result, report_name: tr.dataset.name });
    }));
  }

  // ---------------------------------------------------------------- init
  refreshHealth(); loadFixtures(); loadKb();
  setInterval(refreshHealth, 30000);
})();
