const COLORS = {
  person: "#3ee0a0",
  "cardboard box": "#f5a15a",
  "plastic tote": "#62b6ff",
  cart: "#e98cff",
};
const INCIDENTS = {
  person: { title: "Person density", x: 18, y: 14, hx: 12, hy: 18 },
  "cardboard box": { title: "Parcel surge", x: 84, y: 18, hx: 78, hy: 22 },
  "plastic tote": { title: "Tote cluster", x: 15, y: 50, hx: 14, hy: 58 },
  cart: { title: "Cart congestion", x: 83, y: 52, hx: 76, hy: 64 },
};
const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const mapCanvas = document.getElementById("map-canvas");
const mapCtx = mapCanvas.getContext("2d");
const logEl = document.getElementById("log");
const scrub = document.getElementById("scrub");
const playBtn = document.getElementById("play");

let jobId = null;
let startedAt = 0;
let floatKey = "";
let pinKey = "";
let healthKey = "";
let beamKey = "";
let annotations = { frames: [], duration: 0, stats: {} };
let filter = "all";
let annotated = true;
let pollTimer = null;
let query = "";
const trails = [];

const STARTER = `<div class="action"><span>Waiting for footage</span><em>Standby</em></div>`;

function clsMatch(det) {
  if (query) {
    const q = query.toLowerCase();
    const hay = `${det.class} ${det.track_id || ""}`.toLowerCase();
    if (!hay.includes(q)) return false;
  }
  if (filter === "all") return true;
  if (filter === "parcel") return det.class === "cardboard box" || det.class === "plastic tote";
  return det.class === filter;
}

function renderLog(items) {
  if (!items || !items.length) {
    logEl.innerHTML = STARTER;
    return;
  }
  logEl.innerHTML = items
    .slice(-8)
    .map((item) => {
      const dark = item.kind === "tool" || item.kind === "note" ? " dark" : "";
      return `<div class="action${dark}"><span>${item.text}</span><em>${item.kind || "note"}</em></div>`;
    })
    .join("");
  logEl.scrollTop = logEl.scrollHeight;
}

function setStats(stats) {
  if (!stats || !Object.keys(stats).length) return;
  const all = stats.all || 0;
  const people = stats.employees || 0;
  const parcels = stats.parcels || 0;
  const carts = stats.carts || 0;
  const frames = (annotations.frames || []).length;
  const classes = stats.classes || {};
  const coverage = all ? Math.min(99.9, 70 + Math.min(29, all)) : 0;
  const unlabeled = all ? Math.max(0, Math.round((1 - Math.min(1, all / Math.max(all, 12))) * 1000) / 10) : 0;

  document.getElementById("stat-all").textContent = all;
  document.getElementById("kpi-tracks").textContent = all;
  document.getElementById("ind-tracks").textContent = all;
  document.getElementById("stat-carts").textContent = carts;
  document.getElementById("ind-carts").textContent = carts;
  document.getElementById("stat-parcels").textContent = parcels;
  document.getElementById("ind-parcels").textContent = parcels;
  document.getElementById("stat-employees").textContent = people;
  document.getElementById("ind-people").textContent = people;
  document.getElementById("kpi-frames").textContent = frames;
  document.getElementById("stat-coverage").textContent = `${coverage.toFixed(1)}%`;
  document.getElementById("kpi-sla").textContent = `${coverage.toFixed(0)}%`;
  document.getElementById("kpi-loss").textContent = `${unlabeled}%`;
  const summary = `${people} people, ${parcels} parcels, ${carts} carts on this floor. Grok is holding ${all} tracks across ${frames || 0} sampled frames.`;
  document.getElementById("incident-copy").textContent = summary;
  document.getElementById("health-copy").textContent = summary;
  const setDelta = (id, text, red) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = text;
    el.classList.toggle("red", !!red);
  };
  setDelta("delta-tracks", all ? `↑ ${all} live` : "vs last sample");
  setDelta("delta-people", people ? `↑ ${people} vs idle` : "none", people > 0);
  setDelta("delta-parcels", parcels ? `${parcels} in flow` : "flow");
  setDelta("delta-carts", carts ? `${carts} moving` : "transport");
  document.getElementById("spark-tracks")?.style.setProperty("--p", `${Math.min(100, all * 8)}%`);
  document.getElementById("spark-people")?.style.setProperty("--p", `${Math.min(100, people * 12)}%`);

  const confs = [];
  (annotations.frames || []).forEach((f) =>
    (f.detections || []).forEach((d) => {
      if (typeof d.confidence === "number") confs.push(d.confidence);
    })
  );
  if (confs.length) {
    const avg = confs.reduce((a, b) => a + b, 0) / confs.length;
    document.getElementById("kpi-conf").textContent = `${Math.round(avg * 100)}%`;
  }

  const rows = Object.entries(classes)
    .map(([name, n]) => {
      const cls =
        name === "person" ? "c-person" : name === "cart" ? "c-cart" : name === "plastic tote" ? "c-tote" : "c-box";
      return `<div><i class="${cls}"></i> ${name} <b>${n}</b></div>`;
    })
    .join("");
  document.getElementById("legend").innerHTML =
    `<div class="legend-h">Classes in this view <span>${Object.keys(classes).length}</span></div>` + rows;

  const total = Math.max(1, people + parcels + carts);
  const comps = [
    { name: "People", n: people, id: "person" },
    { name: "Cardboard boxes", n: classes["cardboard box"] || Math.max(0, parcels - (classes["plastic tote"] || 0)), id: "box" },
    { name: "Plastic totes", n: classes["plastic tote"] || 0, id: "tote" },
    { name: "Carts", n: carts, id: "cart" },
  ];
  document.getElementById("components").innerHTML = comps
    .map((c) => {
      const pct = Math.round((c.n / total) * 1000) / 10;
      const bad = c.n === 0;
      return `<div class="comp${bad ? " bad" : ""}"><small>${c.name}</small><strong>${pct}%</strong><div class="bar"><i style="width:${Math.max(4, pct)}%"></i></div></div>`;
    })
    .join("");
}

function lerp(a, b, t) {
  return {
    ...a,
    x: a.x + (b.x - a.x) * t,
    y: a.y + (b.y - a.y) * t,
    w: a.w + (b.w - a.w) * t,
    h: a.h + (b.h - a.h) * t,
    source: "interpolated",
  };
}

function detsAt(t) {
  const frames = annotations.frames || [];
  if (!frames.length) return [];
  if (t <= frames[0].t) return frames[0].detections || [];
  if (t >= frames[frames.length - 1].t) return frames[frames.length - 1].detections || [];
  let lo = 0;
  for (let i = 0; i < frames.length; i++) if (frames[i].t <= t) lo = i;
  const a = frames[lo];
  const b = frames[Math.min(lo + 1, frames.length - 1)];
  const useInterp = document.getElementById("interp-on").checked;
  if (!useInterp || a === b) return a.detections || [];
  const alpha = (t - a.t) / ((b.t - a.t) || 1);
  const mapB = Object.fromEntries((b.detections || []).map((d) => [d.track_id, d]));
  return (a.detections || []).map((da) => {
    const db = mapB[da.track_id];
    return db ? lerp(da, db, alpha) : da;
  });
}

function draw() {
  const w = (canvas.width = video.clientWidth || canvas.clientWidth);
  const h = (canvas.height = video.clientHeight || canvas.clientHeight);
  ctx.clearRect(0, 0, w, h);
  const t = video.currentTime || 0;
  const dur = video.duration || annotations.duration || 0;
  document.getElementById("clock").textContent = `${fmt(t)} / ${fmt(dur)}`;
  if (!annotations.frames.length) return;
  const dets = detsAt(t).filter(clsMatch);
  const groups = annotated ? summarize(dets) : [];
  renderFloats(groups);
  renderBeams(groups);
  renderMapPins(groups);
  renderHealthFloats(groups);
  if (!annotated || !document.getElementById("overlay-on").checked) return;
  const showLabels = document.getElementById("labels-on").checked;
  const showTrails = document.getElementById("trails-on").checked;
  if (showTrails) {
    trails.push(dets.map((d) => ({ ...d })));
    if (trails.length > 18) trails.shift();
    trails.forEach((snap, i) => {
      ctx.globalAlpha = (i + 1) / trails.length / 4;
      snap.filter(clsMatch).forEach((d) => box(d, w, h, false));
    });
    ctx.globalAlpha = 1;
  }
  dets.forEach((d) => box(d, w, h, showLabels));
}

function box(d, w, h, labels) {
  const color = COLORS[d.class] || "#fff";
  const x = d.x * w;
  const y = d.y * h;
  const bw = d.w * w;
  const bh = d.h * h;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.strokeRect(x, y, bw, bh);
  if (labels) {
    const tag = d.track_id || d.class;
    ctx.font = "600 10px Inter, sans-serif";
    const tw = ctx.measureText(tag).width + 8;
    ctx.fillStyle = color;
    ctx.fillRect(x, Math.max(0, y - 14), tw, 14);
    ctx.fillStyle = "#111";
    ctx.fillText(tag, x + 4, Math.max(10, y - 4));
  }
}

function summarize(dets) {
  const out = {};
  dets.forEach((d) => {
    const key = d.class;
    if (!out[key]) out[key] = { class: key, n: 0, conf: 0, x: 0, y: 0 };
    out[key].n += 1;
    out[key].conf += d.confidence || 0;
    out[key].x += d.x + (d.w || 0) / 2;
    out[key].y += d.y + (d.h || 0) / 2;
  });
  return Object.values(out)
    .map((g) => ({
      ...g,
      conf: g.n ? g.conf / g.n : 0,
      x: g.x / g.n,
      y: g.y / g.n,
    }))
    .sort((a, b) => b.n - a.n);
}

function groupKey(groups) {
  return groups.map((g) => `${g.class}:${g.n}`).join("|") + `:${filter}:${query}`;
}

function renderFloats(groups) {
  const key = "f" + groupKey(groups);
  if (key === floatKey) return;
  floatKey = key;
  document.getElementById("floats").innerHTML = groups
    .map((g) => {
      const meta = INCIDENTS[g.class] || { title: g.class, x: 50, y: 30 };
      const conf = Math.round(g.conf * 100);
      return `<div class="float" style="left:${meta.x}%;top:${meta.y}%;transform:translate(-50%,-20%)"><b><i></i> ${meta.title}</b><span>${g.n} track${g.n === 1 ? "" : "s"} · ${conf || "—"}%</span><em>${g.class}</em></div>`;
    })
    .join("");
}

function renderBeams(groups) {
  const key = "b" + groupKey(groups);
  if (key === beamKey) return;
  beamKey = key;
  const lines = groups
    .map((g) => {
      const meta = INCIDENTS[g.class] || { x: 50, y: 30 };
      return `<line x1="52" y1="34" x2="${meta.x}" y2="${meta.y}" stroke="url(#beam)" stroke-width="1.4"/>`;
    })
    .join("");
  document.getElementById("beams").innerHTML =
    `<defs><linearGradient id="beam" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="rgba(226,59,59,.85)"/><stop offset="100%" stop-color="rgba(226,59,59,0)"/></linearGradient></defs>` +
    `<circle cx="52" cy="34" r="2.2" fill="#e23b3b"/><circle cx="52" cy="34" r="6" fill="rgba(226,59,59,.18)"/>` +
    lines;
}

function renderMapPins(groups) {
  const key = "p" + groupKey(groups);
  if (key === pinKey) return;
  pinKey = key;
  const w = (mapCanvas.width = mapCanvas.clientWidth || 0);
  const h = (mapCanvas.height = mapCanvas.clientHeight || 0);
  mapCtx.clearRect(0, 0, w, h);
  groups.forEach((g) => {
    const x = g.x * w;
    const y = g.y * h;
    const rad = mapCtx.createRadialGradient(x, y, 4, x, y, 56);
    rad.addColorStop(0, "rgba(226,59,59,.8)");
    rad.addColorStop(1, "rgba(226,59,59,0)");
    mapCtx.fillStyle = rad;
    mapCtx.beginPath();
    mapCtx.arc(x, y, 56, 0, Math.PI * 2);
    mapCtx.fill();
  });
  document.getElementById("map-pins").innerHTML = groups
    .map((g) => {
      const meta = INCIDENTS[g.class] || { title: g.class };
      const left = Math.min(88, Math.max(12, g.x * 100));
      const top = Math.min(82, Math.max(14, g.y * 100));
      return `<div class="pin" style="left:${left}%;top:${top}%"><i></i><div class="pin-card"><b>${meta.title}</b><span>${g.n} track${g.n === 1 ? "" : "s"} · ${Math.round(g.conf * 100) || "—"}%</span></div></div>`;
    })
    .join("");
}

function renderHealthFloats(groups) {
  const key = "h" + groupKey(groups);
  if (key === healthKey) return;
  healthKey = key;
  const el = document.getElementById("health-floats");
  if (!el) return;
  el.innerHTML = groups
    .map((g) => {
      const meta = INCIDENTS[g.class] || { title: g.class, hx: 50, hy: 30 };
      return `<div class="float" style="left:${meta.hx}%;top:${meta.hy}%;transform:translate(-50%,0)"><b><i></i> ${meta.title}</b><span>${g.n} track${g.n === 1 ? "" : "s"}</span><em>${Math.round(g.conf * 100) || "—"}% conf</em></div>`;
    })
    .join("");
}

function fmt(s) {
  if (!isFinite(s)) s = 0;
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return `${String(m).padStart(2, "0")}:${sec.toFixed(2).padStart(5, "0")}`;
}

function setStatus(job) {
  const chip = document.getElementById("chip-status");
  const level = document.getElementById("chip-level");
  const status = job?.status || "idle";
  if (status === "done") {
    chip.innerHTML = `<b class="ok"></b><span>Ready<small>Camera status</small></span>`;
    level.innerHTML = `<b class="ok"></b><span>Live<small>Task priority</small></span>`;
  } else if (status === "error") {
    chip.innerHTML = `<b class="bad"></b><span>Degraded<small>Camera status</small></span>`;
    level.innerHTML = `<b class="bad"></b><span>Critical<small>Task priority</small></span>`;
  } else if (job) {
    chip.innerHTML = `<b class="warn"></b><span>Labeling<small>Camera status</small></span>`;
    level.innerHTML = `<b class="warn"></b><span>Medium<small>Task priority</small></span>`;
  }
}

async function refreshJob() {
  if (!jobId) return;
  const job = await (await fetch(`/api/jobs/${jobId}`)).json();
  renderLog(job.log);
  setStats(job.stats || {});
  setStatus(job);
  document.getElementById("hint").textContent = `${job.status} · ${Math.round((job.progress || 0) * 100)}%`;
  if (startedAt) document.getElementById("kpi-time").textContent = `${Math.round((Date.now() - startedAt) / 1000)}s`;
  if (job.status === "done" && !annotations.frames.length) {
    annotations = await (await fetch(`/api/jobs/${jobId}/annotations`)).json();
    setStats(annotations.stats || job.stats || {});
    draw();
  }
  if (job.status === "done" || job.status === "error") clearInterval(pollTimer);
}

async function start(file) {
  if (!file) return;
  const body = new FormData();
  body.append("file", file);
  const res = await fetch("/api/jobs", { method: "POST", body });
  const data = await res.json();
  if (!res.ok) {
    document.getElementById("hint").textContent = data.detail || "Failed to start";
    return;
  }
  jobId = data.job_id;
  startedAt = Date.now();
  annotations = { frames: [], duration: 0, stats: {} };
  trails.length = 0;
  floatKey = pinKey = healthKey = beamKey = "";
  document.getElementById("drop").classList.add("has-clip");
  document.getElementById("legend").hidden = false;
  document.getElementById("cam-title").textContent = (file.name || "CAM-01").replace(/\.[^.]+$/, "").slice(0, 18);
  video.src = `/api/jobs/${jobId}/video`;
  video.load();
  video.play().catch(() => {});
  renderLog(data.job.log);
  setStatus(data.job);
  pollTimer = setInterval(refreshJob, 1500);
  refreshJob();
}

function pickFile() {
  document.getElementById("file").click();
}

document.getElementById("file").onchange = (e) => {
  const file = e.target.files[0];
  if (file) start(file);
};
document.getElementById("dropzone").onclick = pickFile;
document.getElementById("composer-upload").onclick = pickFile;
document.getElementById("composer-go").onclick = pickFile;

const drop = document.getElementById("drop");
drop.addEventListener("dragover", (e) => {
  e.preventDefault();
  document.getElementById("dropzone").classList.add("over");
});
drop.addEventListener("dragleave", () => document.getElementById("dropzone").classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  document.getElementById("dropzone").classList.remove("over");
  const file = e.dataTransfer.files[0];
  if (file) start(file);
});

document.querySelectorAll(".asset").forEach((btn) => {
  btn.onclick = () => {
    document.querySelectorAll(".asset").forEach((b) => b.classList.remove("on"));
    btn.classList.add("on");
    filter = btn.dataset.filter;
    const label = btn.querySelector("small")?.textContent || "All assets";
    document.getElementById("view-label").textContent = `${label.toUpperCase()} · CAM 01`;
    draw();
  };
});

document.querySelectorAll(".view-switch button").forEach((btn) => {
  btn.onclick = () => {
    document.querySelectorAll(".view-switch button").forEach((b) => b.classList.remove("on"));
    btn.classList.add("on");
    document.getElementById("stage").dataset.view = btn.dataset.view;
    pinKey = beamKey = "";
    draw();
  };
});

document.getElementById("btn-original").onclick = () => {
  annotated = false;
  document.getElementById("btn-original").classList.add("on");
  document.getElementById("btn-annotated").classList.remove("on");
  draw();
};
document.getElementById("btn-annotated").onclick = () => {
  annotated = true;
  document.getElementById("btn-annotated").classList.add("on");
  document.getElementById("btn-original").classList.remove("on");
  draw();
};
playBtn.onclick = () => {
  if (video.paused) video.play();
  else video.pause();
};
video.addEventListener("play", () => {
  playBtn.textContent = "❚❚";
  const loop = () => {
    if (video.paused) return;
    draw();
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
});
video.addEventListener("pause", () => {
  playBtn.textContent = "▶";
});
video.addEventListener("timeupdate", () => {
  const dur = video.duration || 1;
  scrub.value = Math.round((video.currentTime / dur) * 1000);
  draw();
});
scrub.oninput = () => {
  const dur = video.duration || annotations.duration || 0;
  video.currentTime = (scrub.value / 1000) * dur;
  draw();
};
document.getElementById("prompt").addEventListener("input", (e) => {
  query = e.target.value.trim();
  draw();
});
document.getElementById("rail-toggle").onclick = () => document.body.classList.toggle("rail-open");

fetch("/api/health")
  .then((r) => r.json())
  .then((h) => {
    document.getElementById("model-name").textContent = (h.model || "grok").replace(/grok-?/, "grok ").split("-")[0].trim();
    if (!h.has_key) document.getElementById("hint").textContent = "Grok session missing";
  });

const onboard = document.getElementById("onboard");
onboard.hidden = false;
document.getElementById("onboard-go").onclick = () => {
  onboard.hidden = true;
  pickFile();
};
document.getElementById("onboard-skip").onclick = () => {
  onboard.hidden = true;
};

document.querySelectorAll(".spark").forEach((el) => {
  if (!el.style.getPropertyValue("--p")) el.style.setProperty("--p", `${20 + Math.round(Math.random() * 50)}%`);
});
