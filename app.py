"""
app.py — Sports Analytics Web App
───────────────────────────────────
Serves a polished HTML UI + async job queue so users can:
  1. Paste a YouTube URL
  2. Watch real-time progress
  3. Download the processed video

Jobs run in a background thread so Flask stays responsive.
Status is polled via /status/<job_id> every 2 seconds.
"""

from flask import Flask, request, jsonify, send_file, render_template_string
import os, uuid, threading, time
from collections import OrderedDict

import config
from download_video import download_video
import main as pipeline

app = Flask(__name__)

# ── In-memory job store ───────────────────────────────────────────
# job_id → {"status": "queued|processing|done|error", "message": str, "output": path}
jobs: dict = OrderedDict()
jobs_lock = threading.Lock()

MAX_JOBS = 20  # evict oldest when full


def _evict_old_jobs():
    """Remove oldest job if store is full."""
    with jobs_lock:
        while len(jobs) >= MAX_JOBS:
            oldest = next(iter(jobs))
            old = jobs.pop(oldest)
            # clean up files
            for key in ("input", "output"):
                p = old.get(key)
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass


def _run_job(job_id: str, url: str):
    """Worker: download → process → mark done. Runs in a daemon thread."""

    def update(status: str, message: str):
        with jobs_lock:
            jobs[job_id]["status"]  = status
            jobs[job_id]["message"] = message

    input_path  = os.path.join(config.DATA_DIR,   f"{job_id}.mp4")
    output_path = os.path.join(config.OUTPUT_DIR,  f"{job_id}_out.mp4")
    web_path    = os.path.join(config.OUTPUT_DIR,  f"{job_id}_web.mp4")

    with jobs_lock:
        jobs[job_id]["input"]  = input_path
        jobs[job_id]["output"] = web_path

    try:
        # 1. Download
        update("processing", "⬇️  Downloading video from YouTube…")
        ok = download_video(url, input_path)
        if not ok:
            update("error", "❌ Download failed. Check the URL and try again.")
            return

        # 2. Run pipeline
        update("processing", "🔍 Running detection + tracking pipeline…")
        args = pipeline.parse_args()
        args.video  = input_path
        args.output = output_path
        pipeline.run(args)

        # 3. Re-encode for browser playback (H.264 + faststart)
        update("processing", "🎬 Encoding output for web playback…")
        pipeline.convert_to_web_format(output_path, web_path)

        # Clean up intermediate file
        if os.path.exists(output_path):
            os.remove(output_path)

        update("done", "✅ Done! Your video is ready.")

    except Exception as e:
        update("error", f"❌ Pipeline error: {e}")
    finally:
        # Clean up input
        if os.path.exists(input_path):
            try:
                os.remove(input_path)
            except Exception:
                pass


# ── HTML template ─────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>SportVision — AI Player Analytics</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap" rel="stylesheet"/>
<style>
  :root {
    --bg:       #05060a;
    --surface:  #0d0f18;
    --border:   #1e2235;
    --accent:   #00f0a0;
    --accent2:  #0066ff;
    --warn:     #ff4060;
    --text:     #e8eaf2;
    --muted:    #5a5f7a;
    --mono:     'Space Mono', monospace;
    --sans:     'Syne', sans-serif;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    overflow-x: hidden;
  }

  /* ── grid background ── */
  body::before {
    content: '';
    position: fixed; inset: 0; z-index: 0;
    background-image:
      linear-gradient(rgba(0,240,160,.04) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,240,160,.04) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
  }

  /* ── glow orb ── */
  body::after {
    content: '';
    position: fixed;
    top: -20vh; left: 50%;
    transform: translateX(-50%);
    width: 80vw; height: 60vh;
    background: radial-gradient(ellipse, rgba(0,102,255,.18) 0%, transparent 70%);
    pointer-events: none; z-index: 0;
  }

  .wrap {
    position: relative; z-index: 1;
    width: 100%; max-width: 680px;
    padding: 4rem 2rem 6rem;
    display: flex; flex-direction: column; gap: 2.5rem;
  }

  /* ── header ── */
  header { text-align: center; }
  .logo-tag {
    display: inline-block;
    font-family: var(--mono);
    font-size: .72rem;
    letter-spacing: .2em;
    color: var(--accent);
    border: 1px solid rgba(0,240,160,.35);
    padding: .3rem .9rem;
    border-radius: 2px;
    margin-bottom: 1.4rem;
    text-transform: uppercase;
  }
  h1 {
    font-size: clamp(2.2rem, 8vw, 3.8rem);
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -.02em;
  }
  h1 span { color: var(--accent); }
  .sub {
    margin-top: .9rem;
    font-size: .95rem;
    color: var(--muted);
    line-height: 1.6;
  }

  /* ── card ── */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 2rem;
    display: flex;
    flex-direction: column;
    gap: 1.2rem;
  }

  label {
    font-family: var(--mono);
    font-size: .72rem;
    letter-spacing: .12em;
    color: var(--muted);
    text-transform: uppercase;
  }

  .input-row {
    display: flex;
    gap: .7rem;
  }

  input[type="url"] {
    flex: 1;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    font-family: var(--mono);
    font-size: .85rem;
    padding: .85rem 1rem;
    outline: none;
    transition: border-color .2s;
  }
  input[type="url"]::placeholder { color: var(--muted); }
  input[type="url"]:focus { border-color: var(--accent2); }

  button {
    font-family: var(--sans);
    font-weight: 700;
    font-size: .9rem;
    cursor: pointer;
    border: none;
    border-radius: 8px;
    transition: all .18s;
  }

  .btn-submit {
    background: var(--accent);
    color: #040508;
    padding: .85rem 1.6rem;
    white-space: nowrap;
    letter-spacing: .02em;
  }
  .btn-submit:hover { filter: brightness(1.15); transform: translateY(-1px); }
  .btn-submit:disabled { opacity: .45; cursor: not-allowed; transform: none; }

  /* ── pipeline badges ── */
  .pipeline {
    display: flex;
    flex-wrap: wrap;
    gap: .5rem;
  }
  .badge {
    font-family: var(--mono);
    font-size: .67rem;
    letter-spacing: .08em;
    padding: .25rem .65rem;
    border-radius: 4px;
    border: 1px solid var(--border);
    color: var(--muted);
  }
  .badge.on { border-color: rgba(0,240,160,.4); color: var(--accent); }

  /* ── status card ── */
  #status-card {
    display: none;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 2rem;
    flex-direction: column;
    gap: 1.2rem;
  }
  #status-card.visible { display: flex; }

  .status-header {
    display: flex; align-items: center; gap: .8rem;
  }
  .spinner {
    width: 20px; height: 20px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin .8s linear infinite;
    flex-shrink: 0;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  #status-msg {
    font-size: .95rem;
    font-family: var(--mono);
    color: var(--text);
  }

  /* progress bar */
  .progress-track {
    height: 3px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
  }
  .progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent2), var(--accent));
    border-radius: 2px;
    width: 0%;
    transition: width .6s ease;
    animation: shimmer 1.8s ease infinite;
    background-size: 200% 100%;
  }
  @keyframes shimmer {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
  }

  #job-id-display {
    font-family: var(--mono);
    font-size: .68rem;
    color: var(--muted);
  }

  /* ── download section ── */
  #download-section {
    display: none;
    flex-direction: column;
    gap: 1rem;
    align-items: center;
    text-align: center;
  }
  #download-section.visible { display: flex; }

  .done-icon {
    font-size: 2.5rem;
    line-height: 1;
  }
  .done-text { font-size: 1.1rem; font-weight: 700; }
  .done-sub  { font-size: .85rem; color: var(--muted); margin-top: .2rem; }

  .btn-download {
    background: transparent;
    color: var(--accent);
    border: 1px solid var(--accent);
    padding: .9rem 2.2rem;
    font-size: .95rem;
    border-radius: 8px;
    text-decoration: none;
    display: inline-block;
    font-family: var(--sans);
    font-weight: 700;
    transition: all .18s;
    margin-top: .4rem;
  }
  .btn-download:hover {
    background: var(--accent);
    color: #040508;
    transform: translateY(-2px);
  }

  .btn-reset {
    background: transparent;
    color: var(--muted);
    border: 1px solid var(--border);
    padding: .6rem 1.4rem;
    font-size: .82rem;
  }
  .btn-reset:hover { color: var(--text); border-color: var(--muted); }

  /* ── error state ── */
  .error-msg {
    color: var(--warn);
    font-family: var(--mono);
    font-size: .85rem;
    background: rgba(255,64,96,.07);
    border: 1px solid rgba(255,64,96,.25);
    border-radius: 8px;
    padding: .9rem 1rem;
  }

  /* ── footer ── */
  footer {
    position: relative; z-index: 1;
    font-family: var(--mono);
    font-size: .7rem;
    color: var(--muted);
    text-align: center;
    padding-bottom: 2rem;
  }
</style>
</head>
<body>
<div class="wrap">

  <header>
    <div class="logo-tag">AI Sports Analytics</div>
    <h1>Sport<span>Vision</span></h1>
    <p class="sub">
      Paste a YouTube sports clip. We detect every player,<br>
      track IDs, estimate speed, and read jersey numbers.
    </p>
  </header>

  <!-- Input card -->
  <div class="card">
    <label for="url-input">YouTube Video URL</label>
    <div class="input-row">
      <input type="url" id="url-input"
        placeholder="https://www.youtube.com/watch?v=…"
        autocomplete="off" spellcheck="false"/>
      <button class="btn-submit" id="submit-btn" onclick="submitJob()">
        Analyse →
      </button>
    </div>
    <div class="pipeline">
      <span class="badge on">YOLOv8m</span>
      <span class="badge on">BoT-SORT</span>
      <span class="badge on">Speed km/h</span>
      <span class="badge on">Jersey OCR</span>
      <span class="badge on">Heatmap</span>
      <span class="badge">GPU optional</span>
    </div>
  </div>

  <!-- Status card -->
  <div class="card" id="status-card">
    <div class="status-header">
      <div class="spinner" id="spinner"></div>
      <span id="status-msg">Starting…</span>
    </div>
    <div class="progress-track">
      <div class="progress-fill" id="progress-fill"></div>
    </div>
    <div id="job-id-display"></div>
    <div class="error-msg" id="error-msg" style="display:none"></div>
  </div>

  <!-- Download section (inside status card-like area) -->
  <div class="card" id="download-section">
    <div class="done-icon">🎯</div>
    <div>
      <div class="done-text">Analysis complete</div>
      <div class="done-sub">Your annotated video is ready to download</div>
    </div>
    <a id="download-link" class="btn-download" href="#" download>
      ⬇ Download Video
    </a>
    <button class="btn-reset" onclick="resetUI()">Process another video</button>
  </div>

</div>

<footer>SportVision · YOLOv8 + BoT-SORT · CPU inference</footer>

<script>
let pollInterval = null;
let currentJobId = null;
const PROGRESS_STEPS = {
  "queued":     8,
  "processing": 45,
};

function submitJob() {
  const url = document.getElementById('url-input').value.trim();
  if (!url) { alert('Please enter a YouTube URL.'); return; }

  const btn = document.getElementById('submit-btn');
  btn.disabled = true;

  resetStatus();
  showStatus();

  fetch('/process', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url })
  })
  .then(r => r.json())
  .then(data => {
    if (data.error) { showError(data.error); btn.disabled = false; return; }
    currentJobId = data.job_id;
    document.getElementById('job-id-display').textContent = `Job: ${currentJobId}`;
    pollInterval = setInterval(() => pollStatus(currentJobId), 2500);
  })
  .catch(err => { showError('Network error: ' + err); btn.disabled = false; });
}

function pollStatus(jobId) {
  fetch(`/status/${jobId}`)
    .then(r => r.json())
    .then(data => {
      document.getElementById('status-msg').textContent = data.message || data.status;
      setProgress(data.status);

      if (data.status === 'done') {
        clearInterval(pollInterval);
        showDownload(jobId);
      } else if (data.status === 'error') {
        clearInterval(pollInterval);
        showError(data.message);
        document.getElementById('submit-btn').disabled = false;
      }
    })
    .catch(() => {});
}

function setProgress(status) {
  const fill = document.getElementById('progress-fill');
  const pct = PROGRESS_STEPS[status] || 85;
  fill.style.width = pct + '%';
}

function showStatus() {
  document.getElementById('status-card').classList.add('visible');
  document.getElementById('download-section').classList.remove('visible');
}

function showDownload(jobId) {
  document.getElementById('progress-fill').style.width = '100%';
  document.getElementById('spinner').style.display = 'none';
  document.getElementById('status-card').classList.remove('visible');

  const dl = document.getElementById('download-section');
  dl.classList.add('visible');

  const link = document.getElementById('download-link');
  link.href = `/download/${jobId}`;
}

function showError(msg) {
  const e = document.getElementById('error-msg');
  e.style.display = 'block';
  e.textContent = msg;
  document.getElementById('spinner').style.display = 'none';
}

function resetStatus() {
  document.getElementById('status-msg').textContent = 'Starting…';
  document.getElementById('progress-fill').style.width = '0%';
  document.getElementById('spinner').style.display = 'block';
  document.getElementById('error-msg').style.display = 'none';
  document.getElementById('job-id-display').textContent = '';
}

function resetUI() {
  document.getElementById('download-section').classList.remove('visible');
  document.getElementById('status-card').classList.remove('visible');
  document.getElementById('url-input').value = '';
  document.getElementById('submit-btn').disabled = false;
  if (pollInterval) clearInterval(pollInterval);
  currentJobId = null;
}

// Submit on Enter
document.getElementById('url-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') submitJob();
});
</script>
</body>
</html>"""


# ── Routes ────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/process", methods=["POST"])
def process_video():
    data = request.get_json(silent=True) or {}
    url  = (data.get("url") or "").strip()

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    if not (url.startswith("http://") or url.startswith("https://")):
        return jsonify({"error": "Invalid URL format"}), 400

    _evict_old_jobs()

    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {
            "status":  "queued",
            "message": "⏳ Queued…",
            "output":  None,
            "input":   None,
            "created": time.time(),
        }

    os.makedirs(config.DATA_DIR,   exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    t = threading.Thread(target=_run_job, args=(job_id, url), daemon=True)
    t.start()

    return jsonify({"job_id": job_id, "message": "Job started"})


@app.route("/status/<job_id>")
def status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"status": "error", "message": "Job not found"}), 404
    return jsonify({"status": job["status"], "message": job["message"]})


@app.route("/download/<job_id>")
def download(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "Not ready or not found"}), 404

    path = job.get("output")
    if not path or not os.path.exists(path):
        return jsonify({"error": "Output file missing"}), 404

    return send_file(
        path,
        as_attachment=True,
        download_name="sportvision_output.mp4",
        mimetype="video/mp4",
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)