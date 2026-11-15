/* hanoiFlow — forecast verification page.
 * Upload a long traffic video, get causal +5/+10/+15 minute CI forecasts, then seek the
 * video to each forecast's target time to check it. The browser plays the user's own file
 * locally; only the analysis result comes from the server.
 */
'use strict';

const SEVERE_CI = 0.55;
const BUSY_CI = 0.30;
const LEVEL_LABEL = { free: 'Thông thoáng', busy: 'Đông đúc', gridlock: 'Tắc nghẽn nghiêm trọng' };
const LEVEL_COLOR = { free: '#3f8f5a', busy: '#c4892a', gridlock: '#b5443a' };
const SESSION_CONTRACT = 'hanoiflow-forecast-session/1.0';
const JOB_KEY = 'hanoiflow-forecast-job';

function levelKey(ci) {
  return ci < BUSY_CI ? 'free' : ci < SEVERE_CI ? 'busy' : 'gridlock';
}

function fmtTime(seconds) {
  const s = Math.max(0, Math.round(Number(seconds) || 0));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = String(s % 60).padStart(2, '0');
  return h ? `${h}:${String(m).padStart(2, '0')}:${sec}` : `${String(m).padStart(2, '0')}:${sec}`;
}

function num(value, digits = 2) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return '—';
  return Number(value).toLocaleString('vi-VN', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function pct(value, digits = 0) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return '—';
  return `${num(Number(value) * 100, digits)}%`;
}

/** Last item with item.t <= t (items sorted by t), or null. */
function lastAtOrBefore(items, t) {
  let lo = 0, hi = (items || []).length - 1, found = null;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (items[mid].t <= t + 1e-6) { found = items[mid]; lo = mid + 1; } else hi = mid - 1;
  }
  return found;
}

function skillVerdict(metric) {
  if (!metric || !metric.evaluated) return { tone: 'neutral', title: 'Chưa kiểm chứng được', text: 'Video kết thúc trước khi có thể so sánh chân trời này.' };
  const skill = metric.skill_vs_persistence;
  if (skill === null || skill === undefined) return { tone: 'neutral', title: 'Không xác định', text: `${metric.evaluated} lần kiểm chứng, đường không thay đổi.` };
  if (skill >= 0.05) return { tone: 'good', title: `Tốt hơn ${pct(skill)}`, text: `so với giữ nguyên mức hiện tại · ${metric.evaluated} lần kiểm chứng` };
  if (skill <= -0.05) return { tone: 'bad', title: `Kém hơn ${pct(-skill)}`, text: `so với giữ nguyên mức hiện tại · ${metric.evaluated} lần kiểm chứng` };
  return { tone: 'neutral', title: 'Tương đương', text: `với giữ nguyên mức hiện tại · ${metric.evaluated} lần kiểm chứng` };
}

/** Whether an anchor's horizon outcome may be shown at playback time t. */
function isRevealed(item, t, blind) {
  return !blind || t >= item.target_t - 0.5;
}

function forecastCsv(anchors) {
  const header = ['thoi_diem_du_bao_s', 'ci_luc_du_bao', 'chan_troi_phut', 'thoi_diem_dich_s', 'ci_du_bao', 'ci_thap', 'ci_cao',
    'xac_suat_tac', 'canh_bao', 'ci_thuc_te', 'ci_giu_nguyen', 'ci_xu_huong', 'phuong_phap'];
  const lines = [header.join(',')];
  for (const anchor of anchors || []) {
    for (const item of anchor.horizons || []) {
      lines.push([anchor.t, anchor.observed_ci, item.horizon_minutes, item.target_t, item.ci, item.lower, item.upper,
        item.p_gridlock, item.warning ? 1 : 0, item.actual_ci ?? '', item.persistence_ci, item.trend_ci, item.method]
        .map(v => (typeof v === 'number' ? String(Math.round(v * 10000) / 10000) : String(v))).join(','));
    }
  }
  return '﻿' + lines.join('\r\n') + '\r\n';
}

function validateSession(payload) {
  if (!payload || typeof payload !== 'object') throw new Error('Tệp kết quả không hợp lệ.');
  if (payload.contract !== SESSION_CONTRACT) throw new Error('Tệp kết quả không đúng định dạng hanoiFlow.');
  const mf = payload.minute_forecast;
  if (!mf || !Array.isArray(mf.anchors) || !Array.isArray(mf.observed)) throw new Error('Tệp kết quả thiếu dữ liệu dự báo.');
  return payload;
}

if (typeof module !== 'undefined') {
  module.exports = { levelKey, fmtTime, lastAtOrBefore, skillVerdict, isRevealed, forecastCsv, validateSession, SESSION_CONTRACT };
}

if (typeof document !== 'undefined') {
  const $ = id => document.getElementById(id);
  const state = {
    limits: null,
    file: null,
    objectUrl: null,
    session: null,
    locked: null,
    blind: true,
    busy: false,
    pendingRender: false,
  };
  const player = $('player');
  const chart = $('ciChart');

  const escapeHtml = text => String(text ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  function toast(message) {
    const el = $('toast');
    el.textContent = message;
    el.hidden = false;
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => { el.hidden = true; }, 5000);
  }

  function notice(message, kind = 'info') {
    const el = $('uploadNotice');
    el.hidden = !message;
    el.textContent = message || '';
    el.className = `notice${kind === 'error' ? ' error' : ''}`;
  }

  function setProgress(percent, text) {
    $('progressWrap').hidden = false;
    if (percent !== null) $('progressBar').style.width = `${Math.max(0, Math.min(100, percent))}%`;
    $('progressText').textContent = text;
  }

  async function getJson(path, timeoutMs = 20000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(path, { signal: controller.signal, cache: 'no-store' });
      const text = await response.text();
      let payload = {};
      try { payload = text ? JSON.parse(text) : {}; } catch { const e = new Error('Máy chủ trả dữ liệu hỏng.'); e.status = response.status; throw e; }
      if (!response.ok) { const e = new Error(payload.detail || `HTTP ${response.status}`); e.status = response.status; throw e; }
      return payload;
    } finally { clearTimeout(timer); }
  }

  async function loadStatus() {
    try {
      const status = await getJson('/api/aicc/status');
      state.limits = status;
      const mib = status.upload_limit_mib || Math.floor((status.upload_limit_bytes || 0) / 1048576);
      const minutes = Math.floor((status.upload_max_duration_seconds || 0) / 60);
      const lib = status.forecast_library_enabled ? ` Thư viện học: ${status.forecast_library_videos || 0} video đã phân tích.` : '';
      $('limitText').textContent = `Tối đa ${mib.toLocaleString('vi-VN')} MiB và ${minutes} phút mỗi video.${lib} Cần tối thiểu khoảng 9 phút để có dự báo +5 phút, 25–30 phút để kiểm chứng đủ +15 phút.`;
      const chip = $('serverChip');
      if (status.video_review_enabled) { chip.textContent = 'Sẵn sàng'; chip.className = 'chip ok'; }
      else { chip.textContent = 'Chưa sẵn sàng'; chip.className = 'chip bad'; notice(status.message || 'Máy chủ chưa bật phân tích video.', 'error'); }
      if (minutes && minutes < 10) notice(`Máy chủ này chỉ nhận video tối đa ${minutes} phút, chưa đủ để dự báo +5/+10/+15 phút. Hãy chạy bản local (deploy_demo.ps1) để phân tích video dài.`);
    } catch (error) {
      $('limitText').textContent = 'Không đọc được cấu hình máy chủ.';
      $('serverChip').textContent = 'Mất kết nối';
      $('serverChip').className = 'chip bad';
    }
    renderInputs();
  }

  function renderInputs() {
    const ready = Boolean(state.file) && !state.busy && Boolean(state.limits?.video_review_enabled);
    $('analyzeBtn').disabled = !ready;
  }

  function setVideoFile(file) {
    state.file = file || null;
    if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
    state.objectUrl = file ? URL.createObjectURL(file) : null;
    player.src = state.objectUrl || '';
    $('fileName').textContent = file ? `${file.name} · ${(file.size / 1048576).toLocaleString('vi-VN', { maximumFractionDigits: 1 })} MiB` : 'Chọn video từ máy tính';
    renderInputs();
    if (state.session) checkVideoMatchesSession();
  }

  function upload(form) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', '/api/scenarios/video-review');
      xhr.timeout = 60 * 60 * 1000;
      xhr.upload.onprogress = e => {
        const p = e.lengthComputable ? Math.floor(e.loaded / e.total * 100) : null;
        setProgress(p, `Đang tải video lên${p === null ? '' : `: ${p}%`}`);
      };
      xhr.upload.onload = () => setProgress(100, 'Đã tải lên. Máy chủ đang kiểm tra video…');
      xhr.onload = () => {
        let payload = {};
        try { payload = JSON.parse(xhr.responseText || '{}'); } catch {}
        if (xhr.status !== 202) {
          const msg = xhr.status === 409 ? 'Máy chủ đang phân tích một video khác. Vui lòng thử lại sau.'
            : xhr.status === 413 ? (payload.detail || 'Video vượt giới hạn dung lượng hoặc thời lượng.')
            : (typeof payload.detail === 'string' ? payload.detail : `Không tải được video (HTTP ${xhr.status}).`);
          reject(new Error(msg));
          return;
        }
        if (!/^[a-f0-9]{32}$/.test(payload.job_id || '')) { reject(new Error('Phản hồi tiếp nhận không hợp lệ.')); return; }
        resolve(payload);
      };
      xhr.onerror = () => reject(new Error('Mất kết nối khi tải video.'));
      xhr.ontimeout = () => reject(new Error('Tải video quá thời gian chờ.'));
      xhr.send(form);
    });
  }

  async function poll(jobId) {
    const started = Date.now();
    let failures = 0;
    while (Date.now() - started < 3 * 60 * 60 * 1000) {
      let job;
      try { job = await getJson(`/api/scenarios/video-reviews/${jobId}`, 15000); failures = 0; }
      catch (error) {
        if (error.status && error.status < 500) { sessionStorage.removeItem(JOB_KEY); throw error; }
        if (++failures >= 8) throw new Error('Mất kết nối với máy chủ khi đang phân tích. Tải lại trang để theo dõi tiếp.');
        await new Promise(r => setTimeout(r, 2000));
        continue;
      }
      const p = Number(job.progress_percent || 0);
      if (job.status === 'queued') setProgress(0, 'Đang chờ đến lượt phân tích…');
      else if (job.status === 'running') setProgress(p, job.frames_total ? `Đang phân tích khung hình ${Number(job.frames_decoded || 0).toLocaleString('vi-VN')} / ${Number(job.frames_total).toLocaleString('vi-VN')} · ${p}%` : 'Đang nạp mô hình…');
      else if (job.status === 'succeeded') { sessionStorage.removeItem(JOB_KEY); return job.result; }
      else if (job.status === 'failed') { sessionStorage.removeItem(JOB_KEY); throw new Error(job.error || 'Không phân tích được video.'); }
      await new Promise(r => setTimeout(r, 1200));
    }
    throw new Error('Phân tích quá lâu. Tải lại trang để theo dõi tiếp.');
  }

  async function analyze() {
    if (!state.file || state.busy) return;
    state.busy = true; renderInputs(); notice('');
    try {
      const form = new FormData();
      form.append('file', state.file);
      form.append('preserve_real_timing', $('realTiming').checked ? 'true' : 'false');
      const job = await upload(form);
      sessionStorage.setItem(JOB_KEY, job.job_id);
      const result = await poll(job.job_id);
      openResult(result);
    } catch (error) {
      notice(error.message, 'error');
      setProgress(null, error.message);
    } finally {
      state.busy = false; renderInputs();
    }
  }

  function openResult(review) {
    const mf = review && review.minute_forecast;
    if (!mf) { notice('Kết quả không có phần dự báo theo phút (máy chủ cũ?).', 'error'); return; }
    const session = {
      contract: SESSION_CONTRACT,
      created_at: new Date().toISOString(),
      filename: review.filename || state.file?.name || null,
      video: review.video || {},
      minute_forecast: mf,
    };
    loadSession(session);
  }

  function loadSession(session) {
    validateSession(session);
    const mf = session.minute_forecast;
    state.session = session;
    state.locked = null;
    state.panelKey = null;
    setProgress(100, 'Hoàn tất. Phát video hoặc chọn một khoảnh khắc bên dưới.');
    if (!mf.available) {
      notice(mf.message || 'Video chưa đủ dài để dự báo theo phút.', 'error');
      $('workspace').hidden = true;
      return;
    }
    notice(mf.timing_confirmed ? '' : 'Bạn chưa xác nhận video quay theo thời gian thực: “5 phút” ở đây là 5 phút trên dòng thời gian video.');
    $('workspace').hidden = false;
    const v = session.video || {};
    $('videoMeta').textContent = `${session.filename || 'Video'} · ${fmtTime(v.duration_seconds)} · ${mf.anchors.length.toLocaleString('vi-VN')} thời điểm dự báo, mỗi ${mf.step_seconds} giây`;
    $('methodLine').textContent = 'Dự báo lập tại mỗi thời điểm chỉ dùng phần video trước đó. Khoảng trong ngoặc là khoảng tin cậy 80%.';
    renderHighlights();
    renderMetrics();
    renderLimits();
    checkVideoMatchesSession();
    scheduleRender();
    $('workspace').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function checkVideoMatchesSession() {
    if (!state.session) return;
    if (!player.src) { notice('Chọn file video gốc ở mục 1 để phát và kiểm chứng.'); return; }
    const expected = Number(state.session.video?.duration_seconds || 0);
    player.addEventListener('loadedmetadata', () => {
      if (expected && Math.abs(player.duration - expected) > 3) {
        notice(`Video đang mở dài ${fmtTime(player.duration)}, khác video đã phân tích (${fmtTime(expected)}). Hãy chọn đúng file.`, 'error');
      }
    }, { once: true });
  }

  function duration() {
    return Number(state.session?.video?.duration_seconds) || player.duration || 1;
  }

  function scheduleRender() {
    if (state.pendingRender) return;
    state.pendingRender = true;
    requestAnimationFrame(() => { state.pendingRender = false; render(); });
  }

  function currentAnchor(t) {
    const anchors = state.session?.minute_forecast?.anchors || [];
    return state.locked || lastAtOrBefore(anchors, t);
  }

  function render() {
    if (!state.session?.minute_forecast?.available) return;
    const t = player.currentTime || 0;
    const anchor = currentAnchor(t);
    renderPanel(anchor, t);
    renderChart(anchor, t);
    renderBadge(t);
  }

  function renderBadge(t) {
    const observed = lastAtOrBefore(state.session.minute_forecast.observed.filter(o => o.ci !== null), t);
    const badge = $('nowBadge');
    if (!observed) { badge.hidden = true; return; }
    const key = levelKey(observed.ci);
    badge.hidden = false;
    badge.innerHTML = `<i style="background:${LEVEL_COLOR[key]}"></i>${LEVEL_LABEL[key]} · CI ${num(observed.ci)}`;
  }

  function levelChip(ci) {
    const key = levelKey(ci);
    return `<span class="level ${key}">${LEVEL_LABEL[key]}</span>`;
  }

  function renderPanel(anchor, t) {
    const mf = state.session.minute_forecast;
    $('unlockBtn').hidden = !state.locked;
    // Rebuild only when something visible changes, so buttons stay clickable during playback.
    const key = anchor ? [anchor.t, Boolean(state.locked), state.blind, ...anchor.horizons.map(item => isRevealed(item, t, state.blind))].join('|') : `none|${Math.floor(t)}`;
    if (key === state.panelKey) {
      if (state.locked) $('forecastSub').textContent = `Đang giữ dự báo này để kiểm chứng. Video đang ở ${fmtTime(t)}.`;
      return;
    }
    state.panelKey = key;
    state.panelAnchor = anchor;
    if (!anchor) {
      $('forecastTitle').textContent = `Khởi động · ${fmtTime(t)}`;
      $('forecastSub').textContent = `Hệ thống cần ${Math.round(mf.warmup_seconds / 60)} phút đầu video để có dự báo đầu tiên.`;
      $('nowRow').innerHTML = '';
      $('horizonList').innerHTML = '';
      return;
    }
    $('forecastTitle').textContent = `Dự báo lập lúc ${fmtTime(anchor.t)}`;
    $('forecastSub').textContent = state.locked
      ? `Đang giữ dự báo này để kiểm chứng. Video đang ở ${fmtTime(t)}.`
      : 'Cập nhật theo thời điểm đang xem.';
    $('nowRow').innerHTML = `<span>Lúc lập dự báo:</span>${levelChip(anchor.observed_ci)}<strong>CI ${num(anchor.observed_ci)}</strong>`;
    $('horizonList').innerHTML = anchor.horizons.map((item, index) => {
      const revealed = isRevealed(item, t, state.blind);
      let verify;
      if (item.actual_ci === null || item.actual_ci === undefined) {
        verify = '<div class="verify">Video kết thúc trước thời điểm này; chưa kiểm chứng được.</div>';
      } else if (!revealed) {
        verify = `<button class="btn small ghost" type="button" data-seek-index="${index}">Tua tới ${fmtTime(item.target_t)} để kiểm chứng</button>`;
      } else {
        const sameLevel = levelKey(item.ci) === levelKey(item.actual_ci);
        const warnOk = item.warning ? item.actual_ci >= SEVERE_CI : true;
        const ok = sameLevel && warnOk;
        const warnText = item.warning ? (item.actual_ci >= SEVERE_CI ? ' Cảnh báo tắc: đúng.' : ' Cảnh báo tắc: báo nhầm.') : '';
        verify = `<div class="verify ${ok ? 'hit' : 'miss'}">${ok ? '✓' : '✗'} Thực tế lúc ${fmtTime(item.target_t)}: ${LEVEL_LABEL[levelKey(item.actual_ci)]} (CI ${num(item.actual_ci)}).${warnText}</div>`;
      }
      return `<article class="hcard${item.warning ? ' warning' : ''}">
        <div class="hrow"><span class="htitle">+${item.horizon_minutes} phút</span><span class="htime">→ ${fmtTime(item.target_t)}</span></div>
        <div class="hrow">${levelChip(item.ci)}${item.warning ? '<span class="warn-flag">⚠ Cảnh báo sắp tắc</span>' : ''}</div>
        <div class="hmeta">CI dự báo <strong>${num(item.ci)}</strong> (${num(item.lower)}–${num(item.upper)}) · khả năng tắc ${pct(item.p_gridlock)}</div>
        <div class="pbar" aria-hidden="true"><span style="width:${Math.round(item.p_gridlock * 100)}%"></span></div>
        ${verify}
      </article>`;
    }).join('');
  }

  $('horizonList').addEventListener('click', event => {
    const button = event.target.closest('[data-seek-index]');
    const anchor = state.panelAnchor;
    if (!button || !anchor) return;
    const item = anchor.horizons[Number(button.dataset.seekIndex)];
    state.locked = anchor;
    player.pause();
    player.currentTime = item.target_t;
    scheduleRender();
  });

  function renderChart(anchor, t) {
    const mf = state.session.minute_forecast;
    const width = Math.max(320, Math.round(chart.clientWidth || 1000));
    const height = 240;
    chart.setAttribute('viewBox', `0 0 ${width} ${height}`);
    chart.setAttribute('preserveAspectRatio', 'xMinYMin meet');
    const padL = 44, padR = 14, padT = 10, padB = 28;
    const dur = duration();
    const x = s => padL + (Math.max(0, Math.min(dur, s)) / dur) * (width - padL - padR);
    const y = ci => padT + (1 - Math.max(0, Math.min(1, ci))) * (height - padT - padB);
    const parts = [];
    const band = (lo, hi, color) => `<rect x="${padL}" y="${y(hi)}" width="${width - padL - padR}" height="${y(lo) - y(hi)}" fill="${color}" opacity=".09"/>`;
    parts.push(band(0, BUSY_CI, LEVEL_COLOR.free), band(BUSY_CI, SEVERE_CI, LEVEL_COLOR.busy), band(SEVERE_CI, 1, LEVEL_COLOR.gridlock));
    for (const tick of [0, BUSY_CI, SEVERE_CI, 1]) {
      parts.push(`<line x1="${padL}" x2="${width - padR}" y1="${y(tick)}" y2="${y(tick)}" stroke="#d9dfd8" stroke-dasharray="${tick === 0 || tick === 1 ? '' : '4 4'}"/>`);
      parts.push(`<text x="${padL - 8}" y="${y(tick) + 4}" text-anchor="end" font-size="11" fill="#6f7d76">${num(tick, tick === 0 || tick === 1 ? 0 : 2)}</text>`);
    }
    const stepMin = dur > 3600 ? 10 : 5;
    for (let s = 0; s <= dur; s += stepMin * 60) {
      parts.push(`<text x="${x(s)}" y="${height - 8}" text-anchor="middle" font-size="11" fill="#6f7d76">${fmtTime(s)}</text>`);
    }
    // Observed CI line, cut at the playhead in blind mode.
    const limit = state.blind ? t : dur;
    let d = '', open = false;
    for (const point of mf.observed) {
      if (point.t > limit + 1e-6) break;
      if (point.ci === null) { open = false; continue; }
      d += `${open ? 'L' : 'M'}${x(point.t).toFixed(1)},${y(point.ci).toFixed(1)}`;
      open = true;
    }
    parts.push(`<path d="${d}" fill="none" stroke="#3d4d46" stroke-width="2" stroke-linejoin="round"/>`);
    // Correct congestion warnings that have already happened on screen.
    for (const hit of mf.highlights?.hits || []) {
      if (state.blind && hit.target_t > t) continue;
      parts.push(`<path d="M${x(hit.t)},${y(1) + 2} l6,10 l-12,0 z" fill="${LEVEL_COLOR.gridlock}"><title>Lúc ${fmtTime(hit.t)} cảnh báo +${hit.horizon_minutes} phút sẽ tắc: đúng</title></path>`);
    }
    if (anchor) {
      parts.push(`<line x1="${x(anchor.t)}" x2="${x(anchor.t)}" y1="${padT}" y2="${height - padB}" stroke="#2d5f8f" stroke-width="1" stroke-dasharray="3 3"/>`);
      for (const item of anchor.horizons) {
        parts.push(`<line x1="${x(anchor.t)}" y1="${y(anchor.observed_ci)}" x2="${x(item.target_t)}" y2="${y(item.ci)}" stroke="#2d5f8f" stroke-width="1.5" stroke-dasharray="5 4"/>`);
        parts.push(`<line x1="${x(item.target_t)}" x2="${x(item.target_t)}" y1="${y(item.upper)}" y2="${y(item.lower)}" stroke="#2d5f8f" stroke-width="2" opacity=".45"/>`);
        parts.push(`<circle cx="${x(item.target_t)}" cy="${y(item.ci)}" r="5" fill="#2d5f8f"><title>+${item.horizon_minutes} phút: CI dự báo ${num(item.ci)}</title></circle>`);
        if (item.actual_ci !== null && item.actual_ci !== undefined && isRevealed(item, t, state.blind)) {
          parts.push(`<circle cx="${x(item.target_t)}" cy="${y(item.actual_ci)}" r="6" fill="#fff" stroke="#1f2d28" stroke-width="2"><title>Thực tế: CI ${num(item.actual_ci)}</title></circle>`);
        }
      }
    }
    parts.push(`<line x1="${x(t)}" x2="${x(t)}" y1="${padT}" y2="${height - padB}" stroke="#1f2d28" stroke-width="2"/>`);
    chart.innerHTML = parts.join('');
    chart.dataset.padL = padL; chart.dataset.padR = padR; chart.dataset.width = width;
  }

  function seekFromChart(event) {
    if (!state.session) return;
    const rect = chart.getBoundingClientRect();
    const width = Number(chart.dataset.width || rect.width);
    const scale = width / rect.width;
    const px = (event.clientX - rect.left) * scale;
    const padL = Number(chart.dataset.padL || 44), padR = Number(chart.dataset.padR || 14);
    const ratio = Math.max(0, Math.min(1, (px - padL) / (width - padL - padR)));
    state.locked = null;
    player.currentTime = ratio * duration();
    scheduleRender();
  }

  function momentRow(record, kind) {
    const lead = `Lúc <strong>${fmtTime(record.t)}</strong> đường đang ${LEVEL_LABEL[levelKey(record.observed_ci)].toLowerCase()} (CI ${num(record.observed_ci)}).`;
    const said = kind === 'miss'
      ? `Hệ thống không cảnh báo tắc sau ${record.horizon_minutes} phút (CI dự báo ${num(record.predicted_ci)}, khả năng tắc ${pct(record.p_gridlock)}).`
      : `Hệ thống cảnh báo: sau <strong>${record.horizon_minutes} phút</strong> sẽ tắc (khả năng ${pct(record.p_gridlock)}, CI dự báo ${num(record.predicted_ci)}).`;
    const outcome = kind === 'hit' && state.blind
      ? `Kết quả: tua tới ${fmtTime(record.target_t)} để xem.`
      : `Thực tế lúc ${fmtTime(record.target_t)}: ${LEVEL_LABEL[levelKey(record.actual_ci)]} (CI ${num(record.actual_ci)}).`;
    return `<div class="moment"><div><p>${lead} ${said}</p><p class="muted">${outcome}</p></div>
      <div class="actions"><button class="btn small ghost" type="button" data-at="${record.t}">Xem lúc dự báo</button>
      <button class="btn small primary" type="button" data-at="${record.t}" data-target="${record.target_t}">Tua tới ${fmtTime(record.target_t)}</button></div></div>`;
  }

  function bindMoments(container) {
    container.querySelectorAll('button[data-at]').forEach(button => {
      button.addEventListener('click', () => {
        const anchors = state.session.minute_forecast.anchors;
        state.locked = lastAtOrBefore(anchors, Number(button.dataset.at));
        player.pause();
        player.currentTime = Number(button.dataset.target || button.dataset.at);
        if (!player.src) notice('Chọn file video gốc ở mục 1 để xem hình ảnh.');
        scheduleRender();
        document.querySelector('.player-card').scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
    });
  }

  function renderHighlights() {
    const h = state.session.minute_forecast.highlights || {};
    const hits = h.hits || [];
    $('hitList').innerHTML = hits.length ? hits.map(r => momentRow(r, 'hit')).join('')
      : '<p class="empty">Trong video này hệ thống chưa có lần cảnh báo tắc nào trúng. Xem mục 4 để biết mô hình so với cách đoán đơn giản ra sao.</p>';
    const wrong = [...(h.misses || []).map(r => momentRow(r, 'miss')), ...(h.false_alarms || []).map(r => momentRow(r, 'false'))];
    $('missList').innerHTML = wrong.length ? wrong.join('') : '<p class="empty">Không có trường hợp sai nào được ghi nhận.</p>';
    bindMoments($('hitList'));
    bindMoments($('missList'));
  }

  function renderMetrics() {
    const mf = state.session.minute_forecast;
    const metrics = mf.metrics || [];
    $('verdicts').innerHTML = metrics.map(m => {
      const v = skillVerdict(m);
      return `<div class="verdict ${v.tone}"><span>+${m.horizon_minutes} phút</span><strong>${v.title}</strong><span class="muted">${v.text}</span></div>`;
    }).join('');
    const rows = metrics.map(m => {
      if (!m.evaluated) return `<tr><td>+${m.horizon_minutes} phút</td><td colspan="7" class="muted">Chưa đủ video để kiểm chứng</td></tr>`;
      const skill = m.skill_vs_persistence;
      const onset = m.onset_model || {};
      const trendOnset = m.onset_trend || {};
      const caught = m.actual_onsets ? `${onset.true_positive}/${onset.true_positive + onset.false_negative} (${pct(onset.recall)}) · xu hướng ${pct(trendOnset.recall)}` : 'Không có lần chuyển sang tắc';
      const precision = onset.true_positive + onset.false_positive ? `${onset.true_positive}/${onset.true_positive + onset.false_positive} (${pct(onset.precision)})` : 'Không phát cảnh báo';
      return `<tr><td>+${m.horizon_minutes} phút</td><td>${m.evaluated} (${m.trained_anchors} sau khi đủ dữ liệu học)</td>
        <td>${num(m.mae_model, 3)}</td><td>${num(m.mae_persistence, 3)}</td><td>${num(m.mae_trend, 3)}</td>
        <td class="${skill >= 0.05 ? 'good' : skill <= -0.05 ? 'bad' : ''}">${skill === null ? '—' : pct(skill)}</td>
        <td>${caught}</td><td>${precision}</td></tr>`;
    }).join('');
    document.querySelector('#metricsTable tbody').innerHTML = rows;
    const libVideos = mf.method?.library_videos || 0;
    const lastAnchor = mf.anchors[mf.anchors.length - 1];
    const libPairs = lastAnchor?.horizons?.[0]?.library_pairs || 0;
    $('libraryLine').textContent = libVideos
      ? `Mô hình học thêm từ ${libVideos} video đã phân tích trước (${libPairs.toLocaleString('vi-VN')} cặp mẫu cho +5 phút). Video đang kiểm chứng không bao giờ được dùng để học trước thời điểm dự báo.`
      : 'Mô hình chỉ học từ phần đầu của chính video này. Phân tích trước 2–3 video khác của cùng camera để mô hình có thêm dữ liệu học; các video đó được lưu dưới dạng chuỗi số CI, không lưu hình ảnh.';
    $('metricsSub').textContent = 'Sai số trung bình tính trên thang CI 0–1. “Giữ nguyên” là đoán mức sau 5/10/15 phút bằng đúng mức hiện tại — mốc so sánh tối thiểu một mô hình dự báo phải vượt qua.';
  }

  function renderLimits() {
    const items = state.session.minute_forecast.limitations || [];
    $('limitList').innerHTML = items.map(text => `<li>${escapeHtml(text)}</li>`).join('');
  }

  function download(name, content, type) {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function baseName() {
    return (state.session?.filename || 'video').replace(/\.[^.]+$/, '').replace(/[^\w.-]+/g, '_').slice(0, 60);
  }

  // Wire up events.
  $('videoFile').addEventListener('change', e => setVideoFile(e.target.files?.[0]));
  $('analyzeBtn').addEventListener('click', analyze);
  $('resultFile').addEventListener('change', async e => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      loadSession(validateSession(JSON.parse(await file.text())));
      toast('Đã mở kết quả. Chọn đúng file video gốc ở mục 1 để phát.');
    } catch (error) { notice(error.message, 'error'); }
    e.target.value = '';
  });
  $('blindMode').addEventListener('change', e => { state.blind = e.target.checked; if (state.session) renderHighlights(); scheduleRender(); });
  $('unlockBtn').addEventListener('click', () => { state.locked = null; scheduleRender(); });
  $('exportJson').addEventListener('click', () => state.session && download(`${baseName()}.hanoiflow.json`, JSON.stringify(state.session), 'application/json'));
  $('exportCsv').addEventListener('click', () => state.session && download(`${baseName()}-du-bao.csv`, forecastCsv(state.session.minute_forecast.anchors), 'text/csv'));
  chart.addEventListener('click', seekFromChart);
  ['timeupdate', 'seeked', 'loadedmetadata'].forEach(name => player.addEventListener(name, scheduleRender));
  player.addEventListener('play', () => {
    // Playing from before a locked forecast's anchor means the user is starting over.
    if (state.locked && player.currentTime < state.locked.t) state.locked = null;
  });
  window.addEventListener('resize', scheduleRender);

  loadStatus();
  const pendingJob = sessionStorage.getItem(JOB_KEY);
  if (pendingJob && /^[a-f0-9]{32}$/.test(pendingJob)) {
    state.busy = true; renderInputs();
    setProgress(0, 'Đang theo dõi tiếp tác vụ phân tích trước đó…');
    poll(pendingJob).then(openResult).catch(error => notice(error.message, 'error'))
      .finally(() => { state.busy = false; renderInputs(); });
  }
}
