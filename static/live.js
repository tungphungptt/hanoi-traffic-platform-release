/* hanoiFlow — live page: forecasts of cameras that have a collector sending measurements. */
'use strict';

const LIVE_REFRESH_MS = 15000;
const LIVE_COLOR = { free: '#78a984', busy: '#d3a052', gridlock: '#a94d43' };
const LIVE_LABEL = { free: 'Thông thoáng', busy: 'Đông đúc', gridlock: 'Tắc nghẽn' };

function liveAge(seconds) {
  if (!Number.isFinite(seconds)) return '—';
  if (seconds < 90) return `${Math.round(seconds)} giây trước`;
  if (seconds < 5400) return `${Math.round(seconds / 60)} phút trước`;
  return `${(seconds / 3600).toFixed(1).replace('.', ',')} giờ trước`;
}

function liveSparkline(recent, busy = 0.30, gridlock = 0.55) {
  const pts = (recent || []).filter(p => Number.isFinite(p.ci));
  if (pts.length < 2) return '';
  const w = 260, h = 54, t0 = pts[0].epoch, t1 = pts[pts.length - 1].epoch || t0 + 1;
  const x = t => ((t - t0) / Math.max(1, t1 - t0)) * w;
  const y = v => h - Math.min(1, Math.max(0, v)) * h;
  const path = pts.map((p, i) => `${i ? 'L' : 'M'}${x(p.epoch).toFixed(1)},${y(p.ci).toFixed(1)}`).join(' ');
  return `<svg viewBox="0 0 ${w} ${h}" class="live-spark" role="img" aria-label="CI trong giờ qua" preserveAspectRatio="none">
    <line x1="0" x2="${w}" y1="${y(busy)}" y2="${y(busy)}" stroke="${LIVE_COLOR.busy}" stroke-dasharray="3 3" stroke-width="1"/>
    <line x1="0" x2="${w}" y1="${y(gridlock)}" y2="${y(gridlock)}" stroke="${LIVE_COLOR.gridlock}" stroke-dasharray="3 3" stroke-width="1"/>
    <path d="${path}" fill="none" stroke="#4a3b30" stroke-width="1.6"/></svg>`;
}

function liveCard(cam, esc) {
  const name = esc(cam.name || cam.camera_id);
  if (cam.status === 'warming_up') {
    const need = cam.needed_seconds ? ` (cần khoảng ${Math.ceil(cam.needed_seconds / 60)} phút)` : '';
    return `<article class="panel live-card"><h3>${name}</h3><p class="muted">Đang thu số liệu: ${Math.floor((cam.span_seconds || 0) / 60)} phút${need}. Dự báo sẽ hiện khi đủ số liệu liên tục.</p></article>`;
  }
  const level = cam.level_key;
  const stale = cam.status === 'stale';
  const rows = (cam.horizons || []).map(h => {
    const flag = h.warning ? '<span class="warn-flag">⚠ Cảnh báo sắp tắc</span>' : h.watch ? '<span class="watch-flag">◔ Theo dõi</span>' : '';
    return `<div class="live-h"><span class="live-h-label">+${h.horizon_minutes} phút</span>
      <span class="level-chip ${h.level_key}">${LIVE_LABEL[h.level_key]}</span>
      <span class="muted">CI ${h.ci.toFixed(2)} (${h.lower.toFixed(2)}–${h.upper.toFixed(2)}) · tắc ${Math.round(h.p_gridlock * 100)}%</span>${flag}</div>`;
  }).join('');
  return `<article class="panel live-card${stale ? ' stale' : ''}">
    <div class="live-head"><h3>${name}</h3><span class="level-chip ${level}">${LIVE_LABEL[level]} · CI ${cam.observed_ci.toFixed(2)}</span></div>
    <p class="live-meta">${stale ? '<strong class="bad">Mất tín hiệu</strong> · ' : ''}Cập nhật ${liveAge(cam.age_seconds)} · ${cam.samples.toLocaleString('vi-VN')} mẫu</p>
    ${liveSparkline(cam.recent, cam.thresholds?.busy, cam.thresholds?.gridlock)}
    <div class="live-horizons">${stale ? '<p class="muted">Dự báo dưới đây là của lần đo cuối, không còn là “bây giờ”.</p>' : ''}${rows}</div>
  </article>`;
}

if (typeof module !== 'undefined') module.exports = { liveAge, liveSparkline, liveCard };

if (typeof document !== 'undefined') {
  const $ = id => document.getElementById(id);
  const esc = text => String(text ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  let timer = null;

  async function refresh() {
    try {
      const response = await fetch('/api/live', { cache: 'no-store' });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      $('liveGrid').innerHTML = data.cameras.map(c => liveCard(c, esc)).join('');
      $('liveHelp').hidden = data.cameras.length > 0;
      $('liveHelpTitle').textContent = data.enabled ? 'Chưa có camera nào gửi số liệu' : 'Máy chủ chưa bật nhận số liệu trực tiếp';
    } catch (error) {
      $('liveGrid').innerHTML = `<p class="empty">Không tải được dữ liệu trực tiếp: ${esc(error.message)}</p>`;
    }
  }

  const onPage = page => {
    clearInterval(timer);
    timer = null;
    if (page === 'live') { refresh(); timer = setInterval(refresh, LIVE_REFRESH_MS); }
  };
  window.hanoiFlow.onPage(onPage);
  onPage(window.hanoiFlow.page);  // the page may already be showing when this script loads
}
