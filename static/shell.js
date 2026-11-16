/* hanoiFlow — sidebar navigation (one page per hash), mobile menu, and the model report page. */
'use strict';

const PAGES = { video: 'Dự báo & phương án', overview: 'Bản đồ & thời tiết', report: 'Đánh giá mô hình' };

function pageFromHash(hash) {
  const key = String(hash || '').replace(/^#/, '');
  if (key === 'how') return 'video';
  return Object.prototype.hasOwnProperty.call(PAGES, key) ? key : 'video';
}

if (typeof module !== 'undefined') module.exports = { pageFromHash, PAGES };

if (typeof document !== 'undefined') {
  const listeners = [];
  window.hanoiFlow = { onPage: fn => listeners.push(fn), page: null };
  const $ = id => document.getElementById(id);
  const escapeHtml = text => String(text ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  function show() {
    const page = pageFromHash(location.hash);
    document.querySelectorAll('.page').forEach(el => el.classList.toggle('active', el.id === `page-${page}`));
    document.querySelectorAll('.nav-item').forEach(el => el.classList.toggle('active', el.dataset.page === page));
    $('pageCrumb').textContent = PAGES[page];
    $('sidebar').classList.remove('open');
    if (location.hash === '#how') { $('how').open = true; setTimeout(() => $('how').scrollIntoView({ behavior: 'smooth' }), 50); }
    window.hanoiFlow.page = page;
    listeners.forEach(fn => { try { fn(page); } catch (e) { console.error(e); } });
    if (page === 'report') loadReport();
  }

  let reportLoaded = false;
  async function loadReport() {
    if (reportLoaded) return;
    try {
      const response = await fetch('/api/model/report', { cache: 'no-store' });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const report = await response.json();
      reportLoaded = true;
      const pct = v => (v === null || v === undefined ? '—' : `${v >= 0 ? '+' : '−'}${Math.abs(v * 100).toLocaleString('vi-VN', { maximumFractionDigits: 1 })}%`);
      $('reportSummary').innerHTML = report.summary.map(s => `<article class="metric-card"><span class="metric-label">Dự báo +${s.horizon_minutes} phút · so với giữ nguyên</span>
        <strong class="metric-value${s.skill_vs_persistence > 0 ? '' : ' neutral'}">${s.evaluated ? pct(s.skill_vs_persistence) : '—'}</strong>
        <span class="metric-foot">${s.evaluated ? `${s.evaluated} lần kiểm chứng trên ${s.videos ?? '?'} video · sai số ${s.mae_model.toFixed(3)} so với ${s.mae_persistence.toFixed(3)}` : 'chưa kiểm chứng được'}</span>
        ${s.skill_ci95 ? `<span class="metric-foot">Khoảng tin cậy 95%: ${pct(s.skill_ci95[0])} đến ${pct(s.skill_ci95[1])}</span>` : ''}</article>`).join('');
      $('reportRows').innerHTML = report.records.map(r => {
        const cells = [5, 10, 15].map(h => {
          const m = (r.metrics || []).find(x => x.horizon_minutes === h) || {};
          if (!m.evaluated) return '<td class="muted">—</td>';
          const skill = m.skill_vs_persistence;
          return `<td class="${skill >= 0.05 ? 'good' : skill <= -0.05 ? 'bad' : ''}">${pct(skill)} <span class="muted">(${m.evaluated})</span></td>`;
        }).join('');
        return `<tr><td>${escapeHtml(r.filename)}</td><td>${escapeHtml(r.camera || 'Camera cố định')}</td>${cells}</tr>`;
      }).join('');
      $('reportCount').textContent = `${report.records.length} chuỗi camera`;
      $('reportDate').textContent = report.created_at ? `Tính lúc ${new Date(report.created_at).toLocaleString('vi-VN')}` : '';
    } catch (error) {
      $('reportSummary').innerHTML = `<p class="empty">Không tải được báo cáo: ${escapeHtml(error.message)}</p>`;
    }
  }

  $('mobileMenu').addEventListener('click', () => $('sidebar').classList.toggle('open'));
  $('reloadBtn').addEventListener('click', () => location.reload());
  window.addEventListener('hashchange', show);
  show();
}
