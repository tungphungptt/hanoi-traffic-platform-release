/* hanoiFlow B2B console: reads /api/b2b/overview and /api/routes. Plain script, no dependencies. */
'use strict';
(function () {
  const $ = id => document.getElementById(id);
  const esc = t => String(t ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const LEVEL = { free: 'Thông thoáng', busy: 'Đông', gridlock: 'Ùn tắc' };
  const COLOR = { free: '#4fd1a1', busy: '#f2b84b', gridlock: '#ff6b5e' };
  const PAGES = { overview: 'Tổng quan', cameras: 'Camera', routes: 'Tuyến thay thế', api: 'API & xuất dữ liệu' };
  const fmt = v => (v == null ? '—' : Number(v).toLocaleString('vi-VN', { minimumFractionDigits: 2, maximumFractionDigits: 2 }));
  const levelOf = ci => (ci >= 0.55 ? 'gridlock' : ci >= 0.30 ? 'busy' : 'free');
  let data = null;

  /* ---------- navigation ---------- */
  function show() {
    const key = (location.hash || '#overview').slice(1);
    const page = PAGES[key] ? key : 'overview';
    document.querySelectorAll('.page').forEach(el => el.classList.toggle('active', el.id === `page-${page}`));
    document.querySelectorAll('.side nav a').forEach(a => a.classList.toggle('active', a.dataset.page === page));
    $('crumb').textContent = PAGES[page];
    $('side').classList.remove('open'); $('menu').setAttribute('aria-expanded', 'false');
    window.scrollTo(0, 0);
  }
  $('menu').addEventListener('click', () => { const open = $('side').classList.toggle('open'); $('menu').setAttribute('aria-expanded', String(open)); });
  window.addEventListener('hashchange', show);

  /* ---------- drawing ---------- */
  function spark(series, w, h, level) {
    if (!series || series.length < 2) return '';
    const x = i => (i / (series.length - 1)) * w, y = v => h - 2 - Math.min(1, Math.max(0, v)) * (h - 4);
    const pts = series.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
    const guide = t => `<line x1="0" x2="${w}" y1="${y(t)}" y2="${y(t)}" stroke="#1e3337" stroke-dasharray="3 3"/>`;
    return `<svg class="spark" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="Diễn biến CI">${guide(0.30)}${guide(0.55)}<polyline points="${pts}" fill="none" stroke="${COLOR[level]}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/></svg>`;
  }
  const chip = h => `<span class="chip ${h.level}${h.warning ? ' warn' : ''}" title="${LEVEL[h.level]} · xác suất ùn tắc nặng ${Math.round(h.p_gridlock * 100)}%${h.warning ? ' · cảnh báo' : ''}">${fmt(h.ci)}</span>`;

  /* ---------- overview ---------- */
  function render() {
    const t = data.totals, cams = data.cameras.slice().sort((a, b) => b.now_ci - a.now_ci);
    $('camCount').textContent = t.cameras;
    $('statusNote').textContent = `${t.cameras} camera · bản ghi sẵn`;
    $('kpis').innerHTML = [
      ['Camera theo dõi', t.cameras, 'ok', 'trong các bản ghi đã phân tích'],
      ['Đang ùn tắc', t.gridlock_now, t.gridlock_now ? 'hot' : 'ok', 'CI từ 0,55 trở lên'],
      ['Đang đông', t.busy_now, 'warm', 'CI từ 0,30 đến dưới 0,55'],
      ['Cảnh báo dự báo', t.warnings, t.warnings ? 'hot' : 'ok', 'tổng ở các mốc +5, +10, +15 phút'],
    ].map(([k, v, c, s]) => `<div class="kpi ${c}"><span>${k}</span><strong>${v}</strong><small>${s}</small></div>`).join('');
    $('rows').innerHTML = cams.map(c => `<tr><td class="cam">${esc(c.camera)}</td>
      <td><span class="pill ${c.level}">${fmt(c.now_ci)} <small>${LEVEL[c.level]}</small></span></td>
      <td>${spark(c.series, 120, 30, c.level)}</td>${c.horizons.map(h => `<td>${chip(h)}</td>`).join('')}</tr>`).join('');
    const hot = cams.map(c => ({ c, h: c.horizons.filter(x => x.warning || x.watch).sort((a, b) => b.p_gridlock - a.p_gridlock)[0] })).filter(x => x.h).sort((a, b) => b.h.p_gridlock - a.h.p_gridlock).slice(0, 6);
    $('alerts').innerHTML = hot.length ? hot.map(({ c, h }) => `<li><strong>${esc(c.camera)}</strong><span>${h.warning ? 'Cảnh báo' : 'Theo dõi'} sau ${h.minutes} phút · CI dự báo ${fmt(h.ci)} · xác suất ùn tắc nặng ${Math.round(h.p_gridlock * 100)}%</span></li>`).join('')
      : '<li class="empty">Không có camera nào cần chú ý.</li>';
    $('cams').innerHTML = cams.map(c => `<article class="cam-card"><header><h3>${esc(c.camera)}</h3><span class="pill ${c.level}">${fmt(c.now_ci)}</span></header>
      ${spark(c.series, 300, 70, c.level)}
      <div class="cam-stats"><span>Đỉnh <b>${fmt(c.peak_ci)}</b></span><span>Trung bình <b>${fmt(c.mean_ci)}</b></span><span>${LEVEL[c.level]}</span></div>
      <div class="cam-hz">${c.horizons.map(h => `<div><span>+${h.minutes} phút</span><strong style="color:${COLOR[h.level]}">${fmt(h.ci)}</strong><small>ùn tắc nặng ${Math.round(h.p_gridlock * 100)}%</small></div>`).join('')}</div></article>`).join('');
  }

  /* ---------- routes ---------- */
  const PLACES = [
    ['Hồ Gươm (Hoàn Kiếm)', 21.0285, 105.8523], ['Bến xe Mỹ Đình', 21.0286, 105.7787], ['Sân bay Nội Bài', 21.2212, 105.8072],
    ['Ga Hà Nội', 21.0245, 105.8412], ['Ngã Tư Sở', 21.0080, 105.8196], ['216 Trần Duy Hưng', 21.0075, 105.7990],
    ['Hồ Tùng Mậu', 21.0378, 105.7770], ['109 Xuân Diệu', 21.0650, 105.8285], ['Hà Đông (Quang Trung)', 20.9760, 105.7790],
    ['1111 Giải Phóng', 20.9620, 105.8410], ['Aeon Mall Long Biên', 21.0273, 105.8999],
  ];
  const fmtKm = m => `${(m / 1000).toLocaleString('vi-VN', { maximumFractionDigits: 1 })} km`;
  const fmtMin = s => { const m = Math.round(s / 60); return m >= 60 ? `${Math.floor(m / 60)} giờ ${m % 60} phút` : `${m} phút`; };
  const ROUTE_COLORS = ['#c6f04a', '#4fd1c5', '#f2b84b'];
  PLACES.forEach(([n], i) => { $('from').add(new Option(n, i)); $('to').add(new Option(n, i)); });
  $('from').value = 1; $('to').value = 0;

  function drawRoutes(routes, sel, a, b) {
    const all = routes.flatMap(r => r.geometry.coordinates);
    const lons = all.map(p => p[0]), lats = all.map(p => p[1]);
    const x0 = Math.min(...lons), x1 = Math.max(...lons), y0 = Math.min(...lats), y1 = Math.max(...lats);
    const k = Math.cos(((y0 + y1) / 2) * Math.PI / 180), W = 640, pad = 24;
    const sx = (W - 2 * pad) / Math.max((x1 - x0) * k, 1e-6), sy = sx, H = Math.max(200, (y1 - y0) * sy + 2 * pad);
    const px = ([lon, lat]) => [pad + (lon - x0) * k * sy, H - pad - (lat - y0) * sy];
    const line = r => r.geometry.coordinates.map(p => px(p).map(v => v.toFixed(1)).join(',')).join(' ');
    const order = routes.map((_, i) => i).sort((p, q) => (p === sel) - (q === sel));
    const dot = (p, c) => { const [x, y] = px(p); return `<circle cx="${x}" cy="${y}" r="7" fill="${c}" stroke="#0a1214" stroke-width="3"/>`; };
    $('routeMap').innerHTML = `<svg viewBox="0 0 ${W} ${H.toFixed(0)}" role="img" aria-label="Các tuyến đường">${order.map(i => `<polyline points="${line(routes[i])}" fill="none" stroke="${ROUTE_COLORS[i % 3]}" stroke-width="${i === sel ? 5 : 3}" opacity="${i === sel ? 1 : .55}" stroke-linejoin="round" stroke-linecap="round"/>`).join('')}${dot([a[1], a[0]], '#fff')}${dot([b[1], b[0]], '#ff6b5e')}</svg>`;
  }
  $('go').addEventListener('click', async () => {
    const A = PLACES[$('from').value], B = PLACES[$('to').value];
    if (A === B) { $('routeList').innerHTML = '<p class="note">Chọn hai điểm khác nhau.</p>'; return; }
    $('go').disabled = true; $('routeList').innerHTML = '<p class="note">Đang tìm tuyến…</p>';
    try {
      const r = await fetch(`/api/routes?from=${A[1]},${A[2]}&to=${B[1]},${B[2]}`);
      const p = await r.json();
      if (!r.ok) throw new Error(p.detail || `HTTP ${r.status}`);
      const pick = i => {
        $('routeList').querySelectorAll('.route-opt').forEach((b, j) => { b.classList.toggle('sel', j === i); });
        drawRoutes(p.routes, i, [A[1], A[2]], [B[1], B[2]]);
      };
      $('routeList').innerHTML = p.routes.map((rt, i) => `<button type="button" class="route-opt" data-i="${i}"><strong style="color:${ROUTE_COLORS[i % 3]}">Tuyến ${i + 1} · ${fmtMin(rt.duration_seconds)}</strong><span>${fmtKm(rt.distance_m)}${i === 0 ? ' · nhanh nhất' : ''}</span></button>`).join('') + `<p class="note">${esc(p.note)}</p>`;
      $('routeList').querySelectorAll('.route-opt').forEach(b => b.addEventListener('click', () => pick(Number(b.dataset.i))));
      pick(0);
    } catch (e) {
      $('routeList').innerHTML = `<p class="note">Không tìm được tuyến: ${esc(e.message)}</p>`;
    } finally { $('go').disabled = false; }
  });

  /* ---------- export ---------- */
  function download(name, type, text) {
    const url = URL.createObjectURL(new Blob([text], { type }));
    const a = document.createElement('a'); a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  $('csv').addEventListener('click', () => {
    if (!data) return;
    const q = v => `"${String(v).replace(/"/g, '""')}"`;
    const head = ['camera', 'ci_hien_tai', 'muc', 'ci_dinh', 'ci_tb', 'ci_5p', 'ci_10p', 'ci_15p', 'xs_un_tac_15p'];
    const rows = data.cameras.map(c => [q(c.camera), c.now_ci, c.level, c.peak_ci, c.mean_ci, ...c.horizons.map(h => h.ci), c.horizons[2]?.p_gridlock ?? ''].join(','));
    download('hanoiflow-camera.csv', 'text/csv;charset=utf-8', '﻿' + [head.join(','), ...rows].join('\n'));
  });
  $('json').addEventListener('click', () => data && download('hanoiflow-camera.json', 'application/json', JSON.stringify(data, null, 2)));
  $('curl').textContent = `# Tổng quan camera\ncurl ${location.origin}/api/b2b/overview\n\n# Tuyến Mỹ Đình → Hồ Gươm\ncurl "${location.origin}/api/routes?from=21.0286,105.7787&to=21.0285,105.8523"\n\n# Gửi số liệu trực tiếp (cần LIVE_TOKEN)\ncurl -X POST ${location.origin}/api/live/cam-01/samples \\\n  -H "X-Live-Token: <token>" -H "Content-Type: application/json" \\\n  -d '{"name":"Camera 01","lat":21.0075,"lng":105.799,"samples":[{"t":1700000000,"ci":0.42}]}'`;

  /* ---------- boot ---------- */
  show();
  fetch('/api/b2b/overview').then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
    .then(d => { data = d; render(); })
    .catch(e => { $('statusNote').textContent = 'Không tải được dữ liệu'; $('rows').innerHTML = `<tr><td colspan="6" class="muted">Không tải được dữ liệu: ${esc(e.message)}</td></tr>`; });
})();
