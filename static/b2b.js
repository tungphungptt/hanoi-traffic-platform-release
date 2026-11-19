/* hanoiFlow cho khu công nghiệp. Đọc /api/b2b/overview; cấu hình (giờ ca, người nhận, cổng) lưu trên trình duyệt. */
'use strict';
(function () {
  const $ = id => document.getElementById(id);
  const esc = t => String(t ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const LEVEL = { free: 'Thông thoáng', busy: 'Đông đúc', gridlock: 'Ùn tắc' };
  const COLOR = { free: '#8fd9a0', busy: '#f0a54a', gridlock: '#f27d70' };
  const PAGES = { today: 'Hôm nay', map: 'Bản đồ KCN', report: 'Báo cáo', gates: 'Cổng và camera', shifts: 'Giờ ra vào ca', people: 'Người nhận cảnh báo', api: 'Tích hợp (API)' };
  const fmt = v => (v == null ? '—' : Number(v).toLocaleString('vi-VN', { minimumFractionDigits: 2, maximumFractionDigits: 2 }));
  const pct = v => `${Math.round((v || 0) * 100)}%`;
  const levelOf = ci => (ci >= 0.55 ? 'gridlock' : ci >= 0.30 ? 'busy' : 'free');
  const hhmm = m => { const x = ((Math.round(m) % 1440) + 1440) % 1440; return `${String(Math.floor(x / 60)).padStart(2, '0')}:${String(x % 60).padStart(2, '0')}`; };
  const toMin = s => { const [h, m] = String(s).split(':').map(Number); return h * 60 + m; };

  /* ---------- stored settings ---------- */
  const store = {
    get(key, fallback) { try { const v = JSON.parse(localStorage.getItem(key)); return v ?? fallback; } catch (e) { return fallback; } },
    set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch (e) { /* private mode: keep in memory only */ } },
  };
  const DEFAULT_GATES = [
    { id: 'main', name: 'Cổng chính', cam: 'nguyen_van_cu:0' },
    { id: 'g2', name: 'Cổng 2', cam: 'le_hong_phong:0' },
    { id: 'walk', name: 'Cổng phụ (đi bộ)', cam: 'quang_trung:0' },
    { id: 'feeder', name: 'Nút giao đường gom', cam: 'ho_tung_mau:0' },
  ];
  let gates = store.get('hf_kcn_gates', DEFAULT_GATES);
  let shifts = store.get('hf_kcn_shifts', [{ name: 'Nhà máy A', end: '17:00' }, { name: 'Nhà máy B', end: '17:00' }, { name: 'Nhà máy C', end: '17:30' }]);
  let people = store.get('hf_kcn_people', [{ name: 'Trưởng bảo vệ', contact: '' }, { name: 'Phòng hành chính', contact: '' }]);
  let data = null;
  let parkMap, parkBoundary;

  /* ---------- navigation ---------- */
  function show() {
    const key = (location.hash || '#today').slice(1);
    const page = PAGES[key] ? key : 'today';
    document.querySelectorAll('.page').forEach(el => el.classList.toggle('active', el.id === `page-${page}`));
    document.querySelectorAll('.side nav a').forEach(a => a.classList.toggle('active', a.dataset.page === page));
    $('crumb').textContent = PAGES[page];
    $('side').classList.remove('open'); $('menu').setAttribute('aria-expanded', 'false');
    window.scrollTo(0, 0);
    if (page === 'map') {
      initParkMap();
      if (parkMap) setTimeout(() => parkMap.invalidateSize(), 0);
    }
  }
  $('menu').addEventListener('click', () => { const open = $('side').classList.toggle('open'); $('menu').setAttribute('aria-expanded', String(open)); });
  window.addEventListener('hashchange', show);

  function initParkMap() {
    if (parkMap) return;
    if (typeof L === 'undefined') {
      $('parkMapStatus').textContent = 'Không tải được thư viện bản đồ. Hãy thử tải lại trang.';
      return;
    }
    parkMap = L.map('parkMap', { scrollWheelZoom: false }).setView([21.0450097, 106.0733952], 14);
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a>',
    }).addTo(parkMap);
    $('parkFit').addEventListener('click', () => parkBoundary && parkMap.fitBounds(parkBoundary.getBounds(), { padding: [28, 28], maxZoom: 16 }));
    fetch('/static/industrial_park_thuan_thanh_iii.geojson')
      .then(response => { if (!response.ok) throw new Error(`HTTP ${response.status}`); return response.json(); })
      .then(data => {
        parkBoundary = L.geoJSON(data, {
          style: { color: '#d9b56d', weight: 3, fillColor: '#b08a3e', fillOpacity: 0.3 },
          onEachFeature: (feature, layer) => layer.bindPopup(`<strong>${esc(feature.properties.name)}</strong><br>Ranh giới tham khảo từ OpenStreetMap.`),
        }).addTo(parkMap);
        parkMap.fitBounds(parkBoundary.getBounds(), { padding: [28, 28], maxZoom: 16 });
        $('parkMapStatus').textContent = 'Thuận Thành, Bắc Ninh · OpenStreetMap';
      })
      .catch(error => { $('parkMapStatus').textContent = `Không tải được ranh giới KCN: ${error.message}`; });
  }
  window.addEventListener('leaflet-ready', () => { if ((location.hash || '#today') === '#map') initParkMap(); });

  /* ---------- derived data ---------- */
  function camById(id) { return data && data.cameras.find(c => c.id === id); }
  function enrich() {
    return gates.map(g => {
      const cam = camById(g.cam) || null;
      const h15 = cam ? cam.horizons.find(h => h.minutes === 15) || cam.horizons[cam.horizons.length - 1] : null;
      return { ...g, camData: cam, level: cam ? cam.level : null, now: cam ? cam.now_ci : null, p15: h15 ? h15.p_gridlock : 0,
        lvl15: h15 ? h15.level : null, ci15: h15 ? h15.ci : null, warn: !!(cam && cam.horizons.some(h => h.warning)) };
    });
  }
  function nextShift(now) {
    const nowM = now.getHours() * 60 + now.getMinutes();
    const ends = [...new Set(shifts.map(s => toMin(s.end)))].sort((a, b) => a - b);
    if (!ends.length) return null;
    const end = ends.find(m => m > nowM) ?? ends[0] + 1440;
    return { end, nowM, names: shifts.filter(s => toMin(s.end) === end % 1440).map(s => s.name), left: end - nowM };
  }
  const joinNames = n => (n.length > 2 ? `${n.slice(0, -1).join(', ')} và ${n[n.length - 1]}` : n.join(' và '));

  /* ---------- drawing ---------- */
  function spark(series, w, h, level) {
    if (!series || series.length < 2) return '';
    const x = i => (i / (series.length - 1)) * w, y = v => h - 2 - Math.min(1, Math.max(0, v)) * (h - 4);
    const pts = series.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
    return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="Diễn biến CI"><polyline points="${pts}" fill="none" stroke="${COLOR[level]}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/></svg>`;
  }
  function ring(p) {
    const r = 26, c = 2 * Math.PI * r, col = p >= 0.5 ? COLOR.gridlock : p >= 0.25 ? COLOR.busy : COLOR.free;
    return `<figure class="ring"><svg width="68" height="68" viewBox="0 0 68 68" role="img" aria-label="Nguy cơ ùn tắc ${pct(p)}"><circle cx="34" cy="34" r="${r}" fill="none" stroke="rgba(255,240,210,.12)" stroke-width="6"/><circle cx="34" cy="34" r="${r}" fill="none" stroke="${col}" stroke-width="6" stroke-linecap="round" stroke-dasharray="${(c * Math.min(1, p)).toFixed(1)} ${c.toFixed(1)}" transform="rotate(-90 34 34)"/><text x="34" y="39" text-anchor="middle" font-size="15" font-weight="700" fill="#f5eedf">${pct(p)}</text></svg><figcaption>nguy cơ ùn tắc</figcaption></figure>`;
  }
  const badge = lvl => (lvl ? `<span class="badge ${lvl}">${LEVEL[lvl]}</span>` : '<span class="badge busy">Chưa có camera</span>');

  function renderToday(now) {
    const ns = nextShift(now), g = enrich();
    $('zoneName').textContent = 'KCN Demo';
    $('todaySub').textContent = `${now.toLocaleDateString('vi-VN', { weekday: 'long', day: '2-digit', month: '2-digit' })} · cập nhật ${hhmm(now.getHours() * 60 + now.getMinutes())}`;
    const box = $('nextShift');
    if (!ns) { box.innerHTML = '<span>Chưa có giờ tan ca</span><strong>—</strong><small>Thêm ở mục Giờ ra vào ca</small>'; }
    else box.innerHTML = `<span>Tan ca tiếp theo</span><strong>${hhmm(ns.end)}</strong><small>còn ${ns.left >= 60 ? `${Math.floor(ns.left / 60)} giờ ${ns.left % 60} phút` : `${ns.left} phút`} · ${esc(joinNames(ns.names))}</small>`;
    drawTimeline(ns);
    $('gateGrid').innerHTML = g.map(x => {
      const cam = x.camData;
      return `<article class="gate${x.warn || x.level === 'gridlock' ? ' warn' : ''}"><div class="gate-top"><h3>${esc(x.name)}</h3><span class="cam">${cam ? esc(cam.camera) : 'Chưa gắn camera'}</span></div>
        <div class="gate-mid"><div class="gate-state"><small>Bây giờ</small><strong class="${x.level || 'busy'}">${x.level ? LEVEL[x.level] : '—'}</strong></div>${cam ? ring(x.p15) : ''}</div>
        <div class="gate-foot"><span>15 phút tới <b>${x.lvl15 ? LEVEL[x.lvl15] : '—'}</b></span><span>Cao điểm <b>${ns ? `${hhmm(ns.end)} – ${hhmm(ns.end + 50)}` : '—'}</b></span></div></article>`;
    }).join('');
    const out = renderTodo(g, ns);
    const hr = now.getHours();
    $('greet').textContent = hr < 11 ? 'Chào buổi sáng' : hr < 14 ? 'Chào buổi trưa' : hr < 18 ? 'Chào buổi chiều' : 'Chào buổi tối';
    $('lede').textContent = !ns ? 'Hãy thêm giờ tan ca để hanoiFlow báo trước cho bạn.'
      : `Còn ${ns.left >= 60 ? `${Math.floor(ns.left / 60)} giờ ${ns.left % 60} phút` : `${ns.left} phút`} nữa là tan ca. ` + (out.risky ? `${out.worst.name} có nguy cơ tắc cao (${pct(out.worst.p15)}), nên ${out.first.charAt(0).toLowerCase()}${out.first.slice(1)}.` : 'Các cổng đang thông thoáng, chưa cần can thiệp.');
  }

  function drawTimeline(ns) {
    if (!ns) { $('timeline').innerHTML = '<p class="muted">Thêm giờ tan ca để xem biểu đồ.</p>'; $('tlNote').textContent = ''; return; }
    const start = ns.end - 120, span = 240, p = m => `${((m - start) / span) * 100}%`, w = (a, b) => `${((b - a) / span) * 100}%`;
    let html = `<div class="tl-track"><div class="tl-seg busy" style="left:${p(ns.end - 10)};width:${w(ns.end - 10, ns.end)}"></div><div class="tl-seg jam" style="left:${p(ns.end)};width:${w(ns.end, ns.end + 50)}"></div></div>`;
    for (let m = Math.ceil(start / 60) * 60; m <= start + span; m += 60) html += `<span class="tl-tick" style="left:${p(m)}">${hhmm(m)}</span>`;
    const groups = {};
    shifts.forEach(s => { (groups[s.end] = groups[s.end] || []).push(s.name); });
    Object.entries(groups).sort((a, b) => toMin(a[0]) - toMin(b[0])).forEach(([end, names], gi) => {
      const base = [0, 1440, -1440].map(d => toMin(end) + d).find(m => m >= start - 1 && m <= start + span);
      if (base == null) return;
      html += `<span class="tl-mark ${gi % 2 ? 'r' : 'l'}" style="left:${p(base)}">${esc(joinNames(names))} · ${esc(end)}</span>`;
    });
    if (ns.nowM >= start && ns.nowM <= start + span) html += `<span class="tl-now" style="left:${p(ns.nowM)}"><span>Bây giờ</span></span>`;
    $('timeline').innerHTML = html;
    $('tlNote').textContent = 'Cao điểm tính theo giờ ca bạn cài, không phải dự báo của camera';
  }

  function renderTodo(g, ns) {
    const items = [], withCam = g.filter(x => x.camData);
    const road = withCam.filter(x => x.id !== 'walk');
    const worst = road.slice().sort((a, b) => b.p15 - a.p15)[0], best = road.slice().sort((a, b) => a.p15 - b.p15)[0];
    const walk = g.find(x => x.id === 'walk');
    if (ns && worst) {
      const risky = worst.level !== 'free' || worst.warn || worst.p15 >= 0.2;
      if (walk && risky) items.push({ t: ns.end - 10, h: `Mở ${walk.name} cho công nhân đi bộ`, d: `${worst.name} có ${pct(worst.p15)} khả năng ùn tắc sau 15 phút${walk.level ? `; ${walk.name} đang ${LEVEL[walk.level].toLowerCase()}` : ''}.` });
      if (best && best !== worst) items.push({ t: ns.end - 5, h: `Xe đưa đón ${joinNames(ns.names)} đón tại ${best.name}`, d: `Tránh dòng xe dồn ra ${worst.name} cùng lúc. ${best.name} đang ${LEVEL[best.level].toLowerCase()}.` });
      if (risky) items.push({ t: ns.end, h: `Tăng bảo vệ tại ${worst.name}`, d: `Dự báo ${LEVEL[worst.lvl15].toLowerCase()} sau 15 phút (${pct(worst.p15)} khả năng ùn tắc).` });
    }
    $('todo').innerHTML = items.length ? items.map(i => `<li><time>${hhmm(i.t)}</time><div><strong>${esc(i.h)}</strong><span>${esc(i.d)}</span></div></li>`).join('')
      : '<li><time>—</time><div><strong>Chưa cần can thiệp</strong><span>Các cổng đang thông thoáng và chưa có cảnh báo trong 15 phút tới.</span></div></li>';
    const hot = worst && (worst.warn || worst.p15 >= 0.4);
    const who = people.map(p => p.name).join(', ') || 'chưa có người nhận';
    $('alertPreview').innerHTML = `<div class="card-head"><h2>Tin cảnh báo sẽ gửi</h2><span class="muted">Gửi tới: ${esc(who)}</span></div>
      <div class="alert-card${hot ? '' : ' ok'}"><span class="tag">${hot ? 'SẮP TẮC' : 'ỔN'}</span>
      <p><b>${hot ? `${esc(worst.name)}: ${pct(worst.p15)} khả năng tắc sau 15 phút` : 'Chưa có cổng nào sắp tắc'}</b></p>
      <p>${hot ? `Nên làm: ${esc(items[1] ? items[1].h : items[0] ? items[0].h : 'theo dõi cổng')}.` : 'Hệ thống sẽ nhắn ngay khi một cổng có nguy cơ tắc.'}</p></div>`;
    const risky = !!(worst && (worst.level !== 'free' || worst.warn || worst.p15 >= 0.2));
    return { items, worst, risky, first: items[0] ? items[0].h : 'theo dõi các cổng' };
  }

  /* ---------- report ---------- */
  function buckets(series, n) {
    const out = [];
    for (let i = 0; i < n; i++) {
      const a = Math.floor(i * series.length / n), b = Math.max(a + 1, Math.floor((i + 1) * series.length / n)), part = series.slice(a, b);
      out.push(part.reduce((s, v) => s + v, 0) / part.length);
    }
    return out;
  }
  const heatClass = ci => (ci >= 0.70 ? 'severe' : ci >= 0.55 ? 'jam' : ci >= 0.30 ? 'busy' : 'free');
  function renderReport() {
    const g = enrich().filter(x => x.camData), N = 8;
    $('reportSub').textContent = 'KCN Demo · số đo từ bản ghi camera đã phân tích (không phải số liệu theo tuần)';
    if (!g.length) { $('findings').innerHTML = '<li>Chưa gắn camera nào cho các cổng.</li>'; return; }
    const byPeak = g.slice().sort((a, b) => b.camData.peak_ci - a.camData.peak_ci), byMean = g.slice().sort((a, b) => a.camData.mean_ci - b.camData.mean_ci);
    const warnN = g.filter(x => x.warn).length;
    $('findings').innerHTML = [
      `<b>${esc(byPeak[0].name)}</b> nặng nhất: CI đỉnh ${fmt(byPeak[0].camData.peak_ci)} (${LEVEL[levelOf(byPeak[0].camData.peak_ci)].toLowerCase()}).`,
      `<b>${esc(byMean[0].name)}</b> còn thoáng nhất: CI trung bình ${fmt(byMean[0].camData.mean_ci)}. Có thể điều bớt xe sang cổng này.`,
      warnN ? `<b>${warnN}/${g.length} cổng</b> đang có cảnh báo tắc trong 15 phút tới.` : 'Chưa có cổng nào cảnh báo tắc trong 15 phút tới.',
    ].map(t => `<li>${t}</li>`).join('');
    $('heat').innerHTML = `<thead><tr><th>Cổng</th>${Array.from({ length: N }, (_, i) => `<th>Đoạn ${i + 1}</th>`).join('')}</tr></thead><tbody>${g.map(x => `<tr><td>${esc(x.name)}</td>${buckets(x.camData.series, N).map(v => `<td class="c ${heatClass(v)}">${fmt(v)}</td>`).join('')}</tr>`).join('')}</tbody>`;
    $('sumTable').innerHTML = `<thead><tr><th>Cổng</th><th>Camera</th><th>Bây giờ</th><th>Đỉnh</th><th>Trung bình</th><th>+15 phút</th><th>Khả năng ùn tắc</th></tr></thead><tbody>${g.map(x => `<tr><td><b>${esc(x.name)}</b></td><td>${esc(x.camData.camera)}</td><td>${LEVEL[x.level]} · ${fmt(x.now)}</td><td>${fmt(x.camData.peak_ci)}</td><td>${fmt(x.camData.mean_ci)}</td><td>${LEVEL[x.lvl15]}</td><td class="${x.p15 >= 0.5 ? 'bad' : ''}">${pct(x.p15)}</td></tr>`).join('')}</tbody>`;
  }

  /* ---------- gates, shifts, people ---------- */
  function renderGates() {
    const opts = sel => data.cameras.map(c => `<option value="${esc(c.id)}"${c.id === sel ? ' selected' : ''}>${esc(c.camera)}</option>`).join('');
    const g = enrich();
    $('gateTable').innerHTML = `<thead><tr><th>Cổng</th><th>Camera</th><th>Bây giờ</th><th>Diễn biến</th></tr></thead><tbody>${g.map(x => `<tr><td><b>${esc(x.name)}</b></td><td><select data-gate="${esc(x.id)}" aria-label="Camera cho ${esc(x.name)}">${opts(x.cam)}</select></td><td>${badge(x.level)}</td><td>${x.camData ? spark(x.camData.series, 120, 30, x.level) : ''}</td></tr>`).join('')}</tbody>`;
  }
  $('gateTable').addEventListener('change', e => {
    const id = e.target.dataset && e.target.dataset.gate; if (!id) return;
    gates = gates.map(g => (g.id === id ? { ...g, cam: e.target.value } : g)); store.set('hf_kcn_gates', gates); renderAll();
  });
  function renderShifts() {
    $('shiftTable').innerHTML = `<thead><tr><th>Nhà máy</th><th>Giờ tan ca</th><th></th></tr></thead><tbody>${shifts.map((s, i) => `<tr><td>${esc(s.name)}</td><td>${esc(s.end)}</td><td><button class="btn ghost small" data-del-shift="${i}" type="button">Xóa</button></td></tr>`).join('') || '<tr><td colspan="3" class="muted">Chưa có giờ tan ca.</td></tr>'}</tbody>`;
  }
  $('shiftTable').addEventListener('click', e => { const i = e.target.dataset && e.target.dataset.delShift; if (i == null) return; shifts.splice(Number(i), 1); store.set('hf_kcn_shifts', shifts); renderAll(); });
  $('shiftForm').addEventListener('submit', e => { e.preventDefault(); shifts.push({ name: $('shName').value.trim(), end: $('shTime').value }); store.set('hf_kcn_shifts', shifts); $('shName').value = ''; renderAll(); });
  function renderPeople() {
    $('peopleTable').innerHTML = `<thead><tr><th>Người nhận</th><th>Liên hệ</th><th></th></tr></thead><tbody>${people.map((p, i) => `<tr><td>${esc(p.name)}</td><td>${esc(p.contact) || '<span class="muted">Chưa nhập</span>'}</td><td><button class="btn ghost small" data-del-person="${i}" type="button">Xóa</button></td></tr>`).join('') || '<tr><td colspan="3" class="muted">Chưa có người nhận.</td></tr>'}</tbody>`;
  }
  $('peopleTable').addEventListener('click', e => { const i = e.target.dataset && e.target.dataset.delPerson; if (i == null) return; people.splice(Number(i), 1); store.set('hf_kcn_people', people); renderAll(); });
  $('peopleForm').addEventListener('submit', e => { e.preventDefault(); people.push({ name: $('pName').value.trim(), contact: $('pContact').value.trim() }); store.set('hf_kcn_people', people); $('pName').value = ''; $('pContact').value = ''; renderAll(); });

  /* ---------- export ---------- */
  function download(name, type, text) {
    const url = URL.createObjectURL(new Blob([text], { type }));
    const a = document.createElement('a'); a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  $('csv').addEventListener('click', () => {
    if (!data) return;
    const q = v => `"${String(v).replace(/"/g, '""')}"`;
    const rows = enrich().filter(x => x.camData).map(x => [q(x.name), q(x.camData.camera), x.now, x.level, x.camData.peak_ci, x.camData.mean_ci, x.ci15, x.lvl15, x.p15].join(','));
    download('hanoiflow-kcn-cong.csv', 'text/csv;charset=utf-8', '﻿' + ['cong,camera,ci_hien_tai,muc,ci_dinh,ci_tb,ci_15p,muc_15p,xs_tac_nang_15p', ...rows].join('\n'));
  });
  $('print').addEventListener('click', () => window.print());
  $('json').addEventListener('click', () => data && download('hanoiflow-camera.json', 'application/json', JSON.stringify(data, null, 2)));
  $('curl').textContent = `# Tổng quan camera\ncurl ${location.origin}/api/b2b/overview\n\n# Gửi số liệu trực tiếp (cần LIVE_TOKEN)\ncurl -X POST ${location.origin}/api/live/cong-chinh/samples \\\n  -H "X-Live-Token: <token>" -H "Content-Type: application/json" \\\n  -d '{"name":"Cổng chính","lat":21.0075,"lng":105.799,"samples":[{"t":1700000000,"ci":0.42}]}'`;

  /* ---------- boot ---------- */
  function renderAll() {
    renderShifts(); renderPeople();
    if (!data) return;
    renderToday(new Date()); renderReport(); renderGates();
  }
  show(); renderShifts(); renderPeople();
  setInterval(() => { if (data) renderToday(new Date()); }, 30000);
  fetch('/api/b2b/overview').then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
    .then(d => { data = d; renderAll(); })
    .catch(e => { $('todaySub').textContent = `Không tải được dữ liệu: ${e.message}`; });
})();
