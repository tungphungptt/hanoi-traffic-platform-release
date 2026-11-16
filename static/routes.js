/* hanoiFlow — alternative routes between two points in Hanoi (OSRM on OpenStreetMap). */
'use strict';

function fmtDuration(seconds) {
  const m = Math.round(Number(seconds) / 60);
  return m >= 60 ? `${Math.floor(m / 60)} giờ ${m % 60} phút` : `${m} phút`;
}

function fmtDistance(meters) {
  return `${(Number(meters) / 1000).toLocaleString('vi-VN', { maximumFractionDigits: 1 })} km`;
}

if (typeof module !== 'undefined') module.exports = { fmtDuration, fmtDistance };

if (typeof document !== 'undefined' && typeof L !== 'undefined') {
  const $ = id => document.getElementById(id);
  const escapeHtml = text => String(text ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const COLORS = ['#b52226', '#597e9b', '#c49a38'];
  let map, layers = [], points = { from: null, to: null }, markers = {}, areas = [], nextClick = 'from', ready = false;

  async function init() {
    if (ready) return;
    ready = true;
    const config = await (await fetch('/api/map/config')).json();
    areas = config.areas || [];
    for (const id of ['routeFrom', 'routeTo']) {
      $(id).innerHTML = '<option value="">— chọn khu vực —</option><option value="map">Điểm bấm trên bản đồ</option>' +
        areas.map((a, i) => `<option value="${i}">${escapeHtml(a.name)}</option>`).join('');
    }
    map = L.map('routeMap').setView(config.center, 11);
    window.hanoiFlowTiles(map, config);
    map.on('click', e => {
      setPoint(nextClick, e.latlng.lat, e.latlng.lng, 'Điểm bấm trên bản đồ');
      $(nextClick === 'from' ? 'routeFrom' : 'routeTo').value = 'map';
      nextClick = nextClick === 'from' ? 'to' : 'from';
    });
    $('routeFrom').addEventListener('change', e => pick('from', e.target.value));
    $('routeTo').addEventListener('change', e => pick('to', e.target.value));
    $('routeBtn').addEventListener('click', findRoutes);
  }

  function pick(which, value) {
    if (value === '' || value === 'map') return;
    const a = areas[Number(value)];
    setPoint(which, a.latitude, a.longitude, a.name);
  }

  function setPoint(which, lat, lng, name) {
    points[which] = { lat, lng, name };
    if (markers[which]) markers[which].remove();
    markers[which] = L.marker([lat, lng], { title: which === 'from' ? 'Điểm đi' : 'Điểm đến' })
      .bindTooltip(`${which === 'from' ? 'A · ' : 'B · '}${escapeHtml(name)}`, { permanent: true, direction: 'top', offset: [-15, -10] }).addTo(map);
  }

  async function findRoutes() {
    if (!points.from || !points.to) { $('routeList').innerHTML = '<div class="empty">Chọn đủ điểm đi và điểm đến.</div>'; return; }
    $('routeList').innerHTML = '<div class="empty">Đang tìm tuyến…</div>';
    const q = new URLSearchParams({ from: `${points.from.lat},${points.from.lng}`, to: `${points.to.lat},${points.to.lng}` });
    try {
      const response = await fetch(`/api/routes?${q}`);
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
      layers.forEach(l => l.remove());
      layers = payload.routes.map((r, i) => L.geoJSON(r.geometry, { style: { color: COLORS[i % COLORS.length], weight: i === 0 ? 6 : 4, opacity: i === 0 ? 0.9 : 0.6 } }).addTo(map));
      layers.slice().reverse().forEach(l => l.bringToFront());
      if (layers.length) map.fitBounds(L.featureGroup(layers).getBounds(), { padding: [20, 20] });
      $('routeList').innerHTML = payload.routes.map((r, i) => `<button class="route-option${i === 0 ? ' recommended selected' : ''}" type="button" data-route="${i}">
          ${i === 0 ? '<span class="tag">Nhanh nhất theo mạng đường</span>' : ''}
          <strong style="color:${COLORS[i % COLORS.length]}">Tuyến ${i + 1} · ${fmtDuration(r.duration_seconds)}</strong>
          <span>${fmtDistance(r.distance_m)}${i > 0 ? ` · chậm hơn ${fmtDuration(r.duration_seconds - payload.routes[0].duration_seconds)}` : ''}</span>
        </button>`).join('') + `<p class="fine">${escapeHtml(payload.note)}</p>`;
      $('routeList').querySelectorAll('[data-route]').forEach(btn => btn.addEventListener('click', () => {
        $('routeList').querySelectorAll('.route-option').forEach(b => b.classList.toggle('selected', b === btn));
        const layer = layers[Number(btn.dataset.route)];
        layer.bringToFront();
        map.fitBounds(layer.getBounds(), { padding: [20, 20] });
      }));
    } catch (error) {
      $('routeList').innerHTML = `<div class="empty">Không tìm được tuyến: ${escapeHtml(error.message)}</div>`;
    }
  }

  window.hanoiFlow.onPage(page => {
    if (page !== 'video') return;
    init().then(() => setTimeout(() => map && map.invalidateSize(), 60))
      .catch(error => { $('routeList').innerHTML = `<div class="empty">Không khởi tạo được bản đồ: ${escapeHtml(error.message)}</div>`; });
  });
  if (window.hanoiFlow.page === 'video') init().catch(() => {});
}
