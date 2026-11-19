/* hanoiFlow — alternative routes between two points in Hanoi (OSRM on OpenStreetMap). */
'use strict';

function fmtDuration(seconds) {
  const m = Math.round(Number(seconds) / 60);
  return m >= 60 ? `${Math.floor(m / 60)} giờ ${m % 60} phút` : `${m} phút`;
}

/** Shortest distance in metres from point p [lng, lat] to a GeoJSON LineString. */
function distanceToLine(p, coords) {
  const kx = 111320 * Math.cos(p[1] * Math.PI / 180), ky = 110540;
  let best = Infinity;
  for (let i = 0; i < coords.length - 1; i++) {
    const ax = (coords[i][0] - p[0]) * kx, ay = (coords[i][1] - p[1]) * ky;
    const bx = (coords[i + 1][0] - p[0]) * kx, by = (coords[i + 1][1] - p[1]) * ky;
    const dx = bx - ax, dy = by - ay, den = dx * dx + dy * dy;
    const t = den ? Math.max(0, Math.min(1, -(ax * dx + ay * dy) / den)) : 0;
    best = Math.min(best, Math.hypot(ax + t * dx, ay + t * dy));
  }
  return best;
}

/** Rank routes by network time, adding the forecast delay for routes passing the camera. */
function rankRoutes(routes, cameraPoint, forecastCi, radius = 300) {
  return routes.map((r, i) => {
    const passes = Boolean(cameraPoint) && distanceToLine(cameraPoint, r.geometry.coordinates) <= radius;
    const forecastApplied = passes && Number.isFinite(forecastCi) && forecastCi >= 0.30;
    const penalty = forecastApplied ? 0.35 * forecastCi : 0;
    return { ...r, index: i, passes, forecastApplied, adjusted_seconds: r.duration_seconds * (1 + penalty) };
  }).sort((a, b) => a.adjusted_seconds - b.adjusted_seconds);
}

function decisionGuidance(ci, hasCameraForecast, routes = []) {
  const forecastReady = hasCameraForecast && Number.isFinite(ci) && ci >= 0 && ci <= 1;
  let badge = 'Chờ dữ liệu', tone = '', staff, route, routeDetail;

  if (!forecastReady) {
    staff = 'Chọn camera và mở kết quả dự báo trước khi cân nhắc điều tiết.';
  } else if (ci >= 0.78) {
    badge = 'Ưu tiên xác minh'; tone = 'urgent';
    staff = 'Ưu tiên xác minh hiện trường; cân nhắc bổ sung người điều tiết.';
  } else if (ci >= 0.55) {
    badge = 'Theo dõi sát'; tone = 'watch';
    staff = 'Theo dõi sát; chỉ cân nhắc điều tiết khi hiện trường xác nhận ùn tắc.';
  } else {
    badge = 'Chưa có tín hiệu mạnh';
    staff = 'Chưa đề xuất tăng người điều tiết theo ngưỡng mô phỏng.';
  }

  if (!routes.length) {
    route = 'Chọn điểm đi, điểm đến và tìm tuyến.';
    routeDetail = 'Chưa có phương án tuyến để so sánh.';
  } else if (!forecastReady) {
    route = 'Tuyến 1 nhanh nhất theo mạng đường.';
    routeDetail = 'Chưa có dự báo áp dụng; OSRM không phản ánh tình trạng giao thông trực tiếp.';
  } else if (ci >= 0.55 && !routes[0].passes) {
    route = 'Ưu tiên tuyến 1 để tránh vùng camera có nguy cơ ùn tắc.';
    routeDetail = 'Xếp hạng đã tính CI dự báo; xem tuyến trên bản đồ và xác minh trước khi đi.';
  } else if (ci >= 0.55) {
    route = 'Tuyến 1 có thời gian tham khảo tốt nhất sau khi tính dự báo.';
    routeDetail = 'Nếu tuyến vẫn đi qua vùng camera, hãy kiểm tra tình hình thực tế trước khi chọn.';
  } else {
    route = 'Tuyến 1 nhanh nhất theo dữ liệu mạng đường.';
    routeDetail = 'Chưa có tín hiệu để khẳng định cần đổi tuyến nhằm tránh ùn tắc.';
  }
  return { badge, tone, staff, route, routeDetail };
}

function fmtDistance(meters) {
  return `${(Number(meters) / 1000).toLocaleString('vi-VN', { maximumFractionDigits: 1 })} km`;
}

// Approximate positions of the cameras in the Hanoi broadcast recordings (adjust on the map).
const KNOWN_CAMERAS = [
  { name: '216 Trần Duy Hưng', latitude: 21.0075, longitude: 105.7990 },
  { name: '300 Nguyễn Xiển', latitude: 20.9840, longitude: 105.8070 },
  { name: 'Hồ Tùng Mậu', latitude: 21.0378, longitude: 105.7770 },
  { name: 'Xuân Diệu', latitude: 21.0650, longitude: 105.8285 },
  { name: 'Quang Trung – Lê Trọng Tấn (Hà Đông)', latitude: 20.9760, longitude: 105.7790 },
  { name: 'Lê Hồng Phong', latitude: 21.0330, longitude: 105.8370 },
  { name: '1111 Giải Phóng', latitude: 20.9620, longitude: 105.8410 },
];

if (typeof module !== 'undefined') module.exports = { fmtDuration, fmtDistance, distanceToLine, rankRoutes, decisionGuidance, KNOWN_CAMERAS };

function bootRoutes() {
  if (typeof document === 'undefined' || typeof L === 'undefined' || bootRoutes.done) return;
  bootRoutes.done = true;
  const $ = id => document.getElementById(id);
  const escapeHtml = text => String(text ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const COLORS = ['#b52226', '#597e9b', '#c49a38'];
  let map, layers = [], points = { from: null, to: null }, markers = {}, areas = [], nextClick = 'from', ready = false, rankedRoutes = [];

  async function init() {
    if (ready) return;
    ready = true;
    const config = await (await fetch('/api/map/config')).json();
    areas = config.areas || [];
    for (const id of ['routeFrom', 'routeTo']) {
      $(id).innerHTML = '<option value="">— chọn khu vực —</option><option value="map">Điểm bấm trên bản đồ</option>' +
        areas.map((a, i) => `<option value="${i}">${escapeHtml(a.name)}</option>`).join('');
    }
    $('routeCamera').innerHTML = '<option value="">— không dùng dự báo —</option><option value="map">Bấm bản đồ để đặt vị trí camera</option>' +
      KNOWN_CAMERAS.map((c, i) => `<option value="${i}">${escapeHtml(c.name)} (vị trí gần đúng)</option>`).join('');
    $('routeCamera').addEventListener('change', e => {
      const v = e.target.value;
      if (v === 'map') { nextClick = 'camera'; return; }
      if (v === '') { setCamera(null); return; }
      const c = KNOWN_CAMERAS[Number(v)];
      setCamera({ lat: c.latitude, lng: c.longitude, name: c.name });
    });
    map = L.map('routeMap').setView(config.center, 11);
    window.hanoiFlowTiles(map, config);
    map.on('click', e => {
      if (nextClick === 'camera') {
        setCamera({ lat: e.latlng.lat, lng: e.latlng.lng, name: 'Camera (vị trí đã bấm)' });
        nextClick = points.from ? 'to' : 'from';
        return;
      }
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
    clearRouteResults();
  }

  let cameraPoint = null, cameraMarker = null;
  function clearRouteResults() {
    rankedRoutes = [];
    layers.forEach(layer => layer.remove());
    layers = [];
    $('routeList').innerHTML = '<div class="empty">Tìm lại tuyến sau khi đổi điểm đi, điểm đến hoặc camera.</div>';
    renderDecision();
  }

  function setCamera(c) {
    cameraPoint = c;
    if (cameraMarker) cameraMarker.remove();
    cameraMarker = c ? L.circle([c.lat, c.lng], { radius: 300, color: '#a94d43', weight: 2, fillOpacity: 0.12 })
      .bindTooltip(`📷 ${escapeHtml(c.name)}`, { permanent: true, direction: 'right' }).addTo(map) : null;
    clearRouteResults();
    describeForecast();
  }

  function forecastCi() {
    const f = window.hanoiFlow.forecastNow ? window.hanoiFlow.forecastNow() : null;
    if (!f) return { f: null, ci: NaN };
    const h10 = f.horizons.find(h => h.horizon_minutes === 10) || f.horizons[0];
    return { f, ci: h10 ? h10.ci : NaN, p: h10 ? h10.p_gridlock : NaN };
  }

  function describeForecast() {
    const { f, ci, p } = forecastCi();
    const fmtT = s => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(Math.round(s % 60)).padStart(2, '0')}`;
    $('routeForecast').textContent = !f
      ? 'Chưa có dự báo: mở một kết quả ở bước 1–2 (hoặc bấm "Xem demo Hà Nội") rồi chọn vị trí camera để hệ thống tránh đoạn được dự báo tắc.'
      : !cameraPoint
        ? `Đang có dự báo của ${f.camera} lúc ${fmtT(f.t)} (+10 phút: CI ${ci.toFixed(2)}). Chọn vị trí camera để áp dụng vào tuyến.`
        : `Dùng dự báo của ${f.camera} lập lúc ${fmtT(f.t)} trên video: sau 10 phút CI ${ci.toFixed(2)}, khả năng tắc ${Math.round(p * 100)}%. Tuyến đi qua vùng tròn đỏ sẽ bị cộng thêm thời gian.`;
    renderDecision();
  }

  function renderDecision() {
    const { f, ci } = forecastCi();
    const forecastReady = Boolean(f && cameraPoint && Number.isFinite(ci));
    const guidance = decisionGuidance(ci, forecastReady, rankedRoutes);
    const badge = $('routeDecisionBadge');
    badge.textContent = guidance.badge;
    badge.className = `route-decision-badge ${guidance.tone}`;
    $('routeStaffDecision').textContent = guidance.staff;
    $('routeStaffBasis').textContent = forecastReady
      ? `${f.camera}, +10 phút: CI ${ci.toFixed(2)}. Chỉ hỗ trợ quyết định thử nghiệm.`
      : 'Cần có dự báo và vị trí camera để đánh giá khu vực.';
    $('routeRerouteDecision').textContent = guidance.route;
    $('routeRerouteBasis').textContent = guidance.routeDetail;
  }
  setInterval(() => { if (ready && window.hanoiFlow.page === 'video') describeForecast(); }, 2000);

  async function findRoutes() {
    if (!points.from || !points.to) { $('routeList').innerHTML = '<div class="empty">Chọn đủ điểm đi và điểm đến.</div>'; return; }
    rankedRoutes = [];
    renderDecision();
    $('routeList').innerHTML = '<div class="empty">Đang tìm tuyến…</div>';
    const q = new URLSearchParams({ from: `${points.from.lat},${points.from.lng}`, to: `${points.to.lat},${points.to.lng}` });
    try {
      const response = await fetch(`/api/routes?${q}`);
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
      layers.forEach(l => l.remove());
      const { ci } = forecastCi();
      const ranked = rankRoutes(payload.routes, cameraPoint ? [cameraPoint.lng, cameraPoint.lat] : null, ci);
      rankedRoutes = ranked;
      renderDecision();
      payload.routes = ranked;
      layers = payload.routes.map((r, i) => L.geoJSON(r.geometry, { style: { color: COLORS[i % COLORS.length], weight: i === 0 ? 6 : 4, opacity: i === 0 ? 0.9 : 0.6 } }).addTo(map));
      layers.slice().reverse().forEach(l => l.bringToFront());
      if (layers.length) map.fitBounds(L.featureGroup(layers).getBounds(), { padding: [20, 20] });
      $('routeList').innerHTML = payload.routes.map((r, i) => `<button class="route-option${i === 0 ? ' recommended selected' : ''}" type="button" data-route="${i}">
          ${i === 0 ? `<span class="tag">${payload.routes.some(x => x.forecastApplied) ? 'Đề xuất (đã tính dự báo)' : 'Nhanh nhất theo mạng đường'}</span>` : ''}
          <strong style="color:${COLORS[i % COLORS.length]}">Tuyến ${i + 1} · ${fmtDuration(r.adjusted_seconds)}</strong>
          <span>${fmtDistance(r.distance_m)} · theo mạng đường ${fmtDuration(r.duration_seconds)}</span>
          ${r.search_method === 'via_waypoint' ? '<span class="fine">Phương án bổ sung qua điểm trung gian</span>' : ''}
          ${r.passes ? `<span class="warn">${r.forecastApplied ? `Đi qua camera dự báo đông/tắc: +${fmtDuration(r.adjusted_seconds - r.duration_seconds)}` : 'Đi qua vùng camera; chưa cộng thời gian dự báo.'}</span>` : ''}
        </button>`).join('') + `<p class="fine">${escapeHtml(payload.note)}</p>`;
      $('routeList').querySelectorAll('[data-route]').forEach(btn => btn.addEventListener('click', () => {
        $('routeList').querySelectorAll('.route-option').forEach(b => b.classList.toggle('selected', b === btn));
        const layer = layers[Number(btn.dataset.route)];
        layer.bringToFront();
        map.fitBounds(layer.getBounds(), { padding: [20, 20] });
      }));
    } catch (error) {
      rankedRoutes = [];
      renderDecision();
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
bootRoutes();
if (typeof window !== 'undefined') window.addEventListener('leaflet-ready', bootRoutes);
