/* hanoiFlow — Hanoi map with weather for every area and a detailed forecast at any clicked point. */
'use strict';

const RAINY_CODES = new Set([51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99]);

function fmt(value, unit = '', digits = 0) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return '—';
  return `${Number(value).toLocaleString('vi-VN', { maximumFractionDigits: digits })}${unit}`;
}

function hourLabel(iso) {
  return String(iso || '').slice(11, 16);
}

function dayLabel(iso, index) {
  if (index === 0) return 'Hôm nay';
  if (index === 1) return 'Ngày mai';
  const d = new Date(`${iso}T00:00:00`);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleDateString('vi-VN', { weekday: 'long' });
}

function isRainy(code) {
  return RAINY_CODES.has(Number(code));
}


// ---- Open-Meteo straight from the browser: used when the server cannot reach it (shared hosting IPs are often rate-limited).
const WMO_VI = { 0: 'Trời quang', 1: 'Ít mây', 2: 'Mây rải rác', 3: 'Nhiều mây', 45: 'Sương mù', 48: 'Sương mù đóng băng', 51: 'Mưa phùn nhẹ', 53: 'Mưa phùn', 55: 'Mưa phùn dày',
  56: 'Mưa phùn lạnh', 57: 'Mưa phùn lạnh dày', 61: 'Mưa nhỏ', 63: 'Mưa vừa', 65: 'Mưa to', 66: 'Mưa lạnh', 67: 'Mưa lạnh to', 80: 'Mưa rào nhẹ', 81: 'Mưa rào',
  82: 'Mưa rào rất to', 95: 'Dông', 96: 'Dông kèm mưa đá', 99: 'Dông kèm mưa đá lớn' };
const describeCode = code => WMO_VI[Number(code)] || 'Không rõ';
const OPEN_METEO = 'https://api.open-meteo.com/v1/forecast';
const CUR = 'temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m,wind_direction_10m';
const DAILY_FIELDS = 'weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max';
const omNum = v => (v === null || v === undefined || !Number.isFinite(Number(v)) ? null : Math.round(Number(v) * 10) / 10);

async function openMeteo(params) {
  const response = await fetch(`${OPEN_METEO}?${new URLSearchParams(params)}`);
  if (!response.ok) throw new Error(`Open-Meteo HTTP ${response.status}`);
  return response.json();
}

async function directPoint(lat, lng) {
  const raw = await openMeteo({ latitude: lat, longitude: lng, current: CUR, daily: DAILY_FIELDS, forecast_days: 3, timezone: 'Asia/Bangkok',
    hourly: 'temperature_2m,precipitation_probability,precipitation,weather_code,wind_speed_10m' });
  const c = raw.current || {}, h = raw.hourly || {}, d = raw.daily || {};
  const times = h.time || [];
  const start = Math.max(0, times.findIndex(t => String(t) >= String(c.time || '').slice(0, 13)));
  const hourly = [];
  for (let i = start; i < Math.min(times.length, start + 24); i++) {
    hourly.push({ time: times[i], temperature_c: omNum(h.temperature_2m?.[i]), precipitation_probability_pct: omNum(h.precipitation_probability?.[i]),
      precipitation_mm: omNum(h.precipitation?.[i]), description: describeCode(h.weather_code?.[i]) });
  }
  const daily = (d.time || []).map((date, i) => ({ date, max_c: omNum(d.temperature_2m_max?.[i]), min_c: omNum(d.temperature_2m_min?.[i]),
    precipitation_mm: omNum(d.precipitation_sum?.[i]), precipitation_probability_pct: omNum(d.precipitation_probability_max?.[i]), description: describeCode(d.weather_code?.[i]) }));
  return { current: { temperature_c: omNum(c.temperature_2m), apparent_temperature_c: omNum(c.apparent_temperature), humidity_pct: omNum(c.relative_humidity_2m),
    precipitation_mm: omNum(c.precipitation), wind_kmh: omNum(c.wind_speed_10m), description: describeCode(c.weather_code) }, hourly, daily, direct: true };
}

async function directOverview(areas) {
  const raw = await openMeteo({ latitude: areas.map(a => a.latitude).join(','), longitude: areas.map(a => a.longitude).join(','), current: CUR, daily: DAILY_FIELDS, forecast_days: 1, timezone: 'Asia/Bangkok' });
  const rows = Array.isArray(raw) ? raw : [raw];
  return { areas: areas.map((a, i) => {
    const c = rows[i]?.current || {}, d = rows[i]?.daily || {}, first = k => (d[k] || [null])[0];
    return { name: a.name, latitude: a.latitude, longitude: a.longitude, inner: a.inner !== false, temperature_c: omNum(c.temperature_2m), precipitation_mm: omNum(c.precipitation),
      wind_kmh: omNum(c.wind_speed_10m), weather_code: c.weather_code, description: describeCode(c.weather_code), today_max_c: omNum(first('temperature_2m_max')),
      today_min_c: omNum(first('temperature_2m_min')), today_rain_probability_pct: omNum(first('precipitation_probability_max')) };
  }), attribution: 'Dữ liệu thời tiết: Open-Meteo (CC BY 4.0)', fetched_at: new Date().toISOString() };
}

if (typeof module !== 'undefined') module.exports = { fmt, hourLabel, dayLabel, isRainy, directPoint, directOverview, describeCode };

if (typeof document !== 'undefined') {
  const $ = id => document.getElementById(id);
  const escapeHtml = text => String(text ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  /** Base tiles with an automatic switch to the fallback server when the main one refuses requests. */
  window.hanoiFlowTiles = (target, config) => {
    if (typeof L === 'undefined') return;
    const base = L.tileLayer(config.tile_url_template, { maxZoom: 19, attribution: config.attribution }).addTo(target);
    let loaded = 0, failed = 0;
    base.on('tileload', () => { loaded += 1; });
    base.on('tileerror', () => {
      failed += 1;
      if (failed === 3 && loaded === 0 && config.fallback_tile_url_template) {
        target.removeLayer(base);
        L.tileLayer(config.fallback_tile_url_template, { maxZoom: 19, subdomains: 'abc', attribution: config.fallback_attribution }).addTo(target);
      }
    });
  };

  const FALLBACK_AREAS = [['Hoàn Kiếm', 21.0285, 105.8542], ['Cầu Giấy', 21.0362, 105.7906], ['Long Biên', 21.0465, 105.8885], ['Hoàng Mai', 20.9745, 105.8628],
    ['Hà Đông', 20.9714, 105.7788], ['Bắc Từ Liêm', 21.07, 105.75], ['Đông Anh', 21.137, 105.846], ['Gia Lâm', 21.0213, 105.947], ['Sóc Sơn', 21.256, 105.848],
    ['Sơn Tây', 21.138, 105.505], ['Ba Vì', 21.199, 105.423], ['Chương Mỹ', 20.92, 105.705], ['Mỹ Đức', 20.683, 105.744]].map(([name, latitude, longitude]) => ({ name, latitude, longitude }));
  let map, pinLayer, clickMarker, boundaryLayer, config, started = false;

  async function getJson(url) {
    const response = await fetch(url, { cache: 'no-store' });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
    return payload;
  }

  async function init() {
    if (started) return;
    started = true;
    const now = new Date();
    $('dateToday').textContent = now.toLocaleDateString('vi-VN', { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' }).toUpperCase();
    try { config = await getJson('/api/map/config'); } catch { config = { center: [21.0285, 105.8542] }; }
    buildMap();
    loadOverview();
    showPoint(config.center[0], config.center[1], 'Hoàn Kiếm');
    // Leaflet comes from a CDN: if it arrives late (or from the backup CDN) the map is built then, weather is already shown.
    window.addEventListener('leaflet-ready', () => { buildMap(); loadOverview(); });
    $('refreshBtn').addEventListener('click', () => { loadOverview(); });
  }

  function buildMap() {
    if (map || typeof L === 'undefined') return;
    try {
      map = L.map('map', { zoomControl: false }).setView(config.center, 10);
      $('map').classList.remove('no-map');
      $('map').replaceChildren(...[]);
      window.hanoiFlowTiles(map, config);
      pinLayer = L.layerGroup().addTo(map);
      getJson(config.boundary_url || '/static/hanoi_boundary.geojson').then(boundary => {
        boundaryLayer = L.geoJSON(boundary, { style: { color: '#b52226', weight: 2, fillOpacity: 0.03, fillColor: '#c49a38' }, interactive: false }).addTo(map);
        map.fitBounds(boundaryLayer.getBounds(), { padding: [10, 10] });
      }).catch(() => { /* the map still works without the outline */ });
      const zoomClass = () => $('map').classList.toggle('zoomed-out', map.getZoom() < 11);
      map.on('zoomend', zoomClass);
      zoomClass();
      map.on('click', e => showPoint(e.latlng.lat, e.latlng.lng, 'Điểm đã chọn'));
      $('mapZoomIn').onclick = () => map.zoomIn();
      $('mapZoomOut').onclick = () => map.zoomOut();
      $('mapHanoiFit').onclick = () => boundaryLayer ? map.fitBounds(boundaryLayer.getBounds(), { padding: [10, 10] }) : map.setView(config.center, 10);
      $('mapReset').onclick = () => map.setView(config.center, 13);
    } catch (error) {
      map = null;
      $('overviewMeta').textContent = `Không dựng được bản đồ: ${error.message}`;
    }
  }

  /** Without the map library the areas are still listed, so the weather section never looks broken. */
  function renderAreaList(areas) {
    if (map) return;
    const box = $('map');
    box.classList.add('no-map');
    box.innerHTML = `<div class="area-list">${areas.map(a => `<button type="button" class="area-card" data-lat="${a.latitude}" data-lng="${a.longitude}" data-name="${escapeHtml(a.name)}"><strong>${escapeHtml(a.name)}</strong><span>${fmt(a.temperature_c, '°')} · ${escapeHtml(a.description)}</span><em>Mưa ${fmt(a.today_rain_probability_pct, '%')}</em></button>`).join('')}</div>`;
    box.querySelectorAll('.area-card').forEach(btn => btn.addEventListener('click', () => showPoint(Number(btn.dataset.lat), Number(btn.dataset.lng), btn.dataset.name)));
  }

  async function loadOverview() {
    try {
      let data;
      try { data = await getJson('/api/weather/overview'); }
      catch (serverError) { data = await directOverview((config.areas && config.areas.length ? config.areas : FALLBACK_AREAS)); }
      if (pinLayer) pinLayer.clearLayers();
      renderAreaList(data.areas);
      for (const area of (map ? data.areas : [])) {
        const rainy = isRainy(area.weather_code) || (area.today_rain_probability_pct ?? 0) >= 60;
        const icon = L.divIcon({ className: '', html: `<span class="area-pin${rainy ? ' rain' : ''}${area.inner ? ' inner' : ''}"><b>${escapeHtml(area.name)} </b>${fmt(area.temperature_c, '°')}</span>` });
        L.marker([area.latitude, area.longitude], { icon })
          .bindTooltip(`${escapeHtml(area.name)}: ${escapeHtml(area.description)}, ${fmt(area.today_min_c, '°')}–${fmt(area.today_max_c, '°')}, khả năng mưa ${fmt(area.today_rain_probability_pct, '%')}`)
          .on('click', () => showPoint(area.latitude, area.longitude, area.name))
          .addTo(pinLayer);
      }
      const updated = new Date(data.fetched_at);
      $('overviewMeta').textContent = `${data.areas.length} khu vực · cập nhật ${updated.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' })}${data.stale ? ' (dữ liệu cũ)' : ''}`;
      $('attribution').textContent = data.attribution;
    } catch (error) {
      $('overviewMeta').textContent = `Không tải được thời tiết các khu vực: ${error.message}`;
      if (!loadOverview.retried) { loadOverview.retried = true; setTimeout(loadOverview, 5000); }
    }
  }

  async function showPoint(lat, lng, name, retry = false) {
    if (clickMarker) { clickMarker.remove(); clickMarker = null; }
    if (map) clickMarker = L.circleMarker([lat, lng], { radius: 7, color: '#b52226', weight: 3, fillColor: '#c49a38', fillOpacity: 0.7 }).addTo(map);
    $('placeKicker').textContent = name;
    $('placeMeta').textContent = `${lat.toFixed(4)}, ${lng.toFixed(4)} · Dữ liệu từ Open-Meteo; không thay thế cảnh báo khí tượng chính thức.`;
    const chip = $('weatherChip');
    chip.textContent = 'Đang tải'; chip.className = 'level-chip';
    try {
      let w;
      try { w = await getJson(`/api/weather?latitude=${lat}&longitude=${lng}`); }
      catch (serverError) { w = await directPoint(lat, lng); }
      const c = w.current;
      $('nowTemp').textContent = fmt(c.temperature_c, '°C');
      $('nowDesc').textContent = c.description;
      $('nowFeels').textContent = fmt(c.apparent_temperature_c, '°C');
      $('nowHumidity').textContent = fmt(c.humidity_pct, '%');
      $('nowWind').textContent = fmt(c.wind_kmh, ' km/h');
      $('nowRain').textContent = fmt(c.precipitation_mm, ' mm', 1);
      $('hours').innerHTML = w.hourly.map(h => `<div class="weather-hour"><span>${hourLabel(h.time)}</span><strong>${fmt(h.temperature_c, '°')}</strong>
        <span class="rain" title="${escapeHtml(h.description)}">☂ ${fmt(h.precipitation_probability_pct, '%')}</span></div>`).join('');
      $('days').innerHTML = w.daily.map((d, i) => `<div class="weather-day"><strong>${escapeHtml(dayLabel(d.date, i))}</strong>
        <span>${escapeHtml(d.description)} · ${fmt(d.min_c, '°')}–${fmt(d.max_c, '°')}</span>
        <span class="rain">Mưa ${fmt(d.precipitation_probability_pct, '%')} · ${fmt(d.precipitation_mm, ' mm', 1)}</span></div>`).join('');
      chip.textContent = w.stale ? `Dữ liệu cũ ${w.stale_minutes} phút` : 'Đã cập nhật';
      chip.className = w.stale ? 'level-chip busy' : 'level-chip free';
    } catch (error) {
      $('nowDesc').textContent = error.message;
      $('hours').innerHTML = ''; $('days').innerHTML = '';
      chip.textContent = 'Lỗi'; chip.className = 'level-chip gridlock';
      if (!retry) setTimeout(() => showPoint(lat, lng, name, true), 5000);
    }
  }

  const onShow = page => {
    if (page !== 'overview') return;
    init().then(() => setTimeout(() => map && map.invalidateSize(), 60))
      .catch(error => { $('overviewMeta').textContent = `Không khởi tạo được bản đồ: ${error.message}`; });
  };
  window.hanoiFlow.onPage(onShow);
  onShow(window.hanoiFlow.page);
}
