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

if (typeof module !== 'undefined') module.exports = { fmt, hourLabel, dayLabel, isRainy };

if (typeof document !== 'undefined' && typeof L !== 'undefined') {
  const $ = id => document.getElementById(id);
  const escapeHtml = text => String(text ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  /** Base tiles with an automatic switch to the fallback server when the main one refuses requests. */
  window.hanoiFlowTiles = (target, config) => {
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
    config = await getJson('/api/map/config');
    map = L.map('map', { zoomControl: false }).setView(config.center, 10);
    window.hanoiFlowTiles(map, config);
    try {
      const boundary = await getJson(config.boundary_url);
      boundaryLayer = L.geoJSON(boundary, { style: { color: '#b52226', weight: 2, fillOpacity: 0.03, fillColor: '#c49a38' }, interactive: false }).addTo(map);
      map.fitBounds(boundaryLayer.getBounds(), { padding: [10, 10] });
    } catch { /* the map still works without the outline */ }
    pinLayer = L.layerGroup().addTo(map);
    const zoomClass = () => $('map').classList.toggle('zoomed-out', map.getZoom() < 11);
    map.on('zoomend', zoomClass);
    zoomClass();
    map.on('click', e => showPoint(e.latlng.lat, e.latlng.lng, 'Điểm đã chọn'));
    $('mapZoomIn').addEventListener('click', () => map.zoomIn());
    $('mapZoomOut').addEventListener('click', () => map.zoomOut());
    $('mapHanoiFit').addEventListener('click', () => boundaryLayer ? map.fitBounds(boundaryLayer.getBounds(), { padding: [10, 10] }) : map.setView(config.center, 10));
    $('mapReset').addEventListener('click', () => map.setView(config.center, 13));
    $('refreshBtn').addEventListener('click', () => { loadOverview(); });
    loadOverview();
    showPoint(config.center[0], config.center[1], 'Hoàn Kiếm');
  }

  async function loadOverview() {
    try {
      const data = await getJson('/api/weather/overview');
      pinLayer.clearLayers();
      for (const area of data.areas) {
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
    if (clickMarker) clickMarker.remove();
    clickMarker = L.circleMarker([lat, lng], { radius: 7, color: '#b52226', weight: 3, fillColor: '#c49a38', fillOpacity: 0.7 }).addTo(map);
    $('placeKicker').textContent = name.toUpperCase();
    $('placeMeta').textContent = `${lat.toFixed(4)}, ${lng.toFixed(4)} · Dữ liệu từ Open-Meteo; không thay thế cảnh báo khí tượng chính thức.`;
    const chip = $('weatherChip');
    chip.textContent = 'Đang tải'; chip.className = 'level-chip';
    try {
      const w = await getJson(`/api/weather?latitude=${lat}&longitude=${lng}`);
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
