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
  let map, pinLayer, clickMarker;

  async function getJson(url) {
    const response = await fetch(url, { cache: 'no-store' });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
    return payload;
  }

  async function init() {
    const config = await getJson('/api/map/config');
    map = L.map('map', { zoomControl: true }).setView(config.center, 10);
    const base = L.tileLayer(config.tile_url_template, { maxZoom: 19, attribution: config.attribution }).addTo(map);
    let loaded = 0, failed = 0;
    base.on('tileload', () => { loaded += 1; });
    base.on('tileerror', () => {
      failed += 1;
      if (failed === 3 && loaded === 0 && config.fallback_tile_url_template) {
        map.removeLayer(base);
        L.tileLayer(config.fallback_tile_url_template, { maxZoom: 19, subdomains: 'abc', attribution: config.fallback_attribution }).addTo(map);
      }
    });
    try {
      const boundary = await getJson(config.boundary_url);
      const layer = L.geoJSON(boundary, { style: { color: '#2f6b48', weight: 2, fillOpacity: 0.04 }, interactive: false }).addTo(map);
      map.fitBounds(layer.getBounds(), { padding: [10, 10] });
    } catch { /* the map still works without the outline */ }
    pinLayer = L.layerGroup().addTo(map);
    // Inner districts sit close together: show only their temperature until the user zooms in.
    const zoomClass = () => document.getElementById('map').classList.toggle('zoomed-out', map.getZoom() < 11);
    map.on('zoomend', zoomClass);
    zoomClass();
    map.on('click', e => showPoint(e.latlng.lat, e.latlng.lng, 'Điểm đã chọn'));
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
    }
  }

  async function showPoint(lat, lng, name) {
    if (clickMarker) clickMarker.remove();
    clickMarker = L.circleMarker([lat, lng], { radius: 7, color: '#2d5f8f', weight: 3, fillOpacity: 0.6 }).addTo(map);
    $('placeTitle').textContent = name;
    $('placeMeta').textContent = `${lat.toFixed(4)}, ${lng.toFixed(4)}`;
    const chip = $('weatherChip');
    chip.textContent = 'Đang tải'; chip.className = 'chip';
    try {
      const w = await getJson(`/api/weather?latitude=${lat}&longitude=${lng}`);
      const c = w.current;
      $('now').innerHTML = `<span class="temp">${fmt(c.temperature_c, '°')}</span>
        <span class="desc">${escapeHtml(c.description)}</span>
        <span class="facts">Cảm giác ${fmt(c.apparent_temperature_c, '°')} · ẩm ${fmt(c.humidity_pct, '%')} · gió ${fmt(c.wind_kmh, ' km/h')} · mưa ${fmt(c.precipitation_mm, ' mm', 1)}</span>`;
      $('hours').innerHTML = w.hourly.map(h => `<div class="hour"><span>${hourLabel(h.time)}</span><strong>${fmt(h.temperature_c, '°')}</strong>
        <span class="rain" title="${escapeHtml(h.description)}">${fmt(h.precipitation_probability_pct, '%')}</span></div>`).join('');
      $('days').innerHTML = w.daily.map((d, i) => `<div class="day"><strong>${escapeHtml(dayLabel(d.date, i))}</strong>
        <span>${escapeHtml(d.description)} · ${fmt(d.min_c, '°')}–${fmt(d.max_c, '°')}</span>
        <span class="rain">mưa ${fmt(d.precipitation_probability_pct, '%')} · ${fmt(d.precipitation_mm, ' mm', 1)}</span></div>`).join('');
      chip.textContent = w.stale ? `Dữ liệu cũ ${w.stale_minutes} phút` : 'Đã cập nhật';
      chip.className = w.stale ? 'chip stale' : 'chip ok';
    } catch (error) {
      $('now').innerHTML = `<span class="muted">${escapeHtml(error.message)}</span>`;
      $('hours').innerHTML = ''; $('days').innerHTML = '';
      chip.textContent = 'Lỗi'; chip.className = 'chip bad';
    }
  }

  init().catch(error => { $('overviewMeta').textContent = `Không khởi tạo được bản đồ: ${error.message}`; });
}
