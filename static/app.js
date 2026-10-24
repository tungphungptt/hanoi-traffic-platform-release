const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const state = { dashboard: null, cameras: [], alerts: [], modelStatus: null, providerStatus: null, aiccStatus: null, manifest: null, currentPage: 'video', mapLayer: 'status', scenarioResult: null, scenarioJobId: null, scenarioBusy: false, scenarioPendingJobId: null, routeResult: null, routeMap: null, routeMapLayers: [], selectedRouteId: null, selectedCameraId: null, publicDemoMode: false, leafletMap: null, leafletMarkers: new Map(), baseTileLayer: null, trafficTileLayer: null, mapConfig: null, mapPollSeconds: 45, mapPollBusy: false, mapPollTimer: null };


function canAnalyzeUploadedScenarioVideo() {
  return Boolean(state.aiccStatus?.video_review_enabled);
}

const PAGE_NAMES = { overview: 'Tổng quan', cameras: 'Điểm giám sát', alerts: 'Cảnh báo', analytics: 'Báo cáo & xu hướng', video: 'Dự báo & phương án', models: 'Nguồn xử lý', settings: 'Cài đặt vận hành' };

async function api(path, options = {}) {
  const { timeoutMs = 30000, ...requestOptions } = options;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(path, { ...requestOptions, signal: controller.signal, cache: 'no-store' });
    const raw = await response.text();
    let payload;
    try { payload = raw ? JSON.parse(raw) : {}; }
    catch {
      const error = new Error(`Máy chủ trả phản hồi không phải JSON (HTTP ${response.status}).`);
      error.status = response.status;
      throw error;
    }
    if (!response.ok) {
      const detail = typeof payload?.detail === 'string' ? payload.detail
        : Array.isArray(payload?.detail) ? payload.detail.map(item => item.msg || 'Dữ liệu không hợp lệ').join('; ') : '';
      const error = new Error(`HTTP ${response.status}: ${detail || 'Yêu cầu thất bại.'}`);
      error.status = response.status;
      throw error;
    }
    if (!payload || typeof payload !== 'object') throw new Error('Phản hồi JSON không hợp lệ.');
    return payload;
  } catch (error) {
    if (controller.signal.aborted) throw new Error('Yêu cầu quá thời gian chờ. Kiểm tra kết nối và trạng thái tác vụ trước khi gửi lại.');
    if (error instanceof TypeError) throw new Error('Mất kết nối máy chủ. Kiểm tra backend đang chạy và kết nối mạng.');
    throw error;
  } finally { clearTimeout(timer); }
}
function numberVN(value, digits = 0) { return Number(value ?? 0).toLocaleString('vi-VN', { maximumFractionDigits: digits, minimumFractionDigits: digits }); }
function escapeHtml(text) { return String(text ?? '').replace(/[&<>"']/g, ch => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' }[ch])); }
function toast(title, detail = '', type = 'success') {
  const el = document.createElement('div'); el.className = `toast ${type === 'error' ? 'error' : ''}`;
  el.innerHTML = `<strong>${escapeHtml(title)}</strong>${detail ? `<small>${escapeHtml(detail)}</small>` : ''}`;
  $('#toastStack').appendChild(el); setTimeout(() => el.remove(), 4400);
}
function setPage(page) {
  if (!PAGE_NAMES[page]) return;
  state.currentPage = page;
  $$('.page').forEach(section => section.classList.toggle('active', section.id === `page-${page}`));
  $$('.nav-item').forEach(button => button.classList.toggle('active', button.dataset.page === page));
  $('#pageCrumb').textContent = PAGE_NAMES[page];
  $('#sidebar').classList.remove('open');
  if (page === 'cameras') renderCameraTable();
  if (page === 'alerts') renderAlertFeed();
  if (page === 'analytics') renderAnalytics();
  if (page === 'models') { renderModelStatus(); if (state.manifest && $('#manifestEditor')) $('#manifestEditor').value = JSON.stringify(state.manifest, null, 2); }
  if (page === 'settings') renderProviderStatus();
  if (page === 'video') {
    refreshAiccStatus();
    if (!state.routeMap) initScenarioRouteMap();
    if (state.routeMap) requestAnimationFrame(() => state.routeMap.invalidateSize({ pan: false }));
  }
  if (page === 'overview' && state.leafletMap) {
    requestAnimationFrame(() => state.leafletMap.invalidateSize({ pan: false }));
    if (state.dashboard) void pollCameraMapData();
  }
  if (page === 'overview') { void refreshWeatherForecast(); void refreshSegmentForecasts(true); }
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function severityFromCI(ci) { return ci < .30 ? 'free' : ci < .55 ? 'busy' : ci < .78 ? 'heavy' : 'gridlock'; }
function levelVN(key) { return ({ free:'Thông thoáng', busy:'Đông đúc', heavy:'Ùn ứ cục bộ', gridlock:'Tắc nghẽn nghiêm trọng' })[key] || 'Chưa xác định'; }
function cameraStatusVN(status) { return ({ online:'Có telemetry mới', attention:'Có cảnh báo từ nguồn', offline:'Nguồn mất kết nối', unverified:'Chưa có telemetry' })[status] || status; }
function alertClass(type) { return type === 'critical' ? 'critical' : type === 'warning' ? 'warning' : 'info'; }
function alertIcon(type) { return type === 'critical' ? '!' : type === 'warning' ? '⌁' : 'i'; }
function ciColorClass(ci) { return severityFromCI(Number(ci)); }
function updateDate() {
  const d = new Date();
  $('#dateToday').textContent = d.toLocaleDateString('vi-VN', { day:'numeric', month:'long', year:'numeric' }).toLocaleUpperCase('vi-VN');
}


let weatherLastSuccessMs = 0;
let weatherRequestBusy = false;
let segmentForecastBusy = false;

function fmtValue(value, suffix = '', digits = 1) {
  const num = Number(value);
  return value == null || !Number.isFinite(num) ? '—' : `${numberVN(num, digits)}${suffix}`;
}

function renderWeatherForecast(payload) {
  const current = payload?.current || {};
  $('#weatherTemperature').textContent = fmtValue(current.temperature_c, '°C');
  $('#weatherDescription').textContent = current.description || 'Không có mô tả';
  $('#weatherFeelsLike').textContent = fmtValue(current.apparent_temperature_c, '°C');
  $('#weatherHumidity').textContent = fmtValue(current.relative_humidity_pct, '%', 0);
  $('#weatherWind').textContent = fmtValue(current.wind_speed_kmh, ' km/h');
  $('#weatherRain').textContent = fmtValue(current.precipitation_mm, ' mm');
  const stale = Boolean(payload?.is_stale);
  const fetchedAt = payload?.fetched_at ? new Date(payload.fetched_at) : null;
  $('#weatherUpdatedAt').textContent = `Quan trắc: ${current.time || '—'} · lấy dữ liệu: ${fetchedAt && !Number.isNaN(fetchedAt.getTime()) ? fetchedAt.toLocaleTimeString('vi-VN') : '—'}`;
  $('#weatherStatus').textContent = stale ? `Dữ liệu cache cũ (${fmtValue(payload.stale_age_seconds, ' giây', 0)})` : 'Đã cập nhật';
  $('#weatherStatus').classList.toggle('is-stale', stale);
  const rows = Array.isArray(payload?.hourly) ? payload.hourly.slice(0, 6) : [];
  $('#weatherHourly').innerHTML = rows.length ? rows.map(item => {
    const date = new Date(item.time);
    const hour = Number.isNaN(date.getTime()) ? String(item.time).slice(-5) : date.toLocaleTimeString('vi-VN', {hour:'2-digit', minute:'2-digit'});
    return `<article class="weather-hour"><small>${escapeHtml(hour)}</small><strong>${escapeHtml(fmtValue(item.temperature_c, '°C'))}</strong><span>${escapeHtml(item.description || '—')}</span><small>Mưa ${escapeHtml(fmtValue(item.precipitation_probability_pct, '%', 0))}</small></article>`;
  }).join('') : '<div class="empty-state">Nhà cung cấp chưa trả dữ liệu theo giờ.</div>';
  weatherLastSuccessMs = Date.now();
}

async function refreshWeatherForecast(force = false) {
  if (weatherRequestBusy || !$('#weatherCard')) return;
  if (!force && weatherLastSuccessMs && Date.now() - weatherLastSuccessMs < 10 * 60 * 1000) return;
  weatherRequestBusy = true;
  const status = $('#weatherStatus');
  if (status) status.textContent = 'Đang lấy dữ liệu…';
  try {
    const payload = await api('/api/weather?latitude=21.0285&longitude=105.8542');
    renderWeatherForecast(payload);
  } catch (error) {
    if (status) status.textContent = `Chưa có dữ liệu thời tiết: ${error.message}`;
    $('#weatherDescription').textContent = 'Không thể lấy dữ liệu từ nhà cung cấp';
    $('#weatherHourly').innerHTML = '<div class="empty-state">Kiểm tra kết nối backend/upstream rồi nhấn Cập nhật.</div>';
  } finally {
    weatherRequestBusy = false;
  }
}

function renderSegmentForecasts(payload) {
  const status = $('#segmentForecastStatus');
  const host = $('#segmentForecastList');
  if (!status || !host) return;
  const items = Array.isArray(payload?.items) ? payload.items : [];
  const ready = items.filter(item => item.available && Array.isArray(item.predictions) && item.predictions.length);
  if (!ready.length) {
    status.textContent = payload?.message || 'Chưa có dự báo theo đoạn đường.';
    host.innerHTML = items.length ? items.map(item => `<article class="segment-forecast-row unavailable"><div><strong>${escapeHtml(item.name || item.camera_id)}</strong><small>${escapeHtml(item.area || item.camera_id)}</small></div><span>${escapeHtml(item.message || 'Chưa đủ chuỗi telemetry thật')}</span></article>`).join('') : '';
    return;
  }
  status.textContent = `Có ${ready.length}/${items.length} đoạn đủ điều kiện baseline · chưa được kiểm định độc lập.`;
  host.innerHTML = items.map(item => {
    if (!item.available || !Array.isArray(item.predictions) || !item.predictions.length) {
      return `<article class="segment-forecast-row unavailable"><div><strong>${escapeHtml(item.name || item.camera_id)}</strong><small>${escapeHtml(item.area || item.camera_id)}</small></div><span>${escapeHtml(item.message || 'Chưa đủ dữ liệu')}</span></article>`;
    }
    const predictions = item.predictions.map(pred => `<span class="segment-horizon"><small>+${numberVN(pred.horizon_minutes, 0)} phút</small><strong>${escapeHtml(pred.level || 'Chưa xác định')}</strong><em>CI ${escapeHtml(fmtValue(pred.ci, '', 2))}</em></span>`).join('');
    return `<article class="segment-forecast-row"><div><strong>${escapeHtml(item.name || item.camera_id)}</strong><small>${escapeHtml(item.area || item.camera_id)} · hiện tại ${escapeHtml(fmtValue(item.current_ci, '', 2))}</small></div><div class="segment-horizons">${predictions}</div><small class="segment-method">Baseline ngoại suy tuyến tính · ${numberVN(item.sample_count || 0)} mẫu · không phải mô hình AI đã kiểm định</small></article>`;
  }).join('');
}

async function refreshSegmentForecasts(force = false) {
  if (segmentForecastBusy || !$('#segmentForecastList')) return;
  if (document.hidden || state.currentPage !== 'overview') return;
  segmentForecastBusy = true;
  try {
    const payload = await api('/api/traffic/segment-forecasts');
    renderSegmentForecasts(payload);
  } catch (error) {
    const status = $('#segmentForecastStatus');
    if (status) status.textContent = `Không đọc được dữ liệu dự báo đoạn đường: ${error.message}`;
  } finally {
    segmentForecastBusy = false;
  }
}

async function loadAll({ quiet = false } = {}) {
  const refresh = $('#refreshBtn'); refresh?.classList.add('rotating');
  try {
    const [dashboard, system, cameras, alerts, modelStatus, manifest, providerStatus, aiccStatus] = await Promise.all([
      api('/api/dashboard'), api('/api/system'), api('/api/cameras'), api('/api/alerts'), api('/api/models/status'), api('/api/models/manifest'), api('/api/providers/status'), api('/api/aicc/status')
    ]);
    state.dashboard = dashboard; state.cameras = cameras.items || []; state.alerts = alerts.items || []; state.modelStatus = modelStatus; state.manifest = manifest; state.providerStatus = providerStatus; state.aiccStatus = aiccStatus; state.publicDemoMode = Boolean(system?.public_demo_mode);
    $$('[data-operator-only]').forEach(el => { el.hidden = state.publicDemoMode; });
    renderScenarioCameraOptions();
    renderDashboard(dashboard, system);
    renderCameraTable();
    renderAlertFeed(); renderModelStatus(); renderAnalytics();
    if ($('#manifestEditor')) $('#manifestEditor').value = JSON.stringify(manifest, null, 2);
    renderProviderStatus();
    if (!quiet) toast('Đã cập nhật dữ liệu', modelStatus.mode === 'production' ? 'Runtime đang dùng mô hình dự báo ONNX.' : 'Dữ liệu trên bảng điều khiển hiện là dữ liệu minh họa.');
  } catch (error) {
    toast('Không thể tải dữ liệu', `${error.message}. Vui lòng thử làm mới trang.`, 'error');
  } finally { refresh?.classList.remove('rotating'); }
}

function cameraFeedUrl(feed) {
  if (!feed?.available || !feed?.stream_url) return '';
  return feed.stream_url;
}

function mapCameraTone(cam) {
  if (!cam.telemetry_live) return 'reference';
  if (state.mapLayer === 'forecast') return severityFromCI(Number(cam.ci || 0));
  if (cam.status === 'offline') return 'offline';
  if (cam.status === 'attention') return 'attention';
  return 'online';
}
function mapPopupHtml(cam) {
  const lat = Number(cam.lat), lng = Number(cam.lng);
  const coords = Number.isFinite(lat) && Number.isFinite(lng) ? `${lat.toFixed(5)}, ${lng.toFixed(5)}` : 'Chưa cấu hình tọa độ';
  const status = cameraStatusVN(cam.status);
  const feed = cam.feed || {};
  return `<div class="map-popup"><strong>${escapeHtml(cam.name)}</strong><span>${escapeHtml(cam.area || 'Chưa cấu hình khu vực')} · ${escapeHtml(cam.id)}</span><span class="map-popup-status">${escapeHtml(status)}</span><small>Tọa độ tham chiếu: ${coords}</small><small>Vị trí khu vực tham khảo, chưa xác minh thiết bị camera.</small><small>${feed.available ? 'Đã gán clip cho điểm này' : 'Chưa có feed video được gán'}</small></div>`;
}
function mapMarkerIcon(cam) {
  const tone = mapCameraTone(cam);
  const selected = cam.id === state.selectedCameraId ? ' is-selected' : '';
  const shortId = escapeHtml(String(cam.id).replace(/[^A-Za-z0-9]/g, '').slice(-3));
  return L.divIcon({
    className: 'hanoi-map-marker-shell',
    html: `<span class="hanoi-map-marker ${tone}${selected}" aria-hidden="true"><i></i><b>${shortId}</b></span>`,
    iconSize: [34, 42], iconAnchor: [17, 37], popupAnchor: [0, -34]
  });
}
function updateMapStatus(payload = null) {
  const sync = $('#mapSyncStatus');
  const dataMode = $('#mapDataMode');
  if (sync) sync.textContent = `Đồng bộ API: ${new Date().toLocaleTimeString('vi-VN', { hour:'2-digit', minute:'2-digit', second:'2-digit' })} · ${state.mapPollSeconds}s`;
  if (dataMode) dataMode.textContent = payload?.telemetry_mode === 'rtsp_collector' ? 'Đã nhận telemetry từ collector' : 'Điểm tham chiếu · chưa có telemetry live';
  const count = $('#mapOnlineCount');
  if (count) {
    const online = state.cameras.filter(cam => cam.telemetry_live).length;
    count.textContent = `${online}/${state.cameras.length} điểm có telemetry mới`;
  }
}
function renderMapMarkers(cameras) {
  if (!state.leafletMap || typeof L === 'undefined') return;
  const keep = new Set();
  for (const cam of cameras) {
    const lat = Number(cam.lat), lng = Number(cam.lng);
    // Do not invent a geographic position for a camera without validated coordinates.
    if (!Number.isFinite(lat) || !Number.isFinite(lng) || lat < -90 || lat > 90 || lng < -180 || lng > 180) continue;
    keep.add(cam.id);
    let marker = state.leafletMarkers.get(cam.id);
    if (!marker) {
      marker = L.marker([lat, lng], { icon: mapMarkerIcon(cam), title: `${cam.name} (${cam.id})`, keyboard: true, riseOnHover: true }).addTo(state.leafletMap);
      marker.on('click', () => { selectTrafficCamera(cam.id, false, false); marker.openPopup(); });
      state.leafletMarkers.set(cam.id, marker);
    } else {
      marker.setLatLng([lat, lng]);
      marker.setIcon(mapMarkerIcon(cam));
      marker.options.title = `${cam.name} (${cam.id})`;
    }
    marker.bindPopup(mapPopupHtml(cam), { maxWidth: 260, closeButton: true, autoPan: true });
    marker.bindTooltip(`${cam.name} · ${cam.id}`, { direction: 'top', offset: [0, -25], opacity: 0.96, className: 'hanoi-map-tooltip' });
  }
  for (const [id, marker] of state.leafletMarkers.entries()) {
    if (!keep.has(id)) { marker.remove(); state.leafletMarkers.delete(id); }
  }
  updateMapStatus();
}

const MAP_PROVIDER_FAILOVER_V3 = true;
const BASE_OSM = {
  name: 'OpenStreetMap',
  url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap contributors</a>'
};
const BASE_HOT = {
  name: 'OpenStreetMap HOT',
  url: 'https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png',
  subdomains: 'abc',
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap contributors</a> \u00b7 Tiles courtesy of <a href="https://www.hotosm.org/" target="_blank" rel="noopener">Humanitarian OpenStreetMap Team</a>'
};
const BASE_OSMFR = {
  name: 'OpenStreetMap France',
  url: 'https://{s}.tile.openstreetmap.fr/osmfr/{z}/{x}/{y}.png',
  subdomains: 'abc',
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap contributors</a> \u00b7 Tiles &copy; <a href="https://www.openstreetmap.fr/" target="_blank" rel="noopener">OpenStreetMap France</a>'
};
const BUILTIN_MAP_PROVIDERS = [BASE_OSM, BASE_HOT, BASE_OSMFR];
let baseMapProviders = [...BUILTIN_MAP_PROVIDERS];
let mapVisibilityListenerAdded = false;

const MAP_TILE_RUNTIME = {
  base: {
    mapKey: 'leafletMap', layerKey: 'baseTileLayer',
    errorSelector: '#mapTileError', statusSelector: '#mapBaseStatus',
    timer: null, activeIndex: -1, hasLoaded: false, loadedCount: 0, failedCount: 0
  },
  route: {
    mapKey: 'routeMap', layerKey: 'routeBaseTileLayer',
    errorSelector: '#scenarioRouteMapError', statusSelector: null,
    timer: null, activeIndex: -1, hasLoaded: false, loadedCount: 0, failedCount: 0
  }
};

function mountMapTileLayer(kind, index) {
  const runtime = MAP_TILE_RUNTIME[kind];
  const map = state[runtime.mapKey];
  if (!map || typeof L === 'undefined') return;

  if (runtime.timer) window.clearTimeout(runtime.timer);
  runtime.timer = null;

  const provider = baseMapProviders[index];
  const errorEl = $(runtime.errorSelector);
  const statusEl = runtime.statusSelector ? $(runtime.statusSelector) : null;

  if (!provider) {
    if (errorEl) errorEl.hidden = false;
    if (statusEl) statusEl.textContent = 'B\u1ea3n \u0111\u1ed3 n\u1ec1n kh\u00f4ng kh\u1ea3 d\u1ee5ng';
    return;
  }

  const oldLayer = state[runtime.layerKey];
  if (oldLayer && map.hasLayer(oldLayer)) map.removeLayer(oldLayer);

  runtime.activeIndex = index;
  runtime.hasLoaded = false;
  runtime.loadedCount = 0;
  runtime.failedCount = 0;

  const options = {
    maxZoom: 19,
    attribution: provider.attribution,
    updateWhenIdle: true,
    keepBuffer: 1
  };
  if (provider.subdomains) options.subdomains = provider.subdomains;

  const layer = L.tileLayer(provider.url, options);
  state[runtime.layerKey] = layer;

  if (errorEl) errorEl.hidden = true;
  if (statusEl) statusEl.textContent = `\u0110ang t\u1ea3i b\u1ea3n \u0111\u1ed3 \u00b7 ${provider.name}`;

  layer.on('loading', () => {
    if (runtime.timer) window.clearTimeout(runtime.timer);
    runtime.timer = null;
    runtime.loadedCount = 0;
    runtime.failedCount = 0;
  });

  layer.on('tileload', () => {
    if (state[runtime.layerKey] !== layer) return;
    runtime.loadedCount += 1;
    runtime.hasLoaded = true;
    if (errorEl) errorEl.hidden = true;
    if (statusEl) statusEl.textContent =
      `B\u1ea3n \u0111\u1ed3 \u0111\u01b0\u1eddng ph\u1ed1 \u00b7 ${provider.name}`;
  });

  layer.on('tileerror', () => {
    if (state[runtime.layerKey] !== layer) return;
    runtime.failedCount += 1;
    if (runtime.timer) return;

    runtime.timer = window.setTimeout(() => {
      runtime.timer = null;
      if (state[runtime.layerKey] !== layer) return;

      const total = runtime.loadedCount + runtime.failedCount;
      const errorRatio = total > 0 ? runtime.failedCount / total : 0;
      const partialFailure =
        runtime.failedCount >= 2 && errorRatio >= 0.35;

      if (!partialFailure) return;

      if (index + 1 < baseMapProviders.length) {
        console.warn(
          `Tile provider ${provider.name} has partial failures; trying fallback.`
        );
        mountMapTileLayer(kind, index + 1);
      } else {
        if (errorEl) errorEl.hidden = false;
        if (statusEl) statusEl.textContent =
          `B\u1ea3n \u0111\u1ed3 kh\u00f4ng kh\u1ea3 d\u1ee5ng \u00b7 \u0111\u00e3 th\u1eed ${baseMapProviders.length} ngu\u1ed3n`;
      }
    }, 4000);
  });

  layer.addTo(map);
}

function configureBaseMapProviders(config) {
  const configured = String(config.tile_url_template || '').trim();
  const isCarto = /basemaps\.cartocdn\.com/i.test(configured);
  const isCustom = configured && !isCarto && configured !== BASE_OSM.url;

  const providers = isCustom ? [{
    name: String(config.base_map_name || 'B\u1ea3n \u0111\u1ed3 c\u1ea5u h\u00ecnh'),
    url: configured,
    attribution: String(config.base_attribution || BASE_OSM.attribution),
    ...(configured.includes('{s}') ? { subdomains: 'abc' } : {})
  }, ...BUILTIN_MAP_PROVIDERS] : [...BUILTIN_MAP_PROVIDERS];

  const seen = new Set();
  baseMapProviders = providers.filter(provider => {
    if (!provider.url || seen.has(provider.url)) return false;
    seen.add(provider.url);
    return true;
  });

  for (const kind of ['base', 'route']) {
    const runtime = MAP_TILE_RUNTIME[kind];
    const map = state[runtime.mapKey];
    if (!map) continue;

    const activeLayer = state[runtime.layerKey];
    const activeUrl = activeLayer?._url;
    const urlIsKnown = baseMapProviders.some(provider => provider.url === activeUrl);

    if (isCustom && activeUrl !== configured) {
      mountMapTileLayer(kind, 0);
    } else if (!activeLayer || !urlIsKnown) {
      mountMapTileLayer(kind, 0);
    }
    // Keep a working fallback layer instead of replacing it unnecessarily.
  }
}

function isOverviewPollingAllowed() {
  return !document.hidden &&
    state.currentPage === 'overview' &&
    Boolean($('#page-overview')?.classList.contains('active'));
}

function observeMapContainerResize(map, holder, observerKey) {
  if (state[observerKey]) state[observerKey].disconnect();

  const invalidate = () => {
    if (!holder.isConnected || !map) return;
    const width = holder.clientWidth;
    const height = holder.clientHeight;
    if (width < 1 || height < 1) return;

    const size = map.getSize();
    if (Math.abs(size.x - width) > 1 || Math.abs(size.y - height) > 1) {
      map.invalidateSize({ pan: false, debounceMoveend: true });
    }
  };

  if (typeof ResizeObserver !== 'undefined') {
    const observer = new ResizeObserver(() => requestAnimationFrame(invalidate));
    state[observerKey] = observer;
    observer.observe(holder);
  }

  requestAnimationFrame(invalidate);
  window.setTimeout(invalidate, 250);
  window.setTimeout(invalidate, 800);
}

const HANOI_INNER_CITY = [21.0285, 105.8542];
function initLeafletMap() {
  const holder = $('#leafletMap');
  if (!holder || state.leafletMap) return;

  if (typeof L === 'undefined') {
    const error = $('#mapTileError');
    const status = $('#mapBaseStatus');
    if (error) error.hidden = false;
    if (status) status.textContent = 'Kh\u00f4ng t\u1ea3i \u0111\u01b0\u1ee3c th\u01b0 vi\u1ec7n b\u1ea3n \u0111\u1ed3';
    return;
  }

  state.leafletMap = L.map(holder, {
    center: HANOI_INNER_CITY, zoom: 11, minZoom: 8, maxZoom: 19,
    zoomControl: false, preferCanvas: true, scrollWheelZoom: true, tap: true
  });

  mountMapTileLayer('base', 0);
  observeMapContainerResize(state.leafletMap, holder, 'mapResizeObserver');
  loadMapConfig();
}
function initScenarioRouteMap() {
  const holder = $('#scenarioRouteMap');
  if (!holder || state.routeMap) return;

  if (typeof L === 'undefined') {
    const message = $('#scenarioRouteMapError');
    if (message) message.hidden = false;
    return;
  }

  state.routeMap = L.map(holder, {
    center: HANOI_INNER_CITY, zoom: 11, minZoom: 8, maxZoom: 19,
    zoomControl: true, preferCanvas: true, scrollWheelZoom: true, tap: true
  });

  mountMapTileLayer('route', 0);
  observeMapContainerResize(state.routeMap, holder, 'routeMapResizeObserver');
}
async function loadMapConfig() {
  try {
    const config = await api('/api/map/config');
    state.mapConfig = config;

    const requestedInterval = Number(config.camera_refresh_seconds);
    state.mapPollSeconds = Number.isFinite(requestedInterval)
      ? Math.max(45, Math.min(120, requestedInterval))
      : 45;

    configureBaseMapProviders(config);

    if (state.leafletMap && config.traffic_tile_url_template) {
      if (state.trafficTileLayer) state.leafletMap.removeLayer(state.trafficTileLayer);
      state.trafficTileLayer = L.tileLayer(config.traffic_tile_url_template, {
        opacity: 0.72, maxZoom: 19,
        attribution: config.traffic_attribution || 'Nh\u00e0 cung c\u1ea5p l\u1edbp giao th\u00f4ng',
        updateWhenIdle: true, keepBuffer: 1
      });
      const button = $('#mapTrafficLayerButton');
      if (button) {
        button.title = `L\u1edbp d\u1eef li\u1ec7u t\u1eeb ${config.traffic_layer_name || 'nh\u00e0 cung c\u1ea5p b\u00ean ngo\u00e0i'}`;
        button.setAttribute('aria-label', button.title);
      }
    } else if (state.leafletMap && state.trafficTileLayer) {
      state.leafletMap.removeLayer(state.trafficTileLayer);
      state.trafficTileLayer = null;
    }

    const trafficBtn = $('#mapTrafficLayerButton');
    if (trafficBtn && !config.traffic_layer_enabled) {
      trafficBtn.title = 'Ch\u01b0a c\u1ea5u h\u00ecnh nh\u00e0 cung c\u1ea5p lu\u1ed3ng giao th\u00f4ng tr\u1ef1c ti\u1ebfp; xem .env.example';
    }

    updateMapStatus();
    scheduleMapPolling();
  } catch (error) {
    state.mapConfig = { traffic_layer_enabled: false, camera_refresh_seconds: 45 };
    state.mapPollSeconds = 45;
    scheduleMapPolling();
    console.warn('Map configuration unavailable; using fallback tile providers.', error);
  }
}
function scheduleMapPolling() {
  if (state.mapPollTimer) window.clearInterval(state.mapPollTimer);

  state.mapPollTimer = window.setInterval(() => {
    if (isOverviewPollingAllowed()) void pollCameraMapData();
  }, state.mapPollSeconds * 1000);

  if (!mapVisibilityListenerAdded) {
    document.addEventListener('visibilitychange', () => {
      if (isOverviewPollingAllowed() && state.dashboard) void pollCameraMapData();
    });
    mapVisibilityListenerAdded = true;
  }
}
async function pollCameraMapData() {
  if (state.mapPollBusy) return;
  state.mapPollBusy = true;

  try {
    const payload = await api('/api/cameras');
    if (!Array.isArray(payload.items)) return;

    const previous = state.cameras || [];
    const next = payload.items;

    const signature = (items) => JSON.stringify(items.map((cam) => [
      cam.id, cam.name, cam.area, cam.lat, cam.lng, cam.status, cam.ci,
      cam.telemetry_live, cam.speed, cam.vehicles, cam.occupancy,
      cam.updated, cam.telemetry_source, cam.feed?.available,
      cam.feed?.source_type, cam.feed?.stream_url
    ]));

    updateMapStatus(payload);

    // Payload unchanged: no marker, table, or select rebuilding.
    if (signature(previous) === signature(next)) return;

    const selectedId = state.selectedCameraId;
    const before = previous.find((cam) => cam.id === selectedId);
    const after = next.find((cam) => cam.id === selectedId);

    const selectedView = (cam) => cam ? JSON.stringify([
      cam.id, cam.name, cam.area, cam.lat, cam.lng,
      cam.status, cam.ci, cam.telemetry_live,
      cam.feed?.available, cam.feed?.source_type, cam.feed?.stream_url
    ]) : '';

    state.cameras = next;
    renderMapMarkers(next);
    renderCameraPicker();
    renderScenarioCameraOptions();
    renderCameraTable();

    if (!after && next.length) {
      selectTrafficCamera(next[0].id, false, false);
    } else if (after && selectedView(before) !== selectedView(after)) {
      selectTrafficCamera(selectedId, false, false);
    }

    updateMapStatus(payload);
  } catch (error) {
    const sync = $('#mapSyncStatus');
    if (sync) sync.textContent =
      `M?t k?t n?i API ? ${new Date().toLocaleTimeString('vi-VN')}`;
  } finally {
    state.mapPollBusy = false;
  }
}


function setMapLayer(layer) {
  if (layer === 'traffic' && !state.trafficTileLayer) {
    toast('Chưa có nguồn giao thông trực tiếp', 'Bản đồ đường phố là bản đồ thật; lớp màu ùn tắc cần URL tile/API được cấp phép. Cấu hình TRAFFIC_TILE_URL_TEMPLATE trong .env.', 'error');
    layer = 'status';
  }
  state.mapLayer = layer;
  const legend = $('#mapLegend');
  if (legend) {
    legend.innerHTML = layer === 'forecast'
      ? '<span><i class="legend-dot free"></i>CI thấp · dưới 0,30*</span><span><i class="legend-dot busy"></i>CI trung bình · 0,30–0,54*</span><span><i class="legend-dot heavy"></i>CI cao · từ 0,55*</span>'
      : '<span><i class="legend-dot free"></i>Đang hoạt động*</span><span><i class="legend-dot busy"></i>Cần chú ý*</span><span><i class="legend-dot heavy"></i>Mất kết nối*</span>';
  }
  if (state.leafletMap && state.trafficTileLayer) {
    if (layer === 'traffic') state.trafficTileLayer.addTo(state.leafletMap);
    else state.leafletMap.removeLayer(state.trafficTileLayer);
  }
  $$('[data-map-layer]').forEach(button => button.classList.toggle('selected', button.dataset.mapLayer === layer));
  renderMapMarkers(state.cameras);
  const sync = $('#mapSyncStatus');
  if (sync && layer === 'traffic') sync.textContent = `Lớp giao thông: ${state.mapConfig?.traffic_layer_name || 'nhà cung cấp ngoài'}`;
}

function renderCameraPicker() {
  const select = $('#trafficCameraSelect');
  if (!select) return;
  const current = state.selectedCameraId;
  select.innerHTML = state.cameras.map(cam => `<option value="${escapeHtml(cam.id)}">${escapeHtml(cam.id)} · ${escapeHtml(cam.name)}</option>`).join('');
  if (current && state.cameras.some(cam => cam.id === current)) select.value = current;
  else if (state.cameras.length) select.value = state.cameras[0].id;
}

function selectTrafficCamera(cameraId, forceReload = false, focusMap = false) {
  const cam = state.cameras.find(item => item.id === cameraId);
  if (!cam) return;
  const same = state.selectedCameraId === cameraId;
  state.selectedCameraId = cameraId;
  renderMapMarkers(state.cameras);
  if (focusMap && state.leafletMap && Number.isFinite(Number(cam.lat)) && Number.isFinite(Number(cam.lng))) state.leafletMap.panTo([Number(cam.lat), Number(cam.lng)], { animate: true, duration: 0.45 });
  const select = $('#trafficCameraSelect');
  if (select && select.value !== cameraId) select.value = cameraId;
  $('#trafficCameraTitle').textContent = cam.name;
  $('#trafficCameraArea').textContent = `${cam.area || 'Chưa cấu hình khu vực'} · ${cam.id}`;
  const statusBadge = $('#trafficCameraStatus');
  const feed = cam.feed || {};
  statusBadge.className = `camera-feed-badge ${feed.available ? 'online' : 'offline'}`;
  statusBadge.textContent = feed.available ? (feed.source_type === 'demo_fallback' ? 'CLIP MẪU CHUNG' : 'ĐÃ GÁN CLIP') : 'CHƯA CÓ CLIP';
  const video = $('#trafficCameraVideo');
  const empty = $('#trafficCameraEmpty');
  const tag = $('#trafficVideoTag');
  const upload = $('#trafficFeedUpload');
  const remove = $('#removeCameraFeed');
  if (feed.available) {
    const url = cameraFeedUrl(feed);
    if (forceReload || !same || video.dataset.feedUrl !== url) {
      video.pause();
      video.muted = true;
      video.loop = true;
      video.src = url;
      video.dataset.feedUrl = url;
      video.load();
    }
    video.hidden = false;
    empty.hidden = true;
    if (video.paused) {
  const playAttempt = video.play();
  if (playAttempt && typeof playAttempt.catch === 'function') {
    playAttempt.catch(() => {});
  }
}
    tag.hidden = false;
    tag.textContent = feed.source_type === 'demo_fallback' ? 'CLIP MẪU CHUNG' : 'CLIP ĐÃ GÁN';
    $('#trafficFeedHint').textContent = feed.notice || 'Video đã được gán cho đúng mã camera này.';
    remove.disabled = false;
  } else {
    if (!same || forceReload || video.dataset.feedUrl) {
      video.pause(); video.removeAttribute('src'); delete video.dataset.feedUrl; video.load();
    }
    video.hidden = true;
    empty.hidden = false;
    tag.hidden = true;
    $('#trafficFeedHint').textContent = feed.notice || 'Chưa có clip gắn cho điểm này. Ảnh phía trên chỉ minh họa giao thông đô thị.';
    remove.disabled = true;
  }
  upload.dataset.cameraId = cameraId;
  upload.disabled = state.publicDemoMode;
  $('.camera-feed-upload')?.classList.toggle('is-disabled', state.publicDemoMode);
  remove.disabled = state.publicDemoMode || remove.disabled;

}

async function uploadTrafficCameraClip(file) {
  const cameraId = state.selectedCameraId;
  if (!cameraId || !file) return;
  if (file.size > 500 * 1024 * 1024) { toast('Video quá lớn', 'Giới hạn upload demo là 500 MB.', 'error'); return; }
  $('#trafficFeedHint').textContent = `Đang tải ${file.name} lên điểm ${cameraId}…`;
  const button = $('.camera-feed-upload');
  if (button) button.classList.add('is-uploading');
  try {
    const form = new FormData(); form.append('file', file);
    const result = await api(`/api/cameras/${encodeURIComponent(cameraId)}/feed`, { method:'POST', body:form });
    const cam = state.cameras.find(item => item.id === cameraId);
    if (cam) cam.feed = result.feed;
    selectTrafficCamera(cameraId, true);
    toast('Đã gán video', `${file.name} · ${cameraId}`);
  } catch (error) {
    $('#trafficFeedHint').textContent = `Không thể gán video: ${error.message}`;
    toast('Gán video thất bại', error.message, 'error');
  } finally {
    if (button) button.classList.remove('is-uploading');
    $('#trafficFeedUpload').value = '';
  }
}

async function removeTrafficCameraClip() {
  const cameraId = state.selectedCameraId;
  if (!cameraId) return;
  try {
    const result = await api(`/api/cameras/${encodeURIComponent(cameraId)}/feed`, {method:'DELETE'});
    const cam = state.cameras.find(item => item.id === cameraId);
    if (cam) cam.feed = result.feed;
    selectTrafficCamera(cameraId, true);
    toast('Đã gỡ clip', `${cameraId} · clip đã gán được gỡ khỏi điểm này.`);
  } catch (error) { toast('Không gỡ được clip', error.message, 'error'); }
}

function renderDashboard(data, system) {
  const s = data.summary || {}; const cameras = data.cameras || []; const alerts = data.alerts || [];
  const hasLiveTelemetry = data.data_source === 'live_telemetry';
  const configuredCount = s.camera_total ?? cameras.length;
  $('#metricOnline').textContent = numberVN(s.camera_online || 0);
  $('#metricOnlineFoot').textContent = hasLiveTelemetry ? 'Có telemetry mới trong 180 giây' : 'Chưa có telemetry từ camera';
  $('#onlineProgress').style.width = `${Math.min(100, (s.camera_online || 0) / Math.max(1, configuredCount) * 100)}%`;
  $('#metricVehicles').textContent = s.total_vehicles == null ? '—' : numberVN(s.total_vehicles);
  $('#speedProgress').style.width = hasLiveTelemetry && s.average_speed != null ? `${Math.min(100, Math.max(0, s.average_speed) / 42 * 100)}%` : '0%';
  $('#metricSpeed').textContent = s.average_speed == null ? '—' : numberVN(s.average_speed, 1);
  $('#metricCritical').textContent = s.critical_points == null ? '—' : numberVN(s.critical_points);
  $('#navCameraCount').textContent = configuredCount;
  $('#navAlertCount').textContent = alerts.length;
  $('#mapOnlineCount').textContent = `${s.camera_online || 0}/${configuredCount} điểm có telemetry mới`;
  $('#updatedAt').textContent = `Đồng bộ ${new Date(data.generated_at || Date.now()).toLocaleTimeString('vi-VN', { hour:'2-digit', minute:'2-digit' })}`;
  const noticeTitle = hasLiveTelemetry ? 'Đã nhận telemetry gần đây.' : 'Chưa kết nối nguồn giao thông trực tiếp.';
  const noticeBody = hasLiveTelemetry
    ? 'Một số chỉ số được tính từ telemetry nhận qua collector đã cấu hình. Dự báo chỉ xuất hiện khi có chuỗi đo và forecaster hợp lệ.'
    : 'Các marker là vị trí tham chiếu, không phải camera đã xác minh. Không có dữ liệu đo trực tiếp; chỉ số ùn tắc và dự báo được để trống thay vì mô phỏng.';
  $('#demoNotice').style.display = 'flex';
  $('#demoNotice').innerHTML = `<span class="notice-icon">i</span><span><strong>${noticeTitle}</strong> ${noticeBody}</span><button id="noticeDismiss" aria-label="Ẩn thông báo">×</button>`;
  $('#systemPill').innerHTML = `<span class="status-pulse"></span><span>${hasLiveTelemetry ? 'Có telemetry mới' : 'Chưa có nguồn trực tiếp'}</span>`;
  $('#noticeDismiss').addEventListener('click',()=>{$('#demoNotice').style.display='none';});
  renderForecast(data.forecast, hasLiveTelemetry ? s.average_ci : null);
  renderTrend(data.trend || []);
  renderAlertPreview(alerts);
  state.cameras = cameras;
  if (!state.selectedCameraId || !cameras.some(cam => cam.id === state.selectedCameraId)) state.selectedCameraId = cameras[0]?.id || null;
  renderMapMarkers(cameras);
  renderCameraPicker();
  if (state.selectedCameraId) selectTrafficCamera(state.selectedCameraId, false);
  if (system?.model_status) { state.modelStatus = system.model_status; renderModelStatus(); }
}

function renderForecast(forecast, avgCI) {
  const predictions = Array.isArray(forecast?.predictions) ? forecast.predictions : [];
  const horizonList = $('#horizonList');
  const currentLevel = $('#forecastCurrentLevel');
  const clearPlot = () => {
    $('#forecastLine').setAttribute('d', '');
    $('#forecastArea').setAttribute('d', '');
    $('#forecastPoints').innerHTML = '';
  };
  // The overview may not infer future traffic from fixture measurements. Only a successful
  // observation-driven provider result can populate this chart.
  if (!predictions.length || !['production', 'remote_api'].includes(forecast?.mode)) {
    $('#forecastCurrent').textContent = '—';
    currentLevel.textContent = 'Chưa đủ dữ liệu';
    currentLevel.className = 'level-chip unavailable';
    clearPlot();
    horizonList.innerHTML = '<div class="empty-state">Chưa có dự báo hợp lệ cho +5 / +10 / +15 phút.</div>';
    $('#forecastAdvice').textContent = forecast?.mode === 'error'
      ? 'Nguồn dự báo lỗi. Kiểm tra cấu hình model/API; không sử dụng kết quả cũ.'
      : 'Chưa có chuỗi telemetry và forecaster đã kiểm định. Phân tích frame chỉ mô tả trạng thái trong clip, không tự tạo ra dự báo tương lai.';
    $('#forecastModeFoot').textContent = forecast?.mode === 'error'
      ? 'Dự báo không khả dụng · kiểm tra nguồn xử lý'
      : 'Chỉ hiển thị dự báo từ quan sát và bộ dự báo hợp lệ';
    return;
  }
  $('#forecastCurrent').textContent = avgCI == null ? '—' : numberVN(avgCI, 2);
  currentLevel.className = `level-chip ${avgCI == null ? 'unavailable' : severityFromCI(avgCI)}`;
  currentLevel.textContent = avgCI == null ? 'Chưa có CI hiện tại' : levelVN(severityFromCI(avgCI));
  const points = predictions.slice(0,3).map((p,i)=>({x:[146.7,293.3,440][i],ci:Number(p.ci)}));
  const y = value => 156 - Math.max(0, Math.min(1, value)) * 144;
  let line = '';
  if (avgCI != null) line = `M0,${y(avgCI)} `;
  line += points.map(p => `L${p.x},${y(p.ci)}`).join(' ');
  $('#forecastLine').setAttribute('d', line);
  $('#forecastArea').setAttribute('d', avgCI == null ? '' : `${line} L440,156 L0,156 Z`);
  const circleValues = avgCI == null ? points : [{x:0,ci:Number(avgCI)}, ...points];
  $('#forecastPoints').innerHTML = circleValues.map(p=>`<circle cx="${p.x}" cy="${y(p.ci)}" r="4" fill="#fff" stroke="#719d80" stroke-width="2"/>`).join('');
  horizonList.innerHTML = predictions.map(p => {
    const cls = severityFromCI(p.ci);
    return `<div class="horizon-row"><span class="horizon-time">+${escapeHtml(p.horizon_minutes)} phút</span><span class="horizon-bar"><i class="${cls}" style="width:${Math.round(Number(p.ci)*100)}%;background:${cls==='free'?'#86b28e':cls==='busy'?'#d3a052':cls==='heavy'?'#cb7b68':'#b85a4d'}"></i></span><strong>${numberVN(p.ci,2)}</strong><span class="level-dot ${cls}"></span></div>`;
  }).join('');
  const lastKey = severityFromCI(Number(predictions[predictions.length-1].ci));
  $('#forecastAdvice').textContent = lastKey === 'gridlock' ? 'Có tín hiệu nguy cơ cao trong chân trời dự báo. Xác minh nguồn và hiện trường trước quyết định điều hành.' : lastKey === 'heavy' ? 'Mô hình báo CI cao ở chân trời xa nhất. So sánh telemetry các tuyến thay thế trước khi khuyến nghị.' : lastKey === 'busy' ? 'Mô hình báo xu hướng đông đúc; tiếp tục theo dõi chuỗi đo mới nhất.' : 'Dự báo hiện chưa cho thấy CI cao trong chân trời đã trả về.';
  $('#forecastModeFoot').textContent = forecast.mode === 'production' ? 'Kết quả từ bộ dự báo cục bộ · cần kiểm định độc lập' : `Kết quả từ API dự báo${forecast.latency_ms != null ? ` · ${numberVN(forecast.latency_ms,1)} ms` : ''} · cần kiểm định độc lập`;
}

function renderTrend(trend) {
  const container = $('#trendChart'); if (!container) return;
  const data = Array.isArray(trend) ? trend : [];
  if (data.length < 2) {
    container.innerHTML = '<div class="empty-state trend-empty">Chưa có chuỗi quan sát theo thời gian. Không tạo đường xu hướng từ số liệu mẫu.</div>';
    $('#trendAvg').textContent = '—';
    const note = $('#trendSourceNote'); if (note) note.textContent = 'Cần telemetry có timestamp trước khi vẽ xu hướng.';
    return;
  }
  const width = 680, height = 140, pad = { l: 8, r: 10, t: 12, b: 25 };
  const plotW = width-pad.l-pad.r, plotH = height-pad.t-pad.b;
  const x = i => pad.l + i * plotW / Math.max(1,data.length-1);
  const y = v => pad.t + (1 - Math.max(0,Math.min(1,v))) * plotH;
  const d = data.map((p,i)=>`${i?'L':'M'}${x(i).toFixed(1)},${y(p.ci).toFixed(1)}`).join(' ');
  const area = `${d} L${x(data.length-1)},${height-pad.b} L${x(0)},${height-pad.b} Z`;
  const grid = [0,.25,.5,.75,1].map(v=>`<line x1="${pad.l}" y1="${y(v)}" x2="${width-pad.r}" y2="${y(v)}" stroke="#eef1eb" stroke-dasharray="3 5"/><text x="0" y="${y(v)+3}" fill="#b3bcb4" font-size="9">${v.toFixed(2)}</text>`).join('');
  const labels = data.map((p,i)=>i%2===0 || i===data.length-1 ? `<text x="${x(i)}" y="${height-5}" text-anchor="middle" fill="#aeb7af" font-size="9">${escapeHtml(p.time)}</text>`:'').join('');
  container.innerHTML = `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" role="img" aria-label="Chỉ số ùn tắc đã quan sát theo thời gian">${grid}<path d="${area}" fill="#eaf2e9"/><path d="${d}" fill="none" stroke="#79a284" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>${data.map((p,i)=>`<circle cx="${x(i)}" cy="${y(p.ci)}" r="2.7" fill="#fff" stroke="#79a284" stroke-width="1.8"/>`).join('')}${labels}</svg>`;
  $('#trendAvg').textContent = numberVN(data.reduce((sum,p)=>sum+Number(p.ci),0)/data.length,2);
  const note = $('#trendSourceNote'); if (note) note.textContent = 'CI tính từ telemetry đã quan sát; không phải dự báo.';
}

function renderAlertPreview(alerts) {
  const items = alerts.filter(a => a.status !== 'Đã ghi nhận').slice(0,3);
  $('#alertPreview').innerHTML = items.map(a => `<div class="preview-alert"><span class="alert-symbol ${alertClass(a.type)}">${alertIcon(a.type)}</span><div class="preview-alert-copy"><strong>${escapeHtml(a.title)}</strong><p>${escapeHtml(a.location)} · ${escapeHtml(a.time)}</p></div><span class="preview-alert-time">CI ${numberVN(a.ci,2)}</span></div>`).join('') || '<div class="empty-state">Không có cảnh báo đang mở.</div>';
  $('#alertFooterText').textContent = items.length ? `${items.length} sự kiện cần theo dõi` : 'Không có cảnh báo đang mở';
}

function renderCameraTable() {
  const tbody = $('#cameraTableBody'); if (!tbody) return;
  const query = ($('#cameraSearch')?.value || '').trim().toLocaleLowerCase('vi-VN');
  const status = $('#cameraStatusFilter')?.value || 'all';
  const all = state.cameras || [];
  const filtered = all.filter(cam => {
    const live = Boolean(cam.telemetry_live);
    const statusMatch = status === 'all' || (status === 'live' && live) || (status === 'unverified' && !live);
    return statusMatch && `${cam.id} ${cam.name} ${cam.area}`.toLocaleLowerCase('vi-VN').includes(query);
  });
  tbody.innerHTML = filtered.map(cam => {
    const live = Boolean(cam.telemetry_live);
    const ci = live && Number.isFinite(Number(cam.ci)) ? Number(cam.ci) : null;
    const speed = live && Number.isFinite(Number(cam.speed)) ? Number(cam.speed) : null;
    const vehicles = live && Number.isFinite(Number(cam.vehicles)) ? Number(cam.vehicles) : null;
    const cls = ci == null ? 'reference' : severityFromCI(ci);
    return `<tr><td><div class="location-cell"><span class="camera-thumb">▣</span><div><strong>${escapeHtml(cam.name)}</strong><small>${escapeHtml(cam.id)} · ${escapeHtml(cam.area)}</small></div></div></td><td><span class="status-badge ${live ? 'online' : 'unverified'}">${live ? 'Có telemetry mới' : 'Chưa có telemetry'}</span></td><td>${vehicles == null ? '—' : numberVN(vehicles)}</td><td>${speed == null ? '—' : `${numberVN(speed,1)} km/h`}</td><td><div class="ci-cell"><span class="ci-value">${ci == null ? '—' : numberVN(ci,2)}</span><span class="ci-track"><i class="${cls}" style="width:${ci == null ? 0 : Math.round(ci*100)}%"></i></span></div></td><td>${escapeHtml(live ? (cam.updated || 'Đã nhận telemetry') : 'Vị trí tham chiếu')}</td><td><button class="table-action" data-camera-detail="${escapeHtml(cam.id)}" aria-label="Xem chi tiết">↗</button></td></tr>`;
  }).join('') || '<tr><td colspan="7" style="text-align:center;padding:28px;color:#9ba69e">Không tìm thấy điểm giám sát phù hợp.</td></tr>';
  $('#cameraCountBadge').textContent = `${filtered.length} điểm`;
  $('#cameraTableFoot').textContent = `Hiển thị ${filtered.length} / ${all.length} điểm · số liệu chỉ xuất hiện khi có telemetry`;
  $('#camTotalMetric').textContent = all.length;
  $('#camOnlineMetric').textContent = all.filter(c=>c.telemetry_live).length;
  $('#camIssueMetric').textContent = all.filter(c=>!c.telemetry_live).length;
}

function openCameraModal(cameraId) {
  const cam = state.cameras.find(c => c.id === cameraId); if (!cam) return;
  const live = Boolean(cam.telemetry_live);
  $('#cameraModalTitle').textContent = cam.name;
  const value = v => live && v != null && Number.isFinite(Number(v)) ? numberVN(v, 2) : '—';
  $('#cameraModalContent').innerHTML = `<div class="feed-meta" style="margin-bottom:13px"><strong>${escapeHtml(cam.id)}</strong><span>${escapeHtml(cam.area)}</span><span>${live ? 'Có telemetry mới' : 'Chưa có telemetry'}</span></div><div class="modal-detail-grid"><div class="modal-stat"><span>Phương tiện ghi nhận</span><strong>${value(cam.vehicles)}</strong></div><div class="modal-stat"><span>Tốc độ đo</span><strong>${live && cam.speed != null ? `${numberVN(cam.speed,1)} <small style="font-size:10px">km/h</small>` : '—'}</strong></div><div class="modal-stat"><span>Độ chiếm dụng</span><strong>${live && cam.occupancy != null ? `${numberVN(cam.occupancy*100,0)}<small style="font-size:10px">%</small>` : '—'}</strong></div><div class="modal-stat"><span>Chỉ số ùn tắc</span><strong>${value(cam.ci)}</strong></div></div><div class="modal-note">${live ? 'Số liệu đến từ collector RTSP đã gửi telemetry gần đây. Hãy kiểm tra timestamp, camera calibration và chất lượng nguồn trước khi dùng cho quyết định điều hành.' : 'Đây là điểm tham chiếu trên bản đồ, chưa xác minh vị trí thiết bị camera và chưa có telemetry thật. Hãy gán clip để phân tích nội dung clip hoặc kết nối nguồn camera được cấp quyền.'}</div>`;
  $('#cameraModal').hidden = false;
  $('#cameraModalAlerts').onclick = () => { $('#cameraModal').hidden = true; setPage('alerts'); $('#alertSearch').value = cam.name; renderAlertFeed(); };
}

function renderAlertFeed() {
  const container = $('#alertFeed'); if (!container) return;
  const query = ($('#alertSearch')?.value || '').trim().toLocaleLowerCase('vi-VN');
  const type = $('#alertTypeFilter')?.value || 'all';
  const filtered = state.alerts.filter(a => (type === 'all' || a.type === type) && `${a.title} ${a.location} ${a.message} ${a.camera_id}`.toLocaleLowerCase('vi-VN').includes(query));
  container.innerHTML = filtered.map(a => `<article class="feed-alert"><span class="alert-symbol ${alertClass(a.type)}">${alertIcon(a.type)}</span><div class="feed-main"><h3>${escapeHtml(a.title)}</h3><p>${escapeHtml(a.message)}</p><div class="feed-meta"><span>⌖ <strong>${escapeHtml(a.location)}</strong></span><span>Mã điểm: ${escapeHtml(a.camera_id)}</span><span>CI: <strong>${numberVN(a.ci,2)}</strong></span></div></div><div class="feed-side"><small>${escapeHtml(a.time)}</small><select data-alert-status="${escapeHtml(a.id)}" aria-label="Cập nhật trạng thái"><option ${a.status==='Chưa xử lý'?'selected':''}>Chưa xử lý</option><option ${a.status==='Đang theo dõi'?'selected':''}>Đang theo dõi</option><option ${a.status==='Đã ghi nhận'?'selected':''}>Đã ghi nhận</option></select></div></article>`).join('') || '<div class="empty-state" style="padding:30px;text-align:center;color:#9ba69e">Không tìm thấy cảnh báo phù hợp.</div>';
  $('#alertCountBadge').textContent = `${filtered.length} sự kiện`;
  $('#criticalAlertCount').textContent = state.alerts.filter(a=>a.type==='critical'&&a.status!=='Đã ghi nhận').length;
  $('#watchAlertCount').textContent = state.alerts.filter(a=>a.type==='warning'&&a.status!=='Đã ghi nhận').length;
  $('#resolvedAlertCount').textContent = state.alerts.filter(a=>a.status==='Đã ghi nhận').length;
  $('#navAlertCount').textContent = state.alerts.filter(a=>a.status!=='Đã ghi nhận').length;
}

function renderAnalytics() {
  if (!state.dashboard) return;
  const trend = state.dashboard.trend || [];
  const c = $('#analyticsChart');
  if (!trend.length) {
    if (c) c.innerHTML = '<div class="empty-state">Chưa có telemetry lịch sử. Số liệu mẫu đã được ẩn để tránh nhầm với thực tế.</div>';
    const dist = $('#distributionList'); if (dist) dist.innerHTML = '<div class="empty-state">Chưa có CI thực theo camera.</div>';
    return;
  }
  if (c) {
    const width=700,height=250,pad={l:14,r:12,t:15,b:28},pw=width-pad.l-pad.r,ph=height-pad.t-pad.b;
    const x=i=>pad.l+i*pw/Math.max(1,trend.length-1); const y=v=>pad.t+(1-Math.max(0,Math.min(1,v)))*ph;
    const d=trend.map((p,i)=>`${i?'L':'M'}${x(i)},${y(p.ci)}`).join(' '); const area=`${d} L${x(trend.length-1)},${height-pad.b} L${x(0)},${height-pad.b} Z`;
    const grid=[0,.25,.5,.75,1].map(v=>`<g><line x1="${pad.l}" y1="${y(v)}" x2="${width-pad.r}" y2="${y(v)}" stroke="#edf0eb" stroke-dasharray="4 5"/><text x="0" y="${y(v)+3}" font-size="10" fill="#aeb8b0">${v.toFixed(2)}</text></g>`).join('');
    c.innerHTML=`<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" role="img" aria-label="Diễn biến chỉ số ùn tắc">${grid}<path d="${area}" fill="#edf4eb"/><path d="${d}" fill="none" stroke="#6f9b7b" stroke-width="2.7" stroke-linecap="round" stroke-linejoin="round"/>${trend.map((p,i)=>`<circle cx="${x(i)}" cy="${y(p.ci)}" r="3" fill="white" stroke="#6f9b7b" stroke-width="2"/>`).join('')}${trend.map((p,i)=>i%2===0||i===trend.length-1?`<text x="${x(i)}" y="${height-7}" text-anchor="middle" font-size="10" fill="#aeb8b0">${escapeHtml(p.time)}</text>`:'').join('')}</svg>`;
  }
  const cams=state.cameras.filter(c=>c.status!=='offline'); const groups=[{key:'free',count:0},{key:'busy',count:0},{key:'heavy',count:0},{key:'gridlock',count:0}];
  cams.forEach(cam=>{const g=groups.find(x=>x.key===severityFromCI(cam.ci));if(g)g.count++});
  const dist=$('#distributionList'); if(dist) dist.innerHTML=groups.map(g=>`<div class="distribution-item"><span>${levelVN(g.key)}</span><span class="distribution-track"><i style="width:${cams.length?g.count/cams.length*100:0}%;background:${g.key==='free'?'#7eaa87':g.key==='busy'?'#d3a052':g.key==='heavy'?'#ca7a67':'#b85b4e'}"></i></span><strong>${g.count}</strong></div>`).join('');
}

function renderModelStatus() {
  const status = state.modelStatus || {};
  const provider = state.providerStatus || status.forecast_provider || {};
  const providerName = provider.provider || 'local_onnx';
  const remote = providerName === 'remote_http';
  const configured = Boolean(provider.configured);
  const remoteBlocked = remote && Boolean(provider.public_blocked);
  const remoteReady = remote && configured && !remoteBlocked;
  const localForecasterReady = Boolean(status.forecaster?.ready);
  const demo = remote ? !remoteReady : !localForecasterReady;
  const title = remote ? 'Dịch vụ dự báo bên ngoài' : 'Mô hình cục bộ';
  const message = remote
    ? (remoteBlocked ? 'API đang bị khóa ở chế độ demo công khai. Chỉ mở sau khi có xác thực, hạn mức và giám sát.' : configured ? 'Endpoint đã được cấu hình. Kết nối/đầu ra sẽ được xác nhận khi gửi yêu cầu dự báo.' : 'Chưa cấu hình URL hợp lệ. Hãy điền .env và khởi động lại server.')
    : (localForecasterReady ? 'Forecaster ONNX đã được nạp; cần xác minh trên dữ liệu độc lập trước vận hành.' : 'Chưa nạp forecaster ONNX; hệ thống giữ nhãn dữ liệu minh họa cho đến khi cấu hình mô hình thật.');
  const titleEl = $('#providerUiTitle'); if (titleEl) titleEl.textContent = title;
  const messageEl = $('#providerUiMessage'); if (messageEl) messageEl.textContent = message;
  const statusEl = $('#providerUiStatus'); if (statusEl) statusEl.textContent = remote ? (remoteBlocked ? 'API bị khóa trong chế độ demo' : configured ? 'Đã cấu hình endpoint' : 'Chưa cấu hình endpoint') : (localForecasterReady ? 'Đã nạp forecaster' : 'Chưa có forecaster đã huấn luyện');
  const hostEl = $('#providerEndpointHost'); if (hostEl) hostEl.textContent = remote && provider.endpoint_host ? provider.endpoint_host : 'CPU · ONNX Runtime';
  const dot = $('#providerStatusDot'); if (dot) dot.className = `provider-status-dot ${!demo ? 'ready' : 'demo'}`;
  const chip = $('#modelModeChip'); if (chip) { chip.textContent = remote ? (remoteBlocked ? 'API BỊ KHÓA' : configured ? 'API ĐÃ CẤU HÌNH' : 'CHƯA CẤU HÌNH') : (localForecasterReady ? 'SẴN SÀNG' : 'DỮ LIỆU MẪU'); chip.className = `mode-chip ${demo ? 'demo' : 'production'}`; }
  const det = status.detector || {}, fore = status.forecaster || {};
  const detState = $('#detectorArtifactState'); if (detState) detState.innerHTML = `<span class="tiny-indicator ${det.ready?'green-indicator':'red-indicator'}"></span>${det.ready?'Đã nạp vào runtime':det.exists?'Có file nhưng chưa nạp':'Chưa có file'}`;
  const foreState = $('#forecasterArtifactState'); if (foreState) foreState.innerHTML = `<span class="tiny-indicator ${fore.ready?'green-indicator':'red-indicator'}"></span>${fore.ready?'Đã nạp vào runtime':fore.exists?'Có file nhưng chưa nạp':'Chưa có file'}`;
  const manifestState = $('#manifestArtifactState'); if (manifestState) manifestState.innerHTML = '<span class="tiny-indicator green-indicator"></span>Đã có manifest';
  const detectorName = $('#detectorFileName'); if (detectorName) detectorName.textContent = det.file ? `artifacts/current/${det.file}` : 'artifacts/current/detector.onnx';
  const forecasterName = $('#forecasterFileName'); if (forecasterName) forecasterName.textContent = fore.file ? `artifacts/current/${fore.file}` : 'artifacts/current/forecaster.onnx';
  const runtimeProvider = $('#runtimeProvider'); if (runtimeProvider) runtimeProvider.textContent = status.provider || 'CPUExecutionProvider';
  const runtimeDetector = $('#runtimeDetector'); if (runtimeDetector) runtimeDetector.textContent = det.ready ? 'Đã nạp' : 'Chưa sẵn sàng';
  const detectorIcon = $('#runtimeDetectorIcon'); if (detectorIcon) { detectorIcon.textContent = det.ready ? '✓' : '–'; detectorIcon.className = `runtime-state ${det.ready?'runtime-ok':''}`; }
  const runtimeForecaster = $('#runtimeForecaster'); if (runtimeForecaster) runtimeForecaster.textContent = fore.ready ? 'Đã nạp' : (remoteReady ? 'Đang dùng API ngoài' : 'Chưa sẵn sàng');
  const forecasterIcon = $('#runtimeForecasterIcon'); if (forecasterIcon) { forecasterIcon.textContent = fore.ready || remoteReady ? '✓' : '–'; forecasterIcon.className = `runtime-state ${(fore.ready || remoteReady)?'runtime-ok':''}`; }
  const features = $('#runtimeFeatures'); if (features) features.textContent = status.feature_count ?? 8;
  const horizons = $('#runtimeHorizons'); if (horizons) horizons.textContent = (status.horizons_minutes || [5,10,15]).join(' · ') + ' phút';
  const loadedAt = $('#runtimeLoadedAt'); if (loadedAt) loadedAt.textContent = status.loaded_at ? new Date(status.loaded_at).toLocaleTimeString('vi-VN',{hour:'2-digit',minute:'2-digit',second:'2-digit'}) : '—';
  const dotNav = $('#modelNavDot'); if (dotNav) dotNav.style.background = demo ? '#ce8b4c' : '#70a276';
}

function scenarioWriteBlockReason() {
  if (!state.aiccStatus) return 'Đang kiểm tra khả năng phân tích…';
  if (!canAnalyzeUploadedScenarioVideo()) return 'Tính năng phân tích video hiện chưa sẵn sàng. Vui lòng thử lại sau.';
  return '';
}

function renderScenarioInputState() {
  const file = $('#scenarioVideoFile')?.files?.[0];
  const blocked = scenarioWriteBlockReason();
  const maxMb = Number(state.aiccStatus?.upload_limit_mb || 100);
  const reason = blocked || (state.scenarioBusy ? 'Đang xử lý yêu cầu của bạn…' : '')
    || (state.scenarioPendingJobId ? 'Video đã được tiếp nhận. Hãy theo dõi lại kết quả.' : '')
    || (!file ? 'Chọn một video để bắt đầu.' : '')
    || (file.size > maxMb * 1024 * 1024 ? `Video vượt giới hạn ${maxMb} MB.` : '');
  const button = $('#scenarioAnalyzeUploadBtn');
  if (button) { button.disabled = Boolean(reason); button.title = reason; }
  const input = $('#scenarioVideoFile');
  if (input) input.disabled = state.scenarioBusy;
  const resume = $('#scenarioResumeBtn');
  if (resume) { resume.hidden = !state.scenarioPendingJobId || state.scenarioBusy; resume.disabled = state.scenarioBusy; }
  const hint = $('#scenarioInputHint');
  if (hint) hint.textContent = reason || 'Video đã chọn. Bấm Phân tích video để bắt đầu.';
  const limit = $('#scenarioUploadLimit');
  if (limit) limit.textContent = `Tối đa ${maxMb} MB và ${Math.floor(Number(state.aiccStatus?.upload_max_duration_seconds || 300) / 60)} phút.`;
  const mode = $('#scenarioRuntimeMode');
  if (mode) mode.textContent = blocked ? 'Chưa sẵn sàng' : 'Sẵn sàng nhận video';
}

function renderScenarioCameraOptions() {
  const origin = $('#scenarioOriginSelect');
  const destination = $('#scenarioDestinationSelect');
  if (!origin || !destination || !state.cameras.length) { renderScenarioInputState(); return; }
  const oldOrigin = origin.value;
  const oldDestination = destination.value;
  const optionHtml = state.cameras.filter(cam => cam.lat != null && cam.lng != null && Number.isFinite(Number(cam.lat)) && Number.isFinite(Number(cam.lng)))
    .map(cam => `<option value="${escapeHtml(cam.id)}">${escapeHtml(cam.name)} · ${escapeHtml(cam.area || cam.id)}</option>`).join('');
  origin.innerHTML = `<option value="">Không gắn camera (tùy chọn)</option>${optionHtml || ''}`;
  destination.innerHTML = `<option value="">Chọn điểm đến…</option>${optionHtml || ''}`;
  const preferredOrigin = oldOrigin || state.selectedCameraId || '';
  origin.value = state.cameras.some(cam => cam.id === preferredOrigin) ? preferredOrigin : '';
  const preferredDest = oldDestination || '';
  destination.value = state.cameras.some(cam => cam.id === preferredDest) && preferredDest !== origin.value
    ? preferredDest : '';
  if (destination.value === origin.value) destination.value = '';
  const button = $('#scenarioFindRoutesBtn'); if (button) button.disabled = !origin.value || !destination.value || state.publicDemoMode;
  renderScenarioInputState();
}

function routePointFromCamera(cameraId) {
  const camera = state.cameras.find(item => item.id === cameraId);
  if (!camera || camera.lat == null || camera.lng == null || !Number.isFinite(Number(camera.lat)) || !Number.isFinite(Number(camera.lng))) {
    throw new Error('Điểm đã chọn chưa có tọa độ hợp lệ.');
  }
  return { label: camera.name, camera_id: camera.id, lat: Number(camera.lat), lng: Number(camera.lng) };
}

function renderScenarioReview(result) {
  state.scenarioResult = result;
  state.scenarioJobId = result.job_id || state.scenarioJobId;
  const summary = result.summary || {};
  $('#scenarioCurrentState').textContent = summary.current_state || '—';
  $('#scenarioCurrentCI').textContent = summary.current_ci == null ? '—' : numberVN(summary.current_ci, 2);
  $('#scenarioSnapshotCount').textContent = numberVN((result.frames || []).length);
  $('#scenarioFramesCaption').textContent = `${numberVN(summary.sampled_frames || 0)} mẫu · ${numberVN(summary.saved_frames || 0)} ảnh đã lưu · ${result.filename || 'video'}`;
  const forecast = result.trend_forecast || {};
  const badge = $('#scenarioForecastBadge');
  const text = $('#scenarioForecastText');
  const cards = $('#scenarioForecastCards');
  cards.innerHTML = '';
  if (forecast.available && Array.isArray(forecast.predictions) && forecast.predictions.length) {
    badge.textContent = `Baseline tham khảo · ${forecast.confidence || 'độ tin cậy thấp'}`;
    badge.className = 'status-tag status-warn';
    text.textContent = forecast.message || 'Ngoại suy xu hướng từ clip, chưa phải dự báo được kiểm định.';
    cards.innerHTML = forecast.predictions.filter(item => [5, 10].includes(Number(item.horizon_minutes))).map(item =>
      `<article class="scenario-prediction-card"><small>+${numberVN(item.horizon_minutes)} phút</small><strong>${escapeHtml(item.level || levelVN(item.level_key))}</strong><span>CI ${numberVN(item.ci, 2)}</span></article>`).join('');
  } else {
    badge.textContent = 'Chưa đủ điều kiện'; badge.className = 'status-tag';
    text.textContent = forecast.message || 'Chưa có baseline +5/+10. Không ngoại suy nếu clip không giữ đúng thời gian, quá ngắn hoặc thiếu mẫu.';
  }
  const hanoiForecast = result.hanoi_forecast || {};
  const hanoiBadge = $('#hanoiForecastBadge');
  const hanoiText = $('#hanoiForecastText');
  const hanoiCards = $('#hanoiForecastCards');
  hanoiCards.innerHTML = '';
  if (hanoiForecast.available && Array.isArray(hanoiForecast.predictions)) {
    hanoiBadge.textContent = 'Thử nghiệm · chưa hiệu chuẩn';
    hanoiBadge.className = 'status-tag status-warn';
    hanoiText.textContent = hanoiForecast.message || 'Xác suất tham khảo từ LSTM, chưa được hiệu chuẩn.';
    hanoiCards.innerHTML = hanoiForecast.predictions.map(item =>
      `<article class="scenario-prediction-card"><small>+${numberVN(item.horizon_seconds)} giây</small><strong>${escapeHtml(item.predicted_class === 'Tac_duong' ? 'Có nguy cơ tắc' : 'Thông thoáng')}</strong><span>Xác suất ùn tắc ${numberVN(Number(item.congestion_probability) * 100, 1)}% · chưa hiệu chuẩn</span></article>`).join('');
  } else {
    hanoiBadge.textContent = 'Chưa đủ dữ liệu';
    hanoiBadge.className = 'status-tag';
    hanoiText.textContent = hanoiForecast.message || 'LSTM cần ít nhất 30 giây clip được lấy mẫu ở 1 frame/giây.';
  }
  const frames = result.frames || [];
  const gallery = $('#scenarioFrameGallery');
  gallery.innerHTML = frames.length ? frames.map(frame => {
    const tone = frame.state_key === 'free' ? 'free' : frame.state_key === 'busy' ? 'busy' : 'heavy';
    const rawNote = (frame.raw_state_label && frame.raw_state_label !== frame.state_label) ? ` <small class="text-muted">(thô: ${escapeHtml(frame.raw_state_label)})</small>` : '';
    const confNote = frame.model_confidence != null ? ` · tin cậy: ${numberVN(frame.model_confidence * 100, 0)}%` : '';
    return `<article class="scenario-frame-card"><a class="scenario-frame-image" href="${escapeHtml(frame.url)}" target="_blank" rel="noopener"><img loading="lazy" src="${escapeHtml(frame.url)}" alt="Khung hình ${numberVN(frame.timestamp_seconds, 1)} giây · ${escapeHtml(frame.state_label)}"></a><div class="scenario-frame-copy"><div class="scenario-frame-head"><strong>${escapeHtml(frame.state_label)}</strong>${rawNote}<span class="scenario-state-dot ${tone}"></span></div><small>t = ${numberVN(frame.timestamp_seconds, 1)} giây · CI ${numberVN(frame.ci, 2)}</small><small>${numberVN(frame.vehicle_count)} xe phát hiện · chiếm dụng ảnh ${numberVN((frame.occupancy_ratio || 0) * 100, 1)}%${confNote}</small></div></article>`;
  }).join('') : '<div class="empty-state">Không có ảnh minh chứng trong kết quả.</div>';
  $('#scenarioExportBtn').disabled = !frames.length;
  $('#scenarioJobStatus').textContent = `${result.filename || 'Video'} đã xử lý xong. Nhãn ảnh là baseline thị giác; không phải xác nhận từ camera trực tiếp.`;
  $('#scenarioFindRoutesBtn').disabled = state.publicDemoMode || !$('#scenarioOriginSelect').value || !$('#scenarioDestinationSelect').value;
  if (result.source_camera_id && $('#scenarioOriginSelect').value !== result.source_camera_id) {
    $('#scenarioNotice').innerHTML = '<span class="notice-icon">i</span><span>Clip được gán cho một camera khác với điểm đi hiện chọn. Chọn đúng điểm A nếu muốn đính kèm bối cảnh video vào so sánh tuyến.</span>';
  }
}

function videoRequestError(status, raw) {
  let payload;
  try { payload = raw ? JSON.parse(raw) : {}; } catch { payload = {}; }
  const detail = typeof payload?.detail === 'string' ? payload.detail
    : Array.isArray(payload?.detail) ? payload.detail.map(item => item.msg).join('; ') : '';
  const message = status === 409 ? 'Hệ thống đang phân tích một video khác. Vui lòng thử lại sau'
    : status === 413 ? detail || 'Video vượt giới hạn kích thước hoặc thời lượng.'
    : status === 403 ? 'Tính năng phân tích video hiện đang bị tắt.'
    : detail || `Không thể tải video lên (HTTP ${status}). Vui lòng thử lại.`;
  const error = new Error(message); error.status = status; return error;
}

function uploadScenarioVideo(form, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/scenarios/video-review');
    xhr.timeout = 10 * 60 * 1000;
    xhr.upload.onprogress = event => onProgress(event.loaded, event.lengthComputable ? event.total : null);
    xhr.upload.onload = () => {
      $('#scenarioProgressText').textContent = 'Đã tải video lên. Đang kiểm tra video…';
      $('#scenarioJobStatus').textContent = 'Đã tải video lên. Đang kiểm tra video…';
      $('#scenarioAnalyzeUploadBtn').textContent = 'Đang kiểm tra video…';
    };
    xhr.onload = () => {
      if (xhr.status !== 202) { reject(videoRequestError(xhr.status, xhr.responseText)); return; }
      try {
        const job = JSON.parse(xhr.responseText);
        if (!/^[a-f0-9]{32}$/.test(job.job_id || '')) throw new Error('Phản hồi tiếp nhận video không hợp lệ.');
        resolve(job);
      } catch (error) { reject(error); }
    };
    xhr.onerror = () => reject(new Error('Kết nối tải video bị gián đoạn. Kiểm tra mạng rồi thử lại.'));
    xhr.ontimeout = () => reject(new Error('Tải video quá thời gian chờ. Chưa xác nhận được tác vụ; kiểm tra kết nối trước khi gửi lại.'));
    xhr.onabort = () => reject(new Error('Đã dừng tải video.'));
    xhr.send(form);
  });
}

function rememberScenarioJob(jobId) {
  state.scenarioPendingJobId = jobId;
  try {
    if (jobId) sessionStorage.setItem('scenario-review-job', jobId);
    else sessionStorage.removeItem('scenario-review-job');
  } catch {}
}

async function pollScenarioReviewJob(jobId, filename) {
  const started = Date.now();
  let failures = 0;
  $('#scenarioProgressWrap').hidden = false;
  while (Date.now() - started < 30 * 60 * 1000) {
    let job;
    try {
      job = await api(`/api/scenarios/video-reviews/${encodeURIComponent(jobId)}`, {timeoutMs: 15000});
      failures = 0;
    } catch (error) {
      if (error.status && error.status < 500) { rememberScenarioJob(null); throw error; }
      failures++;
      $('#scenarioProgressText').textContent = 'Kết nối bị gián đoạn. Đang thử kết nối lại để theo dõi kết quả…';
      $('#scenarioJobStatus').textContent = 'Video đã được tiếp nhận; không cần tải lại video.';
      if (failures >= 5) throw new Error('Chưa kết nối lại được. Bấm “Theo dõi lại kết quả” để tiếp tục; không cần tải lại video.');
      await new Promise(resolve => setTimeout(resolve, Math.min(failures * 1000, 5000)));
      continue;
    }
    if (!['queued', 'running', 'succeeded', 'failed'].includes(job.status)) throw new Error('Trạng thái phân tích không hợp lệ. Hãy theo dõi lại kết quả.');
    const pct = Math.max(0, Math.min(100, Number(job.progress_percent || 0)));
    $('#scenarioProgressBar').style.width = `${pct}%`;
    $('#scenarioProgressText').textContent = job.status === 'queued' ? 'Đang chờ phân tích…'
      : !job.frames_total ? 'Đang chuẩn bị phân tích…'
      : `Đang phân tích ${numberVN(job.frames_decoded || 0)} / ${numberVN(job.frames_total)} khung hình · ${pct}%`;
    $('#scenarioJobStatus').textContent = job.status === 'queued' ? 'Video đã được tiếp nhận.' : 'Đang phân tích video của bạn…';
    if (job.status === 'succeeded') {
      rememberScenarioJob(null);
      state.scenarioJobId = jobId;
      const result = { ...job.result, job_id: jobId };
      renderScenarioReview(result);
      $('#scenarioProgressText').textContent = 'Hoàn tất';
      toast('Đã phân tích video', `${numberVN((result.frames || []).length)} ảnh minh chứng.`);
      return result;
    }
    if (job.status === 'failed') { rememberScenarioJob(null); throw new Error(job.error || 'Không xử lý được video.'); }
    await new Promise(resolve => setTimeout(resolve, 900));
  }
  throw new Error('Phân tích đang mất nhiều thời gian. Bấm “Theo dõi lại kết quả” để kiểm tra tiến độ.');
}

async function startScenarioReview() {
  if (state.scenarioBusy) return;
  const pending = state.scenarioPendingJobId;
  const file = $('#scenarioVideoFile').files?.[0];
  if (!pending && (scenarioWriteBlockReason() || !file)) { renderScenarioInputState(); return; }
  if (!pending && file.size > Number(state.aiccStatus?.upload_limit_mb || 100) * 1024 * 1024) { renderScenarioInputState(); return; }
  const button = $('#scenarioAnalyzeUploadBtn');
  state.scenarioBusy = true;
  renderScenarioInputState();
  $('#scenarioProgressWrap').hidden = false;
  try {
    let jobId = pending;
    if (!jobId) {
      button.textContent = 'Đang tải video lên…';
      $('#scenarioProgressBar').style.width = '0%';
      $('#scenarioProgressText').textContent = 'Đang bắt đầu tải video…';
      $('#scenarioJobStatus').textContent = 'Đang tải video của bạn…';
      const form = new FormData(); form.append('file', file);
      const originId = $('#scenarioOriginSelect')?.value;
      if (originId) form.append('camera_id', originId);
      form.append('preserve_real_timing', $('#scenarioPreserveTiming').checked ? 'true' : 'false');
      const job = await uploadScenarioVideo(form, (loaded, total) => {
        const pct = total ? Math.max(0, Math.min(100, Math.floor(loaded / total * 100))) : null;
        if (pct !== null) $('#scenarioProgressBar').style.width = `${pct}%`;
        const progress = pct === null ? `${numberVN(loaded / 1024 / 1024, 1)} MB` : `${pct}% · ${numberVN(loaded / 1024 / 1024, 1)} / ${numberVN(total / 1024 / 1024, 1)} MB`;
        $('#scenarioProgressText').textContent = `Đang tải video lên: ${progress}`;
      });
      jobId = job.job_id;
      rememberScenarioJob(jobId);
    }
    button.textContent = 'Đang phân tích…';
    $('#scenarioProgressBar').style.width = '0%';
    await pollScenarioReviewJob(jobId, file?.name || 'Video');
  } catch (error) {
    $('#scenarioJobStatus').textContent = error.message;
    $('#scenarioProgressText').textContent = error.message;
    toast('Chưa hoàn tất phân tích', error.message, 'error');
  } finally {
    state.scenarioBusy = false;
    button.textContent = 'Phân tích video';
    renderScenarioInputState();
  }
}

function renderScenarioRoutes(payload) {
  state.routeResult = payload;
  const banner = $('#scenarioRouteMessage');
  const forecast = payload.source_analysis || {};
  const lines = [];
  lines.push(payload.message || 'Đã lấy phương án từ dịch vụ chỉ đường.');
  if (forecast.risk_message) lines.push(forecast.risk_message);
  if (!forecast.source_is_origin && forecast.notice) lines.push(forecast.notice);
  banner.textContent = lines.join(' ');
  const list = $('#scenarioRouteList');
  const items = payload.routes || [];
  if (!items.some(route => route.route_id === state.selectedRouteId)) {
    state.selectedRouteId = payload.recommended_route_id || items[0]?.route_id || null;
  }
  list.innerHTML = items.map(route => {
    const recommended = route.route_id === payload.recommended_route_id;
    const status = route.traffic_aware ? 'Có telemetry mới' : 'Thời gian mạng lưới';
    const duration = `${numberVN((route.duration_seconds || 0) / 60, 0)} phút`;
    const distance = `${numberVN((route.distance_m || 0) / 1000, 1)} km`;
    const ci = route.mean_live_ci == null ? 'Chưa có dữ liệu' : numberVN(route.mean_live_ci, 2);
    return `<button type="button" class="scenario-route-option ${recommended ? 'recommended' : ''} ${state.selectedRouteId === route.route_id ? 'selected' : ''}" data-route-id="${escapeHtml(route.route_id)}"><span class="scenario-route-top"><strong>${escapeHtml(route.route_id)} · ${recommended ? 'ƯU TIÊN' : 'Phương án'}</strong><span>${escapeHtml(duration)}</span></span><span class="scenario-route-meta">${escapeHtml(distance)} · ${numberVN(route.live_camera_count || 0)} điểm telemetry mới</span><span class="scenario-route-meta">${escapeHtml(status)} · CI gần tuyến: ${escapeHtml(ci)}</span></button>`;
  }).join('') || '<div class="empty-state">Dịch vụ chỉ đường không có tuyến phù hợp.</div>';
  list.querySelectorAll('[data-route-id]').forEach(button => button.addEventListener('click', () => {
    state.selectedRouteId = button.dataset.routeId;
    renderScenarioRoutesSelectionOnly();
    drawScenarioRouteMap(payload, state.selectedRouteId);
  }));
  renderScenarioRoutesSelectionOnly();
  drawScenarioRouteMap(payload, state.selectedRouteId);
}
function renderScenarioRoutesSelectionOnly() {
  $$('#scenarioRouteList [data-route-id]').forEach(button => {
    button.classList.toggle('selected', button.dataset.routeId === state.selectedRouteId);
  });
}
function drawScenarioRouteMap(payload, selectedRouteId) {
  if (!state.routeMap || typeof L === 'undefined') return;
  state.routeMapLayers.forEach(layer => { try { state.routeMap.removeLayer(layer); } catch {} });
  state.routeMapLayers = [];
  const bounds = [];
  for (const route of payload.routes || []) {
    const coordinates = route.geometry?.coordinates || [];
    const latLngs = coordinates.map(point => [Number(point[1]), Number(point[0])]).filter(pair => Number.isFinite(pair[0]) && Number.isFinite(pair[1]));
    if (latLngs.length < 2) continue;
    const selected = route.route_id === selectedRouteId;
    const color = selected ? '#27764f' : (route.recommended ? '#d39a36' : '#83948a');
    const line = L.polyline(latLngs, { color, weight: selected ? 6 : 3, opacity: selected ? .95 : .62, lineJoin: 'round' }).addTo(state.routeMap);
    line.bindTooltip(`${route.route_id}${route.recommended ? ' · ưu tiên' : ''} · ${numberVN((route.duration_seconds || 0)/60,0)} phút`);
    line.on('click', () => { state.selectedRouteId = route.route_id; renderScenarioRoutesSelectionOnly(); drawScenarioRouteMap(payload, route.route_id); });
    state.routeMapLayers.push(line); bounds.push(...latLngs);
  }
  for (const point of [payload.origin, payload.destination]) {
    if (point && Number.isFinite(Number(point.lat)) && Number.isFinite(Number(point.lng))) {
      const marker = L.circleMarker([Number(point.lat), Number(point.lng)], { radius: 7, color: '#fff', weight: 2, fillColor: point === payload.origin ? '#2b7550' : '#2e6ac2', fillOpacity: 1 }).addTo(state.routeMap);
      marker.bindTooltip(escapeHtml(point.label || 'Điểm')); state.routeMapLayers.push(marker); bounds.push([Number(point.lat), Number(point.lng)]);
    }
  }
  if (bounds.length > 1) state.routeMap.fitBounds(bounds, { padding: [24, 24], maxZoom: 14, animate: true });
}

async function findScenarioRoutes() {
  const button = $('#scenarioFindRoutesBtn');
  button.disabled = true; button.textContent = 'Đang tìm tuyến…';
  try {
    const origin = routePointFromCamera($('#scenarioOriginSelect').value);
    const destination = routePointFromCamera($('#scenarioDestinationSelect').value);
    if (origin.camera_id === destination.camera_id) throw new Error('Điểm đi và điểm đến phải khác nhau.');
    const payload = await api('/api/routes/alternatives', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ origin, destination, analysis_job_id: state.scenarioJobId || null }) });
    renderScenarioRoutes(payload);
    toast('Đã tìm tuyến', payload.traffic_aware ? 'Có telemetry mới gần các phương án; vẫn cần xác minh tại hiện trường.' : 'Chưa có telemetry RTSP mới bao phủ tuyến; kết quả thời gian chỉ là tham chiếu.');
  } catch (error) {
    $('#scenarioRouteMessage').textContent = error.message;
    toast('Không tìm được tuyến', error.message, 'error');
  } finally {
    button.textContent = 'Tìm các tuyến thay thế';
    button.disabled = state.publicDemoMode || !$('#scenarioOriginSelect').value || !$('#scenarioDestinationSelect').value;
  }
}
function exportScenarioFrames() {
  const result = state.scenarioResult;
  if (!result?.frames?.length) return;
  const headers = ['filename','timestamp_seconds','state_label','raw_state_label','ci','raw_ci','vehicle_count','occupancy_ratio','model_confidence','aicc_sequence_label'];
  downloadCsv('hanoi-traffic-video-evidence.csv', headers,
    result.frames.map(frame => [
      frame.filename,
      frame.timestamp_seconds,
      frame.state_label,
      frame.raw_state_label || frame.state_label,
      frame.ci,
      frame.raw_ci ?? frame.ci,
      frame.vehicle_count,
      frame.occupancy_ratio,
      frame.model_confidence != null ? frame.model_confidence : '',
      frame.aicc_sequence_label || ''
    ]));
}

function renderProviderStatus() {
  const status = state.providerStatus || {};
  const nodes = {
    providerDisplayName: status.display_name || 'Mô hình cục bộ',
    providerConfigState: status.public_blocked ? 'Bị khóa trong public demo' : status.configured ? 'Đã cấu hình' : 'Chưa cấu hình',
    providerHost: status.endpoint_host || 'Không dùng API ngoài',
    providerTimeout: status.timeout_seconds ? `${numberVN(status.timeout_seconds, 1)} giây` : 'Không áp dụng',
    providerSecretState: status.secret_configured ? 'Đã đặt trong môi trường' : (status.provider === 'remote_http' ? 'Chưa đặt (nếu API cần khóa)' : 'Không cần cho bộ cục bộ'),
  };
  for (const [id, value] of Object.entries(nodes)) {
    const el = $(`#${id}`); if (el) el.textContent = value;
  }
}

function downloadText(filename, content, type = 'text/plain;charset=utf-8') {
  const blob = new Blob(['\ufeff', content], {type}); const url = URL.createObjectURL(blob); const a = document.createElement('a'); a.href=url; a.download=filename; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
}
function downloadCsv(filename, headers, rows) {
  const csv = [headers, ...rows].map(row => row.map(value => `"${String(value ?? '').replace(/"/g,'""')}"`).join(',')).join('\r\n');
  downloadText(filename, csv, 'text/csv;charset=utf-8'); toast('Đã xuất tệp', filename);
}
function exportCameraCsv(){downloadCsv('hanoi-traffic-cameras.csv',['Mã điểm','Tên điểm','Khu vực','Trạng thái','Phương tiện','Tốc độ km/h','Occupancy','CI'],state.cameras.map(c=>[c.id,c.name,c.area,cameraStatusVN(c.status),c.vehicles,c.speed,c.occupancy,c.ci]));}
function exportAlertCsv(){downloadCsv('hanoi-traffic-alerts.csv',['Mã cảnh báo','Loại','Tiêu đề','Vị trí','Mã camera','Thời điểm','CI','Nội dung','Trạng thái'],state.alerts.map(a=>[a.id,a.type,a.title,a.location,a.camera_id,a.time,a.ci,a.message,a.status]));}
function exportSummary(){const s=state.dashboard?.summary||{};downloadCsv('hanoi-traffic-summary.csv',['Chỉ số','Giá trị'],[['Điểm giám sát',s.camera_total],['Camera trực tuyến',s.camera_online],['Phương tiện ghi nhận',s.total_vehicles],['Tốc độ trung bình km/h',s.average_speed],['Điểm CI cao',s.critical_points],['Chế độ dữ liệu',state.dashboard?.mode||'demo']]);}

async function uploadArtifact(kind, file) {
  if (!file) return;
  const form = new FormData(); form.append('file', file);
  try {
    const result = await api(`/api/models/upload?kind=${encodeURIComponent(kind)}`, {method:'POST',body:form});
    toast('Đã tải tệp lên', `${result.filename} · ${(result.bytes/1024/1024).toFixed(2)} MB. Nạp lại runtime để áp dụng.`);
    const status = await api('/api/models/status'); state.modelStatus=status; renderModelStatus();
  } catch(error) { toast('Tải tệp thất bại',error.message,'error'); }
}
async function reloadModels() {
  const btn=$('#reloadModels'); if(btn){btn.disabled=true;btn.textContent='Đang nạp…';}
  try { const result=await api('/api/models/reload',{method:'POST'}); state.modelStatus=result.status; renderModelStatus(); await loadAll({quiet:true}); toast('Đã nạp lại tệp', result.status.mode==='production'?'Bộ dự báo cục bộ đã sẵn sàng.':'Chưa có forecaster ONNX hợp lệ; dự báo cục bộ vẫn là minh họa.'); }
  catch(error){toast('Không nạp được mô hình',error.message,'error');}
  finally{if(btn){btn.disabled=false;btn.innerHTML='↻ Nạp lại mô hình';}}
}
async function saveManifest() {
  let parsed; try { parsed=JSON.parse($('#manifestEditor').value); }
  catch(error) { $('#manifestValidation').innerHTML='<span class="tiny-indicator red-indicator"></span> JSON không hợp lệ';toast('Không lưu được manifest','Kiểm tra dấu phẩy, dấu ngoặc và định dạng JSON.','error');return; }
  if(!parsed.feature_schema || !Array.isArray(parsed.feature_schema.ordered_input_keys)){toast('Manifest thiếu cấu trúc','Cần feature_schema.ordered_input_keys là một mảng.','error');return;}
  try {const result=await api('/api/models/manifest',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(parsed)});state.manifest=parsed;$('#manifestValidation').innerHTML='<span class="tiny-indicator green-indicator"></span> Đã lưu manifest; nạp lại runtime để áp dụng';toast('Đã lưu cấu hình',result.message);}
  catch(error){toast('Không lưu được manifest',error.message,'error');}
}
async function updateAlertStatus(alertId, status) {
  try {await api(`/api/alerts/${encodeURIComponent(alertId)}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({status})});const item=state.alerts.find(a=>a.id===alertId);if(item)item.status=status;renderAlertFeed();renderAlertPreview(state.alerts);$('#navAlertCount').textContent=state.alerts.filter(a=>a.status!=='Đã ghi nhận').length;toast('Đã cập nhật trạng thái',`${alertId} · ${status}`);}
  catch(error){toast('Không cập nhật được',error.message,'error');}
}

async function refreshAiccStatus() {
  try { state.aiccStatus = await api('/api/aicc/status'); }
  catch { state.aiccStatus = null; }
  renderScenarioInputState();
}

function bindEvents() {
  document.addEventListener('click', event => {
    const nav = event.target.closest('[data-page]'); if (nav) { setPage(nav.dataset.page); return; }
    const detail = event.target.closest('[data-camera-detail]'); if(detail){openCameraModal(detail.dataset.cameraDetail);return;}
  });
  $('#mobileMenu').addEventListener('click',()=>$('#sidebar').classList.toggle('open'));
  $('#scenarioOriginSelect')?.addEventListener('change', () => {
    renderScenarioInputState();
    if (state.scenarioResult?.source_camera_id && state.scenarioResult.source_camera_id !== $('#scenarioOriginSelect').value) {
      $('#scenarioNotice').innerHTML = '<span class="notice-icon">i</span><span>Clip gần nhất thuộc camera khác điểm đi đang chọn. Hãy phân tích lại video tại điểm A nếu muốn gắn dự báo xu hướng với tuyến này.</span>';
    }
    const routes = $('#scenarioFindRoutesBtn'); if (routes) routes.disabled = state.publicDemoMode || !$('#scenarioDestinationSelect').value;
  });
  $('#scenarioDestinationSelect')?.addEventListener('change', () => {
    const routes = $('#scenarioFindRoutesBtn'); if (routes) routes.disabled = state.publicDemoMode || !$('#scenarioOriginSelect').value || !$('#scenarioDestinationSelect').value;
  });
  $('#scenarioVideoFile')?.addEventListener('change', () => {
    const file = $('#scenarioVideoFile').files?.[0];
    const maxMb = Number(state.aiccStatus?.upload_limit_mb || 100);
    renderScenarioInputState();
    if (!state.scenarioBusy) {
      const overLimit = file && file.size > maxMb * 1024 * 1024;
      $('#scenarioJobStatus').textContent = file
        ? overLimit
          ? `${file.name} vượt giới hạn ${maxMb} MB.`
          : `${file.name} · ${(file.size/1024/1024).toFixed(1)} MB · đã chọn.`
        : 'Chưa chọn tệp từ máy tính.';
    }
  });
  $('#scenarioVideoFile')?.addEventListener('input', renderScenarioInputState);
  window.addEventListener('pageshow', renderScenarioInputState);
  renderScenarioInputState();
  $('#scenarioAnalyzeUploadBtn')?.addEventListener('click', () => startScenarioReview());
  $('#scenarioResumeBtn')?.addEventListener('click', () => startScenarioReview());
  $('#scenarioFindRoutesBtn')?.addEventListener('click', findScenarioRoutes);
  $('#scenarioExportBtn')?.addEventListener('click', exportScenarioFrames);
  $('#trafficCameraSelect').addEventListener('change',event=>selectTrafficCamera(event.target.value, false, true));
  $('#trafficFeedUpload').addEventListener('change',event=>uploadTrafficCameraClip(event.target.files?.[0]));
  $('#removeCameraFeed').addEventListener('click',removeTrafficCameraClip);
  $('#refreshBtn').addEventListener('click',()=>loadAll());
  $('#noticeDismiss').addEventListener('click',()=>{$('#demoNotice').style.display='none';});
  $('#cameraSearch').addEventListener('input',renderCameraTable); $('#cameraStatusFilter').addEventListener('change',renderCameraTable);
  $('#alertSearch').addEventListener('input',renderAlertFeed); $('#alertTypeFilter').addEventListener('change',renderAlertFeed);
  $('#alertFeed').addEventListener('change', event=>{const target=event.target.closest('[data-alert-status]');if(target)updateAlertStatus(target.dataset.alertStatus,target.value);});
  $('#cameraTableBody').addEventListener('click',event=>{const button=event.target.closest('[data-camera-detail]');if(button)openCameraModal(button.dataset.cameraDetail);});
  $('#closeCameraModal').addEventListener('click',()=>$('#cameraModal').hidden=true); $('#closeCameraModal2').addEventListener('click',()=>$('#cameraModal').hidden=true);
  $('#cameraModal').addEventListener('click',event=>{if(event.target===$('#cameraModal'))$('#cameraModal').hidden=true;});
  $('#cameraExport').addEventListener('click',exportCameraCsv); $('#alertExport').addEventListener('click',exportAlertCsv); $('#exportOverview').addEventListener('click',exportSummary); $('#analyticsExport').addEventListener('click',exportSummary);
  $$('[data-upload-kind]').forEach(input=>input.addEventListener('change',()=>{const file=input.files?.[0];uploadArtifact(input.dataset.uploadKind,file);input.value='';}));
  $('#reloadModels').addEventListener('click',reloadModels); $('#saveManifest').addEventListener('click',saveManifest);
  $('#resetManifest').addEventListener('click',async()=>{try{state.manifest=await api('/api/models/manifest');$('#manifestEditor').value=JSON.stringify(state.manifest,null,2);$('#manifestValidation').innerHTML='<span class="tiny-indicator green-indicator"></span> Đã khôi phục bản đang lưu trên backend';toast('Đã khôi phục','Đang hiển thị manifest đã lưu ở backend.');}catch(e){toast('Lỗi',e.message,'error');}});
  $('#downloadManifest').addEventListener('click',()=>{const a=document.createElement('a');a.href='/api/models/download-manifest';a.download='manifest.json';a.click();});
  $('#mapZoomIn').addEventListener('click',()=>state.leafletMap?.zoomIn());
  $('#mapZoomOut').addEventListener('click',()=>state.leafletMap?.zoomOut());
  $('#mapReset').addEventListener('click',()=>state.leafletMap?.setView(HANOI_INNER_CITY,11,{animate:true}));
  $('#mapHanoiFit').addEventListener('click',()=>state.leafletMap?.fitBounds([[20.45,105.10],[21.60,106.10]],{padding:[22,22],maxZoom:10,animate:true}));
  $$('[data-map-layer]').forEach(button=>button.addEventListener('click',()=>setMapLayer(button.dataset.mapLayer)));
  $('#manifestEditor').addEventListener('input',()=>{try{JSON.parse($('#manifestEditor').value);$('#manifestValidation').innerHTML='<span class="tiny-indicator green-indicator"></span> JSON hợp lệ · chưa lưu thay đổi';}catch{$('#manifestValidation').innerHTML='<span class="tiny-indicator red-indicator"></span> JSON chưa hợp lệ';}});
}

if (!window.__weatherTrafficPollerV1) {
  window.__weatherTrafficPollerV1 = true;
  window.setInterval(() => {
    if (document.hidden || state.currentPage !== 'overview') return;
    void refreshWeatherForecast();
    void refreshSegmentForecasts(true);
  }, 45_000);
}
$('#weatherRefreshButton')?.addEventListener('click', () => void refreshWeatherForecast(true));
$('#segmentForecastRefresh')?.addEventListener('click', () => void refreshSegmentForecasts(true));
try { const savedJob = sessionStorage.getItem('scenario-review-job'); if (/^[a-f0-9]{32}$/.test(savedJob || '')) state.scenarioPendingJobId = savedJob; } catch {}
updateDate(); bindEvents(); initLeafletMap(); setPage('video'); loadAll({quiet:true});
