const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const page = require('../static/app.js');

test('CI levels use the shared three-level scale', () => {
  for (const [ci, key] of [[0, 'free'], [0.2999, 'free'], [0.30, 'busy'], [0.5499, 'busy'], [0.55, 'gridlock'], [1, 'gridlock']]) {
    assert.equal(page.levelKey(ci), key);
  }
});

test('time formatting and anchor lookup are causal', () => {
  assert.equal(page.fmtTime(0), '00:00');
  assert.equal(page.fmtTime(754), '12:34');
  assert.equal(page.fmtTime(3725), '1:02:05');
  const anchors = [{t: 180}, {t: 195}, {t: 210}];
  assert.equal(page.lastAtOrBefore(anchors, 100), null);
  assert.equal(page.lastAtOrBefore(anchors, 200).t, 195);
  assert.equal(page.lastAtOrBefore(anchors, 999).t, 210);
});

test('blind mode hides outcomes until the target time is reached', () => {
  const item = {target_t: 600};
  assert.equal(page.isRevealed(item, 300, true), false);
  assert.equal(page.isRevealed(item, 600, true), true);
  assert.equal(page.isRevealed(item, 300, false), true);
});

test('verdict compares against persistence honestly', () => {
  assert.equal(page.skillVerdict({evaluated: 0}).tone, 'neutral');
  assert.equal(page.skillVerdict({evaluated: 10, skill_vs_persistence: 0.3}).tone, 'good');
  assert.equal(page.skillVerdict({evaluated: 10, skill_vs_persistence: -0.2}).tone, 'bad');
  assert.equal(page.skillVerdict({evaluated: 10, skill_vs_persistence: 0.01}).tone, 'neutral');
});

test('saved sessions are validated and CSV has one row per horizon', () => {
  assert.throws(() => page.validateSession({contract: 'x'}));
  const session = {contract: page.SESSION_CONTRACT, minute_forecast: {anchors: [{t: 180, observed_ci: 0.4, horizons: [
    {horizon_minutes: 5, target_t: 480, ci: 0.6, lower: 0.4, upper: 0.8, p_gridlock: 0.6, warning: true, actual_ci: 0.7, persistence_ci: 0.4, trend_ci: 0.5, method: 'online_ensemble'},
    {horizon_minutes: 10, target_t: 780, ci: 0.5, lower: 0.3, upper: 0.7, p_gridlock: 0.3, warning: false, actual_ci: null, persistence_ci: 0.4, trend_ci: 0.6, method: 'damped_trend'}]}], observed: []}};
  const checked = page.validateSession(session);
  assert.equal(checked.cameras.length, 1);
  assert.equal(checked.cameras[0].minute_forecast, session.minute_forecast);
  assert.equal(checked.main_camera, 0);
  // Sessions saved by the previous version (single camera, contract 1.0) still open.
  assert.equal(page.validateSession({...session, contract: 'hanoiflow-forecast-session/1.0'}).cameras.length, 1);
  const csv = page.forecastCsv(session.minute_forecast.anchors).trim().split('\r\n');
  assert.equal(csv.length, 3);
  assert.match(csv[1], /,1,0\.7,/);
});

test('multi-camera sessions keep every camera and on-air intervals gate verification', () => {
  const mf = {anchors: [], observed: []};
  const session = page.validateSession({contract: page.SESSION_CONTRACT, main_camera: 1, minute_forecast: mf,
    cameras: [{id: 0, label: 'Camera 1', intervals: [[0, 100]], minute_forecast: mf},
              {id: 1, label: 'Camera 2', intervals: [[100, 400], [700, 900]], minute_forecast: mf}]});
  assert.equal(session.cameras.length, 2);
  assert.equal(session.main_camera, 1);
  assert.throws(() => page.validateSession({contract: page.SESSION_CONTRACT, cameras: [{id: 0, minute_forecast: {}}]}));
  assert.equal(page.onAir([[100, 400], [700, 900]], 250), true);
  assert.equal(page.onAir([[100, 400], [700, 900]], 500), false);
  assert.equal(page.onAir(null, 500), true);
});

test('sidebar pages and route/weather formatting', () => {
  const shell = require('../static/shell.js');
  assert.equal(shell.pageFromHash('#overview'), 'overview');
  assert.equal(shell.pageFromHash(''), 'intro');
  assert.equal(shell.pageFromHash('#intro'), 'intro');
  assert.equal(shell.pageFromHash('#how'), 'video');
  assert.equal(shell.pageFromHash('#nope'), 'video');
  const routes = require('../static/routes.js');
  assert.equal(routes.fmtDuration(600), '10 phút');
  assert.equal(routes.fmtDuration(4500), '1 giờ 15 phút');
  const map = require('../static/map.js');
  assert.equal(map.hourLabel('2026-11-16T14:00'), '14:00');
  assert.equal(map.isRainy(61), true);
  assert.equal(map.isRainy(1), false);
});

test('route ranking adds the forecast delay only to routes passing the camera', () => {
  const routes = require('../static/routes.js');
  const camera = [105.80, 21.00];
  const near = { duration_seconds: 600, geometry: { coordinates: [[105.79, 21.00], [105.81, 21.00]] } };
  const far = { duration_seconds: 700, geometry: { coordinates: [[105.79, 21.05], [105.81, 21.05]] } };
  assert.ok(routes.distanceToLine(camera, near.geometry.coordinates) < 1);
  assert.ok(routes.distanceToLine(camera, far.geometry.coordinates) > 5000);
  const jam = routes.rankRoutes([near, far], camera, 0.7);
  assert.equal(jam[0].duration_seconds, 700);          // detour wins: 600 × (1 + 0.35 × 0.7) = 747 s
  assert.equal(jam[1].passes, true);
  const free = routes.rankRoutes([near, far], camera, 0.1);
  assert.equal(free[0].duration_seconds, 600);         // free-flow forecast: no penalty
  assert.equal(routes.rankRoutes([near, far], null, 0.9)[0].duration_seconds, 600);
});

test('coordination guidance distinguishes missing data, watch, urgent, and alternate routes', () => {
  const routes = require('../static/routes.js');
  const missing = routes.decisionGuidance(Number.NaN, false);
  assert.equal(missing.badge, 'Chờ dữ liệu');
  assert.match(missing.staff, /Chọn camera/);

  const watch = routes.decisionGuidance(0.55, true, [{ passes: false }]);
  assert.equal(watch.badge, 'Theo dõi sát');
  assert.match(watch.staff, /xác nhận ùn tắc/);
  assert.match(watch.route, /tránh vùng camera/);

  const urgent = routes.decisionGuidance(0.78, true, [{ passes: true }]);
  assert.equal(urgent.badge, 'Ưu tiên xác minh');
  assert.match(urgent.staff, /bổ sung người điều tiết/);
  assert.match(urgent.routeDetail, /kiểm tra tình hình thực tế/);

  const normal = routes.decisionGuidance(0.54, true, [{ passes: false }]);
  assert.equal(normal.badge, 'Chưa có tín hiệu mạnh');
  assert.match(normal.route, /mạng đường/);
});

test('route coordination panel is present and discloses its simulation limits', () => {
  const html = fs.readFileSync(require.resolve('../static/index.html'), 'utf8');
  assert.match(html, /id="routeDecision"/);
  assert.match(html, /không tự động điều động/);
  assert.match(html, /không cung cấp dữ liệu làn hoặc tình trạng giao thông trực tiếp/);
});

test('live page helpers format age, draw the history line and flag warnings', () => {
  const live = require('../static/live.js');
  assert.equal(live.liveAge(30), '30 giây trước');
  assert.equal(live.liveAge(600), '10 phút trước');
  assert.equal(live.liveSparkline([{ epoch: 1, ci: 0.2 }]), '');
  assert.match(live.liveSparkline([{ epoch: 1, ci: 0.2 }, { epoch: 2, ci: 0.6 }]), /<path /);
  const esc = t => String(t);
  const cam = { camera_id: 'a', name: 'A', status: 'live', level_key: 'busy', observed_ci: 0.4, age_seconds: 5, samples: 700,
    horizons: [{ horizon_minutes: 5, level_key: 'gridlock', ci: 0.6, lower: 0.5, upper: 0.7, p_gridlock: 0.5, warning: true, watch: false },
               { horizon_minutes: 10, level_key: 'busy', ci: 0.5, lower: 0.4, upper: 0.6, p_gridlock: 0.25, warning: false, watch: true }], recent: [] };
  const html = live.liveCard(cam, esc);
  assert.match(html, /Cảnh báo sắp tắc/); assert.match(html, /Theo dõi/); assert.doesNotMatch(html, /Mất tín hiệu/);
  assert.match(live.liveCard({ ...cam, status: 'stale' }, esc), /Mất tín hiệu/);
  assert.match(live.liveCard({ camera_id: 'b', status: 'warming_up', span_seconds: 120, needed_seconds: 540 }, esc), /9 phút/);
});
