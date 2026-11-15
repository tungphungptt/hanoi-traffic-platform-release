const {test} = require('node:test');
const assert = require('node:assert/strict');
const page = require('../static/forecast.js');

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
  assert.equal(page.validateSession(session), session);
  const csv = page.forecastCsv(session.minute_forecast.anchors).trim().split('\r\n');
  assert.equal(csv.length, 3);
  assert.match(csv[1], /,1,0\.7,/);
});
