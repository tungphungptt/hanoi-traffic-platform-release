const test = require('node:test');
const assert = require('node:assert');
const { directPoint, directOverview } = require('../static/map.js');

test('Open-Meteo answers from the browser are mapped to the page shape', async () => {
  global.fetch = async url => ({ ok: true, json: async () => String(url).includes('forecast_days=1') ? [
    { current: { temperature_2m: 24.04, weather_code: 3, precipitation: 0 }, daily: { temperature_2m_max: [27], temperature_2m_min: [21], precipitation_probability_max: [40] } },
  ] : { current: { time: '2026-11-18T15:00', temperature_2m: 24.3, weather_code: 61, relative_humidity_2m: 71 },
        hourly: { time: ['2026-11-18T14:00', '2026-11-18T15:00', '2026-11-18T16:00'], temperature_2m: [22, 23, 24], precipitation_probability: [10, 20, 30] },
        daily: { time: ['2026-11-18'], temperature_2m_max: [27], temperature_2m_min: [21], weather_code: [61] } } });
  const p = await directPoint(21.0285, 105.8542);
  assert.strictEqual(p.current.temperature_c, 24.3);
  assert.strictEqual(p.current.description, 'Mưa nhỏ');
  assert.strictEqual(p.hourly[0].temperature_c, 23);
  assert.strictEqual(p.daily[0].max_c, 27);
  const o = await directOverview([{ name: 'Hoàn Kiếm', latitude: 21.0285, longitude: 105.8542 }]);
  assert.strictEqual(o.areas[0].today_rain_probability_pct, 40);
  assert.strictEqual(o.areas[0].description, 'Nhiều mây');
});
