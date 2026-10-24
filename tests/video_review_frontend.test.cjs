const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('static/app.js', 'utf8').split('\nif (!window.__weatherTrafficPollerV1)')[0];
function setup() {
  const elements = new Map();
  const el = id => {
    if (!elements.has(id)) elements.set(id, { value: '', files: [], textContent: '', innerHTML: '', style: {}, disabled: false });
    return elements.get(id);
  };
  const context = vm.createContext({ console, AbortController, FormData, setTimeout, clearTimeout,
    document: {querySelector: el}, window: {location:{hostname:'127.0.0.1'}}, fetch: async () => { throw new TypeError('offline'); } });
  vm.runInContext(source, context);
  vm.runInContext("state.aiccStatus = {enabled:true, public_demo_mode:false, video_review_enabled:true, upload_limit_mb:500}; toast = () => {}; renderScenarioReview = result => { state.scenarioResult = result; };", context);
  return { context, el, run: code => vm.runInContext(code, context) };
}
function response(payload, status=200) { return { ok: status < 400, status, text: async () => JSON.stringify(payload) }; }

test('api reports HTTP validation, non-JSON, network, and hung-request errors', async () => {
  const { context, run } = setup();
  for (const status of [403, 404, 409, 413, 422, 500]) {
    context.fetch = async () => response({detail: status === 422 ? [{msg: 'Sai path'}] : 'CPU bận'}, status);
    await assert.rejects(run("api('/test')"), new RegExp(`HTTP ${status}`));
  }
  context.fetch = async () => ({ok: false, status: 500, text: async () => '<html>error</html>'});
  await assert.rejects(run("api('/test')"), /không phải JSON/);
  run("fetch = async () => { throw new TypeError('offline'); };");
  await assert.rejects(run("api('/test')"), /Mất kết nối/);
  context.fetch = (_, {signal}) => new Promise((resolve, reject) => signal.addEventListener('abort', () => reject(new Error('aborted'))));
  await assert.rejects(run("api('/test', {timeoutMs: 1})"), /quá thời gian/);
});

test('empty list disables only data button; selected upload is explicitly local', async () => {
  const {context, el, run} = setup();
  context.fetch = async () => response({items: [], total: 0});
  await run('refreshScenarioVideoList()');
  assert.equal(el('#scenarioAnalyzeDataBtn').disabled, true);
  assert.match(el('#scenarioJobStatus').textContent, /chưa tìm thấy/);
  // Presence of list items must not enable an unselected placeholder.
  context.fetch = async () => response({items: [{path:'nested/clip.avi',size_mb:1}],total:1});
  await run('refreshScenarioVideoList()');
  assert.equal(el('#scenarioAnalyzeDataBtn').disabled, true);
  context.fetch = async () => response({detail:'unavailable'},500);
  await run('refreshScenarioVideoList()');
  assert.equal(el('#scenarioAnalyzeDataBtn').disabled, true);
});

test('both analysis flows submit correct inputs and restore independent buttons on 409', async () => {
  for (const fromData of [false, true]) {
    const {context, el, run} = setup();
    el('#scenarioOriginSelect').value = 'HN-001';
    el('#scenarioDataVideo').value = fromData ? 'nested/clip.avi' : '';
    run("state.scenarioDataVideos = [{path:'nested/clip.avi'}]");
    el('#scenarioVideoFile').files = fromData ? [] : [new Blob(['clip'])];
    let called;
    context.fetch = async (path, options) => {
      called = {path, options};
      assert.match(el('#scenarioJobStatus').textContent, /Đang/);
      assert.equal(el('#scenarioAnalyzeUploadBtn').disabled, true);
      assert.equal(el('#scenarioAnalyzeDataBtn').disabled, true);
      return response({detail:'CPU đang bận; thử lại.'},409);
    };
    await run(`startScenarioReview({fromData:${fromData}})`);
    assert.equal(called.path, fromData ? '/api/scenarios/video-review-from-data' : '/api/scenarios/video-review');
    if (fromData) assert.equal(JSON.parse(called.options.body).path, 'nested/clip.avi');
    else assert.equal(called.options.body.has('file'), true);
    assert.match(el('#scenarioJobStatus').textContent, /HTTP 409.*thử lại/);
    assert.equal(el('#scenarioAnalyzeUploadBtn').disabled, fromData);
    assert.equal(el('#scenarioAnalyzeDataBtn').disabled, !fromData);
  }
});

test('polling renders success and surfaces failed, unknown, 404 and disconnected jobs', async () => {
  const {context, run} = setup();
  context.fetch = async () => response({status:'succeeded', result:{frames:[]}});
  await run("pollScenarioReviewJob('abc', 'clip')");
  assert.ok(run('state.scenarioResult'));
  for (const job of [{status:'failed',error:'Decoder lỗi'}, {status:'surprise'}]) {
    context.fetch = async () => response(job);
    await assert.rejects(run("pollScenarioReviewJob('abc', 'clip')"), /Decoder lỗi|không hợp lệ/);
  }
  context.fetch = async () => response({detail:'Tác vụ đã hết hạn'},404);
  await assert.rejects(run("pollScenarioReviewJob('abc', 'clip')"), /abc.*404/);
  context.fetch = async () => {throw new Error('offline');};
  await assert.rejects(run("pollScenarioReviewJob('abc', 'clip')"), /abc.*offline.*vẫn chạy/);
});


test('accepted job polls queued/running/success without allowing a second submission', async () => {
  const {context, el, run} = setup();
  el('#scenarioOriginSelect').value = 'HN-001';
  el('#scenarioDataVideo').value = 'nested/clip.avi';
  run("state.scenarioDataVideos = [{path:'nested/clip.avi'}]");
  const jobId = 'a'.repeat(32);
  let polls = 0;
  context.setTimeout = (callback, ms) => setTimeout(callback, ms === 900 ? 0 : ms);
  context.fetch = async (path, options) => {
    if (options.method === 'POST') return response({job_id:jobId, filename:'clip.avi'});
    assert.equal(run('state.scenarioBusy'), true);
    assert.equal(el('#scenarioAnalyzeDataBtn').disabled, true);
    const status = ['queued','running','succeeded'][polls++];
    return response({status, frames_total:10, frames_decoded:5, progress_percent:50, result:{frames:[]}});
  };
  await run('startScenarioReview({fromData:true})');
  assert.equal(polls, 3);
  assert.equal(run('state.scenarioBusy'), false);
  assert.equal(el('#scenarioAnalyzeDataBtn').disabled, false);
  assert.equal(el('#scenarioAnalyzeUploadBtn').disabled, true);
  assert.equal(el('#scenarioProgressText').textContent, 'Hoàn tất');
});


test('upload selection enables its button even when data listing is empty or fails', async () => {
  const {context, el, run} = setup();
  el('#scenarioOriginSelect').value = 'HN-001';
  el('#scenarioVideoFile').files = [new Blob(['selected clip'])];
  run('renderScenarioInputState()');
  assert.equal(el('#scenarioAnalyzeUploadBtn').disabled, false);
  assert.equal(el('#scenarioAnalyzeDataBtn').disabled, true);
  context.fetch = async () => response({items: [],total:0});
  await run('refreshScenarioVideoList()');
  assert.equal(el('#scenarioAnalyzeUploadBtn').disabled, false);
  context.fetch = async () => response({detail:'listing unavailable'},500);
  await run('refreshScenarioVideoList()');
  assert.equal(el('#scenarioAnalyzeUploadBtn').disabled, false);
  assert.equal(el('#scenarioAnalyzeDataBtn').disabled, true);
});

test('public opt-in uses upload capability while stored files remain local-only', () => {
  const {el, run} = setup();
  el('#scenarioVideoFile').files = [new Blob(['clip'])];
  run('state.publicDemoMode = true; state.aiccStatus = {enabled:false, public_demo_mode:true, video_review_enabled:true, upload_limit_mb:100}; renderScenarioInputState()');
  assert.equal(el('#scenarioAnalyzeUploadBtn').disabled, false);
  assert.equal(el('#scenarioAnalyzeDataBtn').disabled, true);
  run('state.aiccStatus.video_review_enabled = false; renderScenarioInputState()');
  assert.equal(el('#scenarioAnalyzeUploadBtn').disabled, true);
  assert.match(el('#scenarioInputHint').textContent, /chế độ chỉ đọc/);
  run('state.publicDemoMode = false; state.aiccStatus = {enabled:false,public_demo_mode:false,video_review_enabled:false}; renderScenarioInputState()');
  assert.equal(el('#scenarioAnalyzeUploadBtn').disabled, true);
  assert.match(el('#scenarioInputHint').textContent, /chưa bật phân tích/);
  run('state.aiccStatus = {enabled:true,public_demo_mode:false,video_review_enabled:true}; renderScenarioInputState()');
  assert.equal(el('#scenarioAnalyzeUploadBtn').disabled, false);
});

test('refresh cannot enable either input during an active review', async () => {
  const {context, el, run} = setup();
  el('#scenarioOriginSelect').value = 'HN-001';
  el('#scenarioVideoFile').files = [new Blob(['selected clip'])];
  el('#scenarioDataVideo').value = 'nested/clip.avi';
  context.fetch = async () => response({items:[{path:'nested/clip.avi'}],total:1});
  run('state.scenarioBusy = true; renderScenarioInputState()');
  await run('refreshScenarioVideoList()');
  assert.equal(el('#scenarioAnalyzeUploadBtn').disabled, true);
  assert.equal(el('#scenarioAnalyzeDataBtn').disabled, true);
  run('state.scenarioBusy = false; renderScenarioInputState()');
  assert.equal(el('#scenarioAnalyzeUploadBtn').disabled, false);
  assert.equal(el('#scenarioAnalyzeDataBtn').disabled, false);
});


test('public website explains that starting local does not enable the remote demo', () => {
  const {el, run} = setup();
  run("window.location.hostname = 'hanoi-traffic-platform.onrender.com'; state.aiccStatus = {enabled:false,public_demo_mode:true,video_review_enabled:false}; renderScenarioInputState()");
  assert.equal(el('#scenarioAnalyzeUploadBtn').disabled, true);
  assert.match(el('#scenarioInputHint').textContent, /demo công khai/);
  assert.match(el('#scenarioInputHint').textContent, /không thay đổi trang demo này/);
  assert.equal(el('#scenarioVideoFile').disabled, true);
});

test('fresh backend mode overrides stale read-only dashboard state', () => {
  const {el, run} = setup();
  el('#scenarioOriginSelect').value = 'HN-001';
  el('#scenarioVideoFile').files = [new Blob(['clip'])];
  run('state.publicDemoMode = true; state.aiccStatus = {enabled:true,public_demo_mode:false,video_review_enabled:true}; renderScenarioInputState()');
  assert.equal(el('#scenarioAnalyzeUploadBtn').disabled, false);
  assert.equal(el('#scenarioVideoFile').disabled, false);
  run('state.aiccStatus = null; renderScenarioInputState()');
  assert.equal(el('#scenarioAnalyzeUploadBtn').disabled, true);
  assert.match(el('#scenarioInputHint').textContent, /Chưa xác minh/);
});

test('a data path must belong to the current backend listing', () => {
  const {el, run} = setup();
  el('#scenarioOriginSelect').value = 'HN-001';
  el('#scenarioDataVideo').value = 'stale.mp4';
  run("state.scenarioDataVideos = [{path:'nested/clip.avi'}]; renderScenarioInputState()");
  assert.equal(el('#scenarioAnalyzeDataBtn').disabled, true);
  el('#scenarioDataVideo').value = 'nested/clip.avi';
  run('renderScenarioInputState()');
  assert.equal(el('#scenarioAnalyzeDataBtn').disabled, false);
  assert.equal(el('#scenarioAnalyzeUploadBtn').disabled, true);
});
