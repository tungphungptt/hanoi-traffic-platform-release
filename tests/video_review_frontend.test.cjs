const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('static/app.js','utf8').split('\nif (!window.__weatherTrafficPollerV1)')[0];
const jobId = 'a'.repeat(32);
function response(payload,status=200) {return {ok:status<400,status,text:async()=>JSON.stringify(payload)};}
function setup() {
  const nodes = new Map(), requests=[], storage=new Map();
  const el=id=>{if(!nodes.has(id))nodes.set(id,{value:'',files:[],checked:false,textContent:'',hidden:false,disabled:false,style:{}});return nodes.get(id);};
  const context=vm.createContext({console,AbortController,FormData,Blob,
    setTimeout:(fn,ms)=>setTimeout(fn,ms<=5000?0:ms),clearTimeout,
    document:{querySelector:el},window:{},
    sessionStorage:{setItem:(k,v)=>storage.set(k,v),getItem:k=>storage.get(k),removeItem:k=>storage.delete(k)},
    XMLHttpRequest:class {
      constructor(){this.upload={};}
      open(method,path){this.method=method;this.path=path;}
      send(body){this.body=body;requests.push(this);}
    },fetch:async()=>response({status:'succeeded',result:{frames:[]}})});
  vm.runInContext(source,context);
  const renderReview=vm.runInContext('renderScenarioReview',context);
  vm.runInContext('state.aiccStatus={video_review_enabled:true,upload_limit_bytes:524288000,upload_limit_mb:500,upload_max_duration_seconds:360}; toast=()=>{}; renderScenarioReview=result=>{state.scenarioResult=result;};',context);
  el('#scenarioVideoFile').files=[new Blob(['clip'])];
  return {context,el,requests,storage,renderReview,run:code=>vm.runInContext(code,context)};
}
function accept(xhr) {xhr.upload.onload();xhr.status=202;xhr.responseText=JSON.stringify({job_id:jobId});xhr.onload();}

test('CI labels, colors and manifest agree on three continuous ranges at every boundary',()=>{
  const {run}=setup();
  const labels={free:'Thông thoáng',busy:'Đông đúc',gridlock:'Tắc nghẽn nghiêm trọng'};
  for(const [ci,key] of [[0,'free'],[.29,'free'],[.299999,'free'],[.30,'busy'],[.54,'busy'],[.549999,'busy'],[.55,'gridlock'],[.65,'gridlock'],[.77,'gridlock'],[.78,'gridlock'],[1,'gridlock']]){
    assert.equal(run(`severityFromCI(${ci})`),key);
    assert.equal(run(`levelVN(severityFromCI(${ci}))`),labels[key]);
  }
  assert.equal(run('CI_LEVELS.length'),3);
  assert.equal(new Set(run('CI_LEVELS.map(level=>level.color)')).size,3);
  const thresholds=JSON.parse(fs.readFileSync('artifacts/current/manifest.json','utf8')).forecaster_config.thresholds;
  assert.equal(run('CI_LEVELS[1].min'),thresholds.busy);
  assert.equal(run('CI_LEVELS[2].min'),thresholds.gridlock);
  assert.equal(thresholds.heavy,thresholds.gridlock);
});

test('analytics renders exactly three groups and merges every CI from 0.55 to 1',()=>{
  const {run,el}=setup();
  run(`state.dashboard={trend:[{ci:.55,time:'12:00'}]};
    state.cameras=[.29,.30,.54,.55,.77,.78,1].map(ci=>({ci,status:'online'})); renderAnalytics();`);
  const html=el('#distributionList').innerHTML;
  assert.equal((html.match(/class="distribution-item"/g)||[]).length,3);
  assert.match(html,/Thông thoáng.*?<strong>1<\/strong>/);
  assert.match(html,/Đông đúc.*?<strong>2<\/strong>/);
  assert.match(html,/Tắc nghẽn nghiêm trọng.*?<strong>4<\/strong>/);
});

test('dashboard explanation shows the three new ranges and readable severe styling',()=>{
  const html=fs.readFileSync('static/index.html','utf8');
  const definition=html.match(/<div class="definition-list">(.*?)<\/section>/s)[1];
  assert.equal((definition.match(/class="level-dot /g)||[]).length,3);
  for(const range of ['0,00–0,29','0,30–0,54','0,55–1,00'])assert.ok(definition.includes(range));
  assert.match(definition,/level-dot gridlock.*?Tắc nghẽn nghiêm trọng/);
  const css=fs.readFileSync('static/styles.css','utf8');
  assert.match(css,/\.level-chip\.gridlock\{background:#[a-f0-9]+;color:#[a-f0-9]+\}/);
  assert.match(css,/\.scenario-state-dot\.gridlock\{/);
});

test('video gallery and summary display the official CI label instead of hysteresis metadata',()=>{
  for(const [ci,key,stable] of [[.56,'gridlock','busy'],[.54,'busy','gridlock']]){
    const {run,el,renderReview}=setup();
    const label=run(`levelVN('${key}')`),stableLabel=run(`levelVN('${stable}')`);
    renderReview({summary:{current_ci:ci,current_state:label},frames:[{
      ci,state_key:key,state_label:label,smoothed_state_key:stable,smoothed_state_label:stableLabel,
      timestamp_seconds:1,vehicle_count:1,url:'/frame.jpg',
    }]});
    assert.equal(el('#scenarioCurrentState').textContent,label);
    const gallery=el('#scenarioFrameGallery').innerHTML;
    assert.ok(gallery.includes(label));
    assert.ok(!gallery.includes(stableLabel));
    assert.ok(gallery.includes('scenario-state-dot '+key));
  }
});

test('one upload reports actual byte progress and changes phase before job polling',async()=>{
  const {run,el,requests}=setup();
  const pending=run('startScenarioReview()');const xhr=requests[0];
  assert.equal(xhr.path,'/api/scenarios/video-review');
  assert.equal(xhr.timeout,1800000);
  assert.equal(xhr.body.has('file'),true);assert.equal(xhr.body.has('camera_id'),false);
  assert.equal(el('#scenarioAnalyzeUploadBtn').disabled,true);
  xhr.upload.onprogress({loaded:25,total:100,lengthComputable:true});
  assert.equal(el('#scenarioProgressBar').style.width,'25%');
  assert.match(el('#scenarioProgressText').textContent,/25%/);
  xhr.upload.onprogress({loaded:60,total:100,lengthComputable:true});
  assert.equal(el('#scenarioProgressBar').style.width,'60%');
  xhr.upload.onload();assert.match(el('#scenarioAnalyzeUploadBtn').textContent,/kiểm tra/);
  accept(xhr);await pending;
  assert.equal(el('#scenarioProgressText').textContent,'Hoàn tất');
  assert.equal(el('#scenarioAnalyzeUploadBtn').disabled,false);
  assert.equal(run('state.scenarioPendingJobId'),null);
});

test('unknown transfer length shows measured bytes without invented percentages',async()=>{
  const {run,el,requests}=setup();const pending=run('startScenarioReview()');
  requests[0].upload.onprogress({loaded:1048576,lengthComputable:false});
  assert.match(el('#scenarioProgressText').textContent,/1.*MiB/);
  assert.equal(el('#scenarioProgressBar').style.width,'0%');
  accept(requests[0]);await pending;
});

test('HTTP failures restore controls and show actionable messages',async()=>{
  for(const status of [403,409,413,422,500,503]){
    const {run,el,requests}=setup();const pending=run('startScenarioReview()');
    const xhr=requests[0];xhr.status=status;xhr.responseText=JSON.stringify({detail:'Lỗi từ dịch vụ'});xhr.onload();await pending;
    assert.equal(run('state.scenarioBusy'),false);
    assert.equal(el('#scenarioAnalyzeUploadBtn').disabled,false);
    assert.equal(el('#scenarioVideoFile').disabled,false);
    if(status===409)assert.match(el('#scenarioJobStatus').textContent,/Hệ thống đang phân tích một video khác/);
    else assert.ok(el('#scenarioJobStatus').textContent.length>0);
  }
});

test('connection loss, abort and upload timeout restore controls',async()=>{
  for(const event of ['onerror','ontimeout','onabort']){
    const {run,el,requests}=setup();const pending=run('startScenarioReview()');
    requests[0][event]();await pending;
    assert.equal(run('state.scenarioBusy'),false);
    assert.equal(el('#scenarioAnalyzeUploadBtn').disabled,false);
    assert.ok(el('#scenarioJobStatus').textContent.length>0);
  }
});

test('invalid acceptance JSON fails instead of starting polling',async()=>{
  const {run,el,requests}=setup();const pending=run('startScenarioReview()');
  const xhr=requests[0];xhr.status=202;xhr.responseText='<html>error</html>';xhr.onload();await pending;
  assert.equal(run('state.scenarioBusy'),false);assert.equal(el('#scenarioAnalyzeUploadBtn').disabled,false);
});

test('temporary polling disconnection retries the same job without uploading again',async()=>{
  const {context,run,requests}=setup();let polls=0;
  context.fetch=async()=>{polls++;if(polls<3)throw new Error('offline');return response({status:'succeeded',result:{frames:[]}});};
  const pending=run('startScenarioReview()');accept(requests[0]);await pending;
  assert.equal(polls,3);assert.equal(requests.length,1);assert.ok(run('state.scenarioResult'));
});

test('persistent polling loss keeps a recoverable job; resume never reuploads',async()=>{
  const {context,run,el,requests,storage}=setup();
  context.fetch=async()=>{throw new Error('offline');};
  const pending=run('startScenarioReview()');accept(requests[0]);await pending;
  assert.equal(run('state.scenarioPendingJobId'),jobId);
  assert.equal(storage.get('scenario-review-job'),jobId);
  assert.equal(el('#scenarioResumeBtn').hidden,false);
  assert.equal(el('#scenarioAnalyzeUploadBtn').disabled,true);
  context.fetch=async()=>response({status:'succeeded',result:{frames:[]}});
  await run('startScenarioReview()');
  assert.equal(requests.length,1);assert.equal(run('state.scenarioPendingJobId'),null);
  assert.equal(storage.size,0);assert.equal(el('#scenarioAnalyzeUploadBtn').disabled,false);
});

test('queued/running progress uses server frame counts and percentages',async()=>{
  const {context,run,el}=setup();let polls=0;
  context.fetch=async()=>{polls++;if(polls===1)return response({status:'queued'});
    if(polls===2)return response({status:'running',frames_decoded:38,frames_total:50,progress_percent:76});
    assert.equal(el('#scenarioProgressBar').style.width,'76%');
    assert.match(el('#scenarioProgressText').textContent,/38.*50.*76%/);
    return response({status:'succeeded',result:{frames:[]}});};
  await run(`pollScenarioReviewJob('${jobId}','clip')`);assert.equal(polls,3);
});

test('worker failure and expired job clear pending state to allow another upload',async()=>{
  for(const expired of [false,true]){
    const {context,run,requests,el}=setup();
    context.fetch=async()=>response(expired?{detail:'Tác vụ đã hết hạn'}:{status:'failed',error:'Không đọc được video'},expired?404:200);
    const pending=run('startScenarioReview()');accept(requests[0]);await pending;
    assert.equal(run('state.scenarioPendingJobId'),null);
    assert.equal(el('#scenarioAnalyzeUploadBtn').disabled,false);
    assert.match(el('#scenarioJobStatus').textContent,expired?/hết hạn/:/Không đọc được/);
  }
});

test('public capability and optional camera govern the single upload',()=>{
  const {run,el}=setup();run('renderScenarioInputState()');
  assert.equal(el('#scenarioAnalyzeUploadBtn').disabled,false);
  assert.match(el('#scenarioUploadLimit').textContent,/500 MiB.*6 phút/);
  run('state.aiccStatus.video_review_enabled=false; renderScenarioInputState()');
  assert.equal(el('#scenarioAnalyzeUploadBtn').disabled,true);
  assert.doesNotMatch(el('#scenarioInputHint').textContent,/local|backend|CPU|data\//);
});

test('500 MiB admits files one byte below or at the cap and blocks one byte above',async()=>{
  // Metadata exercises browser admission without allocating or uploading a 500 MiB fixture.
  for(const size of [524287999,524288000,524288001]){
    const {run,el,requests}=setup();
    el('#scenarioVideoFile').files=[{name:'clip.mp4',size}];
    run('renderScenarioInputState()');
    const blocked=size===524288001;
    assert.equal(el('#scenarioAnalyzeUploadBtn').disabled,blocked);
    const pending=run('startScenarioReview()');
    assert.equal(requests.length,blocked?0:1,'submit must enforce the same cap as the button');
    if(blocked)assert.match(el('#scenarioInputHint').textContent,/500 MiB/);
    else accept(requests[0]);
    await pending;
  }
});

test('byte limit overrides rounded legacy MiB and older status responses remain supported',async()=>{
  const {run,el,requests}=setup();
  el('#scenarioVideoFile').files=[{name:'clip.mp4',size:524288000}];
  run('state.aiccStatus.upload_limit_bytes=524287999; renderScenarioInputState()');
  assert.equal(el('#scenarioAnalyzeUploadBtn').disabled,true);
  await run('startScenarioReview()');
  assert.equal(requests.length,0);
  run('delete state.aiccStatus.upload_limit_bytes; state.aiccStatus.upload_limit_mb=100; renderScenarioInputState()');
  assert.equal(el('#scenarioAnalyzeUploadBtn').disabled,true);
  assert.match(el('#scenarioInputHint').textContent,/100 MiB/);
  run('delete state.aiccStatus.upload_limit_mb; renderScenarioInputState()');
  assert.equal(el('#scenarioAnalyzeUploadBtn').disabled,false);
  assert.match(el('#scenarioUploadLimit').textContent,/500 MiB/);
  const pending=run('startScenarioReview()');accept(requests[0]);await pending;
});

test('API polling errors preserve HTTP status and requests have a deadline',async()=>{
  const {context,run}=setup();
  for(const status of [403,404,409,413,422,500]){
    context.fetch=async()=>response({detail:[{msg:'Sai dữ liệu'}]},status);
    await assert.rejects(run("api('/test')"),new RegExp(`HTTP ${status}`));
  }
  context.fetch=(_, {signal})=>new Promise((resolve,reject)=>signal.addEventListener('abort',()=>reject(new Error('aborted'))));
  await assert.rejects(run("api('/test',{timeoutMs:1})"),/quá thời gian/);
});
