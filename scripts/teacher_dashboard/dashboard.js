'use strict';
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const number = (n, digits=3) => Number.isFinite(n) ? n.toFixed(digits) : '—';
const compact = n => !Number.isFinite(n) ? '—' : (Math.abs(n)>0 && Math.abs(n)<.0001 ? n.toExponential(1) : n.toLocaleString('en-US',{maximumFractionDigits:Math.abs(n)<1?4:2}));
const clock = date => date ? new Date(date).toLocaleTimeString('zh-CN',{hour:'2-digit',minute:'2-digit',second:'2-digit'}) : '—';
const duration = hours => { if(!Number.isFinite(hours)) return '—'; if(hours<=0) return '收尾中'; const min=Math.ceil(hours*60); return min<60?`${min} 分钟`:`${Math.floor(min/60)}h ${min%60}m`; };
const phases = {training:'训练中',queued:'排队中',loading:'加载模型',evaluating:'保存 / 评估',finishing:'保存最终结果',completed:'训练完成',incomplete:'提前结束',failed:'训练失败',cancelled:'已停止'};
const hash = new URLSearchParams(location.hash.slice(1));
const state = {snapshot:null,enabled:null,metric:'loss',smooth:20,auto:true,busy:false,domain:hash.get('domain'),step:hash.get('step')||'latest',compare:'previous',key:hash.get('case'),category:'all',search:'',role:'output',cases:null,comparison:null,signature:'',request:0,chart:null,cursor:null};

async function getJSON(url){const response=await fetch(url,{cache:'no-store'});if(!response.ok)throw Error(`数据暂不可用（HTTP ${response.status}）`);return response.json();}
function selectedJob(){return state.snapshot?.jobs.find(j=>j.domain===state.domain);}
function saveView(){const p=new URLSearchParams({domain:state.domain||'',step:state.step});if(state.key)p.set('case',state.key);history.replaceState(null,'','#'+p.toString());}
function setOptions(element, options, value){element.innerHTML=options.map(([v,t])=>`<option value="${esc(v)}">${esc(t)}</option>`).join('');element.value=value;}

async function refresh(){
  if(state.busy)return;
  state.busy=true;$('refresh').disabled=true;
  try{
    const snapshot=await getJSON('/api/status');state.snapshot=snapshot;
    if(!state.enabled)state.enabled=new Set(snapshot.jobs.map(j=>j.domain));
    if(!snapshot.jobs.some(j=>j.domain===state.domain))state.domain=(snapshot.jobs.find(j=>j.validations.length)||snapshot.jobs[0]).domain;
    if(!$('metric').options.length)setOptions($('metric'),snapshot.metrics.map(m=>[m.key,m.label]),state.metric);
    $('connection-error').hidden=true;
    const stale=snapshot.platform_age_sec==null||snapshot.platform_age_sec>180;
    document.querySelector('.live-indicator').classList.toggle('offline',stale);
    $('live-label').textContent=stale?'平台状态更新延迟':state.auto?'正在跟踪训练':'自动刷新已暂停';
    $('updated').textContent=`更新 ${clock(snapshot.updated_at)}`;
    $('run-description').textContent=`Teacher v2 · ${snapshot.project} · 每类 8 张 GPU · 1500 step`;
    if(stale){$('connection-error').hidden=false;$('connection-error').textContent='平台状态超过 3 分钟未更新；曲线仍直接读取训练事件文件。任务状态可能滞后。';}
    renderJobs();renderSeries();renderChart();renderTabs();renderSource();
    const done=snapshot.jobs.filter(j=>j.phase==='completed').length;
    $('footer-status').textContent=`${done} / ${snapshot.jobs.length} 项完成 · 平台状态 ${clock(snapshot.platform_checked_at)}`;
    await loadEval();
  }catch(error){$('connection-error').hidden=false;$('connection-error').textContent=`刷新失败，保留上次可用数据。${error.message}`;$('live-label').textContent='连接暂时中断';document.querySelector('.live-indicator').classList.add('offline');}
  finally{state.busy=false;$('refresh').disabled=false;}
}

function renderJobs(){
  $('job-cards').innerHTML=state.snapshot.jobs.map(job=>{
    const percent=Math.min(100,job.step/job.target*100),last=job.validations.at(-1);
    const eta=job.phase==='completed'?'已完成':job.phase==='queued'?'等待资源':duration(job.eta_hours);
    const checkpoint=job.checkpoint?`Checkpoint ${job.checkpoint.step.toLocaleString()}`:'等待首个 checkpoint';
    return `<article class="job-card" style="--accent:${job.color}"><div class="job-topline"><h2 class="job-name">${esc(job.label)}</h2><span class="badge ${esc(job.phase)}">${esc(phases[job.phase]||job.status)}</span></div><div class="gpu-label">${job.gpu_count} × ${esc(job.gpu_type)} <span>· 独立 Teacher</span></div><div class="progress-label"><strong class="job-step">${job.step.toLocaleString()}</strong><span class="job-target">/ ${job.target.toLocaleString()} step</span><span class="percent">${number(percent,1)}%</span></div><div class="progress-bar" role="progressbar" aria-valuenow="${job.step}" aria-valuemax="${job.target}" aria-label="${esc(job.label)}训练进度"><span style="width:${percent}%"></span></div><div class="job-kpis"><div><div class="kpi-label">当前 loss</div><div class="kpi-value">${number(job.loss,4)}</div></div><div><div class="kpi-label" title="忽略前 5 步，最近 ${job.speed_samples} 步的平均耗时">每步耗时</div><div class="kpi-value">${number(job.seconds_per_step,1)}<span class="kpi-unit">秒</span></div></div><div><div class="kpi-label" title="按最近吞吐估算，不含后续验证、保存和排队">剩余纯训练</div><div class="kpi-value">${eta}</div></div></div><div class="job-bottom"><span>${checkpoint}</span><a href="#eval-title" data-eval-domain="${esc(job.domain)}">${last?`Eval ${last.step} · ${last.samples} cases ↗`:'首次 Eval · step '+job.validation_interval}</a></div><code class="task-name" title="${esc(job.name)}">${esc(job.name)}</code></article>`;
  }).join('');
  document.querySelectorAll('[data-eval-domain]').forEach(a=>a.onclick=()=>selectDomain(a.dataset.evalDomain));
}
function renderSeries(){
  $('series-controls').innerHTML=state.snapshot.jobs.map(j=>`<button class="series-button" style="--accent:${j.color}" data-series="${esc(j.domain)}" aria-pressed="${state.enabled.has(j.domain)}"><i></i>${esc(j.label)}</button>`).join('');
  document.querySelectorAll('[data-series]').forEach(button=>button.onclick=()=>{const d=button.dataset.series;state.enabled.has(d)?state.enabled.delete(d):state.enabled.add(d);renderSeries();renderChart();});
}
function movingAverage(points,n){return points.map((p,i)=>{const recent=points.slice(Math.max(0,i-n+1),i+1).filter(x=>Number.isFinite(x[1]));return [p[0],Number.isFinite(p[1])&&recent.length?recent.reduce((s,x)=>s+x[1],0)/recent.length:null,p[2]];});}
function renderChart(){
  if(!state.snapshot)return;
  const metric=state.snapshot.metrics.find(m=>m.key===state.metric);
  const series=state.snapshot.jobs.filter(j=>state.enabled.has(j.domain)).map(j=>({job:j,raw:j.curves[state.metric]||[],points:movingAverage(j.curves[state.metric]||[],state.smooth)}));
  const all=series.flatMap(s=>s.raw).filter(p=>Number.isFinite(p[1]));
  const W=Math.max(280,$('chart-container').clientWidth),H=$('chart-container').clientHeight,left=62,right=20,top=16,bottom=37,pw=W-left-right,ph=H-top-bottom;
  $('chart').setAttribute('viewBox',`0 0 ${W} ${H}`);
  const maxX=Math.max(10,...all.map(p=>p[0])),minY=Math.min(0,...all.map(p=>p[1])),maxY=Math.max(.00000001,...all.map(p=>p[1]));
  const range=Math.max(1e-9,maxY-minY),yMin=minY<0?minY-range*.06:0,yMax=maxY+range*.10;
  const x=n=>left+n/maxX*pw,y=n=>top+(yMax-n)/(yMax-yMin)*ph;
  let svg='';
  for(let i=0;i<5;i++){const value=yMin+(yMax-yMin)*i/4,py=y(value);svg+=`<line class="chart-grid" x1="${left}" y1="${py}" x2="${W-right}" y2="${py}"/><text class="chart-label" x="${left-11}" y="${py+4}" text-anchor="end">${compact(value)}</text>`;}
  for(let i=0;i<=6;i++){const value=Math.round(maxX*i/6);svg+=`<text class="chart-label" x="${x(value)}" y="${H-13}" text-anchor="middle">${value.toLocaleString()}</text>`;}
  for(const step of [500,1000,1500])if(step<=maxX)svg+=`<line x1="${x(step)}" x2="${x(step)}" y1="${top}" y2="${H-bottom}" stroke="#d1d8d0" stroke-dasharray="4 5"/><text x="${x(step)-5}" y="${top+10}" fill="#9ba59c" font-size="9" text-anchor="end">eval ${step}</text>`;
  const path=points=>{let d='',prev=null;for(const p of points){if(!Number.isFinite(p[1])){prev=null;continue;}d+=`${prev&&p[0]-prev[0]<=1?'L':'M'}${x(p[0]).toFixed(2)},${y(p[1]).toFixed(2)} `;prev=p;}return d;};
  for(const s of series){if(state.smooth>1)svg+=`<path d="${path(s.raw)}" fill="none" stroke="${s.job.color}" stroke-opacity=".19" stroke-width="1" vector-effect="non-scaling-stroke"/>`;svg+=`<path d="${path(s.points)}" fill="none" stroke="${s.job.color}" stroke-width="2" stroke-linejoin="round" vector-effect="non-scaling-stroke"/>`;const p=s.points.at(-1);if(p&&Number.isFinite(p[1]))svg+=`<circle cx="${x(p[0])}" cy="${y(p[1])}" r="3" fill="${s.job.color}"/>`;}
  svg+=`<line id="cursor-line" y1="16" y2="${H-bottom}" stroke="#79877d" stroke-width="1" stroke-dasharray="3 3" visibility="hidden"/>`;
  $('chart').innerHTML=svg;$('chart-unit').textContent=metric.unit==='loss'?'Loss':metric.label+' · '+metric.unit;
  $('chart-empty').hidden=Boolean(all.length);$('chart-empty').textContent=state.enabled.size?'当前指标尚无已记录数据':'请选择至少一个训练类别';
  $('chart-range').textContent=all.length?`1–${maxX.toLocaleString()} optimizer step`:'';
  $('chart-container').setAttribute('aria-label',`${metric.label}随训练步数变化；${series.map(s=>s.job.label).join('、')}；使用左右方向键查看数值`);
  const params=new URLSearchParams({metric:state.metric});for(const d of state.enabled)params.append('domain',d);if(!state.enabled.size)params.append('domain','__none__');$('export').href='/api/export?'+params;
  state.chart={series,maxX,x,y,left,right,W,H};$('chart-tooltip').hidden=true;
}
function showTooltip(fraction){
  if(!state.chart)return;
  const c=state.chart,step=Math.max(1,Math.min(c.maxX,Math.round(fraction*c.maxX)));state.cursor=step;
  const rows=[];
  for(const s of c.series){if(!s.raw.length||step<s.raw[0][0]||step>s.raw.at(-1)[0])continue;let idx=0,dist=Infinity;s.raw.forEach((p,i)=>{if(Math.abs(p[0]-step)<dist){dist=Math.abs(p[0]-step);idx=i;}});const p=s.raw[idx],smooth=s.points[idx];if(dist>Math.max(1,c.maxX*.02))continue;rows.push(`<div class="tooltip-row"><span><i style="background:${s.job.color}"></i>${esc(s.job.label)}</span><b>${compact(p[1])}${state.smooth>1?` <span class="subtle">/ ${compact(smooth[1])}</span>`:''}</b></div>`);}
  if(!rows.length){$('chart-tooltip').hidden=true;return;}
  $('chart-tooltip').innerHTML=`<strong>Step ${step}${state.smooth>1?' · 原始 / 平滑':''}</strong>${rows.join('')}`;
  const bounds=$('chart-container').getBoundingClientRect(),px=c.x(step)/c.W*bounds.width;
  $('chart-tooltip').hidden=false;$('chart-tooltip').style.left=Math.max(0,Math.min(px+13,bounds.width-$('chart-tooltip').offsetWidth-4))+'px';$('chart-tooltip').style.top='24px';
  const line=$('cursor-line');line.setAttribute('x1',c.x(step));line.setAttribute('x2',c.x(step));line.setAttribute('visibility','visible');
}
$('chart-container').onpointermove=e=>{if(!state.chart)return;const box=$('chart-container').getBoundingClientRect(),pos=(e.clientX-box.left)/box.width*state.chart.W;showTooltip((pos-state.chart.left)/(state.chart.W-state.chart.left-state.chart.right));};
$('chart-container').onpointerleave=()=>{$('chart-tooltip').hidden=true;$('cursor-line')?.setAttribute('visibility','hidden');};
$('chart-container').onkeydown=e=>{if(!['ArrowLeft','ArrowRight'].includes(e.key)||!state.chart)return;e.preventDefault();const step=(state.cursor||state.chart.maxX)+(e.key==='ArrowRight'?1:-1)*Math.max(1,Math.round(state.chart.maxX/100));showTooltip(step/state.chart.maxX);};

function renderTabs(){
  $('eval-tabs').innerHTML=state.snapshot.jobs.map(j=>`<button role="tab" aria-selected="${state.domain===j.domain}" data-domain="${esc(j.domain)}">${esc(j.label)}<small>${j.validations.length}</small></button>`).join('');
  document.querySelectorAll('[data-domain]').forEach(b=>b.onclick=()=>selectDomain(b.dataset.domain));
}
function selectDomain(domain){state.domain=domain;state.step='latest';state.key=null;state.category='all';state.search='';$('case-search').value='';state.signature='';renderTabs();loadEval();}
function activeStep(job){return state.step==='latest'?job.validations.at(-1)?.step:Number(state.step);}
function comparisonStep(job,current){if(state.compare==='off')return null;if(state.compare==='previous')return job.validations.filter(v=>v.step<current).at(-1)?.step??null;return Number(state.compare)===current?null:Number(state.compare);}
function emptyEval(title,detail){$('eval-content').hidden=true;$('eval-empty').hidden=false;$('eval-empty').innerHTML=`<div class="empty-icon" aria-hidden="true">▧</div><strong>${esc(title)}</strong><p>${esc(detail)}</p>`;}
async function loadEval(force=false){
  const job=selectedJob();if(!job)return;
  document.documentElement.style.setProperty('--selection',job.color);
  if(!job.validations.length){state.request++;state.signature='';const remaining=Math.max(0,job.validation_interval-job.step),eta=job.seconds_per_step?duration(remaining*job.seconds_per_step/3600):null;emptyEval('尚未生成首轮 Eval',`${job.label}每 ${job.validation_interval} step 评估一次，当前 ${job.step} / ${job.target} step。${eta?'距离首次评估约 '+eta+'（不含评估执行时间）。':''}结果保存完成后会自动展示在这里。`);saveView();return;}
  if(state.step!=='latest'&&!job.validations.some(v=>v.step===Number(state.step)))state.step='latest';
  const current=activeStep(job);
  setOptions($('eval-step'),[['latest',`最新 · Step ${job.validations.at(-1).step}`],...job.validations.map(v=>[String(v.step),'Step '+v.step])],state.step);
  setOptions($('compare-step'),[['previous','前一个 checkpoint'],['off','关闭对比'],...job.validations.filter(v=>v.step!==current).map(v=>[String(v.step),'Step '+v.step])],state.compare);
  if(!$('compare-step').value){state.compare='previous';$('compare-step').value='previous';}
  const compare=comparisonStep(job,current);
  renderEvalSummary(job,current);
  const signature=`${job.domain}:${current}:${compare}`;
  if(!force&&signature===state.signature&&state.cases){$('eval-empty').hidden=true;$('eval-content').hidden=false;return;}
  const request=++state.request;
  emptyEval('正在读取评估案例',`${job.label} · Step ${current}`);
  try{
    const [cases,comparison]=await Promise.all([getJSON('/api/cases?'+new URLSearchParams({domain:job.domain,step:current})),compare!=null?getJSON('/api/cases?'+new URLSearchParams({domain:job.domain,step:compare})):Promise.resolve(null)]);
    if(request!==state.request)return;
    state.cases=cases;state.comparison=comparison;state.signature=signature;
    const categories=[...new Set(cases.samples.map(s=>s.category))];if(!categories.includes(state.category))state.category='all';
    setOptions($('case-category'),[['all','全部子类'],...categories.map(c=>[c,c])],state.category);
    $('eval-content').hidden=false;$('eval-empty').hidden=true;
    renderCases();saveView();
  }catch(error){if(request===state.request)emptyEval('暂时无法读取该轮 Eval',error.message+'；下次刷新会自动重试。');}
}
function renderEvalSummary(job,step){
  const record=job.validations.find(v=>v.step===step),metrics=record?.metrics||{};
  $('eval-summary').innerHTML=`<div><strong>${record?.samples??'—'}</strong>固定案例</div><div><strong>${number(metrics.reference_dino_cosine,4)}</strong>平均 DINO cosine ↑</div><div><strong>${number(metrics.raw_background_mae,4)}</strong>原始背景 MAE</div>`;
  $('history-count').textContent=`${job.validations.length} 次评估`;
  $('history-rows').innerHTML=job.validations.map(v=>`<tr><td>${v.step}</td><td>${v.samples}</td><td>${number(v.metrics.reference_dino_cosine,4)}</td><td>${number(v.metrics.raw_background_mae,5)}</td><td>${new Date(v.published_at).toLocaleString()}</td></tr>`).join('');
}
function filteredCases(){return(state.cases?.samples||[]).filter(s=>(state.category==='all'||s.category===state.category)&&s.key.toLowerCase().includes(state.search.toLowerCase()));}
function renderCases(){
  if(!state.cases)return;
  const samples=filteredCases();if(!samples.some(s=>s.key===state.key))state.key=samples[0]?.key??null;
  $('case-count').textContent=`${samples.length} / ${state.cases.samples.length}`;
  $('case-list').innerHTML=samples.map(sample=>`<button class="case-tile" aria-pressed="${sample.key===state.key}" data-case="${esc(sample.key)}" title="${esc(sample.key)}"><img src="${esc(sample.thumbnail)}" loading="lazy" alt="${esc(sample.key)} 的生成结果缩略图"><span class="tile-label">${esc(sample.category)} · ${esc(sample.stem)}</span></button>`).join('')||'<p class="subtle">没有匹配的案例</p>';
  document.querySelectorAll('[data-case]').forEach(button=>button.onclick=()=>{state.key=button.dataset.case;renderSelectedCase();updateCaseSelection();saveView();});
  renderSelectedCase();saveView();
}
function updateCaseSelection(){document.querySelectorAll('[data-case]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.case===state.key)));}
function renderSelectedCase(){
  const samples=filteredCases(),sample=samples.find(s=>s.key===state.key),idx=samples.indexOf(sample);
  $('previous-case').disabled=idx<=0;$('next-case').disabled=idx<0||idx>=samples.length-1;
  $('case-position').textContent=sample?`${idx+1} / ${samples.length}`:'0 / 0';
  if(!sample){$('case-name').textContent='没有匹配的案例';$('case-meta').textContent='调整子类别或清空搜索';$('case-images').innerHTML='';$('case-metrics').innerHTML='';return;}
  const other=state.comparison?.samples.find(s=>s.key===sample.key),role=state.role;
  $('case-name').textContent=`${sample.category} / ${sample.stem}`;
  const params=state.cases.sampling||{};
  $('case-meta').textContent=`Seed ${sample.seed??'—'} · ${params.num_inference_steps??'—'} 推理步 · CFG ${params.true_cfg_scale??'—'}`;
  const columns=[{label:'参考主体',url:sample.images.reference},{label:'输入场景 / 区域',url:sample.images.source_mask_overlay||sample.images.source}];
  if(state.comparison)columns.push({label:'对比结果',step:state.comparison.step,url:other?.images[role],missing:'该 checkpoint 没有对应的 case'});
  columns.push({label:'当前结果',step:state.cases.step,url:sample.images[role],current:true,missing:'当前结果类型未保存'});
  $('case-images').classList.toggle('three',columns.length===3);
  $('case-images').innerHTML=columns.map(c=>`<figure class="image-column ${c.current?'current':''}"><figcaption class="image-label"><span>${esc(c.label)}</span>${c.step!=null?`<span class="step-tag">STEP ${c.step}</span>`:''}</figcaption>${c.url?`<button class="image-frame" data-large="${esc(c.url)}" data-caption="${esc(sample.key+' · '+c.label+(c.step!=null?' · Step '+c.step:''))}" aria-label="放大${esc(c.label)}"><img src="${esc(c.url)}" alt="${esc(sample.key+' '+c.label)}"></button>`:`<div class="image-frame image-error">${esc(c.missing||'图片未保存')}</div>`}</figure>`).join('');
  document.querySelectorAll('[data-large]').forEach(b=>b.onclick=()=>openImage(b.dataset.large,b.dataset.caption));
  $('case-images').querySelectorAll('img').forEach(img=>img.onerror=()=>{const frame=img.parentElement;frame.innerHTML='<span class="image-error">图片暂不可用，刷新后重试</span>';});
  const metrics=sample.metrics||{},previous=other?.metrics||{},delta=Number.isFinite(metrics.reference_dino_cosine)&&Number.isFinite(previous.reference_dino_cosine)?metrics.reference_dino_cosine-previous.reference_dino_cosine:null;
  $('case-metrics').innerHTML=`<span>当前 case</span><span>参考 DINO cosine<strong>${number(metrics.reference_dino_cosine,4)}</strong>${delta!=null?`<span class="delta">(${delta>=0?'+':''}${delta.toFixed(4)})</span>`:''}</span><span>原始背景 MAE<strong>${number(metrics.raw_background_mae,5)}</strong></span>`;
  $('output-explanation').textContent={output:'融合输出：保留交互区域与柔和过渡。',output_restored_raw:'原始生成：映射回原尺寸、未经掩码合成的模型结果。',composite:'严格掩码合成：掩码外保留输入背景。'}[role];
}
function openImage(url,caption){const u=new URL(url,location.origin);u.searchParams.set('size','1440');$('large-image').src=u.toString();$('large-image').alt=caption;$('image-caption').textContent=caption;u.searchParams.set('size','0');$('original-image').href=u.toString();$('image-dialog').showModal();}
function moveCase(direction){const samples=filteredCases(),i=samples.findIndex(s=>s.key===state.key),next=samples[i+direction];if(!next)return;state.key=next.key;renderSelectedCase();updateCaseSelection();saveView();document.querySelector(`[data-case="${CSS.escape(state.key)}"]`)?.scrollIntoView({block:'nearest',inline:'nearest'});}
function renderSource(){const s=state.snapshot;$('source-content').innerHTML=`<dl><dt>本次运行</dt><dd>${esc(s.run)} · ${esc(s.project)} · ${esc(s.image)}<br>${esc(s.workspace)}</dd><dt>任务状态</dt><dd>${esc(s.source.status)}<br>平台最近检查：${new Date(s.platform_checked_at).toLocaleString()}；后台约每 60 秒同步。</dd><dt>训练曲线</dt><dd>${esc(s.source.curves)}<br>${esc(s.source.step)}${esc(s.source.smoothing)}页面每 15 秒刷新，导出的 CSV 保留原始数值。</dd><dt>预计剩余时间</dt><dd>${esc(s.source.eta)}</dd><dt>Eval 案例与指标</dt><dd>${esc(s.source.cases)}<br>仅在本轮元数据、指标和总览图均保存后展示。不同 step 按 category / stem 匹配同一案例，不按图片序号配对。</dd><dt>指标解释</dt><dd>训练 DINO loss 与评估 DINO cosine 是不同指标。后者越高表示参考身份越一致；原始背景 MAE 只表示背景改动幅度，不能单独证明自然度或整体质量。</dd></dl>`;}

$('metric').onchange=e=>{state.metric=e.target.value;renderChart();};
$('smooth').onchange=e=>{state.smooth=Number(e.target.value);renderChart();};
$('eval-step').onchange=e=>{state.step=e.target.value;loadEval();};
$('compare-step').onchange=e=>{state.compare=e.target.value;loadEval();};
$('output-role').onchange=e=>{state.role=e.target.value;renderSelectedCase();};
$('case-category').onchange=e=>{state.category=e.target.value;renderCases();};
$('case-search').oninput=e=>{state.search=e.target.value;renderCases();};
$('previous-case').onclick=()=>moveCase(-1);$('next-case').onclick=()=>moveCase(1);
$('refresh').onclick=()=>refresh();
$('auto-refresh').onclick=()=>{state.auto=!state.auto;$('auto-refresh').classList.toggle('active',state.auto);$('auto-refresh').setAttribute('aria-pressed',String(state.auto));$('auto-refresh').textContent=state.auto?'自动刷新 · 15s':'已暂停刷新';$('live-label').textContent=state.auto?'正在跟踪训练':'自动刷新已暂停';if(state.auto)refresh();};
$('close-image').onclick=()=>$('image-dialog').close();$('close-source').onclick=()=>$('source-dialog').close();$('source-button').onclick=()=>$('source-dialog').showModal();
for(const id of ['image-dialog','source-dialog'])$(id).onclick=e=>{if(e.target===$(id)){const box=$(id).getBoundingClientRect();if(e.clientX<box.left||e.clientX>box.right||e.clientY<box.top||e.clientY>box.bottom)$(id).close();}};
setInterval(()=>{if(state.auto&&!document.hidden)refresh();},15000);
document.addEventListener('visibilitychange',()=>{if(!document.hidden&&state.auto)refresh();});
new ResizeObserver(()=>renderChart()).observe($('chart-container'));
refresh();
