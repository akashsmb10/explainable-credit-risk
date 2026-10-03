'use strict';
let D;
const repository='https://github.com/akashsmb10/explainable-credit-risk';
const blob=repository+'/blob/main/';
const pct=(v,d=1)=>Number.isFinite(v)?(v*100).toFixed(d)+'%':'—';
const fmt=(v,d=0)=>Number.isFinite(v)?v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d}):'—';
const escapeHTML=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const sum=xs=>xs.reduce((a,b)=>a+b,0);
const mean=xs=>xs.length?sum(xs)/xs.length:NaN;
const bandNames=['LOW','MEDIUM','HIGH','VERY HIGH'];
const bandColors=['#5b9684','#8eaa7b','#c69e63','#b97758'];

function metric(label,value,detail=''){return `<div class="metric"><small>${label}</small><strong>${value}</strong><em>${detail}</em></div>`;}
function panel(title,content,detail=''){return `<section class="panel"><div class="panel-title"><h2>${title}</h2><small>${detail}</small></div>${content}</section>`;}
function hero(eyebrow,title,description,action=''){return `<section class="hero"><div><p class="eyebrow">${eyebrow}</p><h1>${title}</h1><p>${description}</p></div>${action}</section>`;}
function note(text){return `<div class="note">${text}</div>`;}
function table(rows,columns){return `<div class="table-wrap"><table><thead><tr>${columns.map(c=>`<th scope="col">${c[1]}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr${r.candidate===D.summary.selected?' class="selected"':''}>${columns.map(c=>`<td>${escapeHTML(c[2]?c[2](r[c[0]]):r[c[0]])}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;}
function image(name,alt){return `<img class="image-chart" src="assets/${name}" alt="${alt}" loading="lazy">`;}

function decision(threshold){
  const reviewed=D.clients.filter(c=>c.score>=threshold),accepted=D.clients.filter(c=>c.score<threshold);
  const events=sum(D.clients.map(c=>c.event)),tp=sum(reviewed.map(c=>c.event)),fp=reviewed.length-tp,fn=events-tp;
  const tn=D.clients.length-events-fp;
  return {threshold,review_rate:reviewed.length/D.clients.length,acceptance:accepted.length/D.clients.length,
    event_capture:tp/events,accepted_event_rate:mean(accepted.map(c=>c.event)),fp,fn,tp,tn,
    false_positive_rate:fp/(fp+tn),false_negative_rate:fn/(fn+tp)};
}
function decisionFlow(r){return `<div class="decision-flow" role="img" aria-label="Simulated acceptance ${pct(r.acceptance)}; manual review ${pct(r.review_rate)}"><div class="accepted" style="width:${r.acceptance*100}%">${r.acceptance>.1?pct(r.acceptance):''}</div><div class="reviewed" style="width:${r.review_rate*100}%">${r.review_rate>.1?pct(r.review_rate):''}</div></div><div class="legend"><span><i class="green"></i>Simulated acceptance</span><span><i class="orange"></i>Manual review</span></div>`;}
function distributionBars(mode='count'){
  return bandNames.map((band,i)=>{const clients=D.clients.filter(c=>c.band===band);const value=mode==='count'?clients.length:mean(clients.map(c=>c.event));const max=mode==='count'?D.clients.length:1;
    return `<div class="bar-row"><span>${band}</span><div class="bar-track"><div class="bar-fill" style="width:${value/max*100}%;background:${bandColors[i]}"></div></div><span class="bar-number">${mode==='count'?fmt(value):pct(value)}</span></div>`;}).join('');
}
function driverBars(drivers){const max=Math.max(...drivers.map(x=>Math.abs(x.contribution)),1e-8);return drivers.map(x=>`<div class="driver-row"><span class="driver-name">${escapeHTML(x.feature)}</span><div class="driver-scale"><span class="${x.contribution<0?'driver-negative':'driver-positive'}" style="width:${Math.abs(x.contribution)/max*48}%"></span></div><span class="bar-number">${x.contribution>0?'+':''}${x.contribution.toFixed(3)}</span></div>`).join('');}
function lineChart(series){
  const w=620,h=280,left=45,right=15,top=15,bottom=38;
  const x=v=>left+v*(w-left-right),y=v=>h-bottom-v*(h-top-bottom);
  let svg=`<svg class="svg-chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="Holdout simulation: threshold against acceptance, event capture and false positive rate">`;
  for(let i=0;i<=4;i++){const v=i/4;svg+=`<line x1="${left}" x2="${w-right}" y1="${y(v)}" y2="${y(v)}" stroke="#e8ede7"/><text x="${left-8}" y="${y(v)+4}" text-anchor="end" fill="#667780" font-size="10">${pct(v,0)}</text><text x="${x(v)}" y="${h-16}" text-anchor="middle" fill="#667780" font-size="10">${v.toFixed(2)}</text>`;}
  for(const s of series){svg+=`<polyline points="${s.points.map(p=>`${x(p[0])},${y(p[1])}`).join(' ')}" fill="none" stroke="${s.color}" stroke-width="2.4"/>`;}
  svg+='</svg>';return svg+`<div class="legend">${series.map(s=>`<span><i style="background:${s.color}"></i>${s.label}</span>`).join('')}</div>`;
}
function overview(){const s=D.summary,m=s.test_metrics,r=decision(s.review_threshold);return hero('Evidence before decisions','Credit risk, made explainable.','A banking analytics portfolio connecting repayment behaviour, calibrated model scores and transparent decision trade-offs.',`<button class="button" id="explore-decisions">Explore decisions ↗</button>`)
  +`<div class="metrics">${metric('Source cardholders',fmt(s.rows),'Official public UCI dataset')}${metric('Holdout ROC-AUC',fmt(m.roc_auc,4),'Ranking, not probability calibration')}${metric('Average precision',fmt(m.average_precision,4),'Positive-event detection')}${metric('Holdout Brier score',fmt(m.brier,4),'Selected probability model')}</div>`
  +`<div class="grid">${panel('How the holdout portfolio is segmented',distributionBars()+`<p class="caption">Bands come from validation-score quantiles. Counts shown here are the untouched test holdout, not the full source portfolio.</p>`,`${fmt(D.clients.length)} test clients`)}${panel('Review capacity, made explicit',decisionFlow(r)+`<div class="mini-metrics"><div><strong>${pct(r.event_capture)}</strong><small>Events captured in review</small></div><div><strong>${pct(r.accepted_event_rate)}</strong><small>Event rate in acceptance group</small></div></div><p class="caption">Frozen threshold ${s.review_threshold.toFixed(4)}. A retrospective capacity simulation, not a real lending policy.</p>`,'Validation-derived policy')}</div>`
  +panel('From data to decision support',`<div class="steps"><span>Public data</span><b>→</b><span>Statistical evidence</span><b>→</b><span>Calibrated score</span><b>→</b><span>SHAP explanation</span><b>→</b><span>Review simulation</span></div><p class="caption">Base-rate, logistic and XGBoost models are compared objectively. Train, calibration, selection and final testing use separate grouped partitions.</p>`)
  +note('<strong>A useful portfolio, with honest boundaries.</strong> Historical Taiwanese cardholders do not validate Indian bank underwriting. Income, actual loss severity and exposure at default are not observed.');}
function distribution(){return hero('Relative risk','Where does risk concentrate?','Compare observed event rates across validation-defined risk bands and holdout score deciles.')
 +`<div class="grid">${panel('Observed event rate by risk band',distributionBars('event'),'Final holdout')}${panel('Validation-derived boundaries',table(bandNames.map((band,i)=>({band,range:i===0?`score < ${D.summary.band_boundaries[0].toFixed(4)}`:i===3?`score ≥ ${D.summary.band_boundaries[2].toFixed(4)}`:`${D.summary.band_boundaries[i-1].toFixed(4)} ≤ score < ${D.summary.band_boundaries[i].toFixed(4)}`})),[['band','Band'],['range','Score range']])+`<p class="caption">Cutoffs represent relative-risk and review-capacity assumptions. Low risk does not guarantee affordability, repayment or approval.</p>`)}</div>`
 +panel('Band evidence',table(D.bands,[['risk_band','Risk band'],['n','Clients',fmt],['events','Events',fmt],['event_rate','Event rate',pct],['mean_score','Mean score',pct]]))
 +panel('Highest-risk decile first',table(D.deciles,[['decile','Decile'],['n','Clients',fmt],['event_rate','Event rate',pct],['mean_score','Mean score',pct],['lift','Lift',v=>fmt(v,2)+'×'],['cumulative_event_capture','Cumulative capture',pct]]));}
function applicant(){return hero('Held-out cardholders','Look inside a model score.','Explore existing source cardholders, their frozen risk scores and the raw-tree features associated with those scores.')
  +`<div class="controls"><label for="client-select">Source cardholder ID <select id="client-select">${D.clients.map(c=>`<option value="${c.id}"${c.id===D.summary.representative_ids.borderline?' selected':''}>${c.id}</option>`).join('')}</select></label><span class="badge">Public benchmark · Existing cardholders</span></div><div id="client-details"></div>`;}
function clientDetail(c){return `<div class="metrics">${metric('Estimated payment-event score',pct(c.score,2),'Calibrated dataset-event probability')}${metric('Relative risk band',c.band,'Validation-derived cutoffs')}${metric('Review priority',c.score>=D.summary.review_threshold?'Review':'Acceptance','Simulated decision support')}${metric('Assumed-loss scenario',fmt(c.score*.45*100000)+' NTD','Assumed LGD 0.45 · EAD 100,000')}</div><div class="grid">${panel('Observed source features',table([c],[['limit','Credit limit (NTD)',fmt],['status','Latest repayment code'],['bill','Latest bill (NTD)',fmt],['payment','Latest payment (NTD)',fmt]])+`<p class="caption">Limit is not actual exposure at default; repayment-code meaning and negative statements are documented in the data dictionary.</p>`)}${panel('Major raw-tree model associations',driverBars(c.drivers)+`<div class="legend"><span><i class="green"></i>Lower raw log-odds</span><span><i class="orange"></i>Higher raw log-odds</span></div><p class="caption">These exact TreeSHAP contributions explain raw XGBoost log-odds, not the calibrated probability. They are associations, not causes.</p>`)}</div>`;}
function performance(){const m=D.summary.test_metrics;return hero('Independent validation','Ranking is only half the story.','Model discrimination, probability calibration and decision thresholds answer different questions. The winner was selected on validation Brier, before test scoring.')
 +`<div class="metrics">${metric('ROC-AUC',fmt(m.roc_auc,4),'Selected model · Final holdout')}${metric('KS statistic',fmt(m.ks,4),'Separation of conditional score CDFs')}${metric('Brier score',fmt(m.brier,6),'Squared probability error')}${metric('Review recall',pct(m.recall,2),'At frozen review threshold')}</div>`
 +panel('Final holdout comparison',table(D.models,[['candidate','Model / mapping'],['roc_auc','ROC-AUC',v=>fmt(v,4)],['average_precision','Average precision',v=>fmt(v,4)],['brier','Brier',v=>fmt(v,6)],['log_loss','Log loss',v=>fmt(v,6)]]),'Selected model highlighted')
 +note('Sigmoid calibration won on validation. On the test holdout, raw XGBoost has slightly lower Brier error. Both results are reported; the winner is not re-selected on test.')
 +`<div class="grid">${panel('Ranking comparison',image('roc.png','ROC curves of the three raw models on final test data'))}${panel('Probability reliability',image('calibration.png','Holdout calibration curves for raw, sigmoid and isotonic XGBoost probabilities'))}</div>`
 +panel('Conditional bootstrap intervals',table(D.intervals,[['metric','Metric'],['ci_low','Lower bound',v=>fmt(v,4)],['ci_high','Upper bound',v=>fmt(v,4)]])+`<p class="caption">Client-resampled intervals condition on the fitted model and split. They do not capture model-selection uncertainty or guarantee temporal stability.</p>`)
 +panel('Responsible-modeling subgroup audit',table(D.subgroups,[['attribute','Attribute'],['group','Source group'],['n','Clients',fmt],['event_rate','Event rate',pct],['roc_auc','ROC-AUC',v=>fmt(v,4)],['brier','Brier',v=>fmt(v,4)],['calibration_gap','Mean-score gap',pct],['selection_rate','Simulated acceptance',pct]])+`<p class="caption">Demographics are excluded from scoring. Subgroup differences, proxy effects and uncertainty remain. These checks do not certify fairness.</p>`);}
function explainability(){return hero('Exact TreeSHAP','Explain the model. Respect its limits.','Saved global attributions and representative local waterfalls make the raw XGBoost benchmark inspectable.')
 +`<div class="grid">${panel('Global model associations',image('shap_summary.png','Exact TreeSHAP summary: model associations in raw log-odds'))}${panel('Mean absolute SHAP contributions',table(D.global_drivers.slice(0,10),[['feature','Encoded feature'],['mean_abs_shap_log_odds','Mean |SHAP|',v=>fmt(v,4)]])+`<p class="caption">Importance is measured on a reproducible holdout explanation sample. Correlated raw and engineered features can share attribution.</p>`)}</div>`
 +`<div class="controls"><label for="representative-select">Representative profile <select id="representative-select">${Object.keys(D.summary.representative_ids).map(x=>`<option value="${x}">${x}</option>`).join('')}</select></label></div><div id="representative"></div>`
 +note('SHAP explains a fitted function under attribution assumptions. It cannot show that changing a feature will cause repayment or default. Calibration is a separate probability mapping.');}
function representative(label){return panel(`${escapeHTML(label)} profile · Source ID ${D.summary.representative_ids[label]}`,image(`shap_${label}.png`,`${label} representative cardholder raw-tree SHAP waterfall`));}
function decisionOutput(t){const r=decision(t);return `<div class="metrics">${metric('Simulated acceptance',pct(r.acceptance,2),'Clients below review threshold')}${metric('Manual review',pct(r.review_rate,2),'Clients at or above threshold')}${metric('Event capture',pct(r.event_capture,2),'Observed events prioritized for review')}${metric('Accepted event rate',pct(r.accepted_event_rate,2),'Observed events in acceptance group')}</div><div class="grid">${panel('Capacity trade-off',decisionFlow(r)+`<p class="caption">Reducing the threshold increases review load. Higher acceptance can leave more event cases outside review.</p>`)}${panel('Confusion counts at this threshold',table([{label:'Observed event',review:r.tp,accept:r.fn},{label:'Observed non-event',review:r.fp,accept:r.tn}],[['label','Observed label'],['review','Review',fmt],['accept','Acceptance',fmt]])+`<p class="caption">False-positive rate ${pct(r.false_positive_rate,2)}; false-negative rate ${pct(r.false_negative_rate,2)}. These are retrospective outcomes, not actual approvals.</p>`)}</div>`;}
function decisions(){const points=Array.from({length:41},(_,i)=>decision(i/40));return hero('Decision-support simulation','Choose a trade-off, not just a cutoff.','Explore how a review threshold changes capacity, event capture and accepted-portfolio outcomes on the frozen holdout.')
 +`<div class="controls"><label for="threshold">Review threshold <input id="threshold" type="range" min="0" max="1" step="any" value="${D.summary.review_threshold}" aria-describedby="threshold-help"></label><output class="range-value" id="threshold-value">${D.summary.review_threshold.toFixed(4)}</output><button class="button secondary" id="reset-threshold">Reset frozen policy</button></div><p class="caption" id="threshold-help">Score at or above the threshold enters review. Slider exploration never changes the reported validation-selected policy.</p><div id="decision-output"></div>`
 +panel('Holdout threshold sensitivity',lineChart([{label:'Acceptance',color:'#287566',points:points.map(r=>[r.threshold,r.acceptance])},{label:'Event capture',color:'#b56640',points:points.map(r=>[r.threshold,r.event_capture])},{label:'False-positive rate',color:'#839fbe',points:points.map(r=>[r.threshold,r.false_positive_rate])}]))
 +note('A real bank would also need affordability, customer economics, uncertainty, eligibility constraints and model governance. This simulation is not an instruction to approve or reject loans.');}
function loss(){return hero('Explicit scenario assumptions','Same score. Different exposure.','Expected loss depends on probability, severity and exposure. Here probability is illustrative and LGD/EAD are assumed, because actual losses are unavailable.')
 +`<div class="controls"><label for="loss-probability">Illustrative event probability <input id="loss-probability" type="range" min="0" max="1" step="0.01" value="0.2"><output id="loss-p-label" class="range-value">20%</output></label><label for="loss-lgd">Assumed LGD <input id="loss-lgd" type="range" min="0" max="1" step="0.01" value="0.45"><output id="loss-lgd-label" class="range-value">45%</output></label></div><div class="controls"><label for="ead-a">Applicant A assumed EAD (NTD)<input id="ead-a" type="number" min="0" max="10000000" step="10000" value="100000"></label><label for="ead-b">Applicant B assumed EAD (NTD)<input id="ead-b" type="number" min="0" max="10000000" step="10000" value="200000"></label></div><div id="loss-output"></div>`
 +note('Every input on this page is a scenario assumption. Using a payment-event score in place of contractual PD cannot produce real bank expected loss, IFRS 9 provisions or capital requirements. Granted credit limit is not measured EAD.');}
function lossOutput(p,l,a,b){const eads=[50000,100000,200000],lgds=[.25,.45,.65];const scenarios=eads.flatMap(e=>lgds.map(g=>({ead:e,lgd:g,loss:p*g*e})));return `<div class="metrics">${metric('Applicant A scenario loss',fmt(p*l*a)+' NTD','Same illustrative probability and LGD')}${metric('Applicant B scenario loss',fmt(p*l*b)+' NTD','Different assumed exposure')}${metric('Assumed loss severity',pct(l,0),'Conditional LGD fraction')}${metric('Illustrative probability',pct(p,0),'Assumption, not a new model prediction')}</div>`+panel('Scenario sensitivity at this probability',table(scenarios,[['ead','Assumed EAD (NTD)',fmt],['lgd','Assumed LGD',pct],['loss','Scenario loss (NTD)',fmt]]));}

const pages={overview,distribution,applicant,performance,explainability,decisions,loss};
function render(name){
  name=Object.hasOwn(pages,name)?name:'overview';
  document.querySelectorAll('nav button').forEach(b=>b.getAttribute('data-page')===name?b.setAttribute('aria-current','page'):b.removeAttribute('aria-current'));
  const content=document.getElementById('content');content.innerHTML=pages[name]();content.hidden=false;
  document.title=`${name==='overview'?'Portfolio':name.charAt(0).toUpperCase()+name.slice(1)} · Explainable Credit Risk`;
  if(name==='overview')document.getElementById('explore-decisions').onclick=()=>navigate('decisions');
  if(name==='applicant'){
    const select=document.getElementById('client-select');
    const update=()=>document.getElementById('client-details').innerHTML=clientDetail(D.clients.find(c=>c.id===Number(select.value)));
    select.onchange=update;update();
  }
  if(name==='explainability'){
    const select=document.getElementById('representative-select');
    const update=()=>document.getElementById('representative').innerHTML=representative(select.value);
    select.onchange=update;update();
  }
  if(name==='decisions'){
    const input=document.getElementById('threshold');
    const update=()=>{const v=Number(input.value);document.getElementById('threshold-value').textContent=v.toFixed(4);document.getElementById('decision-output').innerHTML=decisionOutput(v);};
    input.oninput=update;
    document.getElementById('reset-threshold').onclick=()=>{input.value=D.summary.review_threshold;update();};
    // Render the precise frozen policy first; subsequent input values obey the range step.
    document.getElementById('decision-output').innerHTML=decisionOutput(D.summary.review_threshold);
  }
  if(name==='loss'){
    const inputs=['loss-probability','loss-lgd','ead-a','ead-b'].map(id=>document.getElementById(id));
    const update=()=>{
      const [p,l,a,b]=inputs.map(i=>Number(i.value));
      document.getElementById('loss-p-label').textContent=pct(p,0);document.getElementById('loss-lgd-label').textContent=pct(l,0);
      if(!inputs.every(i=>i.value!==''&&i.checkValidity())){document.getElementById('loss-output').innerHTML=note('Enter finite nonnegative exposures within the displayed limits.');return;}
      document.getElementById('loss-output').innerHTML=lossOutput(p,l,a,b);
    };
    inputs.forEach(i=>i.oninput=update);update();
  }
}
function navigate(name){if(location.hash==='#'+name)render(name);else location.hash=name;}
document.querySelectorAll('nav button').forEach(button=>button.onclick=()=>navigate(button.dataset.page));
window.addEventListener('hashchange',()=>{if(D)render(location.hash.slice(1));});
fetch('data.json').then(response=>{if(!response.ok)throw new Error('Saved results could not be loaded');return response.json();}).then(data=>{
  D=data;document.getElementById('loading').hidden=true;render(location.hash.slice(1)||'overview');
}).catch(error=>{const loading=document.getElementById('loading');loading.setAttribute('role','alert');loading.textContent='Unable to load this snapshot. Please read the verified reports on GitHub or try again. '+error.message;});
