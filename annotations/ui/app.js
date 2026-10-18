"use strict";
const token = document.querySelector('meta[name="review-token"]').content;
const $ = id => document.getElementById(id);
const labels = {pending:"Ожидает",in_progress:"В работе",reviewed:"Проверено",needs_adjudication:"Повторная проверка"};
let record = null, taxonomy = null;
const ruleCatalog = JSON.parse($("review-rule-catalog").textContent);
let jsonDirty = false;
const editable = ["full_review_complete","verdict","events","negative_windows","visibility_regions","unresolved_items","frames","review_notes","corrections","local_model_provenance"];
function message(text) { $("message").textContent = text; }
async function api(path, options = {}) {
  const response = await fetch(path, {...options, headers:{"X-Review-Token":token,"Content-Type":"application/json",...(options.headers||{})}, cache:"no-store"});
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || "Ошибка локальной операции");
  return body;
}
async function refresh() {
  const queue = await api("/api/queue");
  $("summary").textContent = `Уникальных записей: ${queue.summary.unique_clips}. Проверено: ${queue.summary.review_states.reviewed || 0}.`;
  $("queue").replaceChildren();
  for (const item of queue.records) {
    const button = document.createElement("button");
    button.textContent = `${item.sample_id} · ${labels[item.review_state]} · ${item.split}`;
    button.addEventListener("click", () => openRecord(item.sample_id).catch(error => message(error.message)));
    $("queue").append(button);
  }
}
async function openRecord(id) {
  record = await api(`/api/records/${encodeURIComponent(id)}`);
  $("review").hidden = false;
  $("sample").textContent = record.sample_id;
  $("video").src = `/media/${encodeURIComponent(id)}?token=${encodeURIComponent(token)}`;
  $("video").muted = true;
  $("sources").textContent = JSON.stringify(record.source_records, null, 2);
  $("reviewer").value = record.reviewer_id || "";
  $("state").value = record.review_state;
  $("editor").value = JSON.stringify(Object.fromEntries(editable.filter(key => key in record).map(key => [key,record[key]])), null, 2);
  jsonDirty = false;
  renderForms();
  renderProposals();
  renderMachineSummary();
  message("Запись открыта локально. Проверка еще не сохранена.");
}
function frameUrl(frameId) {
  return `/frames/${encodeURIComponent(record.sample_id)}/${encodeURIComponent(frameId)}?token=${encodeURIComponent(token)}`;
}
function renderMachineSummary(){
  const row=record.machine_first_pass;$("machine-panel").hidden=!row;$("machine-summary").replaceChildren();if(!row)return;
  $("machine-summary").append(element("p",`Машинных окон: ${row.windows.length}. Секундных интервалов: ${(row.per_second||[]).length}. Статус: автоматическая проверка согласованности. Человеческая разметка не требуется для продолжения машинного анализа.`));
  const labels={unknown_no_sampled_frame:"Нет исходного кадра — неопределенно",source_frames_available_unanalyzed:"Кадры есть, вывода модели нет",partial_machine_observation:"Частичное машинное наблюдение",machine_observed:"Есть машинное наблюдение"};
  const limits=element("p",`Ограничения: ${row.limitations.join("; ")}`);limits.className="hint";$("machine-summary").append(limits);
  const provided=row.provided_focus||row.provided_evaluation_focus;if(provided){const points=provided.coarse_anchor_sec.join(", ")||"нет";$("machine-summary").append(element("p",`Предоставленные приблизительные ориентиры: ${points} с. Это отдельные исходные подсказки, не тип события и не его границы. ${row.provided_evaluation_focus?"Только оценка после предсказания, не вход модели.":"Допустимы для выбора обучающего контекста; ответ модели ими не задается."}`));}
  if(row.explanation_ru){const explanation=row.explanation_ru;$("machine-summary").append(element("h4","Что наблюдается и почему вывод ограничен"),element("p",explanation.what_ru),element("p",explanation.why_ru));for(const observation of explanation.physical_observations||[]){const line=element("p",`${observation.text_ru} [${observation.fact_id}; машинное наблюдение]`);const frameId=observation.supporting_frame_ids[0];if(frameId){const link=element("a"," Исходный кадр");link.href=frameUrl(frameId);link.target="_blank";link.rel="noopener";line.append(link);}$("machine-summary").append(line);}for(const explanationEvent of explanation.events||[]){$("machine-summary").append(element("h4",explanationEvent.what_ru),element("p",explanationEvent.why_ru));for(const rule of explanationEvent.rule_candidates||[])$("machine-summary").append(element("p",`${rule.title_ru}, ${rule.clause}: ${rule.required_states_match?"физические признаки совпали, порядок и исключения не подтверждены":"не хватает обязательных признаков: "+rule.unsatisfied_fact_ids.join(", ")}`));}}
  for(const event of row.events||[]){const card=element("div");card.className="proposal-card";card.append(element("h4",taxonomy.event_types.find(value=>value.event_type_id===event.event_type_id)?.name_ru||event.event_type_id),element("p",`${event.start_sec}–${event.end_sec} с; автоматический кандидат, не человеческий эталон.`));if(event.anchor_sec!==null&&event.anchor_sec!==undefined){const jump=element("button",`Перейти к исходному времени ${event.anchor_sec} с`);jump.addEventListener("click",()=>{$("video").currentTime=event.anchor_sec;});card.append(jump);}const facts=element("pre");facts.textContent=JSON.stringify(event.facts||event.observed_facts||[],null,2);card.append(facts);$("machine-summary").append(card);}
  const details=element("details"),summary=element("summary","Каждая секунда: исходные PTS и наблюдаемое покрытие"),table=element("table"),head=element("tr");for(const name of ["Интервал, с","Исходные PTS, с","Статус","Событие"])head.append(element("th",name));table.append(head);
  for(const bin of row.per_second||[]){const tr=element("tr");for(const value of [`${bin.start_sec}–${bin.end_sec}`,bin.source_pts_sec.map(value=>value.toFixed(3)).join(", ")||"—",labels[bin.state]||bin.state,bin.event_state==="machine_candidate_present"?"Машинный кандидат":"Воздержание от вывода"])tr.append(element("td",value));table.append(tr);}details.append(summary,table);$("machine-summary").append(details);
  const raw=element("details"),caption=element("summary","Происхождение и маски машинных наблюдений"),body=element("pre");body.textContent=JSON.stringify({provenance:row.provenance,partition_binding:row.partition_binding,windows:row.windows},null,2);raw.append(caption,body);$("machine-summary").append(raw);
}
function renderProposals() {
  const proposals = record.local_model_proposals || [];
  $("proposal-panel").hidden = proposals.length === 0;
  $("proposals").replaceChildren();
  for (const proposal of proposals) {
    const group = document.createElement("details"); group.open = true;
    const summary = document.createElement("summary");
    summary.textContent = `${proposal.proposal_id} · предложение требует проверки`;
    group.append(summary);
    const provenance = document.createElement("pre");
    provenance.textContent = JSON.stringify({phase:proposal.phase,error_code:proposal.error_code,publication_ready:proposal.publication_ready,provenance:proposal.provenance,coverage:proposal.coverage,unresolved_intervals:proposal.unresolved_intervals,limitations:proposal.limitations},null,2);
    group.append(provenance);
    const events = [...(proposal.verified_events || []),...(proposal.candidate_events || [])];
    for (const event of events) {
      const card = document.createElement("div"); card.className="proposal-card";
      const title = document.createElement("h4");
      title.textContent = `${taxonomy.event_types.find(value=>value.event_type_id===event.event_type_id)?.name_ru || event.event_type_id} · ${event.start_sec.toFixed(3)}–${event.end_sec.toFixed(3)} с`;
      const jump = document.createElement("button"); jump.textContent="Показать интервал";
      jump.addEventListener("click",()=>{$("video").currentTime=event.start_sec;});
      const copy = document.createElement("button"); copy.textContent="Копировать в черновик проверки";
      copy.addEventListener("click",()=>draftProposal(proposal,event));
      const facts = document.createElement("pre"); facts.textContent=JSON.stringify(event.observed_facts || event.supporting_facts || [],null,2);
      const gallery = document.createElement("div"); gallery.className="proposal-frames";
      const ids = event.evidence_frame_ids || event.supporting_frame_ids || [];
      for (const frameId of ids.slice(0,3)) {
        const image = document.createElement("img"); image.src=frameUrl(frameId); image.alt=`Исходный кадр ${frameId}`; image.loading="lazy";
        const link = document.createElement("a"); link.href=image.src; link.target="_blank"; link.rel="noopener"; link.append(image); gallery.append(link);
      }
      card.append(title,jump,copy,gallery,facts); group.append(card);
    }
    if(events.length===0){const notice=document.createElement("p");notice.textContent="Модель не предложила событие. Это не подтверждает отрицательный результат. Проверяйте запись и видимость полностью.";group.append(notice);}
    const raw = document.createElement("details"), label = document.createElement("summary"), body = document.createElement("pre");
    label.textContent="Все локальные решения по окнам (непроверенные)"; body.textContent=JSON.stringify(proposal.window_decisions || proposal.decisions || [],null,2); raw.append(label,body);group.append(raw);
    $("proposals").append(group);
  }
}
function draftProposal(proposal,event) {
  try {
    const patch=JSON.parse($("editor").value), frameMap=new Map((proposal.frames || []).map(frame=>[frame.frame_id,frame]));
    const facts=event.observed_facts || event.supporting_facts || [];
    const sourceIds=[...new Set([...(event.evidence_frame_ids || []),...(event.supporting_frame_ids || []),...facts.flatMap(fact=>fact.supporting_frame_ids || [])])];
    const anchor=frameMap.get(event.anchor_frame_id) || sourceIds.map(id=>frameMap.get(id)).filter(frame=>frame && event.start_sec<=frame.pts_sec && frame.pts_sec<=event.end_sec).sort((a,b)=>Math.abs(a.pts_sec-(event.start_sec+event.end_sec)/2)-Math.abs(b.pts_sec-(event.start_sec+event.end_sec)/2))[0];
    if(!anchor) throw new Error("У предложения нет опорного исходного кадра внутри интервала. Проверьте его вручную.");
    patch.events ||= []; patch.frames ||= [];
    const existing=new Map(patch.frames.map(frame=>[frame.frame_id,frame]));
    for(const id of sourceIds){const frame=frameMap.get(id);if(!frame)throw new Error("Неизвестный исходный кадр предложения.");if(existing.has(id)&&JSON.stringify(existing.get(id))!==JSON.stringify(frame))throw new Error("Конфликт зарегистрированного кадра.");if(!existing.has(id)){patch.frames.push(frame);existing.set(id,frame);}}
    if(!existing.has(anchor.frame_id))patch.frames.push(anchor);
    patch.events.push({event_id:`event_${crypto.randomUUID()}`,event_type_id:event.event_type_id,start_sec:event.start_sec,end_sec:event.end_sec,anchor_sec:anchor.pts_sec,rule_id:event.rule_id || null,rule_grounding_status:"unresolved",temporal_verified:false,exceptions_verified:false,evidence_frame_ids:[anchor.frame_id,...sourceIds.filter(id=>id!==anchor.frame_id)].slice(0,3),facts,local_proposal_id:proposal.proposal_id,limitations:[]});
    patch.verdict=null; patch.full_review_complete=false;
    $("state").value="in_progress"; $("editor").value=JSON.stringify(patch,null,2);
    jsonDirty=false; renderForms();
    message("Предложение скопировано только в несохраненный черновик. Проверьте каждый признак, последовательность и исключения; модель не выполнила человеческую проверку.");
  }catch(error){message(error.message);}
}
function readDraft(){return JSON.parse($("editor").value);}
function updateDraft(change,render=false){try{const patch=readDraft();change(patch);$("editor").value=JSON.stringify(patch,null,2);jsonDirty=false;if(render)renderForms();}catch(error){message(error.message);}}
function element(tag,text){const node=document.createElement(tag);if(text!==undefined)node.textContent=text;return node;}
function field(label,control){const wrapper=element("label",label);control.setAttribute("aria-label",label);wrapper.append(control);return wrapper;}
function checkbox(label,value,onChange){const input=element("input");input.type="checkbox";input.checked=Boolean(value);input.addEventListener("change",()=>onChange(input.checked));const wrapper=field(label,input);wrapper.className="check";return wrapper;}
function numberField(label,value,onChange){const input=element("input");input.type="number";input.min="0";input.step="0.001";input.value=Number.isFinite(value)?value:"";input.addEventListener("change",()=>onChange(input.value===""?null:Number(input.value)));return field(label,input);}
function selectField(label,options,value,onChange,multiple=false){const select=element("select");select.multiple=multiple;if(multiple)select.size=3;for(const [key,title] of options){const option=element("option",title);option.value=key;option.selected=multiple?(value||[]).includes(key):key===(value??"");select.append(option);}select.addEventListener("change",()=>onChange(multiple?[...select.selectedOptions].map(option=>option.value):select.value));return field(label,select);}
function availableFrames(patch){const frames=new Map();for(const frame of [...(record.frames||[]),...(patch.frames||[]),...(record.local_model_proposals||[]).flatMap(proposal=>proposal.frames||[])]){const prior=frames.get(frame.frame_id);if(prior&&(prior.path!==frame.path||prior.pts_sec!==frame.pts_sec))throw new Error("Конфликт исходного кадра; нужна локальная проверка.");frames.set(frame.frame_id,frame);}return frames;}
function registerFrames(patch,ids){const available=availableFrames(patch);patch.frames||=[];const existing=new Set(patch.frames.map(frame=>frame.frame_id));for(const id of ids){if(!available.has(id))throw new Error("Исходный кадр не зарегистрирован.");if(!existing.has(id)){patch.frames.push(available.get(id));existing.add(id);}}}
function ensureFacts(event){const definition=taxonomy.event_types.find(value=>value.event_type_id===event.event_type_id);const rule=ruleCatalog.rules.find(value=>value.rule_id===event.rule_id);const required=new Set([...Object.keys(definition?.required_fact_states||{}),...Object.keys(rule?.required_fact_states||{})]);event.facts||=[];const known=new Set(event.facts.map(fact=>fact.fact_id));for(const fact_id of required)if(!known.has(fact_id))event.facts.push({fact_id,state:"unclear",supporting_frame_ids:[]});}
function syncSimpleForm(){updateDraft(patch=>{patch.full_review_complete=$("full-reviewed").checked;patch.verdict=$("verdict").value||null;patch.review_notes=$("review-notes").value;patch.unresolved_items=$("unresolved-items").value.split("\n").map(value=>value.trim()).filter(Boolean);const proposed=$("proposed-group").value.trim(), reviewed=$("scenario-reviewed").checked, prior=(patch.corrections||[]).findLast(item=>item.kind==="scenario_group_proposal");const unchanged=prior?.proposed_group_id===proposed&&prior?.scenario_reviewed===reviewed;patch.corrections=(patch.corrections||[]).filter(item=>item.kind!=="scenario_group_proposal");if(proposed||reviewed)patch.corrections.push({kind:"scenario_group_proposal",proposed_group_id:proposed,scenario_reviewed:reviewed,reviewer_id:unchanged&&prior?.reviewer_id?prior.reviewer_id:$("reviewer").value.trim()||null});});}
function renderForms(){
  const patch=readDraft();
  $("full-reviewed").checked=Boolean(patch.full_review_complete);$("verdict").value=patch.verdict||"";$("review-notes").value=patch.review_notes||"";$("unresolved-items").value=(patch.unresolved_items||[]).join("\n");
  const scenario=(patch.corrections||[]).findLast(item=>item.kind==="scenario_group_proposal");$("proposed-group").value=scenario?.proposed_group_id||"";$("scenario-reviewed").checked=Boolean(scenario?.scenario_reviewed);$("current-group").textContent=`Текущая защищенная группа: ${record.group_id} · split: ${record.split}`;
  renderNegativeWindows(patch);renderEventForms(patch);
}
function renderNegativeWindows(patch){
  $("negative-windows").replaceChildren();
  (patch.negative_windows||[]).forEach((window,index)=>{
    const card=element("div");card.className="proposal-card";card.append(element("h4",`Интервал ${index+1}`));
    const set=(key,value)=>updateDraft(draft=>draft.negative_windows[index][key]=value);
    const row=element("div");row.className="row";row.append(numberField("Начало, с",window.start_sec,value=>set("start_sec",value)),numberField("Конец, с",window.end_sec,value=>set("end_sec",value)),selectField("Видимость",[["unobservable","Не наблюдается"],["partial","Частично"],["clear","Ясно"]],window.visibility,value=>set("visibility",value)));card.append(row);
    card.append(checkbox("Весь необходимый этап и все зоны доступны проверке",window.context_complete,value=>set("context_complete",value)),checkbox("Я проверил этот интервал: признаки события отсутствуют",window.negative_verified,value=>set("negative_verified",value)));
    const remove=element("button","Удалить интервал");remove.addEventListener("click",()=>updateDraft(draft=>draft.negative_windows.splice(index,1),true));card.append(remove);$("negative-windows").append(card);
  });
}
function renderEventForms(patch){
  $("event-forms").replaceChildren();const frames=availableFrames(patch), options=[...frames.values()].sort((a,b)=>a.pts_sec-b.pts_sec).map(frame=>[frame.frame_id,`${frame.pts_sec.toFixed(3)} с · ${frame.frame_id}`]);
  if(!options.length)$("event-forms").append(element("p","Исходные кадры еще не зарегистрированы. Импортируйте локальное предложение либо добавьте исходные FrameRef в расширенном JSON. Отрицательные интервалы можно проверять по локальному видео."));
  (patch.events||[]).forEach((event,index)=>{
    const card=element("section");card.className="proposal-card";card.append(element("h4",`Событие ${index+1}`));const set=(key,value,render=false)=>updateDraft(draft=>draft.events[index][key]=value,render);
    card.append(selectField("Тип события",taxonomy.event_types.map(value=>[value.event_type_id,value.name_ru]),event.event_type_id,value=>updateDraft(draft=>{draft.events[index].event_type_id=value;draft.events[index].rule_id=null;draft.events[index].rule_grounding_status="unresolved";ensureFacts(draft.events[index]);},true)));
    const row=element("div");row.className="row";row.append(numberField("Начало события, с",event.start_sec,value=>set("start_sec",value)),numberField("Конец события, с",event.end_sec,value=>set("end_sec",value)));card.append(row);
    const matchingRule=ruleCatalog.rules.find(value=>value.rule_id===event.rule_id);
    card.append(selectField("Точная применимая норма",[["","Норма пока не выбрана"],...ruleCatalog.rules.filter(value=>value.event_type_id===event.event_type_id).map(value=>[value.rule_id,`${value.clause||""} · ${value.rule_id}`])],event.rule_id||"",value=>updateDraft(draft=>{draft.events[index].rule_id=value||null;draft.events[index].rule_grounding_status="unresolved";draft.events[index].temporal_verified=false;draft.events[index].exceptions_verified=false;ensureFacts(draft.events[index]);},true)));
    if(matchingRule)card.append(element("p",`${matchingRule.title}, ${matchingRule.clause||""}. Каталог ${ruleCatalog.catalog_version}; окончательную проверку выполняет backend.`));
    const anchor=event.evidence_frame_ids?.find(id=>frames.has(id)&&Math.abs(frames.get(id).pts_sec-event.anchor_sec)<0.000001)||"";
    card.append(selectField("Опорный исходный кадр — точное PTS",[["","Выберите исходный кадр"],...options],anchor,id=>{if(!id)return;updateDraft(draft=>{registerFrames(draft,[id]);const target=draft.events[index];target.anchor_sec=availableFrames(draft).get(id).pts_sec;target.evidence_frame_ids=[id,...(target.evidence_frame_ids||[]).filter(value=>value!==id)].slice(0,3);},true);}),selectField("Кадры доказательств — 1–3",options,event.evidence_frame_ids||[],ids=>{if(ids.length>3){message("Допустимо не более трех кадров доказательств.");renderForms();return;}updateDraft(draft=>{registerFrames(draft,ids);draft.events[index].evidence_frame_ids=ids;});},true));
    card.append(element("p",`Опорное время: ${event.anchor_sec??"не выбрано"} с. Кадр должен лежать внутри интервала события.`));
    (event.facts||[]).forEach((fact,factIndex)=>{
      const definition=taxonomy.facts.find(value=>value.fact_id===fact.fact_id),factCard=element("div");factCard.className="fact-card";const expected=matchingRule?.required_fact_states?.[fact.fact_id];
      factCard.append(element("strong",definition?.question_ru||fact.fact_id));if(expected)factCard.append(element("p",`Для выбранной нормы требуется: ${expected}. Это условие, а не наблюдение.`));
      factCard.append(selectField("Наблюдаемое состояние",[["unclear","Неясно"],["present","Присутствует"],["absent","Проверенно отсутствует"]],fact.state,value=>updateDraft(draft=>draft.events[index].facts[factIndex].state=value)),selectField("Исходные кадры, подтверждающие этот признак",options,fact.supporting_frame_ids||[],ids=>updateDraft(draft=>{registerFrames(draft,ids);draft.events[index].facts[factIndex].supporting_frame_ids=ids;}),true));card.append(factCard);
    });
    card.append(checkbox("Последовательность и полный релевантный контекст проверены мной",event.temporal_verified,value=>set("temporal_verified",value)),checkbox("Исключения применимой нормы проверены мной",event.exceptions_verified,value=>set("exceptions_verified",value)),checkbox("Применимость нормы и все требуемые факты проверены мной",event.rule_grounding_status==="verified_applicable",value=>set("rule_grounding_status",value?"verified_applicable":"unresolved")));
    const remove=element("button","Удалить событие из черновика");remove.addEventListener("click",()=>updateDraft(draft=>draft.events.splice(index,1),true));card.append(remove);$("event-forms").append(card);
  });
}
for(const id of ["full-reviewed","verdict","review-notes","unresolved-items","proposed-group","scenario-reviewed"])$(id).addEventListener("change",syncSimpleForm);
$("editor").addEventListener("input",()=>{jsonDirty=true;});
$("apply-json").addEventListener("click",()=>{try{renderForms();jsonDirty=false;message("JSON применен к формам. Проверка не сохранена.");}catch(error){message(error.message);}});
for(const [id,whole] of [["add-negative",false],["add-whole-negative",true]])$(id).addEventListener("click",()=>updateDraft(patch=>{patch.negative_windows||=[];patch.negative_windows.push({start_sec:0,end_sec:whole?record.duration_sec:0,visibility:"unobservable",context_complete:false,negative_verified:false,criteria_version:taxonomy.schema_version});},true));
$("video").addEventListener("timeupdate", () => { $("clock").textContent = `${$("video").currentTime.toFixed(3)} с`; });
$("video").addEventListener("volumechange", () => { if (!$("video").muted) $("video").muted = true; });
$("video").addEventListener("error", () => message("Браузер не декодирует исходный контейнер. Создайте локальный беззвучный прокси командой python -m annotations.cli proxy или используйте локальный плеер."));
$("jump").addEventListener("click", () => { $("video").currentTime = Number($("seek").value) || 0; });
$("reload").addEventListener("click", () => refresh().catch(error => message(error.message)));
$("save").addEventListener("click", async () => {
  if (!record) return;
  try {
    if(!jsonDirty)syncSimpleForm();
    if($("scenario-reviewed").checked&&!$("proposed-group").value.trim())throw new Error("Укажите ID предлагаемой сценарной группы либо снимите отметку ее проверки.");
    const patch = {...JSON.parse($("editor").value),reviewer_id:$("reviewer").value.trim(),review_state:$("state").value,review_origin:"human"};
    record = await api(`/api/records/${encodeURIComponent(record.sample_id)}`,{method:"POST",body:JSON.stringify({patch,expected_revision:record.revision})});
    await refresh();
    jsonDirty=false;
    message("Проверка сохранена с новой ревизией. Исходные метки и предыдущая версия сохранены в локальном журнале.");
  } catch(error) { message(error.message); }
});
$("template").addEventListener("click", () => {
  try {
    const patch = JSON.parse($("editor").value);
    const definition = taxonomy.event_types.find(event => event.event_type_id === $("event-type").value);
    const timestamp = Number($("video").currentTime.toFixed(3));
    patch.events ||= [];
    patch.events.push({event_id:`event_${crypto.randomUUID()}`,event_type_id:definition.event_type_id,start_sec:timestamp,end_sec:timestamp,anchor_sec:timestamp,rule_id:null,rule_grounding_status:"unresolved",temporal_verified:false,exceptions_verified:false,evidence_frame_ids:[],facts:Object.entries(definition.required_fact_states).map(([fact_id]) => ({fact_id,state:"unclear",supporting_frame_ids:[]})),limitations:[]});
    patch.verdict=null;patch.full_review_complete=false;$("state").value="in_progress";
    $("editor").value = JSON.stringify(patch,null,2);
    jsonDirty=false;renderForms();
    message("Добавлен непроверенный шаблон. Укажите исходные кадры, ненулевой интервал и подтвержденные состояния. Шаблон не является событием или разметкой.");
  } catch(error) { message(error.message); }
});
(async () => {
  taxonomy = await api("/api/taxonomy");
  $("taxonomy").textContent = JSON.stringify(taxonomy,null,2);
  for (const event of taxonomy.event_types) {
    const option = document.createElement("option"); option.value=event.event_type_id; option.textContent=event.name_ru; $("event-type").append(option);
  }
  await refresh();
})().catch(error => message(error.message));
