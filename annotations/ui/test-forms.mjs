// Synthetic browser checks only. All routes are mocked; no private service,
// supplied clip, external page, screenshot or GPU request is involved.
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import {chromium} from "../../web/node_modules/@playwright/test/index.mjs";

const html=await fs.readFile(new URL("index.html",import.meta.url),"utf8"),script=await fs.readFile(new URL("app.js",import.meta.url),"utf8"),css=await fs.readFile(new URL("style.css",import.meta.url),"utf8");
const taxonomy=JSON.parse(await fs.readFile(new URL("../taxonomy.v1.json",import.meta.url),"utf8"));
const rules=JSON.parse(html.match(/<script id="review-rule-catalog" type="application\/json">(.*?)<\/script>/s)[1]);
let record={sample_id:"sample_synthetic",revision:1,group_id:"protected-synthetic-group",split:"unassigned",review_state:"pending",review_origin:null,reviewer_id:null,duration_sec:10,source_records:[],full_review_complete:false,verdict:null,review_notes:"",unresolved_items:["human_local_review_required"],negative_windows:[],events:[],corrections:[],frames:[{frame_id:"synthetic-frame-4",path:"C:/synthetic/frame-4.jpg",pts_sec:4,width:2,height:2},{frame_id:"synthetic-frame-8",path:"C:/synthetic/frame-8.jpg",pts_sec:8,width:2,height:2}]};
record.machine_first_pass={windows:[],events:[],limitations:["synthetic_unknown"],provenance:{model_id:"synthetic-only"},per_second:[{start_sec:0,end_sec:1,state:"unknown_no_sampled_frame",source_pts_sec:[],event_state:"abstain"}]};
let lastSave=null,conflict=false;const errors=[];
const browser=await chromium.launch({channel:"chrome",headless:true});
const page=await browser.newPage({viewport:{width:1440,height:1000}});
page.setDefaultTimeout(8000);
page.on("pageerror",error=>errors.push(error.message));
await page.route("**/*",async route=>{
  const url=new URL(route.request().url()),path=url.pathname;
  if(path==="/")return route.fulfill({status:200,contentType:"text/html",body:html.replace("__SESSION_TOKEN__","synthetic-token")});
  if(path==="/app.js")return route.fulfill({status:200,contentType:"text/javascript",body:script});
  if(path==="/style.css")return route.fulfill({status:200,contentType:"text/css",body:css});
  if(path==="/api/taxonomy")return route.fulfill({json:taxonomy});
  if(path==="/api/queue")return route.fulfill({json:{summary:{unique_clips:1,review_states:{[record.review_state]:1}},records:[{sample_id:record.sample_id,review_state:record.review_state,split:record.split}]}});
  if(path==="/api/records/sample_synthetic"){
    if(route.request().method()==="POST"){
      lastSave=route.request().postDataJSON();
      if(conflict)return route.fulfill({status:409,json:{error:"record changed; reload before saving"}});
      record={...record,...lastSave.patch,revision:record.revision+1};
    }
    return route.fulfill({json:record});
  }
  return route.fulfill({status:204,body:""});
});
let checks=0;
try{
  await page.goto("http://127.0.0.1:9876/");await page.locator("#queue button").click();await page.locator("#machine-panel").waitFor({state:"visible"});
  assert.equal(await page.locator("#full-reviewed").isChecked(),false);assert.equal(await page.locator("#verdict").inputValue(),"");checks++;
  assert.equal(await page.locator("#machine-panel").isVisible(),true);assert.equal(await page.locator("#state").inputValue(),"pending");checks++;
  await page.locator("#reviewer").fill("synthetic-human");await page.locator("#full-reviewed").check();await page.locator("#verdict").selectOption("no_violation");await page.locator("#review-notes").fill("Synthetic reviewed notes");await page.locator("#unresolved-items").fill("");await page.getByText("Предложение по сценарной группе",{exact:true}).click();await page.locator("#proposed-group").fill("proposed-synthetic-group");await page.locator("#scenario-reviewed").check();
  await page.locator("#add-whole-negative").click();const negative=page.locator("#negative-windows .proposal-card");
  assert.equal(await negative.getByLabel("Видимость",{exact:true}).inputValue(),"unobservable");assert.equal(await negative.getByRole("checkbox").nth(1).isChecked(),false);checks++;
  await negative.getByLabel("Видимость",{exact:true}).selectOption("clear");await negative.getByRole("checkbox").nth(0).check();await negative.getByRole("checkbox").nth(1).check();await page.locator("#state").selectOption("reviewed");await page.locator("#save").click();
  await page.waitForFunction(()=>document.getElementById("message").textContent.includes("сохранена"));assert.equal(lastSave.patch.negative_windows[0].end_sec,10);assert.equal(lastSave.patch.negative_windows[0].negative_verified,true);assert.equal(lastSave.patch.verdict,"no_violation");assert.equal(lastSave.patch.review_notes,"Synthetic reviewed notes");assert.equal(lastSave.patch.corrections[0].scenario_reviewed,true);assert.equal(lastSave.patch.corrections[0].proposed_group_id,"proposed-synthetic-group");assert.equal("group_id" in lastSave.patch,false);assert.equal(record.group_id,"protected-synthetic-group");checks++;
  await page.locator("#reviewer").fill("synthetic-second-reviewer");await page.locator("#review-notes").fill("Synthetic later notes");const laterSave=page.waitForResponse(response=>response.url().endsWith("/api/records/sample_synthetic")&&response.request().method()==="POST");await page.locator("#save").click();await laterSave;assert.equal(lastSave.patch.corrections[0].reviewer_id,"synthetic-human");checks++;
  await page.locator("#event-type").selectOption("CASH_UNATTENDED");await page.locator("#template").click();
  const event=page.locator("#event-forms section").first();let draft=JSON.parse(await page.locator("#editor").inputValue());assert.ok(draft.events[0].facts.every(fact=>fact.state==="unclear"));checks++;
  const rule=rules.rules.find(value=>value.event_type_id==="CASH_UNATTENDED");await event.getByLabel("Точная применимая норма",{exact:true}).selectOption(rule.rule_id);
  draft=JSON.parse(await page.locator("#editor").inputValue());assert.ok(Object.keys(rule.required_fact_states).every(id=>draft.events[0].facts.some(fact=>fact.fact_id===id&&fact.state==="unclear")));assert.equal(draft.events[0].rule_grounding_status,"unresolved");checks++;
  await event.getByLabel("Начало события, с",{exact:true}).fill("3");await event.getByLabel("Конец события, с",{exact:true}).fill("5");await event.getByLabel("Опорный исходный кадр — точное PTS",{exact:true}).selectOption("synthetic-frame-4");
  await event.locator(".fact-card").first().getByLabel("Наблюдаемое состояние",{exact:true}).selectOption("present");await event.locator(".fact-card").first().getByLabel("Исходные кадры, подтверждающие этот признак",{exact:true}).selectOption(["synthetic-frame-4"]);
  await event.getByRole("checkbox",{name:"Последовательность и полный релевантный контекст проверены мной",exact:true}).check();await event.getByRole("checkbox",{name:"Исключения применимой нормы проверены мной",exact:true}).check();
  draft=JSON.parse(await page.locator("#editor").inputValue());assert.equal(draft.events[0].anchor_sec,4);assert.deepEqual(draft.events[0].evidence_frame_ids,["synthetic-frame-4"]);assert.equal(draft.events[0].facts.filter(fact=>fact.state==="present").length,1);assert.equal(draft.events[0].rule_grounding_status,"unresolved");assert.equal(draft.events[0].temporal_verified,true);checks++;
  draft.review_notes="Synthetic advanced JSON";draft.full_review_complete=false;await page.getByText("Расширенный JSON",{exact:true}).click();await page.locator("#editor").fill(JSON.stringify(draft));await page.locator("#apply-json").click();assert.equal(await page.locator("#review-notes").inputValue(),"Synthetic advanced JSON");assert.equal(await page.locator("#full-reviewed").isChecked(),false);checks++;
  conflict=true;const before=await page.locator("#editor").inputValue();await page.locator("#save").click();await page.waitForFunction(()=>document.getElementById("message").textContent.includes("record changed"));assert.deepEqual(JSON.parse(await page.locator("#editor").inputValue()),JSON.parse(before));assert.deepEqual(errors,[]);checks++;
  process.stdout.write(JSON.stringify({synthetic_form_checks_passed:checks,private_service_accessed:false,screenshots_created:false})+"\n");
}finally{await browser.close();}
