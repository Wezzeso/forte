"""Read-only private 8B teacher readiness and accepted-job metadata collector.

No inference POST, service change, GPU import or secret output. Accepted hashes
are written only to a new private file; terminal output is aggregate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


REMOTE = r'''
import subprocess,json,datetime
def run(args):
 r=subprocess.run(args,text=True,capture_output=True,timeout=40);r.check_returncode();return r.stdout.strip()
query="BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY; SELECT json_build_object('schema_version','forte.service-content-exposure-inventory.v1','coverage_state',CASE WHEN bool_and(input_sha256 ~ '^[0-9a-f]{64}$') THEN 'complete_snapshot' ELSE 'incomplete_invalid_hash_metadata' END,'scope','all_accepted_jobs_since_service_start','captured_at_utc',CURRENT_TIMESTAMP,'snapshot_isolation','PostgreSQL REPEATABLE READ READ ONLY','accepted_job_count',COUNT(*),'unique_content_count',COUNT(DISTINCT input_sha256),'active_job_count',COUNT(*) FILTER (WHERE status NOT IN ('done','failed','cancelled')),'records',json_agg(json_build_object('content_hash',input_sha256)),'later_uploads_included',false) FROM jobs; COMMIT;"
inventory=json.loads(run(['docker','exec','forte-db-1','psql','-qAt','-U','forte','-d','forte','-c',query]))
containers=json.loads(run(['docker','inspect','forte-model-1','forte-api-1','forte-worker-1']))
states={x['Name'].lstrip('/'):{'status':x['State']['Status'],'paused':x['State']['Paused'],'image':x['Image']} for x in containers}
probe=r"""
import json,os,urllib.request,urllib.error
out={}
headers={}
key=os.environ.get('VLLM_API_KEY') or os.environ.get('FORTE_MODEL_API_KEY')
if key:headers['Authorization']='Bearer '+key
for name,path in [('health','/health'),('ready','/readyz'),('models','/v1/models'),('metrics','/metrics')]:
 try:
  with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8000'+path,headers=headers),timeout=8) as r:
   body=r.read(1048576);out[name+'_status']=r.status
  if name=='ready':
   obj=json.loads(body);out['ready']={k:obj.get(k) for k in ['ready','model_files_verified','model_revision','processor_revision','vllm_version','model_manifest_sha256','tokenizer_sha256','chat_template_sha256']}
  elif name=='models':out['served_model_ids']=[x.get('id') for x in json.loads(body).get('data',[])]
  elif name=='metrics':
   out['model_queue_running_waiting']={k:sum(float(line.split()[-1]) for line in body.decode().splitlines() if line.startswith(k)) for k in ['vllm:num_requests_running','vllm:num_requests_waiting']}
 except urllib.error.HTTPError as e:out[name+'_status']=e.code
 except Exception as e:out[name+'_error_type']=type(e).__name__
print(json.dumps(out))
"""
http=json.loads(run(['docker','exec','forte-model-1','python3','-c',probe])) if states['forte-model-1']['status']=='running' else {'ready_status':0}
print(json.dumps({'inventory':inventory,'attestation':{'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'containers':states,**http,'read_only':True,'inference_posts':0,'gpu_imports':0,'service_mutations':False}}))
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-output", type=Path, required=True)
    parser.add_argument("--attestation-output", type=Path, required=True)
    args = parser.parse_args()
    if "private" not in args.inventory_output.resolve().parts or args.inventory_output.exists() or args.attestation_output.exists():
        parser.error("new private inventory and new attestation paths required")
    result = subprocess.run(["wsl", "-d", "Ubuntu", "--", "ssh", "-T", "-o", "BatchMode=yes", "-o",
        "ConnectTimeout=12", "forte-a100-smoke-20261017", "python3 -"], input=REMOTE, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    if result.returncode:
        print(json.dumps({"status": "failed", "reason_code": "teacher_readonly_attestation_query_failed"}))
        return 2
    payload = json.loads(result.stdout)
    inventory, attestation = payload["inventory"], payload["attestation"]
    ready = attestation.get("ready", {})
    expected_revision = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
    passed = (inventory.get("coverage_state") == "complete_snapshot" and inventory.get("active_job_count") == 0
        and attestation["containers"]["forte-model-1"]["image"] == "sha256:046137dc151b7741b52fbb438de7d606fa74694969c7bb4ab6642d9408b9516e"
        and attestation.get("health_status") == 200 and attestation.get("ready_status") == 200
        and ready.get("ready") is True and ready.get("model_files_verified") is True
        and ready.get("model_revision") == expected_revision and ready.get("processor_revision") == expected_revision
        and ready.get("model_manifest_sha256") == "249977e15be3d72425155475e59adf2385747b16e28654c9d13d45761abf75cc"
        and "Qwen3-VL-8B-Instruct" in attestation.get("served_model_ids", [])
        and attestation.get("model_queue_running_waiting") == {"vllm:num_requests_running": 0.0, "vllm:num_requests_waiting": 0.0})
    for path in (args.inventory_output, args.attestation_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    with args.inventory_output.open("x", encoding="utf-8") as target:
        target.write(json.dumps(inventory, indent=2) + "\n")
    attestation.update(schema_version="forte.teacher-readonly-attestation.v1", baseline_teacher_ready=passed,
        source_inventory_sha256=hashlib.sha256(args.inventory_output.read_bytes()).hexdigest(),
        accepted_jobs=inventory.get("accepted_job_count"), active_jobs=inventory.get("active_job_count"),
        local_captured_at_utc=datetime.now(timezone.utc).isoformat())
    with args.attestation_output.open("x", encoding="utf-8") as target:
        target.write(json.dumps(attestation, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"baseline_teacher_ready": passed, "captured_at_utc": attestation["captured_at_utc"],
        "accepted_jobs": inventory.get("accepted_job_count"), "active_jobs": inventory.get("active_job_count"),
        "inference_posts": 0, "attestation_sha256": hashlib.sha256(args.attestation_output.read_bytes()).hexdigest()}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
