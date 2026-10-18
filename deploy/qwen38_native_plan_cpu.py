"""Actual-CPU metadata binder: hash public sources and receipts; no network/model.

Run inside the intended consumer image with actual source/config/receipt mounts.
The host must independently inspect the image and supply its exact digest via
FORTE_NATIVE_WORKER_IMAGE_SHA256. No key, GPU or private video mount is needed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path,PurePosixPath
import sys
import tarfile


def source_inventory(path,expected):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    if digest.hexdigest()!=expected:raise ValueError('native_CPU_actual_consumer_archive_hash_mismatch')
    files={};total=0
    with tarfile.open(path,'r:*') as archive:
        for member in archive:
            logical=PurePosixPath(member.name)
            if (not member.isfile() or member.issym() or member.islnk() or logical.is_absolute()
                or '..' in logical.parts or '\\' in member.name or ':' in member.name or member.name in files
                or not 0<=member.size<=16*1024*1024):raise ValueError('native_CPU_public_archive_member_invalid')
            total+=member.size
            if total>128*1024*1024 or len(files)>=10000:raise ValueError('native_CPU_public_archive_bound_exceeded')
            files[member.name]=hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    return files


def prepare(spec_path,protocol_path,*,source_root,source_archive,source_proof_output,output):
    root=Path(source_root).resolve()
    sys.path[:0]=[str(root),str(root/'src')]
    try:import native_plan_contract as factory
    except ModuleNotFoundError as error:
        if error.name!='native_plan_contract':raise
        from benchmarks.performance import qwen38_native_plan as factory
    core=factory.parent_contract()
    spec=core.read_json(spec_path);protocol=core.read_json(protocol_path)
    bindings=spec['consumer_runtime_bindings']
    core.require(core._sha(bindings.get('runtime_image_sha256'))
        and os.environ.get('FORTE_NATIVE_WORKER_IMAGE_SHA256')==bindings['runtime_image_sha256'],
        'native_CPU_actual_consumer_image_assertion_required')
    files=source_inventory(source_archive,bindings['source_archive_sha256'])
    paths=spec['artifact_paths']
    core.require(set(paths)==factory.ARTIFACT_NAMES-{'consumer_source_proof'},'native_CPU_exact_receipt_paths_required')
    artifacts={name:factory.artifact(path) for name,path in paths.items()}
    runtime=factory._runtime(artifacts['native_config']['document'])
    modules={'forte.pipeline','forte.model.adapter','forte.video.sampling','forte.video.decode',
        'forte.decision.engine','forte.evidence.assembly',runtime['provider_module'],runtime['identity_module'],
        runtime['profile_module'],runtime['readiness_module']}
    loaded={};origins={}
    for name in sorted(modules):
        module=importlib.import_module(name);origin=Path(module.__file__).resolve()
        member=('src/' if name.startswith('forte.') else '')+name.replace('.','/')+'.py'
        actual=core.file_digest(origin)
        core.require(files.get(member)==actual,'native_CPU_imported_consumer_module_differs_from_actual_archive')
        loaded[name]=actual;origins[name]=str(origin)
    proof={'schema_version':'forte.qwen38-native-consumer-source-proof.v1',
        'source_archive_sha256':bindings['source_archive_sha256'],'runtime_image_sha256':bindings['runtime_image_sha256'],
        'actual_imports_match_source_archive':True,'loaded_module_sha256':loaded,'actual_module_origins':origins,
        'source_archive_member_count':len(files),'image_independently_inspected_here':False,
        'network_calls':0,'model_calls':0,'gpu_operations':0,'private_media_read':False}
    core.write_new(source_proof_output,proof)
    artifacts['consumer_source_proof']=factory.artifact(source_proof_output)
    bindings.update(actual_module_sha256=loaded,consumer_source_proof_sha256=artifacts['consumer_source_proof']['raw_sha256'],
        consumer_prepare_receipt_sha256=artifacts['consumer_prepare']['raw_sha256'])
    metadata={key:spec[key] for key in ('arm','sources','base_url','launch_config_path')}
    metadata.update(artifacts=artifacts,consumer_runtime_bindings=bindings,
        native_admission_addendum=core.read_json(spec['native_admission_addendum_path']))
    for name in ('native_profile_amendment','native_parallel4_public_gate'):
        if name+'_path' in spec:metadata[name]=factory.artifact(spec[name+'_path'])
    plan=factory.bind_native_plan(protocol,metadata)
    factory.validate_native_admission(plan)
    core.write_new(output,plan)
    return {'status':'actual_CPU_native_plan_bound_not_released','source_attempts':6,'reserved_POST_scope':208,
        'native_variant':plan['native_variant'],'model_calls':0,'network_calls':0,'gpu_operations':0,'private_media_read':False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('spec','protocol','source-root','source-archive','source-proof-output','output'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args(argv)
    try:print(json.dumps(prepare(args.spec,args.protocol,source_root=args.source_root,source_archive=args.source_archive,
        source_proof_output=args.source_proof_output,output=args.output),sort_keys=True));return 0
    except Exception as error:
        print(json.dumps({'status':'native_CPU_plan_binding_failed','error_type':type(error).__name__,'model_calls':0}));return 1


if __name__=='__main__':raise SystemExit(main())
