"""CPU-only pinned local Qwen crop processor contract with generated pixels."""
from __future__ import annotations

import argparse
import glob
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_HUB_DISABLE_TELEMETRY='1', WANDB_MODE='disabled',
                      TOKENIZERS_PARALLELISM='false', OTEL_SDK_DISABLED='true')
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from training.public_crop_lora import PublicCropCollator, QUESTION

    assert not torch.cuda.is_initialized() and not glob.glob('/dev/nvidia*')
    root = args.model_dir.resolve()
    config = json.loads((root/'config.json').read_text())
    assert config['model_type']=='qwen3_vl'
    processor = AutoProcessor.from_pretrained(str(root), local_files_only=True,
                                             trust_remote_code=False, max_pixels=65536)
    assert processor.__class__.__name__=='Qwen3VLProcessor'
    for field, token in (('image_token_id','<|image_pad|>'),('video_token_id','<|video_pad|>'),
                         ('vision_start_token_id','<|vision_start|>'),('vision_end_token_id','<|vision_end|>')):
        assert processor.tokenizer.convert_tokens_to_ids(token)==config[field]
    cases=[]
    with tempfile.TemporaryDirectory(prefix='forte-cpu-processor-') as directory:
        temp=Path(directory)
        pixels=bytes(channel for y in range(160) for x in range(320)
                     for channel in ((x*7+y*3)%256,(x*5+y*11)%256,(x*13+y*17)%256))
        Image.frombytes('RGB',(320,160),pixels).save(temp/'synthetic.png')
        collator=PublicCropCollator(processor,temp,config,max_length=256,max_pixels=65536)
        for side in (0,64,32,16):
            row={'_export_path':'synthetic.png','crop_xyxy':[0,0,320,160],
                 'question':QUESTION,'answer':'id_card','_short_side':side}
            batch=collator([row])
            assert all(value.device.type=='cpu' for value in batch.values())
            assert batch['input_ids'].shape[0]==1 and batch['input_ids'].shape[1]<=256
            labels=batch['labels']; completion=labels[labels!=-100]
            assert completion.numel()>0
            assert not any(bool((completion==config[field]).any()) for field in
                           ('image_token_id','video_token_id','vision_start_token_id','vision_end_token_id'))
            cases.append({'short_side':side,'input_ids_shape':list(batch['input_ids'].shape),
                          'pixel_values_shape':list(batch['pixel_values'].shape),
                          'image_grid_thw':batch['image_grid_thw'].tolist(),
                          'supervised_completion_tokens':int(completion.numel()),
                          'visual_tokens':int((batch['input_ids']==config['image_token_id']).sum()),
                          'pixel_values_sha256':__import__('hashlib').sha256(batch['pixel_values'].numpy().tobytes()).hexdigest()})
    assert not torch.cuda.is_initialized() and not glob.glob('/dev/nvidia*')
    report={'schema_version':'forte.qwen2b.cpu-processor-smoke.v1','passed':True,
            'synthetic_pixels_only':True,'model_loaded':False,'cuda_initialized':False,
            'gpu_device_nodes':[],'processor_class':processor.__class__.__name__,
            'max_pixels':65536,'max_length':256,'cases':cases,
            'versions':{name:importlib.metadata.version(name) for name in ('torch','transformers','Pillow')}}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,sort_keys=True))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
