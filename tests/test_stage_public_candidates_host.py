"""Platform-boundary contracts only; no staging, datasets, SSH, or GPU calls."""
from pathlib import Path
import runpy


def test_remote_input_template_is_posix_on_windows_and_linux():
    script = Path(__file__).resolve().parents[1] / 'tools/stage_public_candidates_host.py'
    module = runpy.run_path(str(script), run_name='candidate_stage_contracts')
    template = module['template']()
    pins = [template['cohort']]
    pins.extend(pin for rows in template['members'].values() for pin in rows.values())
    assert len(pins) == 11
    assert all(pin['path'].startswith('/data/claude/training/public-20261018/') for pin in pins)
    assert all('\\' not in pin['path'] for pin in pins)
    assert template['cohort']['sha256'] == '0ab06fd0e9c17c259b8464fc43d7466f8b55cc2e2bfd8182df5ab73942a0a437'
