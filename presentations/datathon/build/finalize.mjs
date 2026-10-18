import fs from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
const SKILL='C:/Users/Wezeso/.codex/plugins/cache/openai-primary-runtime/presentations/26.1011.51657/skills/presentations';
const {finalizePresentation}=await import(pathToFileURL(SKILL+'/container_tools/artifact_tool_utils.mjs'));
const BASE='C:/Users/Wezeso/Desktop/Forte/presentations/datathon';
const r=await finalizePresentation({workspaceDir:'C:/Users/Wezeso/Desktop/Forte',candidatePath:BASE+'/build/candidate-eot.pptx',finalPath:BASE+'/output/FORTE-Panopticon-Datathon.pptx',pythonExecutable:'C:/Users/Wezeso/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:SKILL+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:SKILL+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],explicitTotalSlideCount:15,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[7,9],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:['UpMax']},verifyArtifactToolImport:true,receiptPath:BASE+'/build/validation-final.json'});
console.log(JSON.stringify(r));

