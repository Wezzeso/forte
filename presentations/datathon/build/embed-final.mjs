import fs from 'node:fs/promises';
import {PresentationFile,FileBlob,Presentation} from '@oai/artifact-tool';
const p=await PresentationFile.importPptx(await FileBlob.load('candidate.pptx'));
const pr=p.toProto();pr.fonts=[{name:'UpMax',family:'swiss',embeddedFonts:[{type:1,contentType:'application/vnd.ms-fontobject',data:new Uint8Array(await fs.readFile('../assets/UpMax-regular.eot'))},{type:2,contentType:'application/vnd.ms-fontobject',data:new Uint8Array(await fs.readFile('../assets/UpMax-bold.eot'))}]}];
const q=Presentation.load(pr);await(await PresentationFile.exportPptx(q)).save('candidate-eot.pptx');console.log('EOT font package created');
