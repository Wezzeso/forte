import fs from 'node:fs/promises';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
const p=Presentation.create({slideSize:{width:1280,height:720}});p.slides.add();const pr=p.toProto();
pr.fonts=[{name:'UpMax',family:'swiss',embeddedFonts:[{type:1,contentType:'application/vnd.ms-fontobject',data:new Uint8Array(await fs.readFile('../assets/UpMax-regular.eot'))},{type:2,contentType:'application/vnd.ms-fontobject',data:new Uint8Array(await fs.readFile('../assets/UpMax-bold.eot'))}]}];
try{const p2=Presentation.load(pr);await(await PresentationFile.exportPptx(p2)).save('fonttest.pptx');}catch(e){console.log('ERROR',e.message)}


