import fs from 'node:fs/promises';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
import {GlobalFonts} from '@napi-rs/canvas';
console.log('register', GlobalFonts.registerFromPath('../assets/UpMax-bold.ttf','UpMax'));
const p=Presentation.create({slideSize:{width:1280,height:720}});
const s=p.slides.add();s.background.fill='#A31551';
const sh=s.shapes.add({geometry:'textbox',position:{left:64,top:100,width:1100,height:220},fill:'none',line:{fill:'none',width:0}});
sh.text='Panopticon';sh.text.style={fontSize:144,typeface:'UpMax',bold:true,color:'#F2F2F2',insets:0};
s.speakerNotes.text='Probe';
await (await PresentationFile.exportPptx(p)).save('probe.pptx');
try {let png=await p.export({slide:s,format:'png',scale:1});await fs.writeFile('probe.png',new Uint8Array(await png.arrayBuffer()));console.log('rendered')}catch(e){console.log('rendererr',e.stack?.substring(0,1800))}
