import fs from 'node:fs/promises';
import {createRequire} from 'node:module';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
const require=createRequire(import.meta.resolve('@oai/artifact-tool'));const {FontLibrary}=require('skia-canvas');
FontLibrary.use('UpMax',['regular','medium','bold'].map(s=>'../assets/UpMax-'+s+'.ttf'));
try {
 const p=await PresentationFile.importPptx(await FileBlob.load('../output/FORTE-Panopticon-Datathon.pptx'));
 await fs.mkdir('final-eot-renders',{recursive:true});
 for(let i=0;i<p.slides.items.length;i++){
  const png=await p.export({slide:p.slides.getItem(i),format:'png',scale:1});
  await fs.writeFile('final-eot-renders/slide-'+String(i+1).padStart(2,'0')+'.png',new Uint8Array(await png.arrayBuffer()));
 }
 console.log('Final PPTX imported and rendered',p.slides.items.length);
}catch(e){console.log('RENDER ERROR',e.message);process.exitCode=1;}

