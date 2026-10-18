import {Presentation} from '@oai/artifact-tool';
const p=Presentation.create({slideSize:{width:1280,height:720}});
console.log(p.help('font', {include:'index',maxChars:3500}));
console.log('snapshot',Object.keys(p.toSnapshot()));
console.log('theme',JSON.stringify(p.toSnapshot().theme??{}).slice(0,1600));
