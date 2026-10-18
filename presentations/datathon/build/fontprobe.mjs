import {createRequire} from 'node:module';
import {GlobalFonts} from '@napi-rs/canvas';
const req=createRequire(import.meta.resolve('@oai/artifact-tool')); const {FontLibrary}=req('skia-canvas');
console.log('before',FontLibrary.families.filter(x=>x.toLowerCase().includes('upmax')));
console.log('use',FontLibrary.use('UpMax',['../assets/UpMax-regular.ttf','../assets/UpMax-bold.ttf']));
console.log('after',FontLibrary.families.filter(x=>x.toLowerCase().includes('upmax')));
console.log('global',GlobalFonts.families.filter(f=>f.family.toLowerCase().includes('upmax')));
