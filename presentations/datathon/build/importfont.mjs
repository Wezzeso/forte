import {PresentationFile,FileBlob} from '@oai/artifact-tool';
try{const p=await PresentationFile.importPptx(await FileBlob.load('fonttest.pptx')); console.log('embedded fonts',p.toProto().fonts.map(f=>({name:f.name,faces:f.embeddedFonts.map(e=>({type:e.type,contentType:e.contentType,len:e.data.length}))})));}catch(e){console.log('ERROR',e.message)}
