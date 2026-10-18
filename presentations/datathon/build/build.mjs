import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
import {GlobalFonts} from '@napi-rs/canvas';
import {createRequire} from 'node:module';
const require=createRequire(import.meta.resolve('@oai/artifact-tool'));
const {FontLibrary}=require('skia-canvas');
const ROOT='C:/Users/Wezeso/Desktop/Forte';
const BASE=ROOT+'/presentations/datathon';
const SKILL='C:/Users/Wezeso/.codex/plugins/cache/openai-primary-runtime/presentations/26.1011.51657/skills/presentations';
for (const st of ['regular','medium','bold']) GlobalFonts.registerFromPath(BASE+'/assets/UpMax-'+st+'.ttf','UpMax');
const {applyPresentationChartFont}=await import(pathToFileURL(SKILL+'/container_tools/artifact_tool_utils.mjs'));
FontLibrary.use('UpMax', ['regular','medium','bold'].map(st=>BASE+'/assets/UpMax-'+st+'.ttf'));
const p=Presentation.create({slideSize:{width:1280,height:720}});
p.theme.defaultFont='UpMax';
p.theme.colorScheme={name:'FORTE',themeColors:{accent1:'#A31551',accent2:'#861044',accent3:'#191919',accent4:'#C8C8C8',accent5:'#E8D1DB',accent6:'#6F6F6F',bg1:'#F2F2F2',bg2:'#A31551',tx1:'#191919',tx2:'#F2F2F2',dk1:'#191919',dk2:'#861044',lt1:'#FFFFFF',lt2:'#F2F2F2',hlink:'#A31551',folHlink:'#861044'}};
const C={m:'#A31551',d:'#861044',w:'#F2F2F2',b:'#191919',gray:'#606060'};
const txt=(s,t,x,y,w,h,size=42,color=C.b,bold=false)=>{let a=s.shapes.add({geometry:'textbox',name:t.replaceAll('\n',' ').slice(0,70),position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});a.text=t;a.text.style={fontSize:size,typeface:'UpMax',bold,color,insets:0,lineSpacing:1.0,wrap:'none',autoFit:'none'};return a;};
const img=async(s,f,x,y,w,h,alt,extra={})=>{let src=f.endsWith('.svg')?{svg:await fs.readFile(f,'utf8')}:{blob:new Uint8Array(await fs.readFile(f)),contentType:f.endsWith('.webp')?'image/webp':'image/png'};return s.images.add({...src,alt,position:{left:x,top:y,width:w,height:h},fit:'contain',...extra});};
const zig=async(s,y=235,h=270)=>img(s,ROOT+'/web/public/zimran/zigzag-purple.svg',0,y,1280,h,'Original Zimran zigzag recolored in FORTE burgundy',{fit:'cover'});
const logo=async(s,x=64,y=40,w=180)=>img(s,ROOT+'/web/public/brand/forte-panopticon.svg',x,y,w,w*314/731,'Original FORTE Panopticon logo');
const note=(s,speech,source)=>s.speakerNotes.text=speech+'\n\nИСТОЧНИКИ И ГРАНИЦЫ\n'+source;
const slide=(bg=C.w)=>{let s=p.slides.add();s.background.fill=bg;return s;};
const title=(s,t,color=C.b,size=76)=>txt(s,t,64,40,1152,170,size,color,true);
const link=(s,label,url,x,y,w,size=30,color=C.w)=>{const a=txt(s,label,x,y,w,size*1.5,size,color);a.text.get(label).link={uri:url,isExternal:true};return a;};
const ruleSources='src/forte/fast/catalog.py и src/forte/fast/policy.py. Открытые источники: https://adilet.zan.kz/rus/docs/V1900019680 (НБРК №231), https://adilet.zan.kz/rus/docs/V2000020075 (НБРК №14), https://adilet.zan.kz/rus/docs/V2600038336 (АРРФР №53). Сопоставление выполнено командой и требует проверки применимости специалистом. Это не юридическое заключение.';
const metricsFile=ROOT+'/benchmarks/quality/accuracy-upgrade-candidate-comparison-20261018.json';
const bm=JSON.parse(await fs.readFile(metricsFile,'utf8'));const verified=bm.stages.find(x=>x.stage==='candidate-verified');
const loadFile=ROOT+'/benchmarks/quality/final-public-six-load-20261018T005948Z.json';
const load=JSON.parse(await fs.readFile(loadFile,'utf8'));
// 1 — cover
{
const s=slide(C.m);await zig(s,215,360);await logo(s,64,42,225);
txt(s,'Panopticon',58,240,1160,180,152,C.w,true);
txt(s,'Видеоаналитика кассовых операций',64,445,1110,60,43,C.w);
txt(s,'Datathon 2026',64,626,600,42,29,C.w);
txt(s,'WEZESO',1010,626,200,42,29,C.w,true);
note(s,'Мы сделали FORTE Panopticon для визуальной проверки кассовых операций. Пользователь загружает видео и получает события с исходными кадрами, временем и привязкой к правилу. Проверяющий видит, на чём основан результат. За пять минут покажем продукт, устройство анализа и измеренные результаты.', 'Кейс_ForteBank.docx.pdf, разделы 1–8 и FAQ 19. Название команды: по запросу пользователя WEZESO. Оригинальный логотип: logo/Group 1.svg. Дизайн и шрифт UpMax: https://www.zimran.io/ и web/public/zimran/SOURCES.md. Палитра пользователя #A31551.');
}
// 2 — problem + illustration
{
const s=slide();await img(s,BASE+'/assets/cashier-illustration.png',240,30,1040,690,'Conceptual illustration of a cashier workstation. AI-generated, not dataset footage.',{fit:'cover'});
txt(s,'Касса\nпод контролем',64,65,800,250,90,C.m,true);
txt(s,'Ручной просмотр\nкаждого видео',64,426,650,110,40,C.b);
txt(s,'Поиск конкретного\nмомента',64,562,650,100,40,C.b);
note(s,'В кейсе банка задача проста: ручной просмотр требует времени, а проверяющему нужен конкретный фрагмент. Полного внутреннего регламента и данных банковских систем у команды нет. Поэтому мы ищем только наблюдаемые признаки и показываем основание для дальнейшей проверки.', 'Кейс_ForteBank.docx.pdf, разделы 1, 3, 6, 7. Иллюстрация assets/cashier-illustration.png создана встроенным image_gen. Не является записью банка. Prompt: editorial isometric cashier workstation, security camera, desk, cash, laptop, safe; angular cut-paper style; #A31551/#861044/black/off-white; no people/text/logo.');
}
// 3 — actual product UI
{
const s=slide(C.b);title(s,'Один файл. Одна проверка',C.w,68);
await img(s,ROOT+'/dist/hero-release-20261018/production-hero.png',64,190,890,500,'Actual production FORTE Panopticon homepage screenshot');
txt(s,'Видео',1000,250,240,70,43,C.w,true);txt(s,'Анализ',1000,377,240,70,43,C.w,true);txt(s,'События',1000,504,240,70,43,C.w,true);
note(s,'Это реальный интерфейс продукта. Пользователь загружает MOV-файл, видит статус и получает результат без ручного запуска кода. API работает в два шага: загрузка возвращает job_id, а запрос статуса возвращает результат или ошибку. Ключ передаётся отдельно и не размещается в презентации.', 'Реальный снимок production UI: dist/hero-release-20261018/production-hero.png, https://hackathon.wezeso.me/. Требования: Кейс_ForteBank.docx.pdf, разделы 4, 10. Реализация web/src; docs/integration/acceptance.md. Снимок не является демонстрацией нового анализа.');
}
// 4 — evidence UI (explicitly synthetic)
{
const s=slide();title(s,'Результат можно проверить',C.m,70);
await img(s,ROOT+'/web/qa/screenshots/positive-multiple-desktop.png',64,210,800,410,'Actual result UI with explicitly synthetic fixture, cropped to source and events',{fit:'cover',crop:{left:.025,top:.075,right:.025,bottom:.64}});
txt(s,'Кадр',920,244,280,62,45,C.m,true);txt(s,'Таймкод',920,351,300,62,45,C.m,true);txt(s,'Правило',920,458,300,62,45,C.m,true);
txt(s,'Интерфейс: синтетический пример',64,651,1150,38,25,C.gray);
note(s,'В результате можно открыть несколько событий, перейти к исходной записи, посмотреть JPEG и точное время от начала видео. Справа находятся наблюдение, ссылка на источник и ограничения видимости. На слайде синтетическая проверка интерфейса. Её содержание не подтверждает точность модели. Частные банковские кадры остаются внутри разрешённого сервиса.', 'web/qa/screenshots/positive-multiple-desktop.png. Синтетический fixture review-synthetic.mov с TEST-1 и искусственным изображением. Не реальная кассовая операция, не результат accuracy-теста. docs/submission/pitch.md запрещает помещать частные кадры в публичные слайды.');
}
// 5 — architecture, editable analytical workflow
{
const s=slide(C.m);await zig(s,190,480);title(s,'Локальный анализ',C.w,88);
let rows=[['01','Кадры и сцена','Исходное время записи'],['02','Qwen3-VL','30B-A3B, vLLM, A100'],['03','Временные правила','Порядок действий'],['04','VLM-подтверждение','Исходный JPEG и результат']];
rows.forEach((r,i)=>{let y=228+i*105;txt(s,r[0],64,y,100,85,63,C.w,true);txt(s,r[1],210,y+5,640,70,47,C.w,true);txt(s,r[2],845,y+18,370,58,27,C.w);});
note(s,'Принятый путь использует локальный Qwen3-VL-30B-A3B-Instruct через vLLM на A100. Из видео берём кадры и области сцены, проверяем изменения во времени и повторно подтверждаем кандидатов через VLM. Durable queue отделяет загрузку от обработки. После загрузки весов обработка не требует внешнего AI API. Специализированные дообученные детекторы в live-путь не включены.', 'README.md; docs/submission/pitch.md; docs/integration/accuracy-upgrade-implementation-20261018.md; src/forte/fast/. Принятый профиль: require_corroboration=true, use_verify=true, specialists.enabled=false. Это схема работающего пути, не обещание дообученной системы.');
}
// 6 — rule linking
{
const s=slide();title(s,'Наблюдение + источник',C.m,82);
txt(s,'Кассовые операции',64,244,1000,75,53,C.b,true);txt(s,'НБРК №231',64,333,740,55,34,C.gray);
await img(s,ROOT+'/web/public/zimran/extra-curiosity.svg',984,256,197,324,'Original Zimran curiosity illustration');
txt(s,'Доступ в кассовый узел',64,450,1120,75,53,C.b,true);txt(s,'НБРК №14',64,539,740,55,34,C.gray);
txt(s,'Применимость правила проверяет специалист',64,648,1152,40,28,C.m);
note(s,'Каждое событие содержит наблюдаемый признак и ссылку на открытый источник. Для кассовых процедур каталог использует НБРК №231, для устройства и доступа в кассовый узел НБРК №14. Наличие признака не доказывает полномочия человека, завершение операции или нарушение нормы. Система не получает АБИС и внутренние журналы банка.',ruleSources+' Кейс_ForteBank.docx.pdf, разделы 6–7. Примерная норма по рабочей станции не доказывает обязанность блокировать экран при каждом кратком уходе.');
}
// 7 — native comparison chart
{
const s=slide();title(s,'34 из 40',C.m,106);
txt(s,'верных вердикта',64,191,900,62,43,C.b);
const chart=s.charts.add('bar',{position:{left:70,top:304,width:1110,height:210},categories:['Базовый профиль','С подтверждением'],series:[{name:'Верных вердиктов',values:[33,34],fill:C.m}],hasLegend:false,barOptions:{direction:'bar',grouping:'clustered',gapWidth:65},chartFill:'none',chartLine:{fill:'none',width:0},plotAreaFill:'none',plotAreaLine:{fill:'none',width:0},xAxis:{min:0,max:40,majorUnit:10,visible:true,textStyle:{fontSize:24,fill:C.gray},majorGridlines:{fill:'#D2D2D2',width:1},line:{fill:'none',width:0}},yAxis:{min:0,max:40,majorUnit:10,visible:true,textStyle:{fontSize:28,fill:C.b},line:{fill:'none',width:0},majorGridlines:null},dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:34,fill:C.m,bold:true}}});applyPresentationChartFont(chart,{fontFamily:'UpMax'});
txt(s,'0 ложных тревог',64,544,570,65,43,C.m,true);txt(s,'6 пропусков',694,544,500,65,43,C.b,true);
txt(s,'Те же 40 development-видео. Не независимый тест.',64,652,1152,38,27,C.gray);
note(s,'Свежий базовый прогон дал 33 верных ответа из 40. Профиль с подтверждением дал 34, убрал одну ложную тревогу и сохранил шесть пропусков. Ошибок обработки не было. Это прямой прогон анализатора на тех же предоставленных development-видео. Мы не выдаём его за оценку на новых роликах или новый прогон публичного API на 40 видео.', 'benchmarks/quality/accuracy-upgrade-baseline-20261018.json; '+metricsFile+'. candidate-verified: TP=15, TN=19, FP=0, FN=6, n=40, accuracy=.85, processed=40, failed=0. Baseline: correct=33, FP=1, FN=6. No heldout or cross-validation claim. Precision=15/15, recall=15/21; confidence intervals not measured.');
}
// 8 — localization
{
const s=slide(C.m);await zig(s,330,335);title(s,'Точность времени',C.w,85);
txt(s,'1,5 с',64,226,700,190,170,C.w,true);txt(s,'медианная ошибка',70,440,690,55,41,C.w);
txt(s,'11 / 15',854,261,370,100,76,C.w,true);txt(s,'в пределах 2 с',854,377,360,94,34,C.w);
txt(s,'Было 1,0 с и 12 / 15. Локализация ухудшилась.',64,640,1140,50,30,C.w);
note(s,'Точность вердикта и точность времени нужно разделять. На 15 найденных положительных видео медианная ошибка ближайшего события составляет полторы секунды, 11 попаданий укладываются в две секунды. У базового профиля было лучше: одна секунда и 12 попаданий. Улучшение бинарного вердикта не означает улучшение локализации. В разметке нет типов, поэтому точность классификации по типам мы не заявляем.',metricsFile+'. timecode_scored=15, timecode_err_median=1.5, timecode_within_2s=11. Baseline median=1.0, within2s=12. Time-labeled videos=21; missed videos не включены в эти 15 timing-наблюдений. Ближайшее событие, без сопоставления типов.');
}
// 9 — real six upload chart
{
const s=slide();title(s,'6 загрузок. 6 результатов',C.m,70);
const vals=load.report.results.map(r=>Number(r.elapsed_seconds.toFixed(1)));
const chart=s.charts.add('bar',{position:{left:64,top:218,width:810,height:361},categories:['1','2','3','4','5','6'],series:[{name:'Секунды',values:vals,fill:C.m}],hasLegend:false,barOptions:{direction:'column',grouping:'clustered',gapWidth:90},chartFill:'none',chartLine:{fill:'none',width:0},plotAreaFill:'none',plotAreaLine:{fill:'none',width:0},xAxis:{textStyle:{fontSize:24,fill:C.gray},line:{fill:'none',width:0},majorGridlines:null},yAxis:{min:0,max:180,majorUnit:60,textStyle:{fontSize:24,fill:C.gray},majorGridlines:{fill:'#D2D2D2',width:1},line:{fill:'none',width:0}},dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.0',textStyle:{fontSize:26,fill:C.b}}});applyPresentationChartFont(chart,{fontFamily:'UpMax'});
txt(s,'142,6 с',920,227,310,90,61,C.m,true);txt(s,'максимум',920,325,290,45,31,C.gray);txt(s,'92,1 с',920,431,310,90,61,C.b,true);txt(s,'медиана',920,529,290,45,31,C.gray);
txt(s,'Тёплый public HTTPS с GPU-хоста. Цель 30 с пока не достигнута.',64,637,1152,65,27,C.gray);
note(s,'Отдельно мы проверили полный публичный API: шесть параллельных загрузок, шесть успешных первых попыток, без повторов и ошибок. Максимум 142,568 секунды, медиана 92,076. Проверка подтвердила схему ответа, JPEG и временную шкалу. Клиент работал на GPU-хосте через публичный HTTPS. Это тёплый тест и не измерение загрузки с ноутбука. Цель 30 секунд пока не достигнута.',loadFile+'. 18.10.2026, attempt 20261018T005948Z. parallel=6, passed=6, failed=0, repeated_attempts=0, timeline_validated_jobs=6. Latency seconds: '+load.report.results.map(r=>r.elapsed_seconds).join(', ')+'. Input durations: '+load.selected_source_durations_s.join(', ')+'. Source preparation 19.261137s excluded. No new accuracy claim. Deadline in case 300s; recorded test deadline240s.');
}
// 10 — safe operation
{
const s=slide(C.b);title(s,'Решение за человеком',C.w,88);
txt(s,'Модель предлагает',64,230,1140,100,69,C.w,true);txt(s,'Проверяющий подтверждает',64,365,1140,100,69,C.w,true);
txt(s,'Ошибка обработки остаётся ошибкой',64,565,1140,65,37,'#E7AFCA');
note(s,'Модель формирует признаки для проверки, а человек может записать отдельное операционное решение. Исходный машинный ответ при этом сохраняется. Техническая ошибка не превращается в вердикт об отсутствии нарушения. Звук и идентификация лиц не используются. Мы не выводим факты о проводках или системной сверке, которые невозможно увидеть на записи.', 'Кейс_ForteBank.docx.pdf, раздел 7 и FAQ 28–29. docs/submission/pitch.md, README.md. Реализация раздельного machine/human review в web/src и API. Человеческая проверка опциональна для оператора, не обязательна для автоматического API жюри.');
}
// 11 — next steps
{
const s=slide();title(s,'Следующая проверка',C.m,86);
const rr=[['Новые ракурсы','Независимый набор видео'],['Пропуски событий','Сейчас 6 из 21 положительных'],['Скорость','Цель 30 секунд']];
rr.forEach((r,i)=>{let y=226+i*137;txt(s,r[0],64,y,720,70,51,C.b,true);txt(s,r[1],745,y+14,485,75,31,C.m);});
note(s,'До применения в контуре банка нужны независимые новые видео, другие ракурсы и разметка типов событий. Главный риск сегодня — шесть пропусков из 21 положительного ролика. Следующий этап — снизить их число, перепроверить таймкоды и отдельно измерить холодную обработку и загрузку с обычного клиентского компьютера. План ускорения пока не является полученным результатом.',metricsFile+'; '+loadFile+'; docs/integration/accuracy-upgrade-implementation-20261018.md. Следующие шаги — предложения команды, без обещанных сроков и ненаблюдавшихся результатов. Public-trained specialists находятся в staged disabled, цель95% не выполнена.');
}
// 12 — team cutouts, unique images
{
const s=slide(C.m);await zig(s,213,430);title(s,'Команда WEZESO',C.w,88);
await img(s,ROOT+'/assets/team/hero-hoodie.png',175,228,415,415,'WEZESO team portrait in original white angular stencil');
await img(s,ROOT+'/assets/team/hero-blazer.png',685,228,415,415,'WEZESO team portrait in original white angular stencil');
note(s,'Мы команда WEZESO. Для демонстрации подготовили интерфейс, локальный анализ, API и проверяемые артефакты оценки. Код доступен по ссылке на следующем слайде. Имена и роли не добавлены без подтверждённых данных.', 'Предоставленные пользователем фотографии, обработанные ранее по его запросу. assets/team/hero-hoodie.png, assets/team/hero-blazer.png. Промпты и происхождение assets/team/SOURCES.md. Это отредактированные студийные cutout-портреты, не свидетельство работы модели.');
}
// 13 — closing with intentionally half visible oversized wordmark
{
const s=slide('#000000');await logo(s,64,43,200);
txt(s,'Демо',60,187,1100,170,143,C.w,true);
link(s,'hackathon.wezeso.me','https://hackathon.wezeso.me/',64,390,1000,43);
link(s,'github.com/Wezzeso/forte','https://github.com/Wezzeso/forte',64,462,1000,31,'#D0D0D0');
await img(s,ROOT+'/web/public/brand/wezeso-wordmark.png',0,574,1280,146,'Oversized WEZESO wordmark; intentionally shows the upper half only',{fit:'cover',crop:{left:.02,top:.275,right:.02,bottom:.50}});
note(s,'Откроем разрешённую демонстрацию в продукте. Для жюри используем публичный API и отдельно переданный ключ. Если демонстрационная GPU-среда недоступна, покажем зафиксированные результаты и воспроизводимые инструкции, сохраняя различие между ранее выполненными проверками и новым запуском.', 'Публичный интерфейс https://hackathon.wezeso.me/. Репозиторий предоставлен пользователем: https://github.com/Wezzeso/forte. Актуальная доступность GPU ограничена арендой, слайд не обещает постоянный uptime. Wordmark: web/public/brand/wezeso-wordmark.png, по прямому требованию пользователя oversized и half-visible. Нижняя часть намеренно скрыта через crop внутри слайда.');
}
// 14 — appendix taxonomy
{
const s=slide();title(s,'Каталог событий',C.m,78);
txt(s,'Активный профиль',64,190,550,58,37,C.b,true);
const active=['Документы инкассатора','Пересчёт без клиента','Предыдущая наличность','Ценности без присмотра','Рабочая станция','Открытая дверь','Постороннее лицо','Личные вещи'];
active.forEach((t,i)=>txt(s,t,64+(i>=4?550:0),270+(i%4)*78,535,58,33,C.b));
txt(s,'Ещё 6 типов отключены. Точность по типам не измерена.',64,635,1152,65,28,C.m);
note(s,'Резервный слайд для вопросов о классах. Из 14 категорий каталога шесть отключены. Активные категории перечислены как реализованные кандидаты, а не как проверенные по отдельности классы. Для рабочей станции видимое содержимое экрана — только признак для проверки. Чья это наличность и полномочия присутствующего человека не всегда можно установить визуально.',ruleSources+' Disabled: NO_CLIENT_ID, NO_AUTHENTICITY_CHECK, NO_RECOUNT, KEY_LEFT_IN_SAFE, LEFT_DURING_SERVICE, CASH_HANDOVER_IRREGULAR. Active: COLLECTOR_NO_DOCUMENTS, COUNT_WITHOUT_CLIENT, PREVIOUS_CASH_ON_DESK, CASH_LEFT_UNATTENDED, WORKSTATION_UNLOCKED, CASH_ROOM_DOOR_OPEN, UNAUTHORIZED_PERSON, PERSONAL_ITEMS. Labels lack types, event classification accuracy unmeasured.');
}
// 15 — appendix methods
{
const s=slide();title(s,'Что именно измерили',C.m,75);
txt(s,'40 видео',64,220,500,88,66,C.b,true);txt(s,'Вердикт и время события\nРазработка на том же наборе',64,325,600,110,32,C.gray);
txt(s,'6 загрузок',735,220,500,88,66,C.b,true);txt(s,'Публичный API и схема\nТёплый запуск с GPU-хоста',735,325,500,110,32,C.gray);
txt(s,'Новые ракурсы и типы событий\nещё требуют проверки',64,544,1150,100,43,C.m,true);
note(s,'Резервный слайд для методологии. Dataset: n40, positives21, negatives19. Confusion: TP15, TN19, FP0, FN6. Accuracy34/40=85%, precision15/15=100% на этом наборе, recall15/21=71,4%. Это development, не unseen accuracy. Localization median1.5s на15 detected positive clips. Публичные шесть загрузок проверяют работоспособность и контракт, а не accuracy. Исследовательские RF-DETR и crop-LoRA не активны и не объясняют live-результат.', '1. Кейс_ForteBank.docx.pdf\n2. benchmarks/quality/accuracy-upgrade-baseline-20261018.json\n3. benchmarks/quality/accuracy-upgrade-candidate-comparison-20261018.json\n4. benchmarks/quality/final-public-six-load-20261018T005948Z.json\n5. benchmarks/quality/accuracy-upgrade-rollout-20261018.json\n6. docs/integration/accuracy-upgrade-implementation-20261018.md\n7. docs/integration/public-training-results-20261018.md\n8. docs/submission/pitch.md\n9. src/forte/fast/catalog.py\n10. web/public/zimran/SOURCES.md\n11. assets/team/SOURCES.md\nБенчмарки 18 октября 2026. Частное видео и API-ключи в дек не включены.');
}
await fs.mkdir(BASE+'/build/renders',{recursive:true});
for(let i=0;i<p.slides.items.length;i++){
 const slide=p.slides.getItem(i); const png=await p.export({slide,format:'png',scale:1});
 await fs.writeFile(BASE+'/build/renders/slide-'+String(i+1).padStart(2,'0')+'.png',new Uint8Array(await png.arrayBuffer()));
 const lb=await slide.export({format:'layout'});await fs.writeFile(BASE+'/build/renders/slide-'+String(i+1).padStart(2,'0')+'.json',new Uint8Array(await lb.arrayBuffer()));
 console.log('rendered',i+1);
}
const fontProto=p.toProto();
fontProto.fonts=[{name:'UpMax',family:'swiss',embeddedFonts:[{type:1,contentType:'font/ttf',data:new Uint8Array(await fs.readFile(BASE+'/assets/UpMax-regular.ttf'))},{type:2,contentType:'font/ttf',data:new Uint8Array(await fs.readFile(BASE+'/assets/UpMax-bold.ttf'))}]}];
const embeddedDeck=Presentation.load(fontProto);
await (await PresentationFile.exportPptx(embeddedDeck)).save(BASE+'/build/candidate.pptx');
try {const pdf=await p.export({format:'pdf'});await fs.writeFile(BASE+'/output/FORTE-Panopticon-Datathon-2026.pdf',new Uint8Array(await pdf.arrayBuffer()));console.log('PDF exported')}catch(e){console.log('PDF ERROR',e.message);}
await fs.writeFile(BASE+'/build/proto.json',JSON.stringify(p.toProto()));
console.log('COMPLETE',p.slides.items.length);



