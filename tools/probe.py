"""Ask the model focused questions about crops of specific frames (dev tool).

spec JSON: [{"video": "X-23", "t": 15, "box": [x1,y1,x2,y2] | "laptop" | null, "q": "...", "expand": 1.6}]
"""
import asyncio, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from forte.fast.analyzer import crop, locate_layout  # noqa
from forte.fast.video import decode  # noqa
from forte.fast.vlm import VLMClient, image_part, text_part  # noqa


async def main():
    spec = json.loads(Path(sys.argv[1]).read_text())
    base, model, vdir = sys.argv[2], sys.argv[3], sys.argv[4]
    client = VLMClient(base, model, concurrency=32)
    cache = {}
    layouts = {}

    async def get(name):
        if name not in cache:
            cache[name] = await asyncio.to_thread(decode, f"{vdir}/{name}.mov", 2.0)
            layouts[name] = await locate_layout(client, cache[name])
        return cache[name], layouts[name]

    for item in spec:
        await get(item["video"])

    async def one(item):
        video, layout = await get(item["video"])
        content = []
        if "ts" in item:
            box = item.get("box")
            if isinstance(box, str):
                box = layout.get(box)
            for t in item["ts"]:
                f = video.nearest(t)
                content.append(text_part(f"t={f.t:.1f}s"))
                content.append(image_part(crop(f.image, box, item.get("expand", 1.6), item.get("min_side", 448)) if box else f.image, max_side=768))
            content.append(text_part(item["q"]))
            try:
                ans = await client.chat(content, max_tokens=item.get("max_tokens", 160))
            except Exception as exc:
                ans = repr(exc)
            return f"{item['video']}@{item['ts'][0]}-{item['ts'][-1]} [{item.get('tag','')}] -> {ans.strip()[:400]}"
        frame = video.nearest(item["t"])
        box = item.get("box")
        if isinstance(box, str):
            box = layout.get(box)
        if item.get("full", False) or box is None:
            content.append(image_part(frame.image, max_side=1024))
        if box is not None:
            content.append(image_part(crop(frame.image, box, item.get("expand", 1.6), item.get("min_side", 448)), max_side=768))
        content.append(text_part(item["q"]))
        try:
            ans = await client.chat(content, max_tokens=item.get("max_tokens", 120))
        except Exception as exc:
            ans = repr(exc)
        return f"{item['video']}@{item['t']} [{item.get('tag','')}] -> {ans.strip()[:300]}"

    for line in await asyncio.gather(*(one(i) for i in spec)):
        print(line, flush=True)
    for k, v in layouts.items():
        print(k, {a: [int(x) for x in b] for a, b in v.items()})
    await client.aclose()

asyncio.run(main())
