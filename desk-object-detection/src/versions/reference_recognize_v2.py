"""用本地千问识别图片：python recognize.py /path/to/5.jpeg"""
import argparse
import json
from pathlib import Path

PROMPT = """请检查整张图片，逐个识别清晰可见的物体，尤其留意地面、角落、桌面上的垃圾。
分类规则：
1. 垃圾：明显被丢弃的包装袋、纸屑、烟头、食物残渣、空饮料容器等。
   判断必须结合摆放位置和状态；不能仅因为物体是瓶子或纸张就判为垃圾。
2. 公共设施：垃圾桶、长椅、路灯等供公众使用的设施。垃圾桶本身不是垃圾。
3. 个人物品：仍在使用或有序存放的包、水杯、手机等。
4. 不确定：无法从图片确定是否被丢弃，或物体看不清。不要臆造物体。
必须识别所有清晰可见的独立物体，同时列出垃圾和非垃圾，不能只输出一个物体。逐行扫描全图，检查是否遗漏鼠标、键盘、手机、纸团等实际可见物体；不存在的不要输出。键盘作为整体。揉皱的废纸应考虑为垃圾，证据不足则标为不确定。每个物体分别输出一个对象。只输出一个合法 JSON 对象，不要 Markdown。
格式：{"objects":[{"name":"物体名称","category":"垃圾/公共设施/个人物品/不确定中的一个",
"reason":"结合可见证据的简短理由","bbox":[左,上,右,下]}]}。
bbox 必须为相对于整张图片宽高归一化到 0 至 1000 的整数坐标。
没有可识别物体时 objects 为 []。
"""


def parse_result(text, image_size=None):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    result = json.loads(text)
    if isinstance(result, list):
        result = {"objects": result}
    if not isinstance(result, dict) or not isinstance(result.get("objects"), list):
        raise ValueError("缺少 objects 列表")
    for obj in result["objects"]:
        if not isinstance(obj, dict):
            raise ValueError("物体必须为 JSON 对象")
        if obj.get("category") not in {"垃圾", "公共设施", "个人物品", "不确定"}:
            raise ValueError("分类不符合约定")
        box = obj.get("bbox")
        if image_size is not None:
            width, height = image_size
            if (not isinstance(box, list) or len(box) != 4
                    or any(type(v) not in (int, float) for v in box)):
                raise ValueError("像素 bbox 必须包含四个数字")
            if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
                raise ValueError(f"像素 bbox {box} 超出图片 {width}×{height} 或顺序错误")
            box = [round(box[0] / width * 1000), round(box[1] / height * 1000),
                   round(box[2] / width * 1000), round(box[3] / height * 1000)]
            obj["bbox"] = box
        if not isinstance(box, list) or len(box) != 4 or any(type(v) is not int or not 0 <= v <= 1000 for v in box):
            raise ValueError("bbox 必须包含四个 0 至 1000 的整数")
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError("bbox 左上坐标必须小于右下坐标")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct", help="模型名称或本地模型目录")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--bbox-format", choices=["pixels", "normalized"], default="pixels", help="模型输出的坐标格式，默认 pixels；保存的 JSON 始终使用 0～1000")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"找不到图片：{args.image.resolve()}")
    import torch
    from PIL import Image, ImageDraw, ImageOps
    from transformers import AutoModelForImageTextToText, AutoProcessor

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image = ImageOps.exif_transpose(Image.open(args.image)).convert("RGB")
    model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype="auto", device_map="auto")
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model)
    prompt = PROMPT
    if args.bbox_format == "pixels":
        prompt = prompt.replace("bbox 必须为相对于整张图片宽高归一化到 0 至 1000 的整数坐标。",
                                f"图片宽 {image.width} 像素，高 {image.height} 像素。bbox 使用原始图片像素坐标，横坐标范围 0 至 {image.width}，纵坐标范围 0 至 {image.height}，不要归一化。")
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, ids)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    prefix = args.output_dir / args.image.stem
    prefix.with_suffix(".raw.txt").write_text(raw, encoding="utf-8")
    try:
        result = parse_result(raw, image.size if args.bbox_format == "pixels" else None)
        result["bbox_format"] = "normalized_0_1000"
        result["image_size"] = list(image.size)
    except (ValueError, TypeError, AttributeError) as exc:
        raise SystemExit(f"模型输出格式无效：{exc}。原始输出已保存至 {prefix.with_suffix('.raw.txt')}")
    prefix.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    draw = ImageDraw.Draw(image)
    colors = {"垃圾": "red", "公共设施": "blue", "个人物品": "green", "不确定": "orange"}
    for i, obj in enumerate(result["objects"], 1):
        x1, y1, x2, y2 = obj["bbox"]
        box = (x1 * image.width / 1000, y1 * image.height / 1000, x2 * image.width / 1000, y2 * image.height / 1000)
        draw.rectangle(box, outline=colors[obj["category"]], width=3)
        draw.text((box[0], box[1]), str(i), fill=colors[obj["category"]])
    image.save(prefix.with_suffix(".annotated.jpg"))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"垃圾数量：{sum(o['category'] == '垃圾' for o in result['objects'])}；结果目录：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
