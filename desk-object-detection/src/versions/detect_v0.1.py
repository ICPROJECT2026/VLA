# -*- coding: utf-8 -*-
"""detect_v0.1.py —— 二阶段方案（v0.1）

阶段 1：检测整图所有物体，输出 name + bbox（像素坐标）。
阶段 2：对每个物体裁剪局部图，打包一次调用判定类别。

口径（依据 garbage_handling.md）：
- 垃圾 = 价值低、可随意丢弃（与物理状态无关）；
- 公共物品 = 图书馆基准、书籍除外（暂用临时清单）；
- 检测求准（不漏也不错）、分类从宽（误判可接受）；
- 推理预算：以时间为基准，本方案共 2 轮（检测 1 + 分类 1）。
"""
import json
import os

import torch
from PIL import Image, ImageDraw, ImageFont
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, "6.jpeg")
RESULT_PATH = os.path.join(BASE_DIR, "detection_result_v0.1.json")
ANNOTATED_PATH = os.path.join(BASE_DIR, "6_annotated_v0.1.jpg")

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def build_detect_prompt(width, height):
    return f"""请检测这张图片中的所有物品（图片宽度 {width} 像素、高度 {height} 像素）。要求：
- 准确框出每一个物品，不要遗漏，也不要重复或多报无关物体；
- 对每个物品输出：name（物品名称）和 bbox（边界框，格式 [x1, y1, x2, y2]，(x1, y1) 是左上角，(x2, y2) 是右下角，像素坐标）。

只输出一个 JSON 数组，不要输出任何其他文字、解释或 Markdown 代码块。格式如下：
[
  {{"name": "物品名称", "bbox": [x1, y1, x2, y2]}}
]"""


CLASSIFY_PROMPT = """请按顺序判断上面每张图片中物品的类别，只能是以下三者之一：
- 公共物品：图书馆等公共场所供公众共同使用的物品（如键盘、鼠标、显示器、饮水机、打印机、桌椅等）
- 私人物品：个人私有的物品
- 垃圾：价值低、可以随意丢弃的物品（如矿泉水瓶、纸团、包装袋、纸巾等，即使完好也算垃圾）

只输出一个 JSON 数组，按图片顺序给出每个物品的类别，不要输出任何其他文字、解释或 Markdown 代码块。格式如下：
["公共物品", "私人物品", "垃圾"]
"""


def load_model():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def generate(model, processor, messages, max_new_tokens=1024):
    """执行一次推理，返回解码文本。"""
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def parse_json(text):
    """直接解析模型输出的 JSON，仅做字符串清理，不使用正则。"""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def to_pixel(bbox, width, height):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    scale = max(abs(x1), abs(y1), abs(x2), abs(y2))
    if scale <= 1.5:
        x1, y1, x2, y2 = x1 * width, y1 * height, x2 * width, y2 * height
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return (
        max(0.0, min(width, x1)), max(0.0, min(height, y1)),
        max(0.0, min(width, x2)), max(0.0, min(height, y2)),
    )


def crop_object(image, bbox, pad_ratio=0.1):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    px, py = int(w * pad_ratio), int(h * pad_ratio)
    left = max(0, x1 - px)
    top = max(0, y1 - py)
    right = min(image.width, x2 + px)
    bottom = min(image.height, y2 + py)
    return image.crop((left, top, right, bottom))


def normalize_category(cat):
    if "公共" in cat:
        return "公共物品"
    if "私人" in cat or "私有" in cat:
        return "私人物品"
    if "垃圾" in cat or "废" in cat:
        return "垃圾"
    return cat


def detect_objects(model, processor, image):
    """阶段 1：检测所有物体。"""
    prompt = build_detect_prompt(image.width, image.height)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": IMAGE_PATH},
        {"type": "text", "text": prompt},
    ]}]
    out = generate(model, processor, messages)
    data = parse_json(out)
    if isinstance(data, list):
        raw = data
    elif isinstance(data, dict):
        raw = data.get("objects", data.get("detections", []))
    else:
        raw = []
    objects = []
    for o in raw:
        name = o.get("name", "")
        bbox = o.get("bbox")
        if not name or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = to_pixel(bbox, image.width, image.height)
        objects.append({"name": name, "bbox": [round(x1), round(y1), round(x2), round(y2)]})
    return objects, out


def classify_objects(model, processor, image, objects):
    """阶段 2：裁剪每个物体，打包一次调用分类，返回类别列表。"""
    content = []
    for o in objects:
        content.append({"type": "image", "image": crop_object(image, o["bbox"])})
    content.append({"type": "text", "text": CLASSIFY_PROMPT})
    messages = [{"role": "user", "content": content}]
    out = generate(model, processor, messages, max_new_tokens=512)
    data = parse_json(out)
    cats = [str(c) for c in data] if isinstance(data, list) else []
    cats = [normalize_category(c) for c in cats]
    cats = (cats + ["未知"] * len(objects))[:len(objects)]
    return cats, out


def load_font(size=28):
    for p in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/simsun.ttc"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_boxes(objects, out_path):
    image = Image.open(IMAGE_PATH).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = load_font(28)
    colors = {"公共物品": (34, 139, 34), "私人物品": (30, 120, 255), "垃圾": (230, 57, 70)}
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        color = colors.get(obj["category"], (255, 200, 0))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        label = f"{obj['name']}: {obj['category']}"
        tb = draw.textbbox((x1, y1), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        text_y = y1 - th - 6 if y1 - th - 6 > 0 else y1 + 4
        draw.rectangle([x1, text_y, x1 + tw + 10, text_y + th + 6], fill=color)
        draw.text((x1 + 5, text_y + 3), label, fill=(255, 255, 255), font=font)
    image.save(out_path)


def main():
    print("加载模型中...", flush=True)
    model, processor = load_model()

    with Image.open(IMAGE_PATH) as img:
        image = img.convert("RGB")
        width, height = image.size

    print("阶段 1：检测所有物体（1 轮）...", flush=True)
    objects, det_out = detect_objects(model, processor, image)
    print(f"检测到 {len(objects)} 个物体：{[o['name'] for o in objects]}", flush=True)
    print("检测原始输出：\n" + det_out, flush=True)

    if not objects:
        print("未检测到物体，结束。", flush=True)
        return

    print("阶段 2：逐物体裁剪分类（1 轮，多图打包）...", flush=True)
    cats, cls_out = classify_objects(model, processor, image, objects)
    print("分类原始输出：\n" + cls_out, flush=True)
    for o, c in zip(objects, cats):
        o["category"] = c

    result_objects = []
    for o in objects:
        x1, y1, x2, y2 = o["bbox"]
        result_objects.append({
            "name": o["name"],
            "category": o["category"],
            "bbox": [x1, y1, x2, y2],
            "bbox_normalized": [
                round(x1 / width, 4), round(y1 / height, 4),
                round(x2 / width, 4), round(y2 / height, 4),
            ],
        })

    result = {"image": "6.jpeg", "version": "0.1", "width": width, "height": height, "objects": result_objects}
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    draw_boxes(result_objects, ANNOTATED_PATH)

    print("总轮数：2（检测 1 + 分类 1）", flush=True)
    print(f"结果已保存: {RESULT_PATH}", flush=True)
    print(f"标注图已保存: {ANNOTATED_PATH}", flush=True)


if __name__ == "__main__":
    main()