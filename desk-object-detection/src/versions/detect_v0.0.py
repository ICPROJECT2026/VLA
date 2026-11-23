# -*- coding: utf-8 -*-
"""Qwen2.5-VL-3B 检测 6.jpeg，输出 JSON（name/bbox/category）并绘制 bbox 可视化。"""
import json
import os

import torch
from PIL import Image, ImageDraw, ImageFont
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, "6.jpeg")
RESULT_PATH = os.path.join(BASE_DIR, "detection_result.json")
ANNOTATED_PATH = os.path.join(BASE_DIR, "6_annotated.jpg")

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

PROMPT = """请检测这张图片中的所有物品。对每个物品输出以下字段：
- name：物品名称（中文）
- bbox：边界框，格式为 [x1, y1, x2, y2]，其中 (x1, y1) 是左上角，(x2, y2) 是右下角，坐标为像素坐标（以图片实际像素为单位，左上角为原点）
- category：类别，只能是“公共物品”、“私人物品”、“垃圾”三者之一

只输出一个 JSON 对象，不要输出任何其他文字、解释或 Markdown 代码块。格式如下：
{
  "objects": [
    {"name": "物品名称", "bbox": [x1, y1, x2, y2], "category": "类别"}
  ]
}"""


def load_model():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def run_detection(model, processor):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": IMAGE_PATH},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
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


def to_pixel_bbox(bbox, width, height):
    """将模型输出的 bbox 转为像素坐标 (x1, y1, x2, y2)。"""
    x1, y1, x2, y2 = [float(v) for v in bbox]
    scale = max(abs(x1), abs(y1), abs(x2), abs(y2))
    if scale <= 1.5:  # 兜底：0~1 归一化坐标
        x1, y1, x2, y2 = x1 * width, y1 * height, x2 * width, y2 * height
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return (
        max(0.0, min(width, x1)), max(0.0, min(height, y1)),
        max(0.0, min(width, x2)), max(0.0, min(height, y2)),
    )


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
    print("开始推理...", flush=True)
    output_text = run_detection(model, processor)
    print("模型原始输出：\n" + output_text, flush=True)

    data = parse_json(output_text)
    if isinstance(data, list):
        objects_raw = data
    elif isinstance(data, dict):
        objects_raw = data.get("objects", [])
    else:
        objects_raw = []

    with Image.open(IMAGE_PATH) as img:
        width, height = img.size

    objects = []
    for obj in objects_raw:
        name, category, bbox = obj.get("name", ""), obj.get("category", ""), obj.get("bbox")
        if not name or not bbox:
            continue
        px1, py1, px2, py2 = to_pixel_bbox(bbox, width, height)
        objects.append({
            "name": name,
            "category": category,
            "bbox": [round(px1), round(py1), round(px2), round(py2)],
            "bbox_normalized": [
                round(px1 / width, 4), round(py1 / height, 4),
                round(px2 / width, 4), round(py2 / height, 4),
            ],
        })

    result = {"image": "6.jpeg", "width": width, "height": height, "objects": objects}
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    draw_boxes(objects, ANNOTATED_PATH)
    print(f"检测到 {len(objects)} 个物品；结果: {RESULT_PATH}；标注图: {ANNOTATED_PATH}", flush=True)


if __name__ == "__main__":
    main()