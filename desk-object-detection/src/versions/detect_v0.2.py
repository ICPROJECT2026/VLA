# -*- coding: utf-8 -*-
"""detect_v0.2.py —— v0.2：意义注入 + 结构化扫描

聚焦「识别物体（全检）」这一步。用「桌面清洁者」角色 + 分区域扫描 + 差异检测，
对抗「垃圾被忽略、不确定被省略、小/低对比目标被漏」的问题。

输出 JSON 列表：[{"label":"...","bbox":[x1,y1,x2,y2],"evidence":"..."}]
label 可为物体名称或类别；无法确定时填 "unknown"。
"""
import json
import os

import torch
from PIL import Image, ImageDraw, ImageFont
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, "6.jpeg")
RESULT_PATH = os.path.join(BASE_DIR, "detection_result_v0.2.json")
ANNOTATED_PATH = os.path.join(BASE_DIR, "6_annotated_v0.2.jpg")

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def build_prompt(width, height):
    return f"""你是一名桌面清洁者。现在你将看到一张桌子上的画面（图片宽度 {width} 像素、高度 {height} 像素），你的任务是找出桌面上所有需要处置的物品并决定去向：
- 公共物品 → 放归原位
- 私人物品 → 放到失物招领处
- 垃圾 → 扔进垃圾桶

因此任何物品都不能漏掉——包括看起来像垃圾、没有价值的东西，它们同样必须被找到并扔进垃圾桶。

请严格按以下步骤处理图像：

1. 将图像划分为上/中/下或 2×2 区域，逐区域扫描。

2. 在每个区域内，找出所有与局部背景在颜色、纹理、边缘、亮度上存在差异的候选区域，
   特别关注低对比度、小尺寸、被遮挡的目标；不要只看最明显、最大的几个物体。

3. 对每个候选区域给出边界框 bbox，并说明触发依据（颜色 / 纹理 / 边缘 / 亮度）。

4. 判断每个候选区域是否属于一个独立物体；属于同一物体的多个碎片，合并为一个实例。

5. 无法确定类别的，label 填 "unknown"，不要省略不确定项；某个区域存在但无法归类，也要输出 unknown。

以 JSON 列表输出所有实例，不要只输出最明显的几个。格式：
[{{"label":"...","bbox":[x1,y1,x2,y2],"evidence":"..."}}]"""


def load_model():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def generate(model, processor, messages, max_new_tokens=2048):
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
        start, end = text.find("["), text.rfind("]")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
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


def load_font(size=28):
    for p in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/simsun.ttc"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def label_color(label):
    if "公共" in label:
        return (34, 139, 34)
    if "私人" in label or "私有" in label:
        return (30, 120, 255)
    if "垃圾" in label or "废" in label:
        return (230, 57, 70)
    if "unknown" in label.lower() or "未知" in label:
        return (128, 128, 128)
    return (255, 160, 0)


def draw_boxes(objects, out_path):
    image = Image.open(IMAGE_PATH).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = load_font(24)
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        label = obj.get("label", "unknown")
        color = label_color(label)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
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

    prompt = build_prompt(width, height)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": IMAGE_PATH},
        {"type": "text", "text": prompt},
    ]}]

    print("推理中...", flush=True)
    output_text = generate(model, processor, messages)
    print("模型原始输出：\n" + output_text, flush=True)

    data = parse_json(output_text)
    if isinstance(data, list):
        raw = data
    elif isinstance(data, dict):
        raw = data.get("objects", data.get("instances", []))
    else:
        raw = []

    objects = []
    for o in raw:
        if not isinstance(o, dict):
            continue
        label = o.get("label", "unknown")
        bbox = o.get("bbox")
        evidence = o.get("evidence", "")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = to_pixel(bbox, width, height)
        objects.append({
            "label": label,
            "bbox": [round(x1), round(y1), round(x2), round(y2)],
            "evidence": evidence,
        })

    result = {"image": "6.jpeg", "version": "0.2", "width": width, "height": height, "objects": objects}
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    draw_boxes(objects, ANNOTATED_PATH)

    print(f"检测到 {len(objects)} 个实例", flush=True)
    print(f"结果已保存: {RESULT_PATH}", flush=True)
    print(f"标注图已保存: {ANNOTATED_PATH}", flush=True)


if __name__ == "__main__":
    main()