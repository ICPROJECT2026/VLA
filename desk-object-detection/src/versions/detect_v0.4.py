# -*- coding: utf-8 -*-
"""detect_v0.4.py —— v0.4：YOLO 定位 + 局部图逐框识别 + 规则去重

阶段 A：YOLO-World 定位（conf 过滤，取前 N 个候选框）。
阶段 B：对每个框，仅输入该框的局部裁剪图，Qwen 识别 name + category（逐框）。
阶段 C：代码后处理去重——IoU 高且 name 相同的框，保留置信度最高的框。
"""
import json
import os

import torch
from PIL import Image, ImageDraw, ImageFont
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, "6.jpeg")
RESULT_PATH = os.path.join(BASE_DIR, "detection_result_v0.4.json")
ANNOTATED_PATH = os.path.join(BASE_DIR, "6_annotated_v0.4.jpg")

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
YOLO_MODEL = "yolov8s-worldv2.pt"
MAX_BOXES = 10
YOLO_CONF = 0.05
IOU_THRESH = 0.5

RECOGNIZE_PROMPT = """请识别这张图片中的物体，输出：
- name：物品名称（中文）
- category：类别，只能是"公共物品"、"私人物品"、"垃圾"三者之一

只输出一个 JSON 对象，不要输出任何其他文字、解释或 Markdown 代码块。格式如下：
{"name":"物品名称","category":"类别"}
"""


def load_qwen():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def generate(model, processor, messages, max_new_tokens=1024):
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
        start, end = text.find("["), text.rfind("]")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def run_yolo(image_path, max_boxes, conf):
    """阶段 A：YOLO-World 定位，返回候选框列表。"""
    from ultralytics import YOLO
    model = YOLO(YOLO_MODEL)
    results = model.predict(image_path, conf=conf, verbose=False)
    boxes = []
    for r in results:
        for box in r.boxes:
            xyxy = box.xyxy[0].tolist()
            c = float(box.conf[0])
            try:
                cls_id = int(box.cls[0].item())
            except Exception:
                cls_id = -1
            name = model.names.get(cls_id, "") if isinstance(model.names, dict) else ""
            boxes.append({
                "bbox": [round(v) for v in xyxy],
                "confidence": round(c, 4),
                "yolo_label": name,
            })
    boxes.sort(key=lambda b: b["confidence"], reverse=True)
    boxes = boxes[:max_boxes]
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return boxes


def crop_object(image, bbox, pad_ratio=0.05):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    px, py = int(w * pad_ratio), int(h * pad_ratio)
    left = max(0, x1 - px)
    top = max(0, y1 - py)
    right = min(image.width, x2 + px)
    bottom = min(image.height, y2 + py)
    return image.crop((left, top, right, bottom))


def recognize_one(model, processor, image, box):
    """阶段 B：对单个框，成对输入整图 + 局部图，识别 name + category。"""
    crop = crop_object(image, box["bbox"])
    messages = [{"role": "user", "content": [
        {"type": "image", "image": crop},
        {"type": "text", "text": RECOGNIZE_PROMPT},
    ]}]
    out = generate(model, processor, messages, max_new_tokens=256)
    data = parse_json(out)
    if isinstance(data, dict):
        return {"name": data.get("name", "unknown"), "category": data.get("category", "unknown")}, out
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return {"name": data[0].get("name", "unknown"), "category": data[0].get("category", "unknown")}, out
    return {"name": "unknown", "category": "unknown"}, out


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def deduplicate(objects, iou_thresh):
    """阶段 C：IoU 高且 name 相同的框，保留置信度最高的框。"""
    idx = sorted(range(len(objects)), key=lambda i: -objects[i]["confidence"])
    suppressed = [False] * len(objects)
    result = []
    for i in idx:
        if suppressed[i]:
            continue
        for j in idx:
            if j == i or suppressed[j]:
                continue
            if objects[i]["name"] == objects[j]["name"] and iou(objects[i]["bbox"], objects[j]["bbox"]) > iou_thresh:
                suppressed[j] = True
        result.append(objects[i])
    return result


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
    font = load_font(24)
    colors = {"公共物品": (34, 139, 34), "私人物品": (30, 120, 255), "垃圾": (230, 57, 70)}
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        cat = obj.get("category", "")
        color = colors.get(cat, (255, 160, 0))
        label = f"{obj.get('name', '?')}: {cat}"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        tb = draw.textbbox((x1, y1), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        text_y = y1 - th - 6 if y1 - th - 6 > 0 else y1 + 4
        draw.rectangle([x1, text_y, x1 + tw + 10, text_y + th + 6], fill=color)
        draw.text((x1 + 5, text_y + 3), label, fill=(255, 255, 255), font=font)
    image.save(out_path)


def main():
    print("阶段 A：YOLO-World 定位...", flush=True)
    boxes = run_yolo(IMAGE_PATH, MAX_BOXES, YOLO_CONF)
    print(f"YOLO 输出 {len(boxes)} 个候选框", flush=True)
    for i, b in enumerate(boxes):
        print(f"  [{i}] label={b['yolo_label']} conf={b['confidence']} bbox={b['bbox']}", flush=True)

    if not boxes:
        print("YOLO 未检测到候选框，结束。", flush=True)
        return

    print("阶段 B：加载 Qwen，逐框识别...", flush=True)
    model, processor = load_qwen()
    with Image.open(IMAGE_PATH) as img:
        image = img.convert("RGB")
        width, height = image.size

    objects = []
    for i, b in enumerate(boxes):
        rec, out = recognize_one(model, processor, image, b)
        print(f"  [{i}] -> {rec} (原输出: {out.strip()})", flush=True)
        x1, y1, x2, y2 = b["bbox"]
        objects.append({
            "name": rec["name"],
            "category": rec["category"],
            "bbox": [x1, y1, x2, y2],
            "confidence": b["confidence"],
            "yolo_label": b["yolo_label"],
        })

    print("阶段 C：规则去重...", flush=True)
    before = len(objects)
    objects = deduplicate(objects, IOU_THRESH)
    print(f"去重：{before} -> {len(objects)}", flush=True)

    result = {"image": "6.jpeg", "version": "0.4", "width": width, "height": height, "objects": objects}
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    draw_boxes(objects, ANNOTATED_PATH)

    print(f"结果已保存: {RESULT_PATH}", flush=True)
    print(f"标注图已保存: {ANNOTATED_PATH}", flush=True)


if __name__ == "__main__":
    main()