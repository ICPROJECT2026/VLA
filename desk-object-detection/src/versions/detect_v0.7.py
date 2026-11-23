# -*- coding: utf-8 -*-
"""detect_v0.7.py —— v0.7：v0.6（P0 提速）+ 四点定位（YOLO-World + MobileSAM）

与 v0.6 的区别：
1. 新增四点定位：YOLO-World 出候选框，MobileSAM 用 bbox prompt 出 mask，
   再对 mask 取轮廓 → cv2.minAreaRect → 四个角点。
2. JSON 新增 quad / quad_1000 / mask_bbox / mask_area；bbox 仍保留 YOLO 原始框。
3. 标注图默认画四点四边形；没有有效 mask 时回退成轴对齐四角。
4. Qwen 识别仍沿用 v0.6 的限像素 + batch 方案，不依赖 3B 的定位能力。

阶段 A：YOLO-World 定位。
阶段 B：MobileSAM 出 mask → 四点（约 0.2s，远小于 Qwen）。
阶段 C：逐框裁剪 → 批量送 Qwen 识别 name + category + reason。
阶段 D：规则去重（IoU 高且 name 相同，保留置信度最高的框）。

用法：
  & 'D:\\Anaconda\\envs\\VLM\\python.exe' detect_v0.7.py --image 6.jpeg
  & 'D:\\Anaconda\\envs\\VLM\\python.exe' detect_v0.7.py --image 2.jpg --no-quad
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
YOLO_MODEL = "yolov8s-worldv2.pt"
SAM_MODEL = "mobile_sam.pt"
MAX_BOXES = 10
YOLO_CONF = 0.05
IOU_THRESH = 0.5
DEFAULT_MAX_EDGE = 448
DEFAULT_BATCH_SIZE = 10
DEFAULT_MAX_NEW_TOKENS = 128

RECOGNIZE_PROMPT = """请识别这张裁剪图中唯一的物品。

判断规则：
1. name：物品名称（中文），认不出填 unknown。
2. category：按摆放状态判断，只能是以下三者之一：
   - 垃圾：明显被丢弃的包装袋、纸屑、烟头、空饮料容器、揉皱的废纸等（结合摆放位置和状态，不能仅因为物体是瓶子或纸张就判为垃圾）
   - 公共物品：图书馆等公共场所供公众共同使用的物品（键盘、鼠标、显示器、饮水机、打印机、桌椅等）
   - 私人物品：仍在使用或有序存放的个人物品（包、水杯、手机等）
3. reason：结合可见证据的简短理由（颜色/形状/纹理特征）。

不要臆造物体；没有清晰物品时 name 填 unknown。

只输出一个 JSON 对象，不要输出任何其他文字、解释或 Markdown 代码块。格式：
{"name":"物品名称","category":"垃圾/公共物品/私人物品中的一个","reason":"简短理由"}
"""


def load_qwen():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def configure_processor(processor, max_edge):
    """限制视觉 token 数：Qwen 默认最长边按总像素算，这里改成 max_edge²。"""
    processor.tokenizer.padding_side = "left"
    processor.image_processor.size = {
        "longest_edge": int(max_edge) * int(max_edge),
        "shortest_edge": 28 * 28 * 4,
    }


def parse_json(text):
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


def normalize_category(cat):
    cat = str(cat or "")
    if "垃圾" in cat or "废" in cat:
        return "垃圾"
    if "公共" in cat:
        return "公共物品"
    if "私人" in cat or "私有" in cat or "个人" in cat:
        return "私人物品"
    return "unknown"


def parse_recognize_output(out):
    try:
        data = parse_json(out)
    except Exception:
        return {"name": "unknown", "category": "unknown", "reason": ""}
    if isinstance(data, list) and data and isinstance(data[0], dict):
        data = data[0]
    if isinstance(data, dict):
        return {
            "name": str(data.get("name", "unknown")),
            "category": normalize_category(data.get("category", "")),
            "reason": str(data.get("reason", "")),
        }
    return {"name": "unknown", "category": "unknown", "reason": ""}


def run_yolo(image_path, max_boxes, conf):
    """阶段 A：YOLO-World 定位，返回候选框列表。"""
    from ultralytics import YOLO

    t0 = time.perf_counter()
    model = YOLO(YOLO_MODEL)
    t_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    results = model.predict(image_path, conf=conf, verbose=False)
    t_pred = time.perf_counter() - t0

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
    return boxes, {"yolo_load": t_load, "yolo_predict": t_pred}


def crop_object(image, bbox, pad_ratio=0.05):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    px, py = int(w * pad_ratio), int(h * pad_ratio)
    left = max(0, x1 - px)
    top = max(0, y1 - py)
    right = min(image.width, x2 + px)
    bottom = min(image.height, y2 + py)
    return image.crop((left, top, right, bottom))


def axis_quad(bbox):
    """轴对齐框的四个角点：左上、右上、右下、左下（顺时针）。"""
    x1, y1, x2, y2 = bbox
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def order_quad(points):
    """把四个点按顺时针排序，并从左上角开始。"""
    pts = np.asarray(points, dtype=np.float64)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]
    start = int(np.argmin(pts[:, 0] + pts[:, 1]))
    return np.roll(pts, -start, axis=0)


def fallback_quad(bbox):
    return {"quad": axis_quad(bbox), "mask_bbox": list(bbox), "mask_area": 0, "source": "fallback"}


def compute_quads(image_path, boxes, sam_model_path=SAM_MODEL):
    """阶段 B：YOLO 框作 bbox prompt → MobileSAM mask → 轮廓最小外接矩形 → 四点。"""
    if not boxes:
        return [], {"sam_load": 0.0, "sam_predict": 0.0}

    t0 = time.perf_counter()
    try:
        from ultralytics import SAM

        sam = SAM(sam_model_path)
        t_load = time.perf_counter() - t0
        t0 = time.perf_counter()
        result = sam.predict(image_path, bboxes=[b["bbox"] for b in boxes], verbose=False)[0]
        t_pred = time.perf_counter() - t0
        masks = result.masks.data if result.masks is not None else None
    except Exception as exc:
        print(f"MobileSAM 失败，回退轴对齐四点：{exc}", flush=True)
        return [fallback_quad(b["bbox"]) for b in boxes], {"sam_load": 0.0, "sam_predict": 0.0}

    with Image.open(image_path) as img:
        img_w, img_h = img.size

    quads = []
    for i, b in enumerate(boxes):
        if masks is None or i >= len(masks):
            quads.append(fallback_quad(b["bbox"]))
            continue
        mask = masks[i].cpu().numpy().astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            quads.append(fallback_quad(b["bbox"]))
            continue
        cnt = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(cnt))
        x1, y1, x2, y2 = b["bbox"]
        if area <= 0 or area < 0.01 * max(1.0, (x2 - x1) * (y2 - y1)):
            quads.append(fallback_quad(b["bbox"]))
            continue

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            quads.append(fallback_quad(b["bbox"]))
            continue
        mask_bbox = [
            int(np.clip(xs.min(), 0, img_w - 1)), int(np.clip(ys.min(), 0, img_h - 1)),
            int(np.clip(xs.max(), 0, img_w - 1)), int(np.clip(ys.max(), 0, img_h - 1)),
        ]
        if mask_bbox[2] - mask_bbox[0] < 2 or mask_bbox[3] - mask_bbox[1] < 2:
            quads.append(fallback_quad(b["bbox"]))
            continue

        quad_raw = cv2.boxPoints(cv2.minAreaRect(cnt))
        # 最小外接矩形超出/贴住图像边界，说明 mask 贴边或形状不规则，改用 mask 的轴对齐框
        if (quad_raw[:, 0].min() <= 1 or quad_raw[:, 0].max() >= img_w - 2
                or quad_raw[:, 1].min() <= 1 or quad_raw[:, 1].max() >= img_h - 2):
            quads.append({
                "quad": axis_quad(mask_bbox),
                "mask_bbox": mask_bbox,
                "mask_area": int(round(area)),
                "source": "mask_bbox",
            })
            continue

        quad = order_quad(quad_raw)
        qx1, qy1 = quad.min(axis=0)
        qx2, qy2 = quad.max(axis=0)
        quads.append({
            "quad": [[int(round(x)), int(round(y))] for x, y in quad],
            "mask_bbox": [int(round(qx1)), int(round(qy1)), int(round(qx2)), int(round(qy2))],
            "mask_area": int(round(area)),
            "source": "sam",
        })

    del sam
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return quads, {"sam_load": t_load, "sam_predict": t_pred}


def recognize_batch(model, processor, image, boxes, batch_size, max_new_tokens):
    """阶段 C：逐框裁剪，按 batch 送 Qwen；每个样本独立，顺序与 boxes 一致。"""
    crops = [crop_object(image, b["bbox"]) for b in boxes]
    recs, raws = [], []
    t0 = time.perf_counter()

    for start in range(0, len(crops), batch_size):
        chunk = crops[start:start + batch_size]
        texts, all_images = [], []
        for crop in chunk:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": crop},
                {"type": "text", "text": RECOGNIZE_PROMPT},
            ]}]
            texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            image_inputs, _ = process_vision_info(messages)
            all_images.extend(image_inputs)

        inputs = processor(text=texts, images=all_images, padding=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = generated[:, inputs.input_ids.shape[1]:]
        outs = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for out in outs:
            recs.append(parse_recognize_output(out))
            raws.append(out)

    return recs, raws, time.perf_counter() - t0


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
    """阶段 C：IoU 高且 name 相同（非 unknown）的框，保留置信度最高的框。"""
    idx = sorted(range(len(objects)), key=lambda i: -objects[i]["confidence"])
    suppressed = [False] * len(objects)
    result = []
    for i in idx:
        if suppressed[i]:
            continue
        for j in idx:
            if j == i or suppressed[j]:
                continue
            ni, nj = objects[i]["name"], objects[j]["name"]
            if ni != "unknown" and ni == nj and iou(objects[i]["bbox"], objects[j]["bbox"]) > iou_thresh:
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


def draw_boxes(image_path, objects, out_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = load_font(24)
    colors = {"公共物品": (34, 139, 34), "私人物品": (30, 120, 255), "垃圾": (230, 57, 70)}

    # 先画所有形状，再画所有标签：避免后画的框线从先画的标签文字中间穿过
    for obj in objects:
        cat = obj.get("category", "")
        color = colors.get(cat, (255, 160, 0))
        quad = obj.get("quad")
        if quad:
            pts = [(int(p[0]), int(p[1])) for p in quad]
            draw.line(pts + [pts[0]], fill=color, width=4)
        else:
            x1, y1, x2, y2 = obj["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

    for obj in objects:
        cat = obj.get("category", "")
        color = colors.get(cat, (255, 160, 0))
        quad = obj.get("quad")
        if quad:
            x1 = min(int(p[0]) for p in quad)
            y1 = min(int(p[1]) for p in quad)
        else:
            x1, y1 = obj["bbox"][0], obj["bbox"][1]
        label = f"{obj.get('name', '?')}: {cat}"
        tb = draw.textbbox((x1, y1), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        text_y = y1 - th - 6 if y1 - th - 6 > 0 else y1 + 4
        draw.rectangle([x1, text_y, x1 + tw + 10, text_y + th + 6], fill=color)
        draw.text((x1 + 5, text_y + 3), label, fill=(255, 255, 255), font=font)
    image.save(out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="6.jpeg", help="输入图片路径")
    parser.add_argument("--result", default=None, help="结果 JSON 路径；默认 detection_result_v0.7_<图片名>.json")
    parser.add_argument("--annotated", default=None, help="标注图路径；默认 <图片名>_annotated_v0.7.jpg")
    parser.add_argument("--max-edge", type=int, default=DEFAULT_MAX_EDGE, help="视觉像素上限的边长，默认 448（即 448×448）")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="一次送入 Qwen 的裁剪图数量，默认 10")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="生成 token 上限，默认 128")
    parser.add_argument("--conf", type=float, default=YOLO_CONF, help="YOLO 置信度阈值，默认 0.05")
    parser.add_argument("--max-boxes", type=int, default=MAX_BOXES, help="最多候选框数，默认 10")
    parser.add_argument("--sam", default=SAM_MODEL, help="MobileSAM 权重路径，默认 mobile_sam.pt")
    parser.add_argument("--no-quad", action="store_true", help="关闭 MobileSAM 四点，回退轴对齐四角")
    parser.add_argument("--save-raw", action="store_true", help="把每个框的 Qwen 原始输出写入 JSON")
    args = parser.parse_args()

    image_path = args.image if os.path.isabs(args.image) else os.path.join(BASE_DIR, args.image)
    if not os.path.isfile(image_path):
        parser.error(f"找不到图片：{image_path}")
    stem = os.path.splitext(os.path.basename(image_path))[0]
    result_path = args.result or os.path.join(BASE_DIR, f"detection_result_v0.7_{stem}.json")
    annotated_path = args.annotated or os.path.join(BASE_DIR, f"{stem}_annotated_v0.7.jpg")

    t_total = time.perf_counter()

    print("阶段 A：YOLO-World 定位...", flush=True)
    boxes, yolo_time = run_yolo(image_path, args.max_boxes, args.conf)
    print(f"YOLO 输出 {len(boxes)} 个候选框，定位 {yolo_time['yolo_predict']:.2f}s", flush=True)
    for i, b in enumerate(boxes):
        print(f"  [{i}] label={b['yolo_label']} conf={b['confidence']} bbox={b['bbox']}", flush=True)
    if not boxes:
        print("YOLO 未检测到候选框，结束。", flush=True)
        return

    if args.no_quad:
        quads = [fallback_quad(b["bbox"]) for b in boxes]
        sam_time = {"sam_load": 0.0, "sam_predict": 0.0}
        print("阶段 B：已关闭 MobileSAM，使用轴对齐四角", flush=True)
    else:
        print("阶段 B：MobileSAM 生成四点...", flush=True)
        quads, sam_time = compute_quads(image_path, boxes, args.sam)
        print(f"四点：{len(quads)} 个，mask 生成 {sam_time['sam_predict']:.2f}s"
              f"（加载 {sam_time['sam_load']:.2f}s）", flush=True)
        for i, q in enumerate(quads):
            print(f"  [{i}] {q['source']} quad={q['quad']}", flush=True)

    print("阶段 C：加载 Qwen 并批量识别...", flush=True)
    t0 = time.perf_counter()
    model, processor = load_qwen()
    t_qwen_load = time.perf_counter() - t0
    configure_processor(processor, args.max_edge)
    print(f"Qwen 加载 {t_qwen_load:.2f}s，视觉上限 {args.max_edge}×{args.max_edge}，"
          f"batch={args.batch_size}，max_new_tokens={args.max_new_tokens}", flush=True)

    with Image.open(image_path) as img:
        image = img.convert("RGB")
        width, height = image.size

    recs, raws, t_recognize = recognize_batch(
        model, processor, image, boxes, args.batch_size, args.max_new_tokens
    )
    for i, (rec, raw) in enumerate(zip(recs, raws)):
        print(f"  [{i}] -> {rec}", flush=True)
        print(f"       原始输出: {raw.strip()[:300]}", flush=True)

    objects = []
    for i, (b, rec) in enumerate(zip(boxes, recs)):
        x1, y1, x2, y2 = b["bbox"]
        q = quads[i]
        obj = {
            "name": rec["name"],
            "category": rec["category"],
            "reason": rec["reason"],
            "bbox": [x1, y1, x2, y2],
            "bbox_1000": [
                round(x1 / width * 1000), round(y1 / height * 1000),
                round(x2 / width * 1000), round(y2 / height * 1000),
            ],
            "quad": q["quad"],
            "quad_1000": [
                [round(px / width * 1000), round(py / height * 1000)]
                for px, py in q["quad"]
            ],
            "mask_bbox": q["mask_bbox"],
            "mask_area": q["mask_area"],
            "quad_source": q["source"],
            "confidence": b["confidence"],
            "yolo_label": b["yolo_label"],
        }
        if args.save_raw:
            obj["raw"] = raws[i]
        objects.append(obj)

    print("阶段 D：规则去重...", flush=True)
    before = len(objects)
    objects = deduplicate(objects, IOU_THRESH)
    print(f"去重：{before} -> {len(objects)}", flush=True)

    t_total = time.perf_counter() - t_total
    timing = {
        "yolo_load": round(yolo_time["yolo_load"], 2),
        "yolo_predict": round(yolo_time["yolo_predict"], 2),
        "sam_load": round(sam_time["sam_load"], 2),
        "sam_predict": round(sam_time["sam_predict"], 2),
        "qwen_load": round(t_qwen_load, 2),
        "recognize": round(t_recognize, 2),
        "total": round(t_total, 2),
        "max_edge": args.max_edge,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "quad": not args.no_quad,
    }
    result = {
        "image": os.path.basename(image_path),
        "version": "0.7",
        "width": width,
        "height": height,
        "timing": timing,
        "objects": objects,
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    draw_boxes(image_path, objects, annotated_path)

    print(f"结果已保存: {result_path}", flush=True)
    print(f"标注图已保存: {annotated_path}", flush=True)
    print(f"耗时: YOLO {timing['yolo_predict']:.2f}s + SAM {timing['sam_predict']:.2f}s "
          f"+ Qwen 加载 {timing['qwen_load']:.2f}s + 识别 {timing['recognize']:.2f}s "
          f"= 总计 {timing['total']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
