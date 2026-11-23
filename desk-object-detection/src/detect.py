# -*- coding: utf-8 -*-
"""detect_v1.1.py —— v1.1：v1.0 + 更美观的标注（标签锚定四边形左上角）

与 v1.0 的区别：
1. 标签锚点从「四边形轴对齐 bbox 的 min(x)/min(y)」改为「四边形实际的左上角顶点」。
2. 标签优先放锚点上方，放不下依次换右/左/内部/下方，并做简单避让，避免互相压字。
3. 圆角标签、小锚点、22px 字体，整体更适合展示。
4. 模型/裁剪逻辑完全沿用 v1.0（--crop auto/quad/bbox、--quant、多图共享）。
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
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

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
DEFAULT_QUANT = "bf16"
LARGE_AREA_RATIO = 0.20
MASK_LEAK_RATIO = 0.05
CONTAIN_THRESH = 0.85

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


def load_qwen(quant=DEFAULT_QUANT):
    if quant == "4bit":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID, quantization_config=bnb, device_map="auto"
        )
    else:
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


def crop_quad(image, quad, pad_ratio=0.05):
    """按四边形做透视矫正，裁成矩形后送 Qwen。"""
    pts = order_quad(quad).astype(np.float32)
    center = pts.mean(axis=0)
    pts = center + (pts - center) * (1.0 + pad_ratio)

    w = int(round(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[3] - pts[2]))))
    h = int(round(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2]))))
    w, h = min(max(w, 8), 4096), min(max(h, 8), 4096)

    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(
        np.asarray(image), matrix, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    return Image.fromarray(warped)


def fallback_quad(bbox):
    return {"quad": axis_quad(bbox), "mask_bbox": list(bbox), "mask_area": 0, "source": "fallback"}


def compute_quads(image_path, boxes, sam_model_path=SAM_MODEL,
                  large_area_ratio=LARGE_AREA_RATIO, mask_leak_ratio=MASK_LEAK_RATIO):
    """阶段 B：YOLO 框作 bbox prompt → MobileSAM mask → 轮廓最小外接矩形 → 四点。

    大物体（bbox 面积 > large_area_ratio × 图像面积）且 mask 明显溢出 YOLO 框时，
    不再强行 minAreaRect，直接用 mask 的轴对齐框。
    """
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

        box_area = max(1.0, (x2 - x1) * (y2 - y1))
        large = box_area > large_area_ratio * img_w * img_h
        leak_x = max(0.0, x1 - mask_bbox[0], mask_bbox[2] - x2) / max(1.0, x2 - x1)
        leak_y = max(0.0, y1 - mask_bbox[1], mask_bbox[3] - y2) / max(1.0, y2 - y1)
        if large and max(leak_x, leak_y) > mask_leak_ratio:
            quads.append({
                "quad": axis_quad(mask_bbox),
                "mask_bbox": mask_bbox,
                "mask_area": int(round(area)),
                "source": "large_bbox",
            })
            continue

        quad_raw = cv2.boxPoints(cv2.minAreaRect(cnt))
        # 只有真正超出图像边界才回退；贴边但仍在图内的四边形（如 1.jpg 的笔记本盖）保留
        if (quad_raw[:, 0].min() < 0 or quad_raw[:, 0].max() > img_w - 1
                or quad_raw[:, 1].min() < 0 or quad_raw[:, 1].max() > img_h - 1):
            quads.append({
                "quad": axis_quad(mask_bbox),
                "mask_bbox": mask_bbox,
                "mask_area": int(round(area)),
                "source": "mask_bbox",
            })
            continue

        quad = order_quad(quad_raw)
        quads.append({
            "quad": [[int(round(x)), int(round(y))] for x, y in quad],
            "mask_bbox": mask_bbox,
            "mask_area": int(round(area)),
            "source": "sam",
        })

    del sam
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return quads, {"sam_load": t_load, "sam_predict": t_pred}


def recognize_batch(model, processor, image, boxes, quads, batch_size, max_new_tokens,
                    crop_mode="auto", crop_pad=0.05, auto_quad_ratio=0.15, save_crops_dir=None):
    """阶段 C：按 --crop 裁剪后送 Qwen。

    auto：bbox 面积 >= auto_quad_ratio × 图像面积时用四点透视裁剪，小物体用 bbox 裁剪。
    这样大物体（笔记本）能吃到四点矫正，小物体（瓶子/手机）保留上下文，避免过度放大。
    """
    img_area = max(1, image.width * image.height)
    crops = []
    for b, q in zip(boxes, quads):
        x1, y1, x2, y2 = b["bbox"]
        use_quad = crop_mode == "quad"
        if crop_mode == "auto":
            use_quad = (x2 - x1) * (y2 - y1) >= auto_quad_ratio * img_area
        try:
            if use_quad:
                crops.append(crop_quad(image, q["quad"], crop_pad))
            else:
                crops.append(crop_object(image, b["bbox"]))
        except Exception:
            crops.append(crop_object(image, b["bbox"]))

    if save_crops_dir:
        os.makedirs(save_crops_dir, exist_ok=True)
        for i, crop in enumerate(crops):
            crop.save(os.path.join(save_crops_dir, f"crop_{i:02d}.jpg"))

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


def bbox_area(b):
    return max(0.0, (b[2] - b[0]) * (b[3] - b[1]))


def containment(a, b):
    """小框有多少比例落在大框内。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    small = min(bbox_area(a), bbox_area(b))
    return inter / small if small > 0 else 0.0


def deduplicate(objects, iou_thresh, contain_thresh=CONTAIN_THRESH):
    """阶段 D：先做包含关系合并，再做 IoU/同名去重。

    包含关系：同名、小框 >= contain_thresh 落在大框内、且 IoU <= iou_thresh，
    说明是父子框（如笔记本键盘在笔记本内），保留大框。
    IoU 去重：同名且 IoU 高，保留置信度最高的框。
    """
    n = len(objects)
    suppressed = [False] * n

    order_area = sorted(range(n), key=lambda i: -bbox_area(objects[i]["bbox"]))
    for i in order_area:
        if suppressed[i]:
            continue
        for j in order_area:
            if j == i or suppressed[j]:
                continue
            ni, nj = objects[i]["name"], objects[j]["name"]
            if ni == "unknown" or ni != nj:
                continue
            if bbox_area(objects[j]["bbox"]) >= bbox_area(objects[i]["bbox"]):
                continue
            if iou(objects[i]["bbox"], objects[j]["bbox"]) > iou_thresh:
                continue
            if containment(objects[i]["bbox"], objects[j]["bbox"]) >= contain_thresh:
                suppressed[j] = True

    idx = sorted(range(n), key=lambda i: -objects[i]["confidence"])
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


def rects_overlap(a, b, pad=4):
    return not (a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1])


def clamp_label(lx, ly, lw, lh, width, height, margin=4):
    lx = max(margin, min(width - lw - margin, lx))
    ly = max(margin, min(height - lh - margin, ly))
    return lx, ly


def draw_boxes(image_path, objects, out_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = load_font(22)
    colors = {"公共物品": (34, 139, 34), "私人物品": (30, 120, 255), "垃圾": (230, 57, 70)}
    width, height = image.size

    # 先画所有形状，再画所有标签：避免后画的框线从先画的标签文字中间穿过
    for obj in objects:
        cat = obj.get("category", "")
        color = colors.get(cat, (255, 160, 0))
        quad = obj.get("quad")
        if quad:
            pts = order_quad(quad).astype(int).tolist()
            draw.line(pts + [pts[0]], fill=color, width=3)
        else:
            x1, y1, x2, y2 = obj["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

    placed = []
    for obj in objects:
        cat = obj.get("category", "")
        color = colors.get(cat, (255, 160, 0))
        label = f"{obj.get('name', '?')} · {cat}"

        # 锚点：四边形实际的左上角顶点（order_quad 的第一个点），不再用 bbox 的 min(x)/min(y)
        quad = obj.get("quad")
        if quad:
            pts = order_quad(quad).astype(int).tolist()
            ax, ay = int(pts[0][0]), int(pts[0][1])
        else:
            ax, ay = int(obj["bbox"][0]), int(obj["bbox"][1])

        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        lw, lh = tw + 12, th + 8

        candidates = [
            (ax - 2, ay - lh - 8),          # 锚点上方
            (ax - lw + 2, ay - lh - 8),     # 锚点上方偏左
            (ax + 10, ay - lh // 2),        # 锚点右侧
            (ax - lw - 10, ay - lh // 2),   # 锚点左侧
            (ax + 6, ay + 6),               # 四边形内部
            (ax - lw // 2, ay + 10),        # 锚点下方
        ]

        label_pos = None
        for lx, ly in candidates:
            lx, ly = clamp_label(lx, ly, lw, lh, width, height)
            rect = [lx, ly, lx + lw, ly + lh]
            if not any(rects_overlap(rect, p) for p in placed):
                label_pos = (lx, ly)
                break
        if label_pos is None:
            lx, ly = clamp_label(*candidates[0], lw, lh, width, height)
            label_pos = (lx, ly)
        lx, ly = label_pos
        placed.append([lx, ly, lx + lw, ly + lh])

        draw.rounded_rectangle([lx, ly, lx + lw, ly + lh], radius=6, fill=color)
        draw.text((lx + 6, ly + 4), label, fill=(255, 255, 255), font=font)
        # 小锚点，明确标签对应的是哪个角
        draw.ellipse([ax - 3, ay - 3, ax + 3, ay + 3], fill=color)

    image.save(out_path)


def resolve_path(path):
    """Resolve a user-supplied image path against the working directory, then BASE_DIR."""
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return os.path.abspath(path)
    return os.path.join(BASE_DIR, path)


def run_multi(args):
    """多图模式：Qwen 只加载一次，逐图跑 YOLO/SAM/识别/去重/保存。"""
    if args.result or args.annotated:
        raise SystemExit("多图模式不支持 --result/--annotated，请用 --tag 区分输出")
    tag = f"_{args.tag}" if args.tag else ""
    image_paths = [resolve_path(p) for p in args.images]
    for p in image_paths:
        if not os.path.isfile(p):
            raise SystemExit(f"找不到图片：{p}")

    print(f"多图模式：{len(image_paths)} 张，Qwen 精度 {args.quant}，只加载一次", flush=True)
    t0 = time.perf_counter()
    model, processor = load_qwen(args.quant)
    t_qwen_load = time.perf_counter() - t0
    configure_processor(processor, args.max_edge)
    print(f"Qwen 加载 {t_qwen_load:.2f}s，视觉上限 {args.max_edge}×{args.max_edge}，"
          f"batch={args.batch_size}，max_new_tokens={args.max_new_tokens}", flush=True)

    total_start = time.perf_counter()
    for idx, image_path in enumerate(image_paths):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        result_path = os.path.join(BASE_DIR, f"detection_result_v1.1{tag}_{stem}.json")
        annotated_path = os.path.join(BASE_DIR, f"{stem}_annotated_v1.1{tag}.jpg")
        print(f"\n=== [{idx + 1}/{len(image_paths)}] {os.path.basename(image_path)} ===", flush=True)

        print("阶段 A：YOLO-World 定位...", flush=True)
        boxes, yolo_time = run_yolo(image_path, args.max_boxes, args.conf)
        print(f"YOLO 输出 {len(boxes)} 个候选框，定位 {yolo_time['yolo_predict']:.2f}s", flush=True)

        if args.no_quad:
            quads = [fallback_quad(b["bbox"]) for b in boxes]
            sam_time = {"sam_load": 0.0, "sam_predict": 0.0}
        else:
            print("阶段 B：MobileSAM 生成四点...", flush=True)
            quads, sam_time = compute_quads(
                image_path, boxes, args.sam,
                large_area_ratio=args.large_area_ratio,
                mask_leak_ratio=args.mask_leak_ratio,
            )
            print(f"四点：{len(quads)} 个，mask 生成 {sam_time['sam_predict']:.2f}s"
                  f"（加载 {sam_time['sam_load']:.2f}s）", flush=True)

        with Image.open(image_path) as img:
            image = img.convert("RGB")
            width, height = image.size

        print("阶段 C：Qwen 批量识别...", flush=True)
        if boxes:
            recs, raws, t_recognize = recognize_batch(
                model, processor, image, boxes, quads, args.batch_size, args.max_new_tokens,
                crop_mode=args.crop, crop_pad=args.crop_pad, auto_quad_ratio=args.auto_quad_ratio,
                save_crops_dir=os.path.join(args.save_crops, stem) if args.save_crops else None,
            )
        else:
            recs, raws, t_recognize = [], [], 0.0

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
        objects = deduplicate(objects, IOU_THRESH, args.contain_thresh)
        print(f"去重：{before} -> {len(objects)}", flush=True)

        per_image = yolo_time["yolo_predict"] + sam_time["sam_predict"] + t_recognize
        timing = {
            "yolo_load": round(yolo_time["yolo_load"], 2),
            "yolo_predict": round(yolo_time["yolo_predict"], 2),
            "sam_load": round(sam_time["sam_load"], 2),
            "sam_predict": round(sam_time["sam_predict"], 2),
            "qwen_load_shared": round(t_qwen_load, 2),
            "recognize": round(t_recognize, 2),
            "per_image": round(per_image, 2),
            "first_image_total": round(t_qwen_load + per_image, 2) if idx == 0 else None,
            "candidates": len(boxes),
            "objects": len(objects),
            "max_edge": args.max_edge,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "quant": args.quant,
            "crop": args.crop,
            "crop_pad": args.crop_pad,
            "auto_quad_ratio": args.auto_quad_ratio,
            "large_area_ratio": args.large_area_ratio,
            "mask_leak_ratio": args.mask_leak_ratio,
            "contain_thresh": args.contain_thresh,
            "quad": not args.no_quad,
        }
        result = {
            "image": os.path.basename(image_path),
            "version": "1.1",
            "quant": args.quant,
            "crop": args.crop,
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
        print(f"本图耗时: YOLO {timing['yolo_predict']:.2f}s + SAM {timing['sam_predict']:.2f}s "
              f"+ 识别 {timing['recognize']:.2f}s = {timing['per_image']:.2f}s，"
              f"物体 {before}->{len(objects)}", flush=True)

    wall = time.perf_counter() - total_start
    print(f"\n多图总计 {wall:.2f}s（不含 Qwen 加载 {t_qwen_load:.2f}s），"
          f"平均 {wall / len(image_paths):.2f}s/张", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="6.jpeg", help="单张输入图片路径")
    parser.add_argument("--images", nargs="+", default=None, help="多张图片；只加载一次 Qwen")
    parser.add_argument("--result", default=None, help="结果 JSON 路径；默认 detection_result_v1.1<tag>_<图片名>.json")
    parser.add_argument("--annotated", default=None, help="标注图路径；默认 <图片名>_annotated_v1.1<tag>.jpg")
    parser.add_argument("--max-edge", type=int, default=DEFAULT_MAX_EDGE, help="视觉像素上限的边长，默认 448（即 448×448）")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="一次送入 Qwen 的裁剪图数量，默认 10")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="生成 token 上限，默认 128")
    parser.add_argument("--conf", type=float, default=YOLO_CONF, help="YOLO 置信度阈值，默认 0.05")
    parser.add_argument("--max-boxes", type=int, default=MAX_BOXES, help="最多候选框数，默认 10")
    parser.add_argument("--sam", default=SAM_MODEL, help="MobileSAM 权重路径，默认 mobile_sam.pt")
    parser.add_argument("--quant", choices=["bf16", "4bit"], default=DEFAULT_QUANT, help="Qwen 精度，默认 bf16")
    parser.add_argument("--crop", choices=["bbox", "quad", "auto"], default="auto",
                        help="Qwen 输入裁剪方式：auto（默认，大物体四点/小物体 bbox）、quad、bbox")
    parser.add_argument("--crop-pad", type=float, default=0.05, help="裁剪外扩比例，默认 0.05")
    parser.add_argument("--auto-quad-ratio", type=float, default=0.15,
                        help="auto 模式下，bbox 面积 / 图像面积 超过该值时用四点裁剪，默认 0.15")
    parser.add_argument("--save-crops", default=None, help="保存 Qwen 输入裁剪图到目录")
    parser.add_argument("--tag", default="", help="输出文件名标记，例如 bf16 / 4bit")
    parser.add_argument("--large-area-ratio", type=float, default=LARGE_AREA_RATIO,
                        help="大物体阈值：bbox 面积 / 图像面积，默认 0.20")
    parser.add_argument("--mask-leak-ratio", type=float, default=MASK_LEAK_RATIO,
                        help="mask 超出 YOLO 框的比例阈值，默认 0.05")
    parser.add_argument("--contain-thresh", type=float, default=CONTAIN_THRESH,
                        help="包含关系合并阈值，默认 0.85")
    parser.add_argument("--no-quad", action="store_true", help="关闭 MobileSAM 四点，回退轴对齐四角")
    parser.add_argument("--save-raw", action="store_true", help="把每个框的 Qwen 原始输出写入 JSON")
    args = parser.parse_args()

    if args.images:
        run_multi(args)
        return

    image_path = resolve_path(args.image)
    if not os.path.isfile(image_path):
        parser.error(f"找不到图片：{image_path}")
    stem = os.path.splitext(os.path.basename(image_path))[0]
    tag = f"_{args.tag}" if args.tag else ""
    result_path = args.result or os.path.join(BASE_DIR, f"detection_result_v1.1{tag}_{stem}.json")
    annotated_path = args.annotated or os.path.join(BASE_DIR, f"{stem}_annotated_v1.1{tag}.jpg")

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
        quads, sam_time = compute_quads(
            image_path, boxes, args.sam,
            large_area_ratio=args.large_area_ratio,
            mask_leak_ratio=args.mask_leak_ratio,
        )
        print(f"四点：{len(quads)} 个，mask 生成 {sam_time['sam_predict']:.2f}s"
              f"（加载 {sam_time['sam_load']:.2f}s）", flush=True)
        for i, q in enumerate(quads):
            print(f"  [{i}] {q['source']} quad={q['quad']}", flush=True)

    print("阶段 C：加载 Qwen 并批量识别...", flush=True)
    t0 = time.perf_counter()
    model, processor = load_qwen(args.quant)
    t_qwen_load = time.perf_counter() - t0
    configure_processor(processor, args.max_edge)
    print(f"Qwen({args.quant}) 加载 {t_qwen_load:.2f}s，视觉上限 {args.max_edge}×{args.max_edge}，"
          f"batch={args.batch_size}，max_new_tokens={args.max_new_tokens}", flush=True)

    with Image.open(image_path) as img:
        image = img.convert("RGB")
        width, height = image.size

    recs, raws, t_recognize = recognize_batch(
        model, processor, image, boxes, quads, args.batch_size, args.max_new_tokens,
        crop_mode=args.crop, crop_pad=args.crop_pad, auto_quad_ratio=args.auto_quad_ratio,
        save_crops_dir=os.path.join(args.save_crops, stem) if args.save_crops else None,
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
    objects = deduplicate(objects, IOU_THRESH, args.contain_thresh)
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
        "quant": args.quant,
        "crop": args.crop,
        "crop_pad": args.crop_pad,
        "auto_quad_ratio": args.auto_quad_ratio,
        "large_area_ratio": args.large_area_ratio,
        "mask_leak_ratio": args.mask_leak_ratio,
        "contain_thresh": args.contain_thresh,
        "quad": not args.no_quad,
    }
    result = {
        "image": os.path.basename(image_path),
        "version": "1.1",
        "quant": args.quant,
        "crop": args.crop,
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
