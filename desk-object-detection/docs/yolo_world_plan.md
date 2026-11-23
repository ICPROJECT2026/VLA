# YOLO-World 定位 + Qwen2.5-VL-3B 识别 — 方案计划

> 目标：解决「识别物体」这一步——找出图中所有物体（含垃圾），框对位置，命名准确。
> 分工：YOLO-World 负责**定位**（高召回），Qwen2.5-VL-3B 负责**识别/分类**（强语义）。

## 0. 思路与认可理由

| 之前的问题 | 本方案的解法 |
|------------|--------------|
| 漏检（瓶子、纸） | YOLO-World 是专用检测器，召回远强于 VLM 的「自由输出 JSON」 |
| 位置漂移 | YOLO-World 输出的是稳定、规整的检测框 |
| 命名错误（纸→瓶盖） | Qwen 对每个框**裁剪放大**后再识别，细节更多、更准 |

这套「检测 + 识别」两阶段范式，是 `garbage_handling.md` 里方案 F 的具体落地。

## 1. 技术选型

- **YOLO-World**：经 ultralytics 使用，选用轻量版 `yolov8s-worldv2.pt`（~25MB，适配 8GB 显存）。
- **Qwen2.5-VL-3B-Instruct**：复用现有模型，只做识别/分类，不做定位。

## 2. 整体流程

```
输入图片 6.jpeg
   │
   ▼
[阶段 A] YOLO-World 定位
   - set_classes(宽泛类别集) 或开放词汇
   - predict → 候选框 list（xyxy + confidence）
   - 按 conf 阈值过滤、NMS 去重
   │
   ▼
[释放 YOLO 显存]（分时加载，8GB 够用）
   │
   ▼
[阶段 B] Qwen2.5-VL-3B 识别
   - 对每个候选框裁剪局部图
   - 多图打包一次调用（或逐框）→ 识别 name + category
   │
   ▼
输出 detection_result_v0.3.json + 6_annotated_v0.3.jpg
```

## 3. 环境与依赖

```powershell
& 'D:\Anaconda\envs\VLM\python.exe' -m pip install ultralytics
```

- 权重 `yolov8s-worldv2.pt` 首次运行自动下载（~25MB）。
- 显存策略：**分时加载**——先加载 YOLO 定位，释放后再加载 Qwen 识别，避免两个模型同时占显存。

## 4. 关键设计决策

### 4.1 YOLO 的类别设置（最关键，需你拍板）

YOLO-World 是「文本引导」检测，需要给类别文本。但你说过「场景无从得知具体有什么」。
三种做法：

- **做法 1：宽泛类别集**（推荐）：给一个覆盖「桌面/办公/图书馆 + 垃圾」的几十类文本，
  如 `keyboard, mouse, bottle, cup, paper, plastic bag, bottle cap, tissue, trash, phone, book, ...`。
  不算过拟合，只是给一个较宽的搜索范围。
- **做法 2：默认 COCO 80 类**：零配置，但会漏「纸团、瓶盖」等非标准类别。
- **做法 3：完全开放**：不 set_classes，用 YOLO-World 默认行为，效果不稳定。

### 4.2 置信度阈值

- conf 调低（如 0.05~0.1）保召回，配合 NMS 去重；宁可多框，后续 Qwen 再过滤。

### 4.3 识别方式

- **多图打包**（1 轮）：把所有裁剪图一次给 Qwen，快但顺序有风险；
- **逐框**（N 轮）：顺序准确但慢。倾向先试打包。

## 5. 输出格式

```json
{
  "image": "6.jpeg",
  "version": "0.3",
  "objects": [
    {"name": "键盘", "category": "公共物品", "bbox": [x1,y1,x2,y2], "confidence": 0.92}
  ]
}
```

## 6. 风险与应对

| 风险 | 应对 |
|------|------|
| YOLO 误报（框出背景/阴影） | conf 阈值 + NMS；Qwen 识别时判定「非物体」则丢弃 |
| YOLO 漏检非标准类别 | 用宽泛类别集（做法 1），必要时补类别 |
| 两个模型显存冲突 | 分时加载：YOLO 用完即 `del` + `torch.cuda.empty_cache()` |
| Qwen 识别仍命名错 | 裁剪放大 + 单独识别，细节更多，准确率应高于整图 |
| 轮数/耗时 | YOLO 定位极快（几十 ms）；Qwen 识别 1~N 轮 |

## 7. 我的疑问（需要你确认）

1. **类别设置**：采用做法 1（宽泛类别集）可以吗？还是你有指定的类别清单？
2. **识别输出**：Qwen 识别时要同时输出「名称 + 类别（公共/私人/垃圾）」，还是先只出名称、分类后续再说？
3. **候选框数量上限**：YOLO 可能框出很多，是否设上限（如最多 20 个）再送 Qwen 识别？
4. **显存分时**：接受「YOLO 先跑、释放后 Qwen 再跑」的方式吗？（会多一次模型加载时间）
