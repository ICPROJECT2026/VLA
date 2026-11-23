# 千问 2.5-VL-3B 目标检测程序 — 实施计划

## 1. 目标

使用 **Qwen2.5-VL-3B-Instruct** 视觉语言模型识别图片 `6.jpeg`，输出 JSON 结果，
包含每个检测到的物品的：

- **物品名称**（name）
- **bbox**（边界框坐标）
- **类别**（category：`公共物品` / `私人物品` / `垃圾`）

并在图片上绘制 bbox、标注物品名称与类别，生成一张可视化结果图。

---

## 2. 环境现状

| 项目 | 现状 |
|------|------|
| Python | `D:\Anaconda\envs\VLM\python.exe`（Python 3.12.16，虚拟环境 VLM） |
| GPU | NVIDIA RTX 4060 Laptop GPU，8GB 显存，CUDA 可用（torch 2.6.0+cu124） |
| 已装 | torch 2.6.0+cu124、torchvision 0.21.0+cu124、transformers 5.22.2、qwen-vl-utils 0.0.14、accelerate 1.15.0、bitsandbytes 0.50.2、numpy 2.5.4、pillow 13.0.0 |
| 未装 | 无（所需依赖已全部就绪） |

---

## 3. 运行环境

已确认虚拟环境 `D:\Anaconda\envs\VLM` 中所需依赖**全部就绪，无需额外安装**：

- `torch 2.6.0+cu124`（CUDA 可用，识别到 RTX 4060 Laptop GPU）
- `transformers 5.22.2`
- `qwen-vl-utils 0.0.14`
- `accelerate 1.15.0`
- `bitsandbytes 0.50.2`（备用量化）
- `numpy 2.5.4`、`pillow 13.0.0`、`torchvision 0.21.0+cu124`

程序运行时直接调用该环境解释器：

```powershell
& 'D:\Anaconda\envs\VLM\python.exe' detect.py
```

---

## 4. 模型与推理方案

- **模型**：`Qwen/Qwen2.5-VL-3B-Instruct`（3B 参数，权重约 6GB @ bf16/fp16）
- **精度**：`torch.bfloat16`（首选）或 `torch.float16`，显存不足时降级
- **加载**：`device_map="auto"` + `low_cpu_mem_usage=True`
- **显存应对（8GB 可能偏紧）**，按优先级：
  1. bf16/fp16 + auto 加载（大概率可行，权重约 6GB）
  2. 若 OOM：`max_memory` 指定 + CPU offload（`offload_folder`）
  3. 若仍不足：4bit 量化（`bitsandbytes`，Windows 支持有限）

---

## 5. Prompt 设计（检测 + 分类）

通过 chat template 引导模型输出**结构化 JSON**，要求：

- 检测图片中**所有**物品
- 每个物品给出：名称、归一化 bbox、类别
- 类别限定为三种：`公共物品`、`私人物品`、`垃圾`
- 物品名称、bbox、类别均由模型**对 `6.jpeg` 的实际识别结果**决定，不套用任何预设映射

Prompt 中会明确告知模型：bbox 坐标格式为 `[x1,y1,x2,y2]`，并严格输出 JSON 数组。

> 备选：利用 Qwen2.5-VL 原生 grounding 特殊 token
> `<|box_start|>(x1,y1),(x2,y2)<|box_end|>`（坐标归一化到 0–1000），
> 再用正则解析；若模型直接输出 JSON 更稳定则优先用 JSON 文本方式。

---

## 6. 输出 JSON 格式

`name`、`bbox`、`category` 均为模型对 `6.jpeg` 的**真实检测答案**（以下仅为结构，内容以模型识别结果为准）：

```json
{
  "image": "6.jpeg",
  "objects": [
    {
      "name": "<模型识别到的物品名称>",
      "bbox": [x1, y1, x2, y2],
      "category": "公共物品 | 私人物品 | 垃圾"
    }
  ]
}
```

- `name` / `category`：由模型根据图片实际内容输出
- `bbox`：模型检测出的真实边界框坐标
- bbox 同时保留**归一化坐标**与**像素坐标**（像素坐标用于可视化，归一化用于通用性）
- 结果写入 `detection_result.json`

---

## 7. bbox 可视化

- 用 PIL / OpenCV 读取 `6.jpeg` 得到宽高
- 将归一化 bbox 转成像素坐标
- 为三类分配不同颜色（如：公共物品=绿、私人物品=蓝、垃圾=红）
- 绘制矩形框 + 文字标签（`名称: 类别`）
- 保存为 `6_annotated.jpg`

---

## 8. 代码结构

```
VLM_2/
├── detect.py                  # 主程序
├── detection_plan.md          # 本计划
├── 6.jpeg                     # 输入图片
├── detection_result.json      # 输出：检测结果
└── 6_annotated.jpg            # 输出：bbox 可视化图
```

`detect.py` 主要流程：

1. 加载模型与 processor（含显存策略）
2. 构建多模态 messages（图片 + 检测 prompt）
3. `apply_chat_template` + `process_vision_info` 构造输入
4. `model.generate()` 推理，解码输出文本
5. 直接解析模型输出的 JSON 文本（不使用正则，靠 prompt 约束输出格式）
6. 校验并归一化 bbox、类别
7. 写 `detection_result.json`
8. 画 bbox 并保存 `6_annotated.jpg`

---

## 9. 风险与注意事项

| 风险 | 应对 |
|------|------|
| 8GB 显存不足（权重约 6GB + 激活） | 精度降级 / CPU offload / 4bit 量化 |
| 模型输出非严格 JSON（夹杂解释文字） | 通过 prompt 约束模型只输出 JSON（不使用正则解析） |
| bbox 坐标格式 | prompt 中已明确告知模型输出 `[x1,y1,x2,y2]`，解析时按此格式读取 |
| 模型漏检/误检物品 | prompt 明确"检测所有可见物品"；必要时提高 max_new_tokens |
| 首次需联网下载模型（约 6GB+） | 使用国内镜像 `HF_ENDPOINT=https://hf-mirror.com` |
| Windows + bitsandbytes 兼容性 | 优先不用量化；确需时再评估 |
| transformers 5.x 版本较新，API 可能有差异 | 实现时按 5.22.2 的实际 API（如 `AutoModelForImageTextToText`）编写并验证 |

---

## 10. 交付物

1. `detect.py` — 可运行主程序
2. `detection_result.json` — 结构化检测结果（名称/bbox/类别）
3. `6_annotated.jpg` — bbox 可视化标注图

---

## 下一步

确认本计划后，我将按上述步骤实现 `detect.py` 并实际运行验证。
