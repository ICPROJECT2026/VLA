# Version history

Eleven revisions in one week — twelve runnable scripts, v0.0 to v1.1 — each changing **one
variable** and each written because the previous one failed on a specific photograph. The scripts
are kept verbatim, including the `D:\Anaconda\...` interpreter paths in their usage comments:
they are a record of the process, not production code. Every version takes the command-line
flags it had at the time.

Counts below are objects kept after deduplication on the three test photographs
(`6.jpeg` / `1.jpg` / `2.jpg`); earlier versions only ran on `6.jpeg`.

## Three eras

| Era | Versions | Architecture | Why it ended |
|---|---|---|---|
| 1. The VLM does everything | v0.0 – v0.2 | One prompt, whole image, asks Qwen for boxes and labels | A 3B model cannot localise reliably; recall and naming were both unstable between runs |
| 2. Detector + VLM | v0.3 – v0.5 | YOLO-World finds boxes, Qwen names each crop | Naming was still limited by a one-line prompt; throughput was 23 s per image |
| 3. Geometry + throughput | v0.6 – v1.1 | 448² token cap, batched recognition, four-point masks, quantisation | Open problems are semantic, not geometric |

## Every version

| Version | Date | Change | 6 / 1 / 2 | The failure that forced the next version |
|---|---|---|---|---|
| **v0.0** (`detect.py`) | Nov 10 | Single prompt over the whole image → `{name, bbox, category}` | 3 / – / – | Tissue missed; mouse called personal property. Detection and classification compete in one prompt |
| **v0.1** | Nov 10 | Split in two: detect every object, then classify each crop | 3 / – / – | Tissue found but named "bottle cap"; the real bottle missed entirely |
| **v0.2** | Nov 10 | "Desk cleaner" role prompt + region scanning + difference detection | 2 / – / – | Prompt was ignored: the model reported two objects, not a scan |
| **v0.3** | Nov 11 | **First detector.** YOLO-World localises, Qwen names crops in one multi-image call | 6 / – / – | Qwen returned 2 answers for 6 images, in the wrong order |
| **v0.4** | Nov 11 | One crop per `generate` call, own prompt, IoU deduplication | 4 / – / – | Names still poor — the prompt had no category rules |
| **v0.5** | Nov 11 | Prompt rewrite: detailed rules, `reason` field, no-hallucination clause, 0–1000 coordinates | 4 / – / – | Correct but slow (~23 s/image): full-resolution crops, sequential calls |
| **v0.6** | Nov 15 | Throughput: 448×448 visual cap, batch of 10 crops in one call, 128-token budget | 4 / 8 / 8 | Geometry still axis-aligned, so tilted objects dragged neighbours into the crop |
| **v0.7** | Nov 15 | **Four-point boxes:** MobileSAM mask → `minAreaRect` (plus `--no-quad`) | 4 / 8 / 8 | Large objects produced leaking masks and wild quads |
| **v0.8** | Nov 15 | Large-object fallback (20 % area) + containment-aware deduplication | 4 / 8 / 7 | The quads were only drawn, never used for the crop |
| **v0.9** | Nov 15 | 4-bit quantisation option, multi-image mode with a single model load, `--tag` | 4 / 8 / 7 | 4-bit halves the runtime but degrades names; the crop was still axis-aligned |
| **v1.0** | Nov 15 | **Quads now drive the Qwen crop** — quad above 15 % of the image, bbox below (`--crop auto`) | 4 / 8 / 7 | Laptop misclassification fixed; pure quad over-zooms the bottle and the phone, so the cut-off matters |
| **v1.1** | Nov 15 | Rendering only: labels anchored at each quad's true top-left vertex, with collision avoidance | 4 / 8 / 7 | — |

## Also in this folder

| File | What it is |
|---|---|
| `reference_recognize_v2.py` | A single-model baseline (one prompt does detect + name + classify). Studied, not adopted: its recognition is good but it misses objects. See [`docs/learn_from_recognize_v2.md`](../../docs/learn_from_recognize_v2.md) |

## What each era taught

1. **Ask a model to do what it is good at.** The VLM lost whenever it was asked to localise;
   the detector lost whenever it was asked to name. The split is not decoration.
2. **An optimisation only counts when it reaches the input.** Four-point boxes were computed
   from v0.7 onwards, but the VLM kept receiving axis-aligned crops until v1.0 — so three
   versions "had" the feature and none of them benefited from it.
3. **Measure on all the photographs, every time.** Several versions improved one image and
   damaged another; only re-running all three caught it.
