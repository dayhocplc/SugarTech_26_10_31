#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bản thảo cho Sugar Tech (Springer) — cùng số liệu, khác khung lập luận.

    python3 src/build_manuscript_sugartech.py

Vì sao cần một bản riêng chứ không chỉ đổi định dạng. Bản gửi Research in
Agricultural Engineering dẫn bằng một phê phán phương pháp luận: quy ước loại
nền trong ma trận nhầm lẫn làm F1 nhảy từ 0,47 lên 0,98 trên cùng dự đoán. Đó
là lập luận gửi tới cộng đồng thị giác máy nông nghiệp. Độc giả Sugar Tech là
nhà khoa học cây mía, và năm bài phát hiện đối tượng trên mía mà tạp chí đăng
gần đây đều có dạng "mô hình cải tiến cho một nhiệm vụ mía cụ thể".

Nên ở bản này thứ tự đảo lại: dẫn bằng câu hỏi ứng dụng — máy phun theo điểm
cần gì, và mô hình nào đáp ứng được — rồi đặt phê phán quy ước vào mục Thảo
luận như LỜI GIẢI THÍCH cho việc vì sao con số đã công bố đẹp hơn thực địa.
Không con số nào thay đổi; chỉ thứ tự trình bày thay đổi.

Khác biệt hình thức so với bản RAE:
  * bài nguyên tác ≤ 25 trang GIÃN ĐÔI, lề 2,5 cm, kể cả bảng và hình;
  * abstract 150-250 từ; 4-6 từ khoá;
  * Results và Discussion tách riêng (RAE gộp làm một);
  * tham chiếu kiểu Springer, chuyển bằng src/journal_style.py;
  * mục bắt buộc "Statements and Declarations".
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import discussion as D
from build_manuscript import (AFFIL, AUTHORS, CORRESP, REFERENCES, figure_block,
                              load_review, md_table, methods, to_docx)
from common import (MODEL_LABELS, RAE, RESULTS, clean_fps, load_cfg, log, scope,
                    throughput_requirement)
from journal_style import to_springer

OUT = RAE / "manuscript"

TITLE = ("Detection Completeness and Onboard Throughput of Three Deep Learning "
         "Detectors for Site-Specific Weed Control in Sugarcane")

KEYWORDS = ("Sugarcane; Weed detection; Site-specific herbicide application; "
            "Deep learning; Detection recall")

# Ba bài phát hiện đối tượng trên mía của chính Sugar Tech. Ở một tạp chí
# chuyên cây trồng, đặt công trình cạnh dòng bài sẵn có của tạp chí là việc
# phải làm; chúng cũng đúng chủ đề (YOLO cho máy nông nghiệp thông minh).
EXTRA_REFS = [
    "Chen, Jiasheng, Hongwei Li, Shunsheng Zhang, and Tao Wu. 2026. Integrating "
    "YOLOv10n-seg-p6 segmentation and CycleGAN adversarial augmentation for "
    "smartphone-based precision diagnosis of sugarcane leaf diseases. Sugar Tech 28: "
    "490-503. https://doi.org/10.1007/s12355-025-01640-z",
    "Li, Hongwei, Jingyi Yang, Jiuyang Song, Minshuo Sun, Wei Wei, Shunsheng Zhang, Tao "
    "Wu, and Yanzhou Li. 2026. Stem base detection in sugarcane plants using improved "
    "YOLOv5m model. Sugar Tech 28: 467-477. https://doi.org/10.1007/s12355-025-01590-6",
    "Mo, Yongmei, Hongwei Li, Xindong Lai, Deqiang He, Shunsheng Zhang, and Tao Wu. 2026. "
    "Visual detection of ratoon-emerged sugarcane seedling hill using an improved YOLOv8n "
    "model. Sugar Tech 28: 430-444. https://doi.org/10.1007/s12355-025-01605-2",
]

DECLARATIONS = [
    "## Statements and Declarations",
    "**Competing Interests.** The authors declare that they have no competing financial "
    "or non-financial interests that are directly or indirectly related to the work "
    "submitted for publication.",
    "**Funding.** [TO BE COMPLETED BY THE AUTHORS: grant number and funding body, or an "
    "explicit statement that no funding was received.]",
    "**Author Contributions.** [TO BE COMPLETED BY THE AUTHORS, naming the contribution "
    "of each author.]",
    "**Data Availability.** The annotation files in COCO format, the per-image results "
    "and the per-seed training logs are provided in Online Resource 2, together with "
    "SHA-256 checksums of every image file, so that the image set — supplied by the "
    "corresponding author on request because of its size — can be verified as the one "
    "used here.",
    "**Code Availability.** The complete training, evaluation and analysis code, "
    "including the scripts that regenerate every table and figure in this article and "
    "in Online Resource 1, is provided in Online Resource 2.",
    "**Supplementary Information.** Online Resource 1 (.docx): supplementary tables and "
    "figure supporting the Methods and Results, comprising the per-seed results, the "
    "threshold and IoU sweeps, the size-stratified recall, the checkpoint-protocol "
    "comparison, the resolution control experiment, the polygon-trained models and the "
    "training curves. Online Resource 2 (.zip): source code, COCO annotations, raw "
    "per-model results, generated tables and figures, training and evaluation logs, and "
    "a SHA-256 manifest of all files including the images.",
    "**Ethics Approval.** This study did not involve human participants or animals.",
]


# ───────────────────────────── phần viết riêng ──────────────────────────────

def abstract(stats: dict, rev: dict, cfg: dict) -> str:
    """Abstract 150-250 từ, dẫn bằng ứng dụng thay vì bằng phương pháp luận."""
    prim = ["faster_rcnn", "retinanet", "yolov10"]
    fm, ar = rev["full_metrics"], rev["area_recall"]
    ss = rev["size_strata"]["models"]
    f1s = {k: D.seed_mean(k, "f1") for k in prim}
    ap50 = {k: D.seed_mean(k, "ap50") for k in prim}
    best = max(prim, key=lambda k: ap50[k])
    t = throughput_requirement(cfg)
    fast = [k for k in prim if clean_fps(k)[0] >= t["fps_required_on_workstation"]]
    bb = stats["bbox"]
    lo = lambda d, m: min(d[k]["weed"][m]["mean"] for k in prim)
    hi = lambda d, m: max(d[k]["weed"][m]["mean"] for k in prim)
    return (
        "Site-specific herbicide application in sugarcane requires a detector that finds "
        "the weed present and supplies detections fast enough for the implement to act. "
        "Three deep learning detectors — Faster R-CNN, RetinaNet and YOLOv10 — were "
        f"trained on {bb['_total']['source_images']:,} close-range field images of "
        "sugarcane and weeds from Tay Ninh province, Vietnam, and evaluated on a common "
        f"test partition of {bb['test']['images']} images under one protocol: one input "
        f"resolution, {cfg['train']['epochs']} epochs, final-epoch weights and "
        f"{D.runs_phrase(prim)} per model. Scored under the confusion-matrix convention "
        "commonly reported in this literature, which discards undetected instances, all "
        f"three separated the classes reliably (two-class mean F1-score "
        f"{min(f1s.values()):.2f}-{max(f1s.values()):.2f}). Scored so that undetected "
        "weed and spurious detections are retained, the same predictions gave weed-class "
        f"precision of {lo(fm, 'precision'):.2f}-{hi(fm, 'precision'):.2f} and recall of "
        f"{lo(fm, 'recall'):.2f}-{hi(fm, 'recall'):.2f}; weighted by the ground area a "
        "sprayer would treat, recall was "
        f"{min(ar[k]['area_recall']['mean'] for k in prim):.2f}-"
        f"{max(ar[k]['area_recall']['mean'] for k in prim):.2f}. Recall fell monotonically "
        f"with weed size, from {min(ss[k]['quartile']['Q1']['recall_mean'] for k in prim):.2f} "
        f"in the smallest area quartile to "
        f"{max(ss[k]['quartile']['Q4']['recall_mean'] for k in prim):.2f} in the largest, "
        "and retraining the lightest detector at four times the pixel count did not "
        f"change it. {MODEL_LABELS[best]} obtained the highest AP50 "
        f"({ap50[best]:.1f})"
        + (f", but only {MODEL_LABELS[fast[0]]} met the throughput required at "
           f"{t['sprayer_speed_kmh']:.0f} km h⁻¹, and it treated the least ground"
           if fast else "")
        + ". Detection completeness, not class discrimination, limits this application, "
          "and the engineering lever is acquisition geometry rather than a better "
          "classifier.")


def introduction(stats: dict, rev: dict, cfg: dict) -> list[str]:
    """Dẫn bằng bài toán của máy phun, không bằng phê phán phương pháp."""
    t = throughput_requirement(cfg)
    return [
        "Weeds are among the principal constraints on sugarcane yield and quality "
        "(Suganthi et al. 2019), competing with the crop for nutrients, light, water and "
        "space during establishment and acting as hosts for pests and diseases. The "
        "dominant control is herbicide, applied uniformly across the field, and the cost "
        "of that uniformity is both economic and environmental (Donga and Eklo 2018). "
        "Site-specific application — treating only the locations where weeds actually "
        "occur — reduces the quantity applied without reducing efficacy, and it is the "
        "motivation for putting a camera and a detection model on a sprayer.",

        "That shifts the problem upstream, onto the detector, and it imposes two "
        "requirements that a laboratory accuracy figure does not express. The first is "
        "completeness: a weed patch the model does not find is a patch the implement "
        "drives past untreated, so what matters is the proportion of the weed present "
        "that is detected, not the proportion of detections that are correct. The second "
        "is throughput: detections must arrive fast enough that the implement does not "
        "travel past the target before the nozzle can act. At a working speed of "
        f"{t['sprayer_speed_kmh']:.0f} km h⁻¹, with successive frames required to advance "
        f"no more than {t['max_frame_advance_m']:.2f} m, the onboard hardware must supply "
        f"at least {t['fps_required_on_device']:.1f} frames per second.",

        "Deep learning object detectors are the natural candidates, and recent work in "
        "this crop has applied them to tasks closely related to the present one: stem "
        "base detection for intelligent harvesting (Li et al. 2026), detection of "
        "ratoon-emerged seedling hills for replanting (Mo et al. 2026), and smartphone "
        "diagnosis of foliar disease (Chen et al. 2026). For weed detection specifically, "
        "Sun et al. (2024) combined multi-scale convolution with a transformer backbone "
        "in sugarcane fields, Singh et al. (2024) performed pixel-level classification of "
        "weed patches from drone imagery, and Mini et al. (2020) applied Mask R-CNN to "
        "aerial sugarcane imagery. Reported accuracies in the wider weed-detection "
        "literature are consistently high, frequently above 98% (dos Santos Ferreira et "
        "al. 2017; Yu et al. 2019; Moazzam et al. 2021; Zhang et al. 2023), which would "
        "suggest the detection half of the problem is close to solved.",

        "Whether that impression survives the two requirements above has not been tested "
        "for sugarcane on close-range imagery of the kind a boom-mounted camera would "
        "produce. The question is practical rather than architectural: not which detector "
        "is most accurate in the abstract, but how much of the weed in a sugarcane field "
        "a current detector actually finds, what limits that figure, and whether any of "
        "the candidates could run at the rate the implement demands.",

        "This study addresses those questions on imagery collected under production "
        "conditions at a commercial farm in Tay Ninh province, Vietnam. It (i) builds and "
        "releases an annotated close-range dataset of sugarcane and its associated weed "
        "flora; (ii) trains and compares three detectors spanning the two-stage, "
        "single-stage dense and anchor-free real-time families — Faster R-CNN, RetinaNet "
        "and YOLOv10 — under a single protocol, at a single input resolution, with a "
        "single checkpoint rule and repeated training runs; (iii) reports detection "
        "completeness both by instance and by the ground area a sprayer would treat, "
        "together with the size dependence that governs it; and (iv) screens the three "
        "against the throughput the implement requires. Two further models trained on a "
        "polygon annotation of the same images are reported separately and are not ranked "
        "against the three.",
    ]


def results_blocks(cfg, stats, res, seedvar, boot, comp, rev, lk) -> list[str]:
    """Mục Results: chỉ số liệu, phần diễn giải rộng để sang Discussion."""
    primary, secondary = scope(cfg)
    order = sorted(primary, key=lambda k: -D.seed_mean(k, "ap50"))
    fm = rev["full_metrics"]
    B = []

    # --- Bảng 1: chất lượng phát hiện ---
    rows = []
    for k in order:
        cells = [MODEL_LABELS[k]]
        for metric, dec in (("ap", 1), ("ap50", 2), ("ap75", 1)):
            m_, sd_ = D.seed_mean(k, metric), D.seed_sd(k, metric)
            cells.append(f"{m_:.{dec}f} ± {sd_:.{dec}f}" if sd_ is not None
                         else f"{m_:.{dec}f}")
        lo, hi = boot["models"][k]["ap50"]["ci95"]
        cells.append(f"{lo:.1f}-{hi:.1f}")
        cells.append(str(len([e for e in seedvar[k]["runs"] if e.get("status") == "ok"])))
        rows.append(cells)
    B.append(
        "**Detection quality.** Average precision on the common test partition is given "
        "in Table 1. Ranked by AP50 the order was "
        + D._and_list([f"{MODEL_LABELS[k]} ({D.seed_mean(k, 'ap50'):.2f} ± "
                       f"{D.seed_sd(k, 'ap50'):.2f})" for k in order])
        + ". The stricter criteria do not reproduce that ordering: on AP averaged over "
          "intersection-over-union thresholds from 0.50 to 0.95 the values were "
        + D._and_list([f"{D.seed_mean(k, 'ap'):.1f} for {MODEL_LABELS[k]}" for k in order])
        + ", and on AP75 "
        + D._and_list([f"{D.seed_mean(k, 'ap75'):.1f}" for k in order])
        + ", so the architecture leading on AP50 does not lead once localisation is "
          "required to be more precise.")
    B.append(md_table(
        ["Model", "AP (mean ± SD)", "AP50 (mean ± SD)", "AP75 (mean ± SD)",
         "AP50 95% CI", "Runs"], rows,
        "**Table 1** Average precision of the three bounding-box detectors on the common "
        f"test partition of {stats['bbox']['test']['images']} images under the COCO "
        "protocol. Means and standard deviations are over each model's converged training "
        "runs; the confidence interval is a separate 95% percentile interval from "
        "resampling test images."))

    # --- Bảng 2: hai biến thể ---
    rows = []
    for k in order:
        w, c = fm[k]["weed"], fm[k]["sugarcane"]
        rows.append([
            MODEL_LABELS[k],
            f"{D.seed_mean(k, 'weed_precision'):.3f}",
            f"{D.seed_mean(k, 'weed_recall'):.3f}",
            f"{w['precision']['mean']:.3f} ± {w['precision']['sd']:.3f}",
            f"{w['recall']['mean']:.3f} ± {w['recall']['sd']:.3f}",
            f"{w['f1']['mean']:.3f} ± {w['f1']['sd']:.3f}",
            f"{c['recall']['mean']:.3f} ± {c['recall']['sd']:.3f}",
        ])
    f1s = {k: D.seed_mean(k, "f1") for k in primary}
    B.append(
        "**How much weed is found.** The confusion matrix over matched detections can be "
        "scored in two ways, and Table 2 gives both for the same predictions. Deleting "
        "the background row and column before computing the metrics — the convention most "
        "often quoted in this literature — counts only instances given the wrong one of "
        "the two classes, and on that basis all three detectors separated sugarcane from "
        f"weeds reliably, with two-class mean F1-scores from {min(f1s.values()):.3f} to "
        f"{max(f1s.values()):.3f}. Retaining the background row and column, so that an "
        "undetected instance counts as a false negative and a detection with no matching "
        "annotation as a false positive, gives for the weed class Precision of "
        + D._and_list([f"{fm[k]['weed']['precision']['mean']:.3f}" for k in order])
        + ", Recall of "
        + D._and_list([f"{fm[k]['weed']['recall']['mean']:.3f}" for k in order])
        + " and F1-score of "
        + D._and_list([f"{fm[k]['weed']['f1']['mean']:.3f}" for k in order])
        + " for "
        + D._and_list([MODEL_LABELS[k] for k in order])
        + " respectively. Expressed in the terms that matter to a sprayer: about half the "
          "weed annotated in these images was not found, and about half of what the models "
          "reported as weed had no annotated weed under it.")
    B.append(md_table(
        ["Model", "Precision (weed, bg-excl.)", "Recall (weed, bg-excl.)",
         "Precision (weed, full)", "Recall (weed, full)", "F1 (weed, full)",
         "Recall (sugarcane, full)"], rows,
        "**Table 2** The same predictions scored under both confusion-matrix conventions, "
        f"at a confidence threshold of {cfg['eval']['conf_threshold']:.2f} and an "
        f"intersection-over-union threshold of {cfg['eval']['iou_threshold']:.2f}. "
        "Background-excluded columns describe class discrimination among localised "
        "objects; full columns retain undetected instances and spurious detections. Values "
        "are means over each model's converged runs."))

    # --- Hình 2: ma trận nhầm lẫn, hình trung tâm của bài ---
    cm = {k: res[k]["confusion_matrix"] for k in order}
    k0 = order[0]
    tp0, fp_c0, fn_c0 = cm[k0][1][1], cm[k0][0][1], cm[k0][1][0]
    fp_b0, fn_b0 = cm[k0][2][1], cm[k0][1][2]
    f1 = lambda fp, fn: 2 * tp0 / (2 * tp0 + fp + fn)
    B.append(
        "**Where the difference comes from.** Fig. 2 shows the confusion matrices "
        "behind both columns of Table 2, with the band that the background-excluded "
        f"convention removes marked. Taking {MODEL_LABELS[k0]} as the worked example: of "
        f"the {tp0 + fn_c0 + fn_b0} annotated weed instances in the test partition, "
        f"{tp0} were detected and classified as weed, {fn_c0} were detected but called "
        f"sugarcane, and {fn_b0} produced no detection at all. In the other direction the "
        f"model placed {fp_b0} weed boxes where no instance was annotated and {fp_c0} on "
        "sugarcane. Scoring with the whole matrix gives a weed-class F1-score of "
        f"{f1(fp_c0 + fp_b0, fn_c0 + fn_b0):.2f}. Deleting the background row and column "
        f"first removes {fn_b0} false negatives and {fp_b0} false positives from the "
        f"arithmetic — the two largest off-diagonal entries — and the same predictions "
        f"then score {f1(fp_c0, fn_c0):.2f}.")
    B.append(
        "The deleted band is not a technicality. Its column holds the weed the implement "
        "would drive past untreated and its row holds the ground it would spray for "
        "nothing; between them they are the whole of the field performance. What survives "
        "deletion is the question of whether an object already found is labelled "
        "correctly, which these models answer almost perfectly and which no sprayer asks. "
        "The three matrices also show the behavioural difference reported earlier: the "
        "background row grows from "
        + D._and_list([f"{cm[k][2][0] + cm[k][2][1]} for {MODEL_LABELS[k]}" for k in order])
        + " spurious detections, in the same order as the number of boxes each model "
          "proposes.")
    B.append(figure_block(
        "fig6_confusion",
        "**Fig. 2** Confusion matrices on the test partition for the first training run "
        "(seed 0), rows ground truth and columns prediction. The dashed outline marks the "
        "background row and column, which the background-excluded convention deletes "
        "before computing its metrics; the background column counts annotated instances "
        "that produced no detection and the background row counts detections that match "
        "no annotation. Weed-class F1-scores under both conventions are printed beneath "
        "each matrix. Counts are from a single run, so they differ slightly from the "
        "seed-averaged values in Table 2"))

    # --- diện tích phủ: con số có nghĩa với máy phun ---
    ar = rev["area_recall"]
    B.append(
        "**What a sprayer would actually treat.** Counting instances is not the same as "
        "covering ground, and for an implement the second is the operative quantity. "
        "Weighting each annotated weed instance by its bounding-box area rather than "
        "counting it once gives a recall of "
        + D._and_list([f"{ar[k]['area_recall']['mean']:.3f} for {MODEL_LABELS[k]}"
                       for k in order])
        + ", against "
        + D._and_list([f"{ar[k]['instance_recall']['mean']:.3f}" for k in order])
        + " by instance count — "
        f"{min((ar[k]['area_recall']['mean'] - ar[k]['instance_recall']['mean']) * 100 for k in primary):.0f} "
        f"to {max((ar[k]['area_recall']['mean'] - ar[k]['instance_recall']['mean']) * 100 for k in primary):.0f} "
        "percentage points higher in every case, because the instances that go undetected "
        "are the small ones. Both figures are reported: the instance count measures how "
        "many weed patches the system is blind to, the area-weighted figure how much of "
        "the weed problem a sprayer driven by these detections would treat.")

    # --- Hình 2: PR / FROC / ROC ---
    cur = rev.get("curves")
    if cur:
        at30 = {k: min(cur["froc"][k], key=lambda d: abs(d["conf"] - 0.30))
                for k in order}
        mx = {k: max(cur["froc"][k], key=lambda d: d["recall"]) for k in order}
        B.append(
            "**The whole operating curve, not one point.** Every figure above is taken at "
            f"a single confidence threshold of {cfg['eval']['conf_threshold']:.2f}. "
            "Fig. 3 shows the complete trade-off. Fig. 3a gives the precision-recall "
            "curve of the weed class in the full variant: the three architectures trace "
            "closely spaced curves, differing in precision by a median of 0.075 over the "
            "recall range they share, and they differ mainly in where on that common "
            "trade-off each one sits rather than in how favourable the trade-off is. "
            "Fig. 3b restates it in the units a sprayer operator would use, recall "
            "against false weed detections per image. At the working threshold the models "
            "produce "
            + D._and_list([f"{at30[k]['fp_per_image']:.1f} for {MODEL_LABELS[k]}"
                           for k in order])
            + " spurious weed detections per image, and buying further recall is "
              "expensive: "
            + D._and_list([f"{MODEL_LABELS[k]} reaches {mx[k]['recall']:.2f} only at "
                           f"{mx[k]['fp_per_image']:.1f} false detections per image"
                           for k in order])
            + ".")
        B.append(
            "Fig. 3c asks whether the confidence score itself can separate correct weed "
            "detections from spurious ones. This is the one place in a detection problem "
            "where a receiver operating characteristic is well defined, because the "
            "negatives are a finite, identifiable set — the weed detections that match no "
            "annotated weed — rather than the unbounded set of image locations containing "
            "no object. The areas under the curve are "
            + D._and_list([f"{cur['roc'][k]['auc']:.2f} for {MODEL_LABELS[k]}"
                           for k in order])
            + ". Separation is moderate, not strong, and that is the reason the spurious "
              "detections cannot simply be thresholded away: raising the confidence "
              "threshold discards correct detections at almost the same rate as incorrect "
              "ones, which is why the F1-score in Supplementary Table S2 is nearly flat "
              "across the threshold range for two of the three models.")
        B.append(figure_block(
            "fig8_curves",
            "**Fig. 3** Operating characteristics of the three detectors for the weed "
            "class in the full variant, first training run (seed 0). **a** "
            "precision-recall curve, with the filled circle marking the working threshold "
            f"of {cfg['eval']['conf_threshold']:.2f}. **b** free-response operating "
            "characteristic: recall against the mean number of false weed detections per "
            "image, the form of operating characteristic that is defined for detection "
            "tasks. **c** receiver operating characteristic for the confidence score as a "
            "discriminator between weed detections that match an annotated weed and those "
            "that do not, with the area under each curve; the dotted diagonal is chance"))

    # --- cái gì giới hạn: kích thước, định vị, điều kiện ---
    B.extend(D.what_limits_detection(rev, lk, cfg))
    B.extend(D.resolution_control(rev, cfg))

    # --- Bảng 3: chi phí suy luận ---
    t = throughput_requirement(cfg)
    rows = []
    for k in order:
        c = comp[k]
        fps = clean_fps(k)[0]
        rows.append([MODEL_LABELS[k], f"{c['params'] / 1e6:.1f}",
                     f"{c['gmacs_median']:.1f}", f"{c['peak_vram_mb']:.0f}",
                     f"{fps:.1f}",
                     "yes" if fps >= t["fps_required_on_workstation"] else "no"])
    B.extend(D.inference_cost(res, cfg, comp, rev, cost_table=3))
    B.append(md_table(
        ["Model", "Parameters (M)", "GMACs", "Peak VRAM (MB)", "FPS", "Screen passed"],
        rows,
        "**Table 3** Inference cost on the workstation GPU and the outcome of the "
        f"throughput screen. The threshold is the {t['fps_required_on_workstation']:.0f} "
        "frames per second derived in the Materials and Methods. Compute is given as "
        "multiply-accumulate operations at each model's own input resolution and is "
        "therefore not directly comparable between rows."))

    # --- Bảng 4: mọi thang đo cạnh nhau ---
    mr = rev.get("metric_ranking")
    if mr:
        rows, prev = [], None
        for m in mr["metrics"]:
            grp = m["group"] if m["group"] != prev else ""
            prev = m["group"]
            cells = [grp, m["label"] + ("" if m["better"] == "hi" else " \u2193")]
            for k in order:
                v = m["values"][k]
                txt = f"{v:,.0f}" if abs(v) >= 100 else (
                    f"{v:.1f}" if abs(v) >= 10 else f"{v:.3f}")
                cells.append(f"**{txt}**" if m["rank"][k] == 1 else txt)
            rows.append(cells)
        B.append(
            "**Which model is best depends on the metric.** Table 4 places every measure "
            "used in this study side by side. "
            + (f"All {mr['n_models_winning']} architectures rank first on at least one of "
               f"the {mr['n_metrics']} measures" if mr["n_models_winning"] == len(order)
               else f"{mr['n_models_winning']} of the {len(order)} architectures rank "
                    f"first on at least one of the {mr['n_metrics']} measures")
            + ", and the two that lead most often do so on "
            + D._and_list([f"{mr['wins'][k]} measures for {MODEL_LABELS[k]}"
                           for k in sorted(order, key=lambda x: -mr['wins'][x])])
            + ". The split is not random. "
            + f"{MODEL_LABELS[max(order, key=lambda k: mr['wins'][k])]} leads wherever "
              "finding objects is what is rewarded, and the lighter architecture leads "
              "wherever precise placement, restraint in reporting, or computational cost "
              "is what is rewarded. This follows directly from how many detections each "
              "returns, reported in the Materials and Methods: medians of 14, 50 and 9 "
              "per image. A detector that proposes more finds more and is wrong more "
              "often; one that proposes less is right more often and misses more.")
        B.append(md_table(
            ["Metric family", "Measure"] + [MODEL_LABELS[k] for k in order], rows,
            "**Table 4** Every measure reported in this study, for the three "
            "bounding-box models. Bold marks the best value in each row; \u2193 marks a "
            "measure for which lower is better. Values are means over each model's "
            "converged training runs, except throughput and compute, which are "
            "architectural properties. The table is given because no single row of it "
            "determines the ranking of the others."))

    # --- thống kê ---
    B.extend(D.statistics_section(seedvar, boot, cfg, rev))

    # --- hai mô hình đa giác ---
    B.append(
        "**Models trained on the polygon convention.** For completeness, "
        + " and ".join(MODEL_LABELS[k] for k in secondary)
        + " were evaluated on the polygon test partition of "
        + f"{stats['polygon']['test']['images']} images under the same protocol, reaching "
          "AP50 of "
        + D._and_list([f"{res[k]['coco']['AP50']:.1f}" for k in secondary])
        + " respectively (Supplementary Table S10). Those values are computed against "
          "ground truth that resolves weed patches into constituent parts and describe a "
          "different task; each model was trained once, so no training variability is "
          "available for them.")
    return B


def discussion(cfg, stats, res, rev, lk) -> list[str]:
    """Thảo luận. Phê phán quy ước đặt Ở ĐÂY, làm lời giải thích."""
    prim = ["faster_rcnn", "retinanet", "yolov10"]
    fm = rev["full_metrics"]
    ar = rev["area_recall"]
    f1s = {k: D.seed_mean(k, "f1") for k in prim}
    ap50 = {k: D.seed_mean(k, "ap50") for k in prim}
    order = sorted(prim, key=lambda k: -(ap50[k] or 0))
    cm = {k: res[k]["confusion_matrix"] for k in prim}
    t = throughput_requirement(cfg)
    fast = [k for k in prim if clean_fps(k)[0] >= t["fps_required_on_workstation"]]
    out = []

    out.append(
        "**Why published figures look better than field performance.** The two sets of "
        "columns in Table 2 come from one set of predictions on one set of images and "
        "differ by roughly a factor of two. The difference is a reporting convention. "
        "When the background row and column of the confusion matrix are deleted, an "
        "annotated weed that produced no detection leaves the denominator with them, "
        "and so does a detection that matched no annotation. What remains measures one "
        "thing: given that an object has been localised, is it labelled correctly? "
        "That is a reasonable question and the answer here is encouraging, a two-class "
        f"mean F1-score between {min(f1s.values()):.2f} and {max(f1s.values()):.2f}. "
        "It is not the question a sprayer asks, and a figure of that magnitude quoted "
        "without stating which convention produced it will be read as though it were. "
        "On the same predictions the full variant gives weed-class F1-scores between "
        f"{min(fm[k]['weed']['f1']['mean'] for k in prim):.2f} and "
        f"{max(fm[k]['weed']['f1']['mean'] for k in prim):.2f}. The convention is "
        "widespread and its effect is large enough to change what a reader concludes "
        "about whether the problem is solved.")

    out.append(
        "**Comparison with detection results reported in sugarcane.** Published "
        "figures for this crop are mostly not commensurable with the present ones. "
        "Singh et al. (2024) report 90.5% accuracy for pixel-level classification of "
        "weed patches from drone imagery, a per-pixel figure for a segmentation task "
        "at a different acquisition scale; Mini et al. (2020) report an AP50 of 65.5% "
        "with Mask R-CNN against mask annotations on aerial imagery. By the same rule "
        "this study applies to its own two annotation conventions, neither can be "
        "placed on one scale with the AP50 of "
        + D._and_list([f"{ap50[k]:.1f}% for {MODEL_LABELS[k]}" for k in order])
        + " obtained here. The direction is nonetheless worth stating: close-range "
          "nadir acquisition at 140 cm did not by itself yield higher average "
          "precision than the published aerial result, which is evidence against the "
          "assumption that moving the camera closer resolves the detection problem.")

    out.append(
        "**Implications for sprayer design.** Three consequences follow for anyone "
        "building the implement. First, the binding constraint is detection, not "
        "classification: once a weed patch is localised its class is almost always "
        "assigned correctly, and the counts of sugarcane reported as weed ("
        + D._and_list([str(cm[k][0][1]) for k in order])
        + ") and weed reported as sugarcane ("
        + D._and_list([str(cm[k][1][0]) for k in order])
        + ") are negligible beside the numbers undetected; of those two, sugarcane "
          "reported as weed is the more harmful, because that plant is sprayed. "
          "Second, the shortfall has two halves that cost differently — weed left "
          "untreated, and herbicide applied where no weed was annotated — and a single "
          "accuracy figure hides the trade-off between them. Third, the engineering "
          "lever is acquisition geometry rather than architecture: recall falls "
          "monotonically with object size, and retraining the lightest model at the "
          "input resolution of the other two did not change its detection quality, so "
          "lowering the camera or tiling the frame, which change the pixels falling on "
          "a weed patch, are the indicated routes.")

    mr = rev.get("metric_ranking")
    if mr:
        wins = mr["wins"]
        top = sorted(prim, key=lambda k: -wins[k])
        out.append(
            "**The ranking is a property of the yardstick, not only of the model.** "
            f"Across the {mr['n_metrics']} measures collected in Table 4, each of the "
            "three architectures ranks first on at least one, and the two that lead most "
            "often do so on "
            + D._and_list([f"{wins[k]} measures for {MODEL_LABELS[k]}" for k in top])
            + ". A reader who takes AP50 as the criterion concludes that "
            f"{MODEL_LABELS[top[0]]} is the better detector; one who takes AP75 or "
            "weed-class precision concludes the opposite; one who takes weed-class "
            "F1-score in the full variant concludes that neither leads. All three "
            "readings are correct about their own measure, and none of them is correct "
            "about the field.")
        out.append(
            "The pattern behind the split is mechanical rather than mysterious, and it "
            "is visible in how many detections each model returns: medians of 14, 50 and "
            "9 per image for Faster R-CNN, RetinaNet and YOLOv10. A detector that "
            "proposes freely finds more of what is present and is wrong more often, so "
            "it leads on recall, on average recall and on the area-weighted measure, and "
            "trails on precision. A detector that proposes sparingly is right more often "
            "and misses more, so it leads on precision, on the background-excluded "
            "F1-score and on the stricter overlap threshold. Neither behaviour is better "
            "in the abstract; which is better depends on whether a weed left untreated "
            "or herbicide applied to bare ground is the more expensive error on a given "
            "farm. That is an agronomic and economic question, not a question about "
            "architectures, and it cannot be settled by reporting one number. The "
            "practical recommendation that follows is to state the operating point and "
            "the convention alongside any figure quoted for this task, and to choose the "
            "model against the cost structure of the intended operation rather than "
            "against a leaderboard.")

    out.append(
        "**Choosing between the three.** No architecture was best on every criterion, "
        "and the honest outcome is a stated trade-off. "
        + (f"{MODEL_LABELS[fast[0]]} was the only model to clear the throughput "
           "screen, and it also recovered the fewest weed instances and the smallest "
           "treated area; the two architectures that found more are, on this hardware, "
           "too slow at the assumed working speed. " if fast else
           "No model cleared the throughput screen. ")
        + "Where completeness governs, the heavier architectures are indicated and the "
          "throughput problem must be solved another way — a slower working speed, a "
          "wider field of view, or a dedicated accelerator. Where onboard throughput "
          "binds, the lighter architecture is the only candidate and the weed it "
          "misses must be accepted or handled by a second pass. The area-weighted "
          "figures are the ones to use in that calculation: "
        + D._and_list([f"{ar[k]['area_recall']['mean']:.2f} for {MODEL_LABELS[k]}"
                       for k in order])
        + " of the weed-covered ground would be treated.")

    # Đoạn "shortfall ... two halves" của limitations_v2 nay trùng với mục
    # Implications viết riêng ở trên, nên bỏ khỏi bản này. Bản RAE không có mục
    # Discussion riêng nên vẫn cần nó.
    lim = D.limitations_v2(res, json.loads((RESULTS / "seed_variance.json").read_text()),
                           stats, cfg, rev, lk)
    out.extend(x for x in lim if "two halves that should not be conflated" not in x)
    return out


def conclusion(cfg, stats, res, rev) -> list[str]:
    prim = ["faster_rcnn", "retinanet", "yolov10"]
    fm = rev["full_metrics"]
    ar = rev["area_recall"]
    ss = rev["size_strata"]["models"]
    f1s = {k: D.seed_mean(k, "f1") for k in prim}
    ap50 = {k: D.seed_mean(k, "ap50") for k in prim}
    order = sorted(prim, key=lambda k: -(ap50[k] or 0))
    t = throughput_requirement(cfg)
    fast = [k for k in prim if clean_fps(k)[0] >= t["fps_required_on_workstation"]]
    return [
        "Three deep learning detectors were trained on close-range field imagery of "
        "sugarcane and weeds from Tay Ninh province, Vietnam, and evaluated against the "
        "two requirements a site-specific sprayer imposes. All three assigned the correct "
        "class to almost every object they localised, with two-class mean F1-scores from "
        f"{min(f1s.values()):.2f} to {max(f1s.values()):.2f} under the convention commonly "
        "reported in this literature. Scored so that undetected weed and spurious "
        "detections are retained, the same predictions gave weed-class Precision of "
        f"{min(fm[k]['weed']['precision']['mean'] for k in prim):.2f} to "
        f"{max(fm[k]['weed']['precision']['mean'] for k in prim):.2f} and Recall of "
        f"{min(fm[k]['weed']['recall']['mean'] for k in prim):.2f} to "
        f"{max(fm[k]['weed']['recall']['mean'] for k in prim):.2f}; weighted by the ground "
        "area a sprayer would treat, recall was "
        f"{min(ar[k]['area_recall']['mean'] for k in prim):.2f} to "
        f"{max(ar[k]['area_recall']['mean'] for k in prim):.2f}.",

        "Detection completeness, not class discrimination, is therefore what limits this "
        "application, and it is governed by object size: recall rose from "
        f"{min(ss[k]['quartile']['Q1']['recall_mean'] for k in prim):.2f}-"
        f"{max(ss[k]['quartile']['Q1']['recall_mean'] for k in prim):.2f} in the smallest "
        "quartile of annotated weed instances to "
        f"{min(ss[k]['quartile']['Q4']['recall_mean'] for k in prim):.2f}-"
        f"{max(ss[k]['quartile']['Q4']['recall_mean'] for k in prim):.2f} in the largest. "
        "Retraining the lightest detector at four times the pixel count left its detection "
        "quality unchanged, so the remedy indicated is acquisition geometry — camera "
        "height or tiled inference — rather than a larger network input or a better "
        "classifier. "
        + (f"Of the three, only {MODEL_LABELS[fast[0]]} met the throughput the implement "
           f"requires at {t['sprayer_speed_kmh']:.0f} km h⁻¹, and it is also the model "
           "that treats the least ground, so the choice is a trade-off rather than a "
           "ranking. " if fast else "")
        + "Confirming feasibility on a working implement requires measurement on the "
          "target hardware, which this study did not have.",
    ]


# ───────────────────────────────── dựng bài ─────────────────────────────────

def build(cfg, stats, res, seedvar, boot, comp) -> str:
    D.use_seed_variance(seedvar)
    rev, lk = load_review()
    primary, _ = scope(cfg)
    tr = cfg["train"]

    # Hướng dẫn của tạp chí: trang bìa mang tiêu đề, thông tin tác giả, abstract,
    # từ khoá; lời cảm ơn cũng đặt ở trang bìa. Không có yêu cầu làm mù, nên tên
    # tác giả nằm trong chính tệp bản thảo (khác RAE).
    L = [f"# {TITLE}", "",
         AUTHORS, "",
         AFFIL, "",
         CORRESP, "",
         "*ORCID*: [TO BE COMPLETED BY THE AUTHORS — the 16-digit ORCID of each author, "
         "if available.]", "",
         "**Abstract** " + abstract(stats, rev, cfg), "",
         f"**Keywords** {KEYWORDS}", "",
         "**Acknowledgements** [TO BE COMPLETED BY THE AUTHORS — people, grants and "
         "funds; write the names of funding organisations in full. Delete this section "
         "if there is nothing to acknowledge.]", "",
         "## Introduction", ""]
    L += [p + "\n" for p in introduction(stats, rev, cfg)]

    L += ["## Materials and Methods", ""]
    meth = methods(cfg, stats, res, seedvar, boot, journal="sugartech")
    # Hình 1 (quy trình) đi ngay sau đoạn mô tả địa điểm
    blocks = []
    for p in meth:
        blocks.append(p)
        if p.startswith("**Study area and image dataset."):
            blocks.append(figure_block(
                "fig1_workflow",
                "**Fig. 1** Workflow of the study, from image acquisition and annotation "
                "through model training to evaluation. The two annotation conventions run "
                "in parallel to separate test partitions and are never ranked against "
                "each other"))
    L += [b + "\n" for b in blocks]

    L += ["## Results", ""]
    for b in results_blocks(cfg, stats, res, seedvar, boot, comp, rev, lk):
        L.append(b + "\n")
    for b in D.qualitative_results(res, fig="Fig. 4"):
        L.append(b + "\n")
    L.append(figure_block(
        "fig5_qualitative",
        "**Fig. 4** Detection results on the test set, first training run (seed 0), for "
        "Faster R-CNN (first column), RetinaNet (second column) and YOLOv10 (third "
        "column). **a**-**c** ground-truth annotation of the success case for each "
        "model; **d**-**f** the corresponding predictions; **g**-**i** ground-truth "
        "annotation of the failure case; **j**-**l** the corresponding predictions. "
        "Green boxes sugarcane, red boxes weed; predicted boxes carry the confidence "
        "score. For each model the success case is the test image at the median of "
        "per-image weed-class recall and the failure case the image at the tenth "
        "percentile, so neither is the most favourable nor the most unfavourable "
        "example available") + "\n")

    L += ["## Discussion", ""]
    for b in discussion(cfg, stats, res, rev, lk):
        L.append(b + "\n")

    L += ["## Conclusion", ""]
    for b in conclusion(cfg, stats, res, rev):
        L.append(b + "\n")

    L += [""] + [d + "\n" for d in DECLARATIONS]

    refs, bad = to_springer(REFERENCES)
    if bad:
        log(f"cảnh báo: {len(bad)} tài liệu giữ nguyên kiểu RAE — {bad[:2]}")
    refs = sorted(refs + EXTRA_REFS, key=str.lower)
    L += ["## References", ""] + [r + "\n" for r in refs]

    out = "\n".join(L)

    # Kiểm tại chỗ: bảng phải đánh số liên tục và không rơi mất.
    caps = [int(m.group(1)) for m in re.finditer(r"^\*\*Table (\d)\*\*", out, re.M)]
    n_tbl = out.count("\n|---")
    if caps != list(range(1, len(caps) + 1)):
        raise SystemExit(f"Đánh số bảng không liên tục: {caps}")
    if n_tbl != len(caps):
        raise SystemExit(f"{len(caps)} chú thích bảng nhưng {n_tbl} bảng thật")
    log(f"kiểm tra bảng: {len(caps)} bảng, đánh số liên tục 1–{len(caps)}")
    return out


def main():
    cfg = load_cfg()
    res = {k: json.loads((RESULTS / f"{k}.json").read_text()) for k in MODEL_LABELS}
    seedvar = json.loads((RESULTS / "seed_variance.json").read_text())
    boot = json.loads((RESULTS / "bootstrap.json").read_text())
    comp = json.loads((RESULTS / "complexity.json").read_text())
    stats = json.loads((RESULTS / "dataset_stats.json").read_text())

    md = build(cfg, stats, res, seedvar, boot, comp)
    p = OUT / "SugarTech_manuscript.md"
    p.write_text(md, encoding="utf-8")
    log(f"{len(md.split())} từ → {p}")
    # Sugar Tech: Times Roman 10 pt, giãn đôi, lề 2,5 cm, đánh số trang tự động.
    d = to_docx(p, double_spaced=True, font_pt=10, page_numbers=True)
    if d:
        final = OUT / "SugarTech_manuscript.docx"
        if d != final:
            d.replace(final)
        log(f"→ {final}")


if __name__ == "__main__":
    main()
