#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phụ lục trực tuyến của bài RAE.

    python3 src/build_supplementary.py

Tạp chí giới hạn bài nguyên tác ở 15 trang chuẩn KỂ CẢ bảng và hình, nhưng vòng
phản biện yêu cầu nhiều phân tích độ nhạy mà bài không thể chứa hết. Mục
"ix. Supplementary Material" của hướng dẫn tác giả cho phép đưa bảng và dữ liệu
bổ sung ra một tệp riêng, chỉ xuất bản ở bản điện tử, đánh số S1, S2, …

Nguyên tắc tạp chí nêu rõ: bản thảo chính phải ĐỨNG ĐỘC LẬP được. Vì vậy mỗi
bảng ở đây đều có một câu tóm tắt tương ứng trong bài, và phụ lục chỉ chứa chi
tiết đầy đủ chứ không chứa kết luận mới.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from build_manuscript import md_table, to_docx
from common import MODEL_LABELS, RAE, RESULTS, load_cfg, log, scope

OUT = RAE / "manuscript"
TITLE = "Supplementary file – for online publication only"


def main():
    cfg = load_cfg()
    primary, _ = scope(cfg)
    rev = json.loads((RESULTS / "review_analysis.json").read_text())
    comp = json.loads((RESULTS / "complexity.json").read_text())
    sv = json.loads((RESULTS / "seed_variance.json").read_text())
    order = sorted(primary, key=lambda k: -rev["coco_detail"]["models"][k]["AP50"]["mean"])

    L = [f"# {TITLE}", "",
         "Supplementary material for: *Detection completeness limits deep learning weed "
         "detection in field-grown sugarcane*.", "",
         "These tables give in full the sensitivity analyses summarised in the main text. "
         "They are referred to there as Supplementary Table S1 to S6. The main text stands "
         "alone without them.", ""]

    # ── S1: cấu hình huấn luyện và suy luận đầy đủ ──
    rows = []
    for k in order:
        spec = cfg["models"][k]
        rows.append([
            MODEL_LABELS[k],
            "Detectron2 v0.6" if spec["framework"] == "detectron2" else "Ultralytics v8.3",
            f"{spec['imgsz']} × {spec['imgsz']}",
            "COCO-pretrained",
            "100" if spec["framework"] == "detectron2" else "300",
            "0.50 (head), 0.70 (RPN)" if k == "faster_rcnn"
            else ("0.50" if spec["framework"] == "detectron2" else "none (end-to-end head)"),
            "final epoch",
        ])
    L.append(md_table(
        ["Model", "Framework", "Input size", "Initialisation",
         "Max detections per image", "NMS IoU threshold", "Checkpoint evaluated"], rows,
        "**Supplementary Table S1. Training and inference configuration of the three "
        "bounding-box models, in full. All were trained for 50 epochs with AdamW, an "
        "initial learning rate of 0.001, cosine annealing, weight decay 0.0001, a "
        "three-epoch linear warm-up and an effective batch of 16 realised by gradient "
        "accumulation. Predictions scoring below 0.05 were discarded before any metric was "
        "computed. A seed fixes the Python, NumPy and PyTorch generators, and therefore the "
        "initialisation of the detection heads, the data ordering and the sampling of the "
        "training-time horizontal flip; the data partition and the augmented images are "
        "identical across seeds.**"))

    # ── S2: quét ngưỡng tin cậy ──
    cs = rev["conf_sweep"]
    rows = [[f"{r['conf']:.2f}"] + [f"{cs[k][i]['recall']:.3f} / {cs[k][i]['precision']:.3f}"
                                    f" / {cs[k][i]['f1']:.3f}" for k in order]
            for i, r in enumerate(cs[order[0]])]
    L.append(md_table(["Confidence threshold"] + [f"{MODEL_LABELS[k]} (R / P / F1)" for k in order],
                      rows,
                      "**Supplementary Table S2. Weed-class Recall, Precision and F1-score "
                      "in the full variant of the confusion matrix, as a function of the "
                      "confidence threshold, at a fixed IoU threshold of 0.50. Values are "
                      "means over each model's converged training runs. Confidence scores "
                      "are not calibrated to a common scale between frameworks, so a single "
                      "threshold does not represent the same operating point for every "
                      "model.**"))

    # ── S3: quét ngưỡng IoU ──
    iou = rev["iou_sweep"]
    rows = [[f"{r['iou']:.2f}"] + [f"{iou[k][i]['recall']:.3f} / {iou[k][i]['precision']:.3f}"
                                   f" / {iou[k][i]['f1']:.3f}" for k in order]
            for i, r in enumerate(iou[order[0]])]
    L.append(md_table(["IoU threshold"] + [f"{MODEL_LABELS[k]} (R / P / F1)" for k in order],
                      rows,
                      "**Supplementary Table S3. The same metrics as a function of the IoU "
                      "threshold at which a prediction is accepted as matching an "
                      "annotation, at a fixed confidence threshold of 0.30. The difference "
                      "between the rows measures how much of the apparent shortfall in "
                      "Recall is imprecise localisation rather than failure to detect.**"))

    # ── S4: recall phân tầng kích thước ──
    ss = rev["size_strata"]
    rows = []
    for band, label in (("small", "COCO small (< 32 × 32 px)"),
                        ("medium", "COCO medium (32–96 px)"),
                        ("large", "COCO large (> 96 × 96 px)")):
        rows.append([label, str(ss["coco_band_counts"][band])]
                    + [f"{ss['models'][k]['coco'][band]['recall_mean']:.3f}" for k in order])
    q = ss["quartile_edges_px2"]
    edges = ["< {:,.0f}".format(q[0]), "{:,.0f}–{:,.0f}".format(q[0], q[1]),
             "{:,.0f}–{:,.0f}".format(q[1], q[2]), "> {:,.0f}".format(q[2])]
    for i, nm in enumerate(("Q1", "Q2", "Q3", "Q4")):
        rows.append([f"{nm} ({edges[i]} px²)", str(ss["quartile_band_counts"][nm])]
                    + [f"{ss['models'][k]['quartile'][nm]['recall_mean']:.3f}" for k in order])
    L.append(md_table(["Size band", "Weed instances"] + [MODEL_LABELS[k] for k in order], rows,
                      "**Supplementary Table S4. Weed-class Recall in the full variant, "
                      "stratified by the area of the annotated bounding box, at a "
                      "confidence threshold of 0.30 and an IoU threshold of 0.50. The upper "
                      "block uses the absolute size bands of the COCO protocol; the lower "
                      "block uses quartiles of the area distribution of this test "
                      "partition, which divide the data evenly and are therefore the more "
                      "reliable instrument here. Areas are measured on the 1280 px exported "
                      "frame.**"))

    # ── S5: COCO AP/AR theo dải kích thước ──
    cd = rev["coco_detail"]
    keys = [("AP", "AP"), ("AP50", "AP50"), ("AP75", "AP75"),
            ("AP_small", "AP small"), ("AP_medium", "AP medium"), ("AP_large", "AP large"),
            ("AR100", "AR@100"), ("AR_small", "AR small"),
            ("AR_medium", "AR medium"), ("AR_large", "AR large")]
    rows = [[lab] + [f"{cd['models'][k][key]['mean']:.1f}" for k in order] for key, lab in keys]
    L.append(md_table(["Metric"] + [MODEL_LABELS[k] for k in order], rows,
                      "**Supplementary Table S5. Average precision and average recall under "
                      "the COCO protocol, by size band, averaged over each model's "
                      "converged training runs. The ground-truth test partition contains "
                      f"{cd['gt_area_bands']['small']} small, "
                      f"{cd['gt_area_bands']['medium']} medium and "
                      f"{cd['gt_area_bands']['large']} large instances across both classes, "
                      "so the small-band figures rest on few objects and are reported for "
                      "the trend rather than as values.**"))

    # ── S6: độ nhạy với giao thức checkpoint ──
    cs6 = rev.get("checkpoint_sensitivity", {}).get("yolov10", {})
    if cs6.get("last") and cs6.get("best_on_val"):
        rows = [["Final epoch (protocol used here)",
                 f"{cs6['last']['mean']:.2f} ± {cs6['last']['sd']:.2f}", str(cs6['last']['n'])],
                ["Best epoch on the validation partition",
                 f"{cs6['best_on_val']['mean']:.2f} ± {cs6['best_on_val']['sd']:.2f}",
                 str(cs6['best_on_val']['n'])]]
        L.append(md_table(["Checkpoint rule", "AP50 (mean ± SD across seeds)", "Runs"], rows,
                          "**Supplementary Table S6. Sensitivity of YOLOv10 to the choice "
                          "of evaluated checkpoint. Detectron2 writes only a final "
                          "checkpoint, so the two Detectron2 models admit no equivalent "
                          "choice; taking each framework's default would therefore have "
                          f"given YOLOv10 an advantage of {cs6['delta_mean']:.1f} AP50 and "
                          f"inflated its between-seed standard deviation by a factor of "
                          f"{cs6['sd_ratio']:.0f}. The final-epoch rule is used throughout "
                          "the main text.**"))

    # ── S7: kiểm định theo seed, đầy đủ ba chỉ số × ba cặp ──
    rows = []
    for metric, lab in (("ap50", "AP50"),
                        ("weed_recall_full", "Weed Recall (full)"),
                        ("weed_f1_full", "Weed F1 (full)")):
        for key, v in rev["tests"][metric].items():
            a_, b_ = key.split("__")
            rows.append([lab, f"{MODEL_LABELS[a_]} − {MODEL_LABELS[b_]}",
                         f"{v['diff']:+.3f}", f"{v['p_raw']:.3f}", f"{v['p_holm']:.3f}",
                         f"{v['hedges_g']:+.2f} [{v['g_ci95'][0]:+.2f}, {v['g_ci95'][1]:+.2f}]"])
    L.append(md_table(["Metric", "Comparison", "Difference", "P (raw)", "P (Holm)",
                       "Hedges' g [95% CI]"], rows,
                      "**Supplementary Table S7. Welch's t-test on the seed-level values "
                      "of each metric, for all three pairwise comparisons, with P-values "
                      "adjusted across the three comparisons by the Holm–Bonferroni "
                      "procedure. These tests hold the test partition fixed and therefore "
                      "reflect training variability only; the two-level bootstrap reported "
                      "in the main text releases both sources at once and is the basis for "
                      "the paper's statements about whether architectures differ.**"))

    # ── S8: bootstrap hai tầng, đầy đủ ──
    tl = rev["two_level_bootstrap"]
    rows = []
    for key, v in tl["differences"].items():
        a_, b_ = key.split("__")
        for met, lab in (("recall", "Recall"), ("f1", "F1-score")):
            d = v[met]
            rows.append([f"Weed {lab}", f"{MODEL_LABELS[a_]} − {MODEL_LABELS[b_]}",
                         f"{d['mean']:+.3f}",
                         f"[{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]",
                         "yes" if d["excludes_zero"] else "no"])
    L.append(md_table(["Metric", "Comparison", "Difference", "95% interval",
                       "Excludes zero"], rows,
                      f"**Supplementary Table S8. Two-level bootstrap over {tl['n_boot']} "
                      f"iterations, resampling the {tl['n_images']} test images and the "
                      "converged training runs together and taking each difference within "
                      "an iteration. This is the interval the main text refers to wherever "
                      "it states that two architectures do or do not differ.**"))

    # ── S9: kết quả theo buổi chụp (chuyển từ bài chính để giữ giới hạn trang) ──
    lk = json.loads((RESULTS / "leakage_analysis.json").read_text())
    se = lk.get("sessions") or {}
    if se:
        rows = []
        for name, d in se.items():
            cells = [name, f"{d['n_train']} / {d['n_valid']} / {d['n_test']}"]
            for k in order:
                v = d["models"][k]
                cells.append(f"{v['recall']:.3f} / {v['precision']:.3f} / {v['f1']:.3f}")
            rows.append(cells)
        L.append(md_table(
            ["Capture session", "Source images (train / valid / test)"]
            + [f"{MODEL_LABELS[k]} (R / P / F1)" for k in order], rows,
            "**Supplementary Table S9. Weed-class Recall, Precision and F1-score in the "
            "full variant, computed separately on each capture session of the test "
            "partition, with the number of source images each session contributed to the "
            "three partitions. Values are means over each model's converged runs. The "
            "sessions differ simultaneously in plot, crop age and illumination, so the "
            "spread between them cannot be attributed to any single factor. Note that the "
            "random partitioning left the validation partition with no images from the "
            "first session.**"))

    # ── S10: hai mô hình huấn luyện trên nhãn đa giác ──
    res = {k: json.loads((RESULTS / f"{k}.json").read_text())
           for k in ("mask_rcnn", "yolov8_seg")
           if (RESULTS / f"{k}.json").exists()}
    if res:
        from common import clean_fps
        rows = []
        for k, d in res.items():
            f_ = d["metrics"]["filtered"]
            wb = d["metrics"]["with_background"]["per_class"]
            rows.append([MODEL_LABELS[k], f"{d['coco']['AP50']:.1f}",
                         f"{wb['weed']['recall']:.3f}", f"{f_['macro']['f1']:.3f}",
                         f"{f_['overall_accuracy']:.3f}", f"{clean_fps(k)[0]:.1f}"])
        L.append(md_table(
            ["Model", "AP50", "Weed recall (full)", "F1 (two-class mean)", "Accuracy",
             "FPS"], rows,
            "**Supplementary Table S10. The two polygon-trained models on the polygon test "
            "partition (50 images) under the same protocol. These values are not "
            "comparable with the main-text tables, which are computed on a different test "
            "partition under a different annotation convention, and each model was trained "
            "once only.**"))

    # ── hình hỗ trợ: đường học (ma trận nhầm lẫn ở bài chính) ──
    from build_manuscript import figure_block
    L.append(figure_block(
        "fig4_training",
        "Supplementary Figure S1. Training curves for the three bounding-box models, "
        "first training run (seed 0). Left axis: training loss; right axis, for YOLOv10: "
        "validation mAP@50. The abscissa is epochs for all three panels, the Detectron2 "
        "logs converted at 181 iterations per epoch; that framework computes no "
        "validation loss under this configuration. The training loss of all three is "
        "still falling at the final epoch, so none had reached a plateau, and the "
        "validation mAP@50 of YOLOv10 stops improving after roughly the tenth epoch. The "
        "break in the YOLOv10 curve marks two epochs at which the logged box-regression "
        "loss was non-finite; training recovered in the following epoch."))

    md = "\n\n".join(L) + "\n"
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="RAE",
                    help="tiền tố tên tệp; dùng SugarTech cho bản nộp Sugar Tech")
    stem = ap.parse_args().prefix
    p = OUT / f"{stem}_supplementary.md"
    p.write_text(md, encoding="utf-8")
    log(f"{len(md.split())} từ → {p}")
    d = to_docx(p)
    if d:
        final = OUT / f"{stem}_supplementary.docx"
        if d != final:
            d.replace(final)
        log(f"→ {final}")


if __name__ == "__main__":
    main()
