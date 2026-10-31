#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Các phân tích bổ sung do phản biện yêu cầu (vòng major revision).

    python3 src/review_analysis.py

Phản biện nêu rằng kết luận trung tâm của bài phụ thuộc vào điểm vận hành và
vào định nghĩa recall, mà bài chỉ báo cáo một lát cắt duy nhất (conf 0,30,
IoU 0,50, recall dạng full, không có precision dạng full). Script này tính lại
từ detection thô đã lưu trong results/dets/, cho MỌI lượt chạy đã hội tụ:

  R1  precision/recall/F1 dạng full cho cả hai lớp, trung bình theo seed ± SD.
      Bài chỉ báo cáo recall dạng full — thiếu precision là báo cáo chọn lọc,
      và phản biện tính đúng: F1 dạng full gần như hoà giữa ba mô hình.
  R2  quét ngưỡng tin cậy 0,05-0,90 ở IoU 0,50  → điểm vận hành có quyết định
      thứ hạng không.
  R3  quét ngưỡng IoU 0,10/0,30/0,50 ở conf 0,30 → "bỏ sót" có phải chỉ là
      lệch hộp không.
  R4  đường PR cho lớp cỏ (dựng từ chính detection, không qua pycocotools).
  R5  COCO AR và AP theo kích thước (AP_S/M/L), cùng số instance mỗi dải — để
      biết dải nào thật sự có dữ liệu.
  R6  recall theo DIỆN TÍCH (tỷ lệ diện tích cỏ được phủ bởi hộp phát hiện),
      vì bài diễn giải recall theo instance thành "lượng cỏ phun sót".
  R7  recall phân tầng theo kích thước đối tượng → kiểm chứng giả thuyết
      "vật thể nhỏ" thay vì chỉ so trung vị.
  R8  Holm-Bonferroni cho ba so sánh cặp, và kiểm định Welch cho recall.
  R9  bootstrap hai tầng (ảnh × seed) → khoảng tin cậy thật sự gộp hai nguồn
      biến thiên, thay cho công thức "căn tổng hai phương sai" đã nêu mà
      không dùng.

Kết quả ghi ra results/review_analysis.json.
"""

from __future__ import annotations

import collections
import itertools
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import MODEL_LABELS, RESULTS, load_cfg, log
from evaluate import load_test_set
from metrics import box_iou_batch, _drop_extra_matches

CONF_GRID = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
IOU_GRID = [0.10, 0.30, 0.50]
PRIMARY = ["faster_rcnn", "retinanet", "yolov10"]
WEED, CANE = 1, 0
N_BOOT = 1000
RNG = np.random.default_rng(20260301)


# ─────────────────────────────── ghép hộp ────────────────────────────────────

def match_image(gt_boxes, gt_cls, det_boxes, det_cls, det_conf, conf, iou_thr):
    """Ghép như metrics.confusion_matrix_image nhưng trả về chi tiết từng nhãn.

    Trả (hit, pred_of_true, matched_det): hit[i] = nhãn thật i có được ghép
    không; pred_of_true[i] = lớp mà dự đoán ghép được gán cho nó (-1 nếu không
    ghép); matched_det = mảng bool trên dự đoán.
    """
    keep = det_conf > conf
    db, dc = det_boxes[keep], det_cls[keep]
    hit = np.zeros(len(gt_cls), dtype=bool)
    pred_of_true = np.full(len(gt_cls), -1, dtype=int)
    used_det = np.zeros(len(dc), dtype=bool)
    if len(gt_cls) and len(dc):
        iou = box_iou_batch(gt_boxes, db)
        h = np.asarray(iou > iou_thr).nonzero()
        if h[0].shape[0]:
            mt = _drop_extra_matches(np.stack((h[0], h[1], iou[h]), axis=1))
            ti, di = mt[:, 0].astype(int), mt[:, 1].astype(int)
            hit[ti] = True
            pred_of_true[ti] = dc[di]
            used_det[di] = True
    return hit, pred_of_true, dc, used_det


def counts_full(items, dets_by_img, conf, iou_thr):
    """TP/FP/FN dạng full (giữ hàng và cột background) cho từng lớp, theo ảnh.

    Trả về mảng (n_images, n_class, 3) để bootstrap theo ảnh dùng lại được.
    """
    out = np.zeros((len(items), 2, 3), dtype=np.int64)
    for k, (meta, _path, gb, gc) in enumerate(items):
        d = dets_by_img.get(meta["id"], ([], [], []))
        db = np.array(d[0], float).reshape(-1, 4)
        dc = np.array(d[1], int)
        ds = np.array(d[2], float)
        hit, pred_of_true, kept_cls, used = match_image(gb, gc, db, dc, ds, conf, iou_thr)
        for c in (CANE, WEED):
            is_c = gc == c
            tp = int(((pred_of_true == c) & is_c).sum())
            # FN: nhãn thật lớp c không được gán đúng lớp c (sót hẳn hoặc gán sai lớp)
            fn = int(is_c.sum()) - tp
            # FP: dự đoán lớp c không ghép được, cộng dự đoán lớp c ghép vào lớp khác
            fp_bg = int(((kept_cls == c) & (~used)).sum())
            fp_cls = int(((pred_of_true == c) & (~is_c)).sum())
            out[k, c] = (tp, fp_bg + fp_cls, fn)
    return out


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


# ───────────────────────────── nạp dữ liệu ──────────────────────────────────

def load_runs(cfg):
    """Mọi lượt chạy đã hội tụ của ba mô hình hộp, kèm detection thô."""
    sv = json.loads((RESULTS / "seed_variance.json").read_text())
    runs = collections.defaultdict(list)
    for m in PRIMARY:
        for r in sv[m]["runs"]:
            if r.get("status") != "ok":
                continue
            seed = r["seed"]
            stem = m if seed == 0 else f"{m}_seed{seed}"
            f = RESULTS / "dets" / f"{stem}.json"
            if not f.exists():
                log(f"thiếu detection thô: {f.name} — bỏ qua seed {seed} của {m}")
                continue
            det = json.loads(f.read_text())["detections"]
            by_img = collections.defaultdict(lambda: ([], [], []))
            for d in det:
                x, y, w, h = d["bbox"]
                t = by_img[d["image_id"]]
                t[0].append([x, y, x + w, y + h])
                t[1].append(d["category_id"])
                t[2].append(d["score"])
            runs[m].append({"seed": seed, "stem": stem, "by_img": dict(by_img)})
    return runs


# ──────────────────────────── R1-R3: quét ──────────────────────────────────

def r1_full_metrics(items, runs):
    """Precision/Recall/F1 dạng full ở điểm vận hành của bài (0,30 / 0,50)."""
    out = {}
    for m, rs in runs.items():
        per_seed = {c: {"precision": [], "recall": [], "f1": []} for c in ("weed", "sugarcane")}
        for r in rs:
            cnt = counts_full(items, r["by_img"], 0.30, 0.50).sum(axis=0)
            for name, ci in (("sugarcane", CANE), ("weed", WEED)):
                p, rec, f = prf(*cnt[ci])
                per_seed[name]["precision"].append(p)
                per_seed[name]["recall"].append(rec)
                per_seed[name]["f1"].append(f)
        out[m] = {
            c: {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                    "n": len(v), "values": [float(x) for x in v]}
                for k, v in d.items()}
            for c, d in per_seed.items()
        }
    return out


def r2_conf_sweep(items, runs):
    """Thứ hạng có đổi theo ngưỡng tin cậy không."""
    out = {}
    for m, rs in runs.items():
        rows = []
        for conf in CONF_GRID:
            vals = {"precision": [], "recall": [], "f1": []}
            for r in rs:
                cnt = counts_full(items, r["by_img"], conf, 0.50).sum(axis=0)
                p, rec, f = prf(*cnt[WEED])
                vals["precision"].append(p); vals["recall"].append(rec); vals["f1"].append(f)
            rows.append({"conf": conf,
                         **{k: float(np.mean(v)) for k, v in vals.items()}})
        out[m] = rows
    return out


def r3_iou_sweep(items, runs):
    """Bỏ sót có phải chỉ là lệch hộp không: nới IoU xuống 0,30 và 0,10."""
    out = {}
    for m, rs in runs.items():
        rows = []
        for iou in IOU_GRID:
            vals = {"precision": [], "recall": [], "f1": []}
            for r in rs:
                cnt = counts_full(items, r["by_img"], 0.30, iou).sum(axis=0)
                p, rec, f = prf(*cnt[WEED])
                vals["precision"].append(p); vals["recall"].append(rec); vals["f1"].append(f)
            rows.append({"iou": iou, **{k: float(np.mean(v)) for k, v in vals.items()}})
        out[m] = rows
    return out


def r4_pr_curve(items, runs):
    """Đường PR cho lớp cỏ, lấy seed 0 để hình vẽ được một đường mỗi mô hình."""
    out = {}
    grid = np.round(np.arange(0.05, 0.96, 0.025), 4)
    for m, rs in runs.items():
        r = next((x for x in rs if x["seed"] == 0), rs[0])
        pts = []
        for conf in grid:
            cnt = counts_full(items, r["by_img"], float(conf), 0.50).sum(axis=0)
            p, rec, f = prf(*cnt[WEED])
            pts.append({"conf": float(conf), "precision": p, "recall": rec, "f1": f})
        out[m] = {"seed": r["seed"], "points": pts}
    return out


# ─────────────────── R6-R7: diện tích và phân tầng kích thước ───────────────

def _gt_weed_areas(items):
    """Diện tích hộp cỏ (px²) và diện tích khung, theo ảnh."""
    areas, frames = [], []
    for meta, _p, gb, gc in items:
        for b, c in zip(gb, gc):
            if c == WEED:
                areas.append(float((b[2] - b[0]) * (b[3] - b[1])))
                frames.append(float(meta["width"] * meta["height"]))
    return np.array(areas), np.array(frames)


def r6_area_recall(items, runs):
    """Recall theo diện tích: phần diện tích cỏ nằm trong hộp phát hiện được.

    Bài viết "how much weed cover goes untreated" nhưng chỉ đo recall theo
    instance. Vì instance bị sót nhỏ hơn hẳn, hai con số lệch nhau nhiều.
    """
    out = {}
    for m, rs in runs.items():
        inst, area = [], []
        for r in rs:
            num_a = den_a = 0.0
            num_i = den_i = 0
            for meta, _p, gb, gc in items:
                d = r["by_img"].get(meta["id"], ([], [], []))
                db = np.array(d[0], float).reshape(-1, 4)
                dc = np.array(d[1], int); ds = np.array(d[2], float)
                hit, pred_of_true, _kc, _u = match_image(gb, gc, db, dc, ds, 0.30, 0.50)
                for b, c, ok in zip(gb, gc, hit):
                    if c != WEED:
                        continue
                    a = float((b[2] - b[0]) * (b[3] - b[1]))
                    den_a += a; den_i += 1
                    if ok:
                        num_a += a; num_i += 1
            inst.append(num_i / den_i if den_i else 0.0)
            area.append(num_a / den_a if den_a else 0.0)
        out[m] = {"instance_recall": {"mean": float(np.mean(inst)),
                                      "sd": float(np.std(inst, ddof=1)) if len(inst) > 1 else None},
                  "area_recall": {"mean": float(np.mean(area)),
                                  "sd": float(np.std(area, ddof=1)) if len(area) > 1 else None},
                  "n_runs": len(inst)}
    return out


def r7_size_strata(items, runs):
    """Recall phân tầng theo diện tích hộp cỏ.

    Hai cách chia, vì phản biện nói đúng rằng dải COCO được định nghĩa cho ảnh
    ~640 px còn ảnh ở đây là 1280 px:
      - dải COCO tuyệt đối: nhỏ <32², vừa 32²-96², lớn >96²
      - tứ phân vị diện tích của chính bộ dữ liệu này
    """
    areas, frames = _gt_weed_areas(items)
    qs = np.quantile(areas, [0.25, 0.50, 0.75])
    coco_edges = [0, 32 ** 2, 96 ** 2, np.inf]
    quart_edges = [0, *qs, np.inf]

    def band(a, edges):
        for i in range(len(edges) - 1):
            if edges[i] <= a < edges[i + 1]:
                return i
        return len(edges) - 2

    out = {"quartile_edges_px2": [float(x) for x in qs],
           "n_weed_instances": int(len(areas)),
           "coco_band_counts": {}, "quartile_band_counts": {}, "models": {}}
    cnt_c = collections.Counter(band(a, coco_edges) for a in areas)
    cnt_q = collections.Counter(band(a, quart_edges) for a in areas)
    out["coco_band_counts"] = {["small", "medium", "large"][i]: int(cnt_c.get(i, 0)) for i in range(3)}
    out["quartile_band_counts"] = {f"Q{i+1}": int(cnt_q.get(i, 0)) for i in range(4)}

    for m, rs in runs.items():
        acc_c = np.zeros((len(rs), 3, 2)); acc_q = np.zeros((len(rs), 4, 2))
        for j, r in enumerate(rs):
            for meta, _p, gb, gc in items:
                d = r["by_img"].get(meta["id"], ([], [], []))
                db = np.array(d[0], float).reshape(-1, 4)
                dc = np.array(d[1], int); ds = np.array(d[2], float)
                hit, _pot, _kc, _u = match_image(gb, gc, db, dc, ds, 0.30, 0.50)
                for b, c, ok in zip(gb, gc, hit):
                    if c != WEED:
                        continue
                    a = float((b[2] - b[0]) * (b[3] - b[1]))
                    for acc, edges in ((acc_c, coco_edges), (acc_q, quart_edges)):
                        i = band(a, edges)
                        acc[j, i, 1] += 1
                        if ok:
                            acc[j, i, 0] += 1
        def summarise(acc, names):
            res = {}
            for i, nm in enumerate(names):
                rec = [acc[j, i, 0] / acc[j, i, 1] if acc[j, i, 1] else float("nan")
                       for j in range(len(rs))]
                rec = [x for x in rec if not np.isnan(x)]
                res[nm] = {"recall_mean": float(np.mean(rec)) if rec else None,
                           "recall_sd": float(np.std(rec, ddof=1)) if len(rec) > 1 else None,
                           "n_instances": int(acc[0, i, 1])}
            return res
        out["models"][m] = {"coco": summarise(acc_c, ["small", "medium", "large"]),
                            "quartile": summarise(acc_q, ["Q1", "Q2", "Q3", "Q4"])}
    return out


# ──────────────────────── R8: hiệu chỉnh đa so sánh ────────────────────────

def holm(pvals: dict) -> dict:
    """Holm-Bonferroni. Trả P đã hiệu chỉnh, giữ tính đơn điệu."""
    items_ = sorted(pvals.items(), key=lambda kv: kv[1])
    k = len(items_)
    adj, running = {}, 0.0
    for i, (key, p) in enumerate(items_):
        val = min(1.0, (k - i) * p)
        running = max(running, val)
        adj[key] = running
    return adj


def r8_tests(runs, r1):
    from scipy import stats as st
    sv = json.loads((RESULTS / "seed_variance.json").read_text())
    out = {}
    for metric, getter in (
        ("ap50", lambda m: [r["ap50"] for r in sv[m]["runs"] if r.get("status") == "ok"]),
        ("weed_recall_full", lambda m: r1[m]["weed"]["recall"]["values"]),
        ("weed_f1_full", lambda m: r1[m]["weed"]["f1"]["values"]),
    ):
        raw, detail = {}, {}
        for a, b in itertools.combinations(PRIMARY, 2):
            va, vb = getter(a), getter(b)
            t, p = st.ttest_ind(va, vb, equal_var=False)
            key = f"{a}__{b}"
            raw[key] = float(p)
            # Hedges' g với hiệu chỉnh mẫu nhỏ, kèm CI xấp xỉ
            na, nb = len(va), len(vb)
            sa, sb = np.var(va, ddof=1), np.var(vb, ddof=1)
            sp = np.sqrt(((na - 1) * sa + (nb - 1) * sb) / (na + nb - 2))
            d = (np.mean(va) - np.mean(vb)) / sp if sp else float("nan")
            g = d * (1 - 3 / (4 * (na + nb) - 9))
            se_g = np.sqrt((na + nb) / (na * nb) + g ** 2 / (2 * (na + nb - 3.94)))
            detail[key] = {"p_raw": float(p), "t": float(t),
                           "diff": float(np.mean(va) - np.mean(vb)),
                           "hedges_g": float(g),
                           "g_ci95": [float(g - 1.96 * se_g), float(g + 1.96 * se_g)],
                           "n": [na, nb]}
        adj = holm(raw)
        for key in detail:
            detail[key]["p_holm"] = float(adj[key])
            detail[key]["sig_raw"] = bool(detail[key]["p_raw"] < 0.05)
            detail[key]["sig_holm"] = bool(adj[key] < 0.05)
        out[metric] = detail
    return out


# ───────────────────── R9: bootstrap hai tầng (ảnh × seed) ─────────────────

def r9_two_level(items, runs):
    """Khoảng tin cậy gộp hai nguồn: lấy mẫu lại ảnh VÀ lấy mẫu lại seed.

    Mỗi vòng: rút lại danh sách ảnh (có hoàn lại) và rút lại một seed (có hoàn
    lại), rồi tính chỉ số trên tổ hợp đó. Đây là cách thực sự gộp hai nguồn
    biến thiên, thay cho công thức "căn tổng hai phương sai" mà bài nêu nhưng
    không dùng ở đâu.
    """
    # đếm trước theo ảnh cho mỗi seed, để vòng bootstrap chỉ còn là cộng mảng
    pre = {}
    for m, rs in runs.items():
        pre[m] = np.stack([counts_full(items, r["by_img"], 0.30, 0.50) for r in rs])
    n_img = len(items)
    out = {"n_boot": N_BOOT, "n_images": n_img, "models": {}, "differences": {}}

    draws = {}
    for m, arr in pre.items():
        n_seed = arr.shape[0]
        rec = np.empty(N_BOOT); f1 = np.empty(N_BOOT); prec = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = RNG.integers(0, n_img, n_img)
            s = RNG.integers(0, n_seed)
            tp, fp, fn = arr[s, idx, WEED].sum(axis=0)
            p, r_, f_ = prf(tp, fp, fn)
            prec[b], rec[b], f1[b] = p, r_, f_
        draws[m] = {"precision": prec, "recall": rec, "f1": f1}
        out["models"][m] = {
            k: {"mean": float(v.mean()),
                "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
            for k, v in draws[m].items()}

    # hiệu giữa các mô hình: ghép cặp trên cùng mẫu ảnh
    for a, b in itertools.combinations(PRIMARY, 2):
        if a not in pre or b not in pre:
            continue
        aa, bb = pre[a], pre[b]
        d_rec = np.empty(N_BOOT); d_f1 = np.empty(N_BOOT)
        for i in range(N_BOOT):
            idx = RNG.integers(0, n_img, n_img)
            sa = RNG.integers(0, aa.shape[0]); sb = RNG.integers(0, bb.shape[0])
            _p1, r1_, f1_ = prf(*aa[sa, idx, WEED].sum(axis=0))
            _p2, r2_, f2_ = prf(*bb[sb, idx, WEED].sum(axis=0))
            d_rec[i] = r1_ - r2_; d_f1[i] = f1_ - f2_
        out["differences"][f"{a}__{b}"] = {
            m_: {"mean": float(v.mean()),
                 "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                 "excludes_zero": bool(np.percentile(v, 2.5) > 0 or np.percentile(v, 97.5) < 0)}
            for m_, v in (("recall", d_rec), ("f1", d_f1))}
    return out


# ───────────────────────────── R5: COCO AR/AP_S ────────────────────────────

def r5_coco_detail(cfg, runs):
    """AR và AP theo dải kích thước, kèm số instance thật có trong mỗi dải."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from common import coco_json
    from common import remap_categories
    import contextlib, io

    gt_raw = remap_categories(cfg, json.loads(coco_json(cfg, "bbox", "test").read_text()))
    tmp = RESULTS / "_tmp_gt.json"
    tmp.write_text(json.dumps(gt_raw))
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(tmp))
    tmp.unlink()

    areas = np.array([a["bbox"][2] * a["bbox"][3] for a in gt_raw["annotations"]])
    bands = {"small": int((areas < 32 ** 2).sum()),
             "medium": int(((areas >= 32 ** 2) & (areas < 96 ** 2)).sum()),
             "large": int((areas >= 96 ** 2).sum())}

    keys = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
            "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]
    out = {"gt_area_bands": bands, "models": {}}
    for m, rs in runs.items():
        per_seed = collections.defaultdict(list)
        for r in rs:
            det = json.loads((RESULTS / "dets" / f"{r['stem']}.json").read_text())["detections"]
            if not det:
                continue
            with contextlib.redirect_stdout(io.StringIO()):
                dt = coco_gt.loadRes([dict(d) for d in det])
                e = COCOeval(coco_gt, dt, "bbox")
                e.evaluate(); e.accumulate(); e.summarize()
            for k, v in zip(keys, e.stats):
                per_seed[k].append(float(v) * 100)
        out["models"][m] = {k: {"mean": float(np.mean(v)),
                                "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None}
                            for k, v in per_seed.items()}
    return out


# ───────────────────────────────── main ────────────────────────────────────

def r10_checkpoint_sensitivity() -> dict:
    """AP50 khi chấm checkpoint cuối so với checkpoint tốt nhất trên validation.

    Chỉ áp dụng cho nhánh Ultralytics, vì Detectron2 không lưu checkpoint theo
    validation. Các tệp *_bestckpt*.json là bản lưu trước khi chuyển sang giao
    thức epoch cuối.
    """
    import re
    out = {}
    for model in ("yolov10",):
        got = {}
        for label, pat in (("last", "{m}{s}.json"), ("best_on_val", "{m}_bestckpt{s}.json")):
            vals = []
            for f in sorted((RESULTS).glob(pat.format(m=model, s="*"))):
                if label == "last" and "_bestckpt" in f.name:
                    continue
                if not re.fullmatch(rf"{model}(_bestckpt)?(_seed\d+)?\.json", f.name):
                    continue
                d = json.loads(f.read_text())
                ap = d.get("coco", {}).get("AP50")
                if ap is not None:
                    vals.append(float(ap))
            if vals:
                got[label] = {"mean": float(np.mean(vals)),
                              "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                              "n": len(vals)}
        if len(got) == 2:
            got["delta_mean"] = got["best_on_val"]["mean"] - got["last"]["mean"]
            got["sd_ratio"] = (got["best_on_val"]["sd"] / got["last"]["sd"]
                               if got["last"]["sd"] else None)
        out[model] = got
    return out


def r11_training_anomalies() -> dict:
    """Epoch có loss không hữu hạn trong log Ultralytics.

    Bài báo cáo khá kỹ việc RetinaNet phân kỳ, nên để cân xứng cũng phải nêu
    hiện tượng nhẹ hơn ở YOLOv10: một vài epoch ghi train/box_loss = inf rồi
    lượt chạy tự phục hồi và hoàn tất. Khác hẳn phân kỳ (lượt chạy hỏng hẳn),
    nhưng vẫn là bất ổn số học và người đọc có quyền biết.
    """
    import csv
    from common import RUNS
    out = {}
    for d in sorted(RUNS.glob("yolo*")):
        f = d / "results.csv"
        if not f.exists():
            continue
        rows = list(csv.DictReader(f.open()))
        if not rows:
            continue
        keys = {k.strip(): k for k in rows[0]}
        cols = [k for k in keys if k.startswith("train/") and k.endswith("loss")]
        bad = {}
        for r in rows:
            for c in cols:
                v = r[keys[c]].strip().lower()
                if v in ("inf", "-inf", "nan"):
                    bad.setdefault(c, []).append(int(float(r[keys["epoch"]])))
        out[d.name] = {"n_epochs": len(rows), "non_finite_epochs": bad}
    n_runs = sum(1 for k, v in out.items()
                 if k.startswith("yolov10") and "img" not in k)
    n_bad = sum(1 for k, v in out.items()
                if k.startswith("yolov10") and "img" not in k and v["non_finite_epochs"])
    out["_summary"] = {"yolov10_runs": n_runs, "yolov10_runs_with_non_finite_epoch": n_bad}
    return out


def r12_resolution_control() -> dict:
    """YOLOv10 huấn luyện lại ở 1024 px: confound độ phân giải có thật không?

    Phản biện lập luận rằng YOLOv10 chạy 640 px trong khi hai mô hình kia chạy
    1024 px, nên "đánh đổi tốc độ - recall" có thể chỉ là hiệu ứng kích thước
    đầu vào. Cách duy nhất để biết là huấn luyện lại YOLOv10 ở 1024 px và so.
    """
    import re
    out = {}
    for label, pat in (("imgsz_640", r"yolov10(_seed\d+)?\.json"),
                       ("imgsz_1024", r"yolov10_img1024_seed\d+\.json")):
        rows = []
        for f in sorted(RESULTS.glob("yolov10*.json")):
            if not re.fullmatch(pat, f.name):
                continue
            d = json.loads(f.read_text())
            cm = d["confusion_matrix"]
            tp = cm[WEED][WEED]
            fp = sum(cm[r][WEED] for r in range(3) if r != WEED)
            fn = sum(cm[WEED][c] for c in range(3) if c != WEED)
            p_, r_, f_ = prf(tp, fp, fn)
            rows.append({"ap50": d["coco"]["AP50"], "ap": d["coco"]["AP"],
                         "recall": r_, "precision": p_, "f1": f_})
        if rows:
            out[label] = {k: {"mean": float(np.mean([r[k] for r in rows])),
                              "sd": (float(np.std([r[k] for r in rows], ddof=1))
                                     if len(rows) > 1 else None)}
                          for k in rows[0]}
            out[label]["n"] = len(rows)
    f = RESULTS / "edge_yolov10_img1024.json"
    if f.exists():
        d = json.loads(f.read_text())
        out["fps_1024"] = {"mean": d["fps"]["mean"],
                           "latency_ms": d["latency_ms"]["mean"],
                           "n_images": d["n_images"], "repeats": d["repeats"]}
    if "imgsz_640" in out and "imgsz_1024" in out:
        out["delta_1024_minus_640"] = {
            k: out["imgsz_1024"][k]["mean"] - out["imgsz_640"][k]["mean"]
            for k in ("ap50", "ap", "recall", "precision", "f1")}
    return out


def r13_metric_ranking(rev_so_far: dict) -> dict:
    """Xếp hạng ba mô hình trên MỌI thang đo đã dùng trong bài.

    Bài đo mười một đại lượng khác nhau nhưng chưa bao giờ đặt chúng cạnh nhau.
    Khi đặt cạnh nhau thì thấy điều đáng nói: mỗi mô hình đứng nhất ở ít nhất
    một thang, và việc chọn thang nào quyết định hoàn toàn thứ hạng. Đó chính
    là luận điểm của bài về quy ước báo cáo, lần này đo được thay vì nói suông.

    Mỗi thang kèm hướng tốt: "hi" là cao hơn tốt hơn, "lo" là thấp hơn tốt hơn.
    """
    import numpy as np
    from common import clean_fps

    sv = json.loads((RESULTS / "seed_variance.json").read_text())
    comp = json.loads((RESULTS / "complexity.json").read_text())
    fm = rev_so_far["full_metrics"]
    ar = rev_so_far["area_recall"]
    cd = rev_so_far["coco_detail"]["models"]

    def seed_mean(k, m):
        v = [e[m] for e in sv[k]["runs"] if e.get("status") == "ok" and m in e]
        return float(np.mean(v)) if v else None

    # (nhãn, nhóm, hướng tốt, hàm lấy giá trị)
    rows = [
        ("AP",                 "COCO average precision", "hi", lambda k: seed_mean(k, "ap")),
        ("AP50",               "COCO average precision", "hi", lambda k: seed_mean(k, "ap50")),
        ("AP75",               "COCO average precision", "hi", lambda k: seed_mean(k, "ap75")),
        ("AR@100",             "COCO average recall",    "hi", lambda k: cd[k]["AR100"]["mean"]),
        ("F1, two-class mean", "Background-excluded",    "hi", lambda k: seed_mean(k, "f1")),
        ("Weed precision",     "Full variant",           "hi",
         lambda k: fm[k]["weed"]["precision"]["mean"]),
        ("Weed recall",        "Full variant",           "hi",
         lambda k: fm[k]["weed"]["recall"]["mean"]),
        ("Weed F1",            "Full variant",           "hi",
         lambda k: fm[k]["weed"]["f1"]["mean"]),
        ("Area-weighted recall", "Agronomic",            "hi",
         lambda k: ar[k]["area_recall"]["mean"]),
        ("Throughput (FPS)",   "Deployment",             "hi", lambda k: clean_fps(k)[0]),
        ("Compute (GMACs)",    "Deployment",             "lo",
         lambda k: comp[k]["gmacs_median"]),
    ]

    out = {"metrics": [], "wins": {k: 0 for k in PRIMARY}}
    for label, group, better, fn in rows:
        vals = {k: fn(k) for k in PRIMARY}
        order = sorted(PRIMARY, key=lambda k: vals[k], reverse=(better == "hi"))
        best = order[0]
        out["wins"][best] += 1
        out["metrics"].append({
            "label": label, "group": group, "better": better,
            "values": {k: float(vals[k]) for k in PRIMARY},
            "rank": {k: order.index(k) + 1 for k in PRIMARY},
            "best": best,
        })
    out["n_metrics"] = len(rows)
    out["n_models_winning"] = sum(1 for v in out["wins"].values() if v)
    return out


def _weed_detection_labels(items, by_img, iou_thr=0.50):
    """Với MỌI dự đoán lớp cỏ: (điểm tin cậy, có khớp một nhãn cỏ thật không).

    Đây là bài toán nhị phân có tập âm tính xác định — các dự đoán cỏ không
    khớp nhãn cỏ nào — nên ROC và AUC ở đây có nghĩa, khác với ROC trên toàn
    ảnh vốn không định nghĩa được vì số vị trí "không vật thể" là vô hạn.
    """
    score, is_tp = [], []
    for meta, _path, gb, gc in items:
        d = by_img.get(meta["id"], ([], [], []))
        db = np.array(d[0], float).reshape(-1, 4)
        dc = np.array(d[1], int)
        ds = np.array(d[2], float)
        sel = dc == WEED
        db, ds = db[sel], ds[sel]
        if len(db) == 0:
            continue
        gt = gb[gc == WEED] if len(gc) else np.zeros((0, 4))
        hit = np.zeros(len(db), dtype=bool)
        if len(gt):
            iou = box_iou_batch(gt, db)
            h = np.asarray(iou > iou_thr).nonzero()
            if h[0].shape[0]:
                mt = _drop_extra_matches(np.stack((h[0], h[1], iou[h]), axis=1))
                hit[mt[:, 1].astype(int)] = True
        score.extend(ds.tolist())
        is_tp.extend(hit.tolist())
    return np.array(score), np.array(is_tp, dtype=bool)


def r14_curves(items, runs) -> dict:
    """Dữ liệu cho ba đường cong minh hoạ kết quả.

      PR    đường precision-recall của lớp cỏ (biến thể giữ nền)
      FROC  recall theo số báo nhầm trung bình mỗi ảnh — dạng "ROC" dùng được
            cho bài toán phát hiện, vì trục hoành là số FP tuyệt đối chứ không
            phải tỷ lệ cần tập âm tính
      ROC   điểm tin cậy có tách được dự đoán cỏ ĐÚNG khỏi dự đoán cỏ THỪA
            không; ở đây tập âm tính xác định nên AUC có nghĩa
    """
    grid = np.round(np.arange(0.05, 0.96, 0.01), 4)
    n_img = len(items)
    out = {"pr": {}, "froc": {}, "roc": {}, "n_images": n_img}
    for m, rs in runs.items():
        r = next((x for x in rs if x["seed"] == 0), rs[0])
        cnt = {c: counts_full(items, r["by_img"], float(c), 0.50).sum(axis=0)[WEED]
               for c in grid}
        pr, froc = [], []
        for c in grid:
            tp, fp, fn = cnt[c]
            p_, r_, _f = prf(tp, fp, fn)
            pr.append({"conf": float(c), "precision": p_, "recall": r_})
            froc.append({"conf": float(c), "recall": r_, "fp_per_image": fp / n_img})
        out["pr"][m] = pr
        out["froc"][m] = froc

        sc, tp_mask = _weed_detection_labels(items, r["by_img"])
        order = np.argsort(-sc)
        y = tp_mask[order]
        P, N = int(y.sum()), int((~y).sum())
        tpr = np.concatenate(([0.0], np.cumsum(y) / max(P, 1)))
        fpr = np.concatenate(([0.0], np.cumsum(~y) / max(N, 1)))
        auc = float(np.trapz(tpr, fpr))
        # thưa bớt để tệp JSON không phình
        step = max(1, len(fpr) // 300)
        out["roc"][m] = {
            "auc": auc, "n_pos": P, "n_neg": N,
            "points": [{"fpr": float(fpr[i]), "tpr": float(tpr[i])}
                       for i in range(0, len(fpr), step)] +
                      [{"fpr": float(fpr[-1]), "tpr": float(tpr[-1])}],
        }
    return out


def main():
    cfg = load_cfg()
    items, _ = load_test_set(cfg, "bbox", None)
    log(f"tập test: {len(items)} ảnh")
    runs = load_runs(cfg)
    for m in PRIMARY:
        log(f"{MODEL_LABELS[m]}: {len(runs[m])} lượt chạy có detection thô")

    out = {"n_test_images": len(items),
           "runs_used": {m: [r["seed"] for r in rs] for m, rs in runs.items()}}
    log("R1 chỉ số dạng full…");        out["full_metrics"] = r1_full_metrics(items, runs)
    log("R2 quét ngưỡng tin cậy…");     out["conf_sweep"] = r2_conf_sweep(items, runs)
    log("R3 quét IoU…");                out["iou_sweep"] = r3_iou_sweep(items, runs)
    log("R4 đường PR…");                out["pr_curve"] = r4_pr_curve(items, runs)
    log("R5 COCO AR và AP_S/M/L…");     out["coco_detail"] = r5_coco_detail(cfg, runs)
    log("R6 recall theo diện tích…");   out["area_recall"] = r6_area_recall(items, runs)
    log("R7 phân tầng kích thước…");    out["size_strata"] = r7_size_strata(items, runs)
    log("R8 Holm và effect size…");     out["tests"] = r8_tests(runs, out["full_metrics"])
    log("R9 bootstrap hai tầng…");      out["two_level_bootstrap"] = r9_two_level(items, runs)
    log("R10 độ nhạy checkpoint…");    out["checkpoint_sensitivity"] = r10_checkpoint_sensitivity()
    log("R11 bất thường huấn luyện…"); out["training_anomalies"] = r11_training_anomalies()
    log("R12 đối chứng độ phân giải…"); out["resolution_control"] = r12_resolution_control()
    log("R13 xếp hạng theo từng thang đo…")
    out["metric_ranking"] = r13_metric_ranking(out)
    log("R14 đường PR / FROC / ROC…"); out["curves"] = r14_curves(items, runs)

    p = RESULTS / "review_analysis.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    log(f"→ {p}")


if __name__ == "__main__":
    main()
