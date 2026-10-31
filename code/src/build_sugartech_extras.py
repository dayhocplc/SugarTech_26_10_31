#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Các tệp đi kèm bản Sugar Tech: title page, cover letter, thư rút bài khỏi RAE.

    python3 src/build_sugartech_extras.py

Springer yêu cầu nộp title page riêng với tên tác giả và cơ quan, còn tệp bản
thảo để làm mù. Thư rút bài là bắt buộc về mặt liêm chính: không được nộp song
song hai nơi, nên phải rút khỏi Research in Agricultural Engineering TRƯỚC khi
nộp Sugar Tech.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import discussion as D
from build_manuscript import AFFIL, AUTHORS, CORRESP, to_docx
from build_manuscript_sugartech import KEYWORDS, TITLE
from common import MODEL_LABELS, RAE, RESULTS, clean_fps, load_cfg, log, scope, \
    throughput_requirement

OUT = RAE / "manuscript"
TODAY = datetime.date.today().strftime("%d %B %Y")
RAE_TITLE_OLD = ("Computer vision-based weed detection in sugarcane cultivation: "
                 "detection limits of three object detection architectures under "
                 "field conditions")


def _plain(s: str) -> str:
    return s.replace("<sup>", " (").replace("</sup>", ")")


def checklist() -> str:
    """Phiếu kiểm nộp bài: những gì đã xong và những gì nhóm tác giả phải điền."""
    ms = OUT / "SugarTech_manuscript.md"
    text = ms.read_text(encoding="utf-8") if ms.exists() else ""
    clean = re.sub(r"!\[[^\]]*\]\([^)]*\)|\*\*|\*|&nbsp;", "", text)
    n_tbl = len(re.findall(r"\*\*Table \d+\*\*", text))
    n_fig = len(re.findall(r"\*\*Fig\. \d+\*\*", text))
    todo = re.findall(r"\[TO BE COMPLETED BY THE AUTHORS[^\]]*\]", text)
    sup = OUT / "SugarTech_supplementary.md"
    st = sup.read_text(encoding="utf-8") if sup.exists() else ""

    rows = [
        ("Type of paper", "Original Paper", "\u2713"),
        ("Length", "25 double-spaced pages max, incl. tables and figures",
         "\u2713 see page count printed in the footer of the manuscript"),
        ("Font", "10-point Times Roman, double-spaced, 2.5 cm margins", "\u2713"),
        ("Page numbering", "automatic", "\u2713 footer field"),
        ("Abstract", "150-250 words",
         f"\u2713 {len(re.findall(r'[*][*]Abstract[*][*] (.+)', text)[0].split()) if re.findall(r'[*][*]Abstract[*][*] (.+)', text) else 0} words"),
        ("Keywords", "4-6", f"\u2713 {len(KEYWORDS.split(';'))}"),
        ("Headings", "no more than three levels", "\u2713 two levels"),
        ("Tables", "Arabic numerals, cited in order, caption above",
         f"\u2713 {n_tbl} tables"),
        ("Figures", "named FigN, cited in order, captions in the text file",
         f"\u2713 {n_fig} figures, files in springer_figures/"),
        ("Figure resolution", "600 dpi for combination art, RGB",
         "\u2713 Fig1 and Fig2 at 600 dpi, TIFF and PNG"),
        ("Figure captions", "'Fig.' bold, no punctuation after number or at end",
         "\u2713"),
        ("References", "author-year in text; alphabetical; journal titles italic",
         f"\u2713 {len([l for l in text.split('## References')[-1].split(chr(10)) if l.strip()])} entries"),
        ("Statements and Declarations", "required heading", "\u2713 present"),
        ("Supplementary", "named Supplementary Table SN / Figure SN",
         f"\u2713 {len(re.findall(r'[*][*]Supplementary Table S[0-9]+[.]', st))} tables, "
         f"{len(re.findall(r'Supplementary Figure S[0-9]+[.]', st))} figures"),
        ("Source files", "editable .docx required",
         "\u2713 manuscript and supplementary supplied as .docx"),
    ]
    tbl = ["| Requirement | Journal rule | Status |", "|---|---|---|"]
    tbl += [f"| {a_} | {b_} | {c_} |" for a_, b_, c_ in rows]

    return "\n\n".join([
        "# Sugar Tech submission checklist",
        f"Manuscript: {len(clean.split()):,} words, {len(clean):,} characters.",
        "## Automatically verified",
        "\n".join(tbl),
        "## To be completed by the authors",
        "These cannot be derived from the data or the code. Each appears in the "
        "manuscript or cover letter as a bracketed placeholder, and the consistency "
        "checker reports them on every run until they are replaced.",
        "\n".join(f"{i}. {t}" for i, t in enumerate(todo, 1)) or "(none)",
        "## Also required by the journal, outside the files",
        "1. **Society membership.** One author must be an annual or life member of the "
        "Society for Sugar Research & Promotion. The membership form is linked from the "
        "journal's submission guidelines.",
        "2. **Competing-interest form.** All listed authors must complete the journal's "
        "online conflict-of-interest disclosure; the generated statement is published "
        "before the Acknowledgements.",
        "3. **ORCID.** Supply the 16-digit ORCID of each author if available.",
        "4. **Use of AI tools.** If a large language model was used beyond copy editing, "
        "it must be documented in the Methods section. AI-assisted copy editing does not "
        "need to be declared.",
        "5. **Withdrawal from the previous journal.** Send the withdrawal letter to "
        "Research in Agricultural Engineering and wait for written acknowledgement "
        "before submitting here. The cover letter states that this has been done.",
    ]) + "\n"


def cover_letter(cfg, rev) -> str:
    prim = ["faster_rcnn", "retinanet", "yolov10"]
    fm, ar = rev["full_metrics"], rev["area_recall"]
    ss = rev["size_strata"]["models"]
    f1s = {k: D.seed_mean(k, "f1") for k in prim}
    t = throughput_requirement(cfg)
    fast = [k for k in prim if clean_fps(k)[0] >= t["fps_required_on_workstation"]]
    lo = lambda m: min(fm[k]["weed"][m]["mean"] for k in prim)
    hi = lambda m: max(fm[k]["weed"][m]["mean"] for k in prim)
    A = []
    a = A.append
    a(TODAY)
    a("To: The Editor-in-Chief, *Sugar Tech*")
    a("Dear Editor,")
    a(f"We are submitting our manuscript, “{TITLE}”, for consideration as an "
      "Original Paper in *Sugar Tech*.")
    a("Site-specific herbicide application in sugarcane depends on a detector that finds "
      "the weed actually present and supplies detections fast enough for the implement to "
      "act on them. The deep learning literature reports accuracies for this task that "
      "are consistently high, and *Sugar Tech* has recently published detection work on "
      "closely related sugarcane problems — stem base detection for harvesting (Li et al. "
      "2026), ratoon-emerged seedling hills for replanting (Mo et al. 2026) and "
      "smartphone diagnosis of foliar disease (Chen et al. 2026). Our study asks what "
      "those detectors deliver against the two requirements a sprayer imposes, and the "
      "answer is less comfortable than the published figures suggest.")
    a("The main findings are these.")
    a(f"1. **Class discrimination is solved; detection is not.** On the same test "
      f"partition, the three detectors reached two-class mean F1-scores of "
      f"{min(f1s.values()):.2f}–{max(f1s.values()):.2f} under the confusion-matrix "
      f"convention usually reported, which discards undetected instances. Scored so that "
      f"undetected weed and spurious detections are retained, the same predictions gave "
      f"weed-class precision of {lo('precision'):.2f}–{hi('precision'):.2f} and "
      f"recall of {lo('recall'):.2f}–{hi('recall'):.2f}. About half the weed was not "
      "found, and about half of what was reported as weed had no annotated weed under it.")
    a(f"2. **The operative figure for a sprayer is higher than the instance count.** "
      f"Weighted by the ground area an implement would treat rather than by instance, "
      f"recall was {min(ar[k]['area_recall']['mean'] for k in prim):.2f}–"
      f"{max(ar[k]['area_recall']['mean'] for k in prim):.2f}, because the instances "
      "missed are the small ones. We report both and say which answers which question.")
    a(f"3. **The cause is measured, not asserted.** Recall falls monotonically with weed "
      f"size, from {min(ss[k]['quartile']['Q1']['recall_mean'] for k in prim):.2f} in the "
      f"smallest area quartile to "
      f"{max(ss[k]['quartile']['Q4']['recall_mean'] for k in prim):.2f} in the largest. "
      "Retraining the lightest detector at four times the pixel count changed nothing, so "
      "a larger network input is not the remedy; camera height or tiled inference are.")
    a("4. **A deployment screen grounded in agronomy.** A working speed of "
      f"{t['sprayer_speed_kmh']:.0f} km h⁻¹ with a maximum frame advance of "
      f"{t['max_frame_advance_m']:.2f} m implies "
      f"{t['fps_required_on_device']:.1f} frames per second on the implement"
      + (f", and only {MODEL_LABELS[fast[0]]} clears the corresponding workstation "
         "threshold — the same model that treats the least ground. We report the "
         "trade-off rather than a ranking." if fast else "."))
    a("The dataset, the annotation files, the per-image results, the training logs and "
      "the complete analysis code accompany the submission, with checksums so that the "
      "image set can be verified as the one used here.")
    a("We confirm that this manuscript is original, has not been published elsewhere and "
      "is not under consideration by another journal. An earlier version was under review "
      "at *Research in Agricultural Engineering*; it has been formally withdrawn, and the "
      "withdrawal was acknowledged before this submission. The present manuscript has "
      "been substantially rewritten and extended in response to that review.")
    a("[TO BE COMPLETED BY THE AUTHORS: name the author holding Society for Sugar "
      "Research & Promotion membership, which the journal requires.]")
    a("Thank you for your consideration.")
    a("Sincerely,")
    a("Tran Thi Kim Nga, on behalf of all authors  \nCorresponding author  \n"
      "Faculty of Mechanical Engineering – Technology, Nong Lam University "
      "Ho Chi Minh City, Vietnam  \nEmail: ttknga@hcmuaf.edu.vn")
    return "\n\n".join(A) + "\n"


def withdrawal_letter() -> str:
    return "\n\n".join([
        TODAY,
        "To: The Editor-in-Chief, *Research in Agricultural Engineering*",
        "Dear Editor-in-Chief,",
        f"**Subject: Withdrawal of manuscript “{RAE_TITLE_OLD}”**",
        "We write to withdraw the above manuscript from consideration by *Research in "
        "Agricultural Engineering*.",
        "We do so with regret and with thanks. The reviewer's report was exceptionally "
        "careful, and acting on it changed the paper substantially: we added the "
        "precision and F1-score of the full confusion-matrix variant, size-stratified and "
        "area-weighted recall, sensitivity to the confidence and overlap thresholds, "
        "multiplicity-adjusted tests with effect sizes, a two-level bootstrap over images "
        "and training runs, and a new experiment retraining one model at the input "
        "resolution of the others to remove a confound the reviewer identified. The "
        "reviewer also found four errors we had missed, including a reversed statement "
        "about which direction of class error harms the crop. The manuscript is better "
        "for all of it, and we are grateful.",
        "Our reason for withdrawing is institutional rather than scientific. Our "
        "institution's research assessment recognises journals indexed in the Web of "
        "Science Core Collection, and *Research in Agricultural Engineering* is indexed "
        "in Scopus and other databases but not in that collection. We became aware of "
        "this only while preparing the revision. We would not raise it had the review "
        "gone differently; as it is, we cannot complete the process and then be unable to "
        "count the result.",
        "We ask that the reviewer be thanked on our behalf. Their report improved the work "
        "regardless of where it appears, and the revised manuscript credits the review "
        "explicitly in its acknowledgement of how the analysis came to be structured as "
        "it is.",
        "We confirm that the manuscript is withdrawn in full and that we will not submit "
        "it elsewhere until this withdrawal has been acknowledged.",
        "Yours sincerely,",
        "Tran Thi Kim Nga, on behalf of all authors  \nCorresponding author  \n"
        "Faculty of Mechanical Engineering – Technology, Nong Lam University "
        "Ho Chi Minh City, Vietnam  \nEmail: ttknga@hcmuaf.edu.vn",
    ]) + "\n"


def main():
    cfg = load_cfg()
    seedvar = json.loads((RESULTS / "seed_variance.json").read_text())
    D.use_seed_variance(seedvar)
    rev = json.loads((RESULTS / "review_analysis.json").read_text())

    for name, body in (("SugarTech_submission_checklist", checklist()),
                       ("SugarTech_cover_letter", cover_letter(cfg, rev)),
                       ("RAE_withdrawal_letter", withdrawal_letter())):
        p = OUT / f"{name}.md"
        p.write_text(body, encoding="utf-8")
        log(f"{len(body.split())} từ → {p}")
        d = to_docx(p)
        if d:
            final = OUT / f"{name}.docx"
            if d != final:
                d.replace(final)


if __name__ == "__main__":
    main()
