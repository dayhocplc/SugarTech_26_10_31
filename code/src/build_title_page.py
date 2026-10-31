#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Title Page nộp riêng, kèm bản thảo đã làm mù.

    python3 src/build_title_page.py

Hướng dẫn tác giả của tạp chí yêu cầu nộp ba tệp: (i) Title Page có loại bài,
tiêu đề, tên đầy đủ các tác giả theo thứ tự sẽ in, cơ quan, email tác giả liên
hệ, SỐ KÝ TỰ của bản thảo, và phần ghi công; (ii) tệp bản thảo đã làm mù; (iii)
tệp hình. Vì vậy tên tác giả không còn nằm trong RAE_manuscript.docx.

Số ký tự lấy trực tiếp từ bản thảo đã dựng, nên không thể lệch với tệp thật.
"""

from __future__ import annotations

import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from build_manuscript import AFFIL, AUTHORS, CORRESP, KEYWORDS, TITLE, to_docx
from common import RAE, log

OUT = RAE / "manuscript"


def main():
    ms = OUT / "RAE_manuscript.md"
    if not ms.exists():
        raise SystemExit("Chưa có RAE_manuscript.md — chạy src/build_manuscript.py trước")
    text = ms.read_text(encoding="utf-8")
    clean = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    clean = re.sub(r"\*\*|\*|&nbsp;", "", clean)
    n_chars = len(clean)
    n_words = len(clean.split())
    n_tbl = len(re.findall(r"^\*\*Table \d+\.", text, re.M))
    n_fig = len(re.findall(r"^Figure \d+\.", text, re.M))
    # Đếm phụ lục từ chính tệp phụ lục, không viết cứng.
    sup = OUT / "RAE_supplementary.md"
    stext = sup.read_text(encoding="utf-8") if sup.exists() else ""
    n_stbl = len(re.findall(r"\*\*Supplementary Table S\d+\.", stext))
    n_sfig = len(re.findall(r"Supplementary Figure S\d+\.", stext))

    L = [
        "# Title page", "",
        "**Type of document:** original paper", "",
        f"**Title:** {TITLE}", "",
        f"**Running title:** Detection completeness in sugarcane weed detection", "",
        f"**Authors:** {AUTHORS.replace('<sup>', ' (').replace('</sup>', ')')}", "",
        f"**Affiliation:** {AFFIL.replace('<sup>1</sup>', '1 — ')}", "",
        f"**Corresponding author:** {CORRESP.replace('*Corresponding author*: ', '')}", "",
        f"**Keywords:** {KEYWORDS}", "",
        f"**Number of characters (manuscript file, including spaces, excluding "
        f"figures):** {n_chars:,}", "",
        f"**Approximate word count:** {n_words:,}", "",
        f"**Tables:** {n_tbl}  ·  **Figures:** {n_fig}  ·  "
        f"**Supplementary tables:** {n_stbl}  ·  "
        f"**Supplementary figures:** {n_sfig}", "",
        "**Acknowledgement:** "
        "[TO BE COMPLETED BY THE AUTHORS, or delete this line if there is none.]", "",
        "**Funding:** "
        "[TO BE COMPLETED BY THE AUTHORS: grant number and funding body, or an explicit "
        "statement that the work received no external funding.]", "",
        "**Author contributions:** "
        "[TO BE COMPLETED BY THE AUTHORS, in the journal's CRediT categories.]", "",
        "**Conflict of interest:** The authors declare no conflict of interest.", "",
        "**Use of artificial intelligence:** "
        "[TO BE COMPLETED BY THE AUTHORS. The journal requires disclosure of AI tools "
        "used in preparing the manuscript, except tools used solely for literature search "
        "or reference management. Tools used as part of the research process are disclosed "
        "in the Methods instead.]", "",
    ]
    md = "\n".join(L)
    p = OUT / "RAE_title_page.md"
    p.write_text(md, encoding="utf-8")
    log(f"bản thảo: {n_chars:,} ký tự, {n_words:,} từ, {n_tbl} bảng, {n_fig} hình")
    log(f"→ {p}")
    d = to_docx(p)
    if d:
        final = OUT / "RAE_title_page.docx"
        if d != final:
            d.replace(final)
        log(f"→ {final}")


if __name__ == "__main__":
    main()
