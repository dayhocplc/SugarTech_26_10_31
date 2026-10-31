#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chuyển danh mục tài liệu giữa hai kiểu trình bày của hai tạp chí.

Research in Agricultural Engineering dùng:
    Ren S., He K., Girshick R., Sun J. (2017): Faster R-CNN: Towards ...
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 39: 1137–1149.

Sugar Tech (Springer) in ra dạng khác — kiểm trên một bài đã đăng của chính
tạp chí (doi 10.1007/s12355-025-01605-2):
    Esteban, D.A.A., Z.M. de Souza, C.A. Tormena, ... 2019. Soil compaction, ...
    Geoderma 347: 1-10.

Khác biệt: tác giả đầu đảo "Họ, Chữ-viết-tắt"; các tác giả sau để chữ viết tắt
trước họ; trước tác giả cuối thêm "and"; năm đứng sau danh sách tác giả, không
ngoặc đơn; sau năm là dấu chấm; tên tạp chí in nghiêng và KHÔNG có dấu phẩy
trước số tập.

Hàm ở đây làm việc chuyển đổi đó một cách máy móc và báo lại những mục không
khớp mẫu, để không có mục nào bị đổi sai mà im lặng.
"""

from __future__ import annotations

import re

# "Ren S., He K., Girshick R., Sun J. (2017): phần còn lại"
_RE_ENTRY = re.compile(r"^(?P<authors>.+?)\s*\((?P<year>\d{4}[ab]?)\):\s*(?P<rest>.+)$",
                       re.S)
# một tác giả: "Ren S." / "de Souza M.F." / "Lin T.-Y." / "NVIDIA Corporation"
_RE_AUTHOR = re.compile(r"^(?P<sur>.+?)\s+(?P<ini>(?:[A-ZÀ-Ý]\.(?:-[A-ZÀ-Ý]\.)*)+)$")


def _springer_authors(block: str) -> str | None:
    """Đổi danh sách tác giả sang thứ tự Springer, hoặc None nếu không nhận ra."""
    parts = [p.strip() for p in block.split(",") if p.strip()]
    out = []
    for i, p in enumerate(parts):
        m = _RE_AUTHOR.match(p)
        if not m:
            # tổ chức làm tác giả (ví dụ "NVIDIA Corporation") — giữ nguyên
            if i == 0 and len(parts) == 1:
                return p
            return None
        sur, ini = m.group("sur"), m.group("ini")
        out.append(f"{sur}, {ini}" if i == 0 else f"{ini} {sur}")
    if len(out) == 1:
        return out[0]
    if len(out) == 2:
        return f"{out[0]}, and {out[1]}"
    return ", ".join(out[:-1]) + f", and {out[-1]}"


def _springer_rest(rest: str) -> str:
    """Chuẩn hoá phần sau năm.

    Hai việc: bỏ dấu phẩy trước số tập ('Journal, 39: 1137-1149.' → 'Journal 39:
    …'), và in NGHIÊNG tên tạp chí như hướng dẫn yêu cầu. Tên tạp chí là đoạn
    nằm giữa dấu chấm kết thúc nhan đề bài và số tập; nếu không nhận được dạng
    đó thì để nguyên chứ không đoán.
    """
    rest = re.sub(r",\s*(\d+(?:\(\d+\))?):", r" \1:", rest)
    m = re.search(r"\.\s+([A-Z][^.]{2,80}?)\s+(\d+(?:\(\d+\))?):", rest)
    if m:
        return rest[:m.start(1)] + f"*{m.group(1)}*" + rest[m.end(1):]
    # Kỷ yếu hội nghị: hướng dẫn yêu cầu in nghiêng cả tên sách, mà mục kỷ yếu
    # không có số tập nên mẫu trên bỏ qua. Bắt riêng dạng "Proceedings of ...".
    m = re.search(r"(In: )?(Proceedings of [^,]+?)(,|\.)", rest)
    if m:
        return rest[:m.start(2)] + f"*{m.group(2)}*" + rest[m.end(2):]
    # Chương sách: "In: Biên-tập-viên (eds): Tên sách. Nhà xuất bản, Nơi."
    m = re.search(r"\(eds?\):?\s+([^.]{4,100}?)\.", rest)
    if m:
        return rest[:m.start(1)] + f"*{m.group(1)}*" + rest[m.end(1):]
    return rest


def to_springer(refs: list[str]) -> tuple[list[str], list[str]]:
    """Trả (danh mục đã chuyển, danh sách mục không chuyển được).

    Mục không khớp mẫu được giữ NGUYÊN VĂN và báo ra, chứ không đoán bừa.
    """
    done, failed = [], []
    for r in refs:
        one = " ".join(r.split())
        m = _RE_ENTRY.match(one)
        if not m:
            done.append(one); failed.append(one[:70]); continue
        auth = _springer_authors(m.group("authors"))
        if auth is None:
            done.append(one); failed.append(one[:70]); continue
        done.append(f"{auth}. {m.group('year')}. {_springer_rest(m.group('rest'))}")
    return sorted(done, key=str.lower), failed


if __name__ == "__main__":
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from build_manuscript import REFERENCES
    out, bad = to_springer(REFERENCES)
    for r in out:
        print(" •", r[:150])
    print(f"\n{len(out)} mục; {len(bad)} mục giữ nguyên văn vì không khớp mẫu:")
    for b in bad:
        print("   !", b)
