#!/usr/bin/env python3
"""
Render a PDF page to a high-resolution PNG for README / docs.

GitHub Markdown does not reliably preview embedded PDF figures; export to PNG
(or SVG from a vector editor) and point README at that file instead.

Usage (from repo root):
  pip install pymupdf
  python Evo_depth/scripts/export_readme_figure_from_pdf.py \\
    --pdf Evo_depth/assets/main3.pdf --page 0 --scale 3 \\
    --out Evo_depth/assets/model_overview.png
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Export PDF page to PNG for README.")
    p.add_argument("--pdf", type=Path, required=True, help="Input PDF path.")
    p.add_argument("--page", type=int, default=0, help="0-based page index (default: 0).")
    p.add_argument(
        "--scale",
        type=float,
        default=3.0,
        help="Zoom vs PDF default 72 dpi (e.g. 3 => ~216 dpi equivalent width).",
    )
    p.add_argument("--out", type=Path, required=True, help="Output PNG path.")
    args = p.parse_args()

    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise SystemExit(
            "Missing dependency: pip install pymupdf\n"
            "(package import name is `fitz`.)"
        ) from e

    doc = fitz.open(args.pdf)
    if args.page < 0 or args.page >= len(doc):
        raise SystemExit(f"page {args.page} out of range (0..{len(doc) - 1})")
    page = doc[args.page]
    mat = fitz.Matrix(args.scale, args.scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pix.save(args.out.as_posix())
    doc.close()
    print(f"Wrote {args.out} ({pix.width}x{pix.height})")


if __name__ == "__main__":
    main()
