#!/usr/bin/env python3
"""Build a duplex-friendly PDF: one spectrogram page, then one metadata page.

Input:
- spectrogram manifest CSV produced by the spectrogram pipeline
- family summary CSV with Nest Father / Genetic Father / HR Birds / XF Birds

Output:
- a PDF with alternating pages:
  Page 1: spectrogram image
  Page 2: metadata block
  Page 3: spectrogram image
  Page 4: metadata block
  ...

This is intentionally simple and stable for printing / manual review.
"""

from __future__ import annotations

import argparse
import html
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
DEFAULT_BASE_DIR = "/Users/annietaylor/Documents/ucsf/brainard/globus/song_phenotyping"
CSV_BASE_DIR = "/Users/annietaylor/Documents/ucsf/brainard/x-foster/"
DEFAULT_MANIFEST = os.path.join(DEFAULT_BASE_DIR, "file_management", "xfoster_specs", "spectrogram_manifest.csv")
DEFAULT_FAMILY = os.path.join(CSV_BASE_DIR, "nest_gen_pair_offspring_summary.csv")
DEFAULT_OUTPUT = os.path.join(DEFAULT_BASE_DIR, "file_management", "xfoster_specs", "song_spectrogram_book.pdf")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
@dataclass
class Record:
    bird: str
    filepath: str
    output_path: str
    threshold: Optional[float]
    n_segments: Optional[int]
    nest_father: str
    genetic_father: str
    status: str


def parse_bird_list(cell: object) -> List[str]:
    if cell is None:
        return []
    s = str(cell).strip()
    if not s or s.lower() == "nan":
        return []
    return [b.strip() for b in s.split(";") if b.strip()]


def fmt_num(x: object, digits: int = 4) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def fmt_int(x: object) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    try:
        return str(int(float(x)))
    except Exception:
        return str(x)


def path_to_html(path: str) -> str:
    """Make a long filepath wrap in a Paragraph by inserting breaks at separators."""
    if not path:
        return ""
    p = Path(path).as_posix()
    # Break after each slash; this makes very long paths readable in a narrow cell.
    return html.escape(p).replace("/", "/<br/>")


def build_family_lookup(family_df: pd.DataFrame) -> Dict[str, Tuple[str, str]]:
    """Map bird -> (nest father, genetic father), preferring direct bird membership over father-only matches."""
    best: Dict[str, Tuple[int, int, str, str]] = {}
    # tuple = (priority, row_index, nest_father, genetic_father)
    # priorities: direct family member 3, nest/gen father 2
    for idx, row in family_df.iterrows():
        nest = str(row.get("Nest Father", "")).strip()
        gen = str(row.get("Genetic Father", "")).strip()

        for bird in parse_bird_list(row.get("HR Birds", "")) + parse_bird_list(row.get("XF Birds", "")):
            cur = best.get(bird)
            cand = (3, idx, nest, gen)
            if cur is None or cand[:2] > cur[:2]:
                best[bird] = cand

        if nest:
            cur = best.get(nest)
            cand = (2, idx, nest, gen)
            if cur is None or cand[:2] > cur[:2]:
                best[nest] = cand

        if gen:
            cur = best.get(gen)
            cand = (2, idx, nest, gen)
            if cur is None or cand[:2] > cur[:2]:
                best[gen] = cand

    return {bird: (nest, gen) for bird, (_, __, nest, gen) in best.items()}


def load_records(manifest_csv: str, family_csv: str) -> List[Record]:
    manifest = pd.read_csv(manifest_csv)
    family_df = pd.read_csv(family_csv)
    fam_lookup = build_family_lookup(family_df)

    # Accept either status naming convention.
    status_ok = {"song", "saved"}
    if "status" not in manifest.columns:
        raise ValueError("Manifest CSV must contain a 'status' column.")

    rows: List[Record] = []
    for _, row in manifest.iterrows():
        status = str(row.get("status", "")).strip()
        if status not in status_ok:
            continue

        bird = str(row.get("bird", "")).strip()
        filepath = str(row.get("filepath", "")).strip()
        output_path = str(row.get("output_path", "")).strip()
        threshold = row.get("threshold", None)
        n_segments = row.get("n_segments", None)

        nest, gen = fam_lookup.get(bird, ("", ""))
        rows.append(
            Record(
                bird=bird,
                filepath=filepath,
                output_path=output_path,
                threshold=None if pd.isna(threshold) else float(threshold),
                n_segments=None if pd.isna(n_segments) else int(float(n_segments)),
                nest_father=nest,
                genetic_father=gen,
                status=status,
            )
        )

    # Stable sort for easy review.
    rows.sort(key=lambda r: (r.bird, r.filepath))
    return rows


def resolve_image_path(rec: Record) -> Optional[str]:
    candidates = []
    if rec.output_path:
        candidates.append(rec.output_path)
    # Fallback to same folder conventions if output_path is stale/blank.
    if rec.bird and rec.filepath:
        stem = Path(rec.filepath).stem
        candidates.append(os.path.join(DEFAULT_BASE_DIR, "file_management", "xfoster_specs", rec.bird, f"{stem}.png"))
        candidates.append(os.path.join(DEFAULT_BASE_DIR, "file_management", "xfoster_specs", rec.bird, f"{Path(rec.filepath).name}.png"))

    for cand in candidates:
        if cand and os.path.exists(cand):
            return cand
    return None


def fit_image_box(img_w: float, img_h: float, max_w: float, max_h: float) -> Tuple[float, float]:
    scale = min(max_w / img_w, max_h / img_h)
    return img_w * scale, img_h * scale


# -----------------------------------------------------------------------------
# Page drawing
# -----------------------------------------------------------------------------
def draw_front_page(c: canvas.Canvas, rec: Record, page_num: int, total_pages: int) -> None:
    page_w, page_h = letter
    margin = 0.55 * inch
    usable_w = page_w - 2 * margin
    usable_h = page_h - 2 * margin

    c.setTitle("Song spectrogram book")
    c.setAuthor("OpenAI")
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, page_h - margin + 0.05 * inch, f"Bird: {rec.bird}")
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.grey)
    c.drawRightString(page_w - margin, page_h - margin + 0.05 * inch, f"Page {page_num}/{total_pages}")
    c.setFillColor(colors.black)

    title_y = page_h - margin - 0.18 * inch
    c.setFont("Helvetica", 10)
    c.drawString(margin, title_y, f"Song spectrogram")

    img_path = resolve_image_path(rec)
    img_top = title_y - 0.12 * inch
    img_bottom = margin + 0.2 * inch
    max_img_h = img_top - img_bottom

    if img_path is None:
        # Placeholder box if the image is missing.
        c.setStrokeColor(colors.red)
        c.setFillColor(colors.whitesmoke)
        c.rect(margin, img_bottom, usable_w, max_img_h, fill=1, stroke=1)
        c.setFillColor(colors.red)
        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(page_w / 2, img_bottom + max_img_h / 2 + 8, "MISSING IMAGE")
        c.setFont("Helvetica", 9)
        c.drawCentredString(page_w / 2, img_bottom + max_img_h / 2 - 8, rec.output_path or rec.filepath)
        c.setFillColor(colors.black)
        return

    # Fit image into the available area while preserving aspect.
    with PILImage.open(img_path) as im:
        img_w_px, img_h_px = im.size

    # Convert ratio-only dimensions into page units.
    img_w_pt, img_h_pt = fit_image_box(img_w_px, img_h_px, usable_w, max_img_h)
    x = margin + (usable_w - img_w_pt) / 2
    y = img_bottom + (max_img_h - img_h_pt) / 2

    c.drawImage(ImageReader(img_path), x, y, width=img_w_pt, height=img_h_pt, preserveAspectRatio=True, mask='auto')

    # Small footer line.
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.grey)
    c.drawString(margin, margin - 0.15 * inch, f"{Path(img_path).name}")
    c.setFillColor(colors.black)


def draw_back_page(c: canvas.Canvas, rec: Record, page_num: int, total_pages: int) -> None:
    page_w, page_h = letter
    margin = 0.55 * inch
    title_style = ParagraphStyle(
        "title",
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#222222"),
    )
    body_style = ParagraphStyle(
        "body",
        fontName="Helvetica",
        fontSize=9,
        leading=11,
        alignment=TA_LEFT,
        textColor=colors.black,
    )
    body_style_small = ParagraphStyle(
        "body_small",
        fontName="Helvetica",
        fontSize=8.5,
        leading=10,
        alignment=TA_LEFT,
        textColor=colors.black,
    )

    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, page_h - margin + 0.05 * inch, f"Metadata: {rec.bird}")
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.grey)
    c.drawRightString(page_w - margin, page_h - margin + 0.05 * inch, f"Page {page_num}/{total_pages}")
    c.setFillColor(colors.black)

    # Build table content.
    data = [
        [Paragraph("<b>Bird ID</b>", body_style), Paragraph(html.escape(rec.bird), body_style)],
        [Paragraph("<b>Filepath</b>", body_style), Paragraph(path_to_html(rec.filepath), body_style_small)],
        [Paragraph("<b>Threshold</b>", body_style), Paragraph(fmt_num(rec.threshold, digits=4), body_style)],
        [Paragraph("<b>Number of segments</b>", body_style), Paragraph(fmt_int(rec.n_segments), body_style)],
        [Paragraph("<b>Nest father</b>", body_style), Paragraph(html.escape(rec.nest_father or ""), body_style)],
        [Paragraph("<b>Genetic father</b>", body_style), Paragraph(html.escape(rec.genetic_father or ""), body_style)],
    ]

    label_w = 1.9 * inch
    value_w = page_w - 2 * margin - label_w
    table = Table(data, colWidths=[label_w, value_w], repeatRows=0)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#444444")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#A0A0A0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (0, -1), "LEFT"),
                ("ALIGN", (1, 0), (1, -1), "LEFT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )

    tw, th = table.wrapOn(c, page_w - 2 * margin, page_h - 2 * margin)
    x = margin
    y = page_h - margin - 0.75 * inch - th
    table.drawOn(c, x, y)

    # Small note.
    note = Paragraph(
        "<i>Print duplex. Page 1 is the spectrogram; page 2 is the metadata block.</i>",
        ParagraphStyle(
            "note",
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=10,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#555555"),
        ),
    )
    nw, nh = note.wrap(page_w - 2 * margin, 0.6 * inch)
    note.drawOn(c, margin, margin - 0.02 * inch)


# -----------------------------------------------------------------------------
# Build PDF
# -----------------------------------------------------------------------------
def build_pdf(records: List[Record], output_pdf: str) -> None:
    if not records:
        raise ValueError("No records to write. Check your manifest filters and image paths.")

    out_dir = os.path.dirname(output_pdf)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    c = canvas.Canvas(output_pdf, pagesize=letter)
    total_pages = len(records) * 2
    page_num = 1

    for rec in records:
        draw_front_page(c, rec, page_num, total_pages)
        c.showPage()
        page_num += 1

        draw_back_page(c, rec, page_num, total_pages)
        c.showPage()
        page_num += 1

    c.save()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Build a duplex-friendly PDF with spectrogram pages and metadata pages.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="Path to spectrogram manifest CSV")
    parser.add_argument("--family-summary", default=DEFAULT_FAMILY, help="Path to family summary CSV")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output PDF path")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of records to include")
    args = parser.parse_args()

    records = load_records(args.manifest, args.family_summary)
    if args.limit is not None:
        records = records[: args.limit]

    build_pdf(records, args.output)
    print(f"Wrote PDF: {args.output}")
    print(f"Records included: {len(records)}")


if __name__ == "__main__":
    main()
