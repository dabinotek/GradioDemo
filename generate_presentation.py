#!/usr/bin/env python3
"""Generate a 16:9 PowerPoint deck from Seminar_presentation.md.

Usage:
    python generate_presentation.py
    python generate_presentation.py --input Seminar_presentation.md --output Seminar_presentation.pptx

Install dependency:
    python -m pip install python-pptx

The slide source is Markdown: each slide begins with `## Slide NN — Title`.
Bullets become editable PowerPoint text; `> Visual:` lines become a small
italic design cue at the bottom of the slide. The deck uses a white canvas and
slim maroon top banner. This generates .pptx (the current PowerPoint format).
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

MAROON = RGBColor(112, 31, 57)
NAVY = RGBColor(40, 58, 78)
INK = RGBColor(35, 43, 52)
MUTED = RGBColor(96, 108, 120)
PALE = RGBColor(247, 248, 250)
ACCENT = RGBColor(177, 137, 83)
WHITE = RGBColor(255, 255, 255)


@dataclass
class SlideData:
    title: str
    lines: List[str] = field(default_factory=list)
    visual: str = ""


def parse_markdown(text: str) -> List[SlideData]:
    slides: List[SlideData] = []
    current = None
    for raw in text.splitlines():
        line = raw.strip()
        match = re.match(r"^##\s+Slide\s+\d+\s+[—–-]\s+(.+)$", line)
        if match:
            if current:
                slides.append(current)
            current = SlideData(title=match.group(1).strip())
            continue
        if current is None or not line:
            continue
        if line.startswith("> Visual:"):
            current.visual = line.removeprefix("> Visual:").strip()
        elif line.startswith("- "):
            current.lines.append(line[2:].strip())
        elif line.startswith("> "):
            current.lines.append(line[2:].strip())
        elif line.startswith("#") or line.startswith("---"):
            continue
        else:
            current.lines.append(line)
    if current:
        slides.append(current)
    return slides


def add_text(slide, text: str, left, top, width, height, *, size=18, color=INK,
             bold=False, italic=False, font="Aptos", align=None):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(0.02)
    tf.margin_right = Inches(0.02)
    tf.margin_top = Inches(0.01)
    tf.margin_bottom = Inches(0.01)
    p = tf.paragraphs[0]
    if align is not None:
        p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return box


def add_banner(slide, title: str, slide_number: int, total: int):
    # White background with a deliberately slim, consistent maroon header.
    banner = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(0.34)
    )
    banner.fill.solid()
    banner.fill.fore_color.rgb = MAROON
    banner.line.fill.background()
    add_text(slide, "", Inches(0.45), Inches(0.40), Inches(0.1), Inches(0.1), size=1)
    add_text(slide, title, Inches(0.62), Inches(0.62), Inches(12.05), Inches(0.72),
             size=27, color=NAVY, bold=True)
    # Short gold rule adds a restrained academic accent.
    rule = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.64), Inches(1.38), Inches(0.72), Inches(0.035)
    )
    rule.fill.solid()
    rule.fill.fore_color.rgb = ACCENT
    rule.line.fill.background()
    add_text(slide, f"ACL/EMNLP-inspired seminar  ·  {slide_number:02d}/{total:02d}",
             Inches(9.55), Inches(7.20), Inches(3.15), Inches(0.18),
             size=9, color=MUTED, align=PP_ALIGN.RIGHT)


def add_bullet(slide, text: str, y, font_size=19):
    # Each item is its own text box to keep wrapping and spacing predictable.
    box = slide.shapes.add_textbox(Inches(0.92), y, Inches(11.45), Inches(0.69))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(0.02)
    tf.margin_right = Inches(0.02)
    tf.margin_top = Inches(0.025)
    tf.margin_bottom = Inches(0.015)
    p = tf.paragraphs[0]
    p.space_after = Pt(0)
    bullet = p.add_run()
    bullet.text = "•  "
    bullet.font.name = "Aptos"
    bullet.font.size = Pt(font_size)
    bullet.font.bold = True
    bullet.font.color.rgb = MAROON
    body = p.add_run()
    body.text = re.sub(r"\*\*(.*?)\*\*", r"\1", text).replace("`", "")
    body.font.name = "Aptos"
    body.font.size = Pt(font_size)
    body.font.color.rgb = INK
    return box


def render_slide(prs: Presentation, data: SlideData, index: int, total: int):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = WHITE
    add_banner(slide, data.title, index, total)

    # Keep body compact and readable; body text is editable in PowerPoint.
    body_lines = data.lines
    if len(body_lines) > 6:
        body_lines = body_lines[:6]
    start_y = 1.72
    available = 4.7 if data.visual else 5.15
    line_height = min(0.77, available / max(1, len(body_lines)))
    font_size = 19 if len(body_lines) <= 5 else 17
    for offset, line in enumerate(body_lines):
        add_bullet(slide, line, Inches(start_y + offset * line_height), font_size)

    if data.visual:
        # A soft panel carries the design cue, rather than silently dropping it.
        y = 6.25 if len(body_lines) <= 5 else 6.12
        panel = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(0.70), Inches(y), Inches(11.95), Inches(0.62)
        )
        panel.fill.solid()
        panel.fill.fore_color.rgb = PALE
        panel.line.color.rgb = RGBColor(220, 224, 229)
        panel.line.width = Pt(0.7)
        tf = panel.text_frame
        tf.clear()
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_left = Inches(0.18)
        tf.margin_right = Inches(0.15)
        p = tf.paragraphs[0]
        label = p.add_run()
        label.text = "FIGURE CUE  "
        label.font.name = "Aptos"
        label.font.size = Pt(10)
        label.font.bold = True
        label.font.color.rgb = MAROON
        cue = p.add_run()
        cue.text = data.visual
        cue.font.name = "Aptos"
        cue.font.size = Pt(11)
        cue.font.italic = True
        cue.font.color.rgb = MUTED
    return slide


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="Seminar_presentation.md", help="Markdown slide source")
    parser.add_argument("--output", default="Seminar_presentation.pptx", help="Output PowerPoint file")
    args = parser.parse_args()

    source = Path(args.input)
    if not source.exists():
        raise SystemExit(f"Input Markdown not found: {source}")
    slides = parse_markdown(source.read_text(encoding="utf-8"))
    if len(slides) != 25:
        raise SystemExit(f"Expected 25 slides, found {len(slides)}")

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    prs.core_properties.title = "Zero-Shot Machine Translation: Models, Failure Modes, and Practical Diagnostics"
    prs.core_properties.subject = "ACL/EMNLP-inspired seminar presentation"
    prs.core_properties.author = ""
    for i, data in enumerate(slides, start=1):
        render_slide(prs, data, i, len(slides))
    output = Path(args.output)
    prs.save(output)
    print(f"Wrote {len(slides)} slides to {output.resolve()}")


if __name__ == "__main__":
    main()
