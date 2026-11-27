#!/usr/bin/env python3
"""Create a blank 25-slide ACL/EMNLP-inspired PowerPoint template.

The template uses a white 16:9 canvas, a slim maroon title banner, editable
slide-title placeholders inside that banner, a clean blank content area, and
subtle slide numbers. Output is .pptx, the modern PowerPoint format.

Usage:
    python -m pip install -r requirements-presentation.txt
    python Seminar_presentation_template.py
    python Seminar_presentation_template.py --slides 25 --output Seminar_presentation_template.pptx

Note: python-pptx writes .pptx files, not the legacy binary .ppt format.
"""

import argparse
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

MAROON = RGBColor(112, 31, 57)
WHITE = RGBColor(255, 255, 255)
INK = RGBColor(35, 43, 52)
MUTED = RGBColor(112, 122, 132)

SLIDE_WIDTH = 13.333
SLIDE_HEIGHT = 7.5
BANNER_HEIGHT = 0.62


def add_title_banner(slide, slide_number: int, total_slides: int) -> None:
    """Add maroon title band with an editable title placeholder."""
    banner = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        0,
        0,
        Inches(SLIDE_WIDTH),
        Inches(BANNER_HEIGHT),
    )
    banner.fill.solid()
    banner.fill.fore_color.rgb = MAROON
    banner.line.fill.background()

    # Put the title text inside the band so the content area remains generous.
    title_box = slide.shapes.add_textbox(
        Inches(0.62), Inches(0.075), Inches(11.9), Inches(0.46)
    )
    title_frame = title_box.text_frame
    title_frame.clear()
    title_frame.word_wrap = False
    title_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    title_frame.margin_left = Inches(0.02)
    title_frame.margin_right = Inches(0.02)
    title_frame.margin_top = 0
    title_frame.margin_bottom = 0
    paragraph = title_frame.paragraphs[0]
    paragraph.alignment = PP_ALIGN.LEFT
    run = paragraph.add_run()
    run.text = "[Slide title]"
    run.font.name = "Aptos Display"
    run.font.size = Pt(24)
    run.font.bold = True
    run.font.color.rgb = WHITE

    # Small, unobtrusive pagination in the lower-right corner.
    footer = slide.shapes.add_textbox(
        Inches(11.7), Inches(7.16), Inches(1.0), Inches(0.18)
    )
    footer_tf = footer.text_frame
    footer_tf.clear()
    footer_tf.margin_left = footer_tf.margin_right = 0
    footer_tf.margin_top = footer_tf.margin_bottom = 0
    footer_tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = footer_tf.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    r = p.add_run()
    r.text = f"{slide_number:02d} / {total_slides:02d}"
    r.font.name = "Aptos"
    r.font.size = Pt(9)
    r.font.color.rgb = MUTED


def build_template(slide_count: int, output_path: Path) -> None:
    if slide_count < 1:
        raise ValueError("Slide count must be at least 1")

    presentation = Presentation()
    presentation.slide_width = Inches(SLIDE_WIDTH)
    presentation.slide_height = Inches(SLIDE_HEIGHT)
    presentation.core_properties.title = "ACL/EMNLP Seminar Presentation Template"
    presentation.core_properties.subject = "Blank white presentation with maroon title banners"
    presentation.core_properties.author = ""

    blank_layout = presentation.slide_layouts[6]
    for number in range(1, slide_count + 1):
        slide = presentation.slides.add_slide(blank_layout)
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = WHITE
        add_title_banner(slide, number, slide_count)
        # The remaining white canvas is intentionally empty for slide content.

    output_path.parent.mkdir(parents=True, exist_ok=True)
    presentation.save(output_path)
    print(f"Created {slide_count}-slide template: {output_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slides", type=int, default=25, help="Number of blank slides (default: 25)"
    )
    parser.add_argument(
        "--output",
        default="Seminar_presentation_template.pptx",
        help="Output PowerPoint .pptx path",
    )
    args = parser.parse_args()
    build_template(args.slides, Path(args.output))


if __name__ == "__main__":
    main()
