# Seminar presentation template

The template generator creates a blank, 25-slide, 16:9 PowerPoint deck with a white background, slim maroon header banner, editable `[Slide title]` text inside each banner, empty content areas, and discreet slide numbers.

## Generate the PowerPoint

```bash
python -m pip install -r requirements-presentation.txt
python Seminar_presentation_template.py
```

This writes `Seminar_presentation_template.pptx` in the current directory. To choose another slide count or output path:

```bash
python Seminar_presentation_template.py --slides 25 --output Seminar_presentation_template.pptx
```

Open the `.pptx` in PowerPoint or LibreOffice, replace `[Slide title]` in each maroon banner, and add content in the blank white canvas. The script generates `.pptx`, the current PowerPoint format; it does not generate legacy `.ppt` files.
