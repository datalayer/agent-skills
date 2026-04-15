---
name: pdf
description: PDF form-processing toolkit for checking fillable forms, extracting form field metadata, validating bounding boxes, converting pages to images, and filling forms.
license: Proprietary. LICENSE.txt has complete terms
---

# PDF Skill

## Required Environment Variables

- None required.

## Invocation Contract

- Call `run_skill_script` with:
  - `skill_name`: `pdf`
  - `script_name`: one of `check_fillable_fields`, `extract_form_field_info`, `fill_fillable_fields`, `convert_pdf_to_images`, `check_bounding_boxes`, `create_validation_image`, `fill_pdf_form_with_annotations`
- These scripts are positional-argument based. Use `args`, not `kwargs`, for required parameters.

## Scripts API

### `script_name: check_fillable_fields`

- `args`: `["<input_pdf>"]`
- Output: prints whether the PDF has fillable fields.

### `script_name: extract_form_field_info`

- `args`: `["<input_pdf>", "<output_json>"]`
- Output: JSON file with discovered fields and metadata.

### `script_name: fill_fillable_fields`

- `args`: `["<input_pdf>", "<field_values_json>", "<output_pdf>"]`
- Input contract: field values JSON structure documented in `forms.md`.

### `script_name: convert_pdf_to_images`

- `args`: `["<input_pdf>", "<output_directory>"]`
- Output: PNG images (one per page).

### `script_name: check_bounding_boxes`

- `args`: `["<fields_json>"]`
- Input contract: `fields.json` format documented in `forms.md`.
- Output: validation messages.

### `script_name: create_validation_image`

- `args`: `["<page_number>", "<fields_json>", "<input_image>", "<output_image>"]`
- Output: image with overlay boxes for inspection.

### `script_name: fill_pdf_form_with_annotations`

- `args`: `["<input_pdf>", "<fields_json>", "<output_pdf>"]`
- Input contract: annotated form JSON schema in `forms.md`.

## `run_skill_script` Examples

- Extract field info:

```json
{
  "skill_name": "pdf",
  "script_name": "extract_form_field_info",
  "args": ["form.pdf", "fields.json"]
}
```

- Fill fillable fields:

```json
{
  "skill_name": "pdf",
  "script_name": "fill_fillable_fields",
  "args": ["form.pdf", "field_values.json", "filled.pdf"]
}
```

## Direct CLI Examples

```bash
python agent_skills/skills/pdf/scripts/check_fillable_fields.py form.pdf
python agent_skills/skills/pdf/scripts/extract_form_field_info.py form.pdf fields.json
python agent_skills/skills/pdf/scripts/fill_fillable_fields.py form.pdf field_values.json filled.pdf
python agent_skills/skills/pdf/scripts/convert_pdf_to_images.py form.pdf out_images/
python agent_skills/skills/pdf/scripts/check_bounding_boxes.py fields.json
python agent_skills/skills/pdf/scripts/create_validation_image.py 1 fields.json out_images/page_1.png validated.png
python agent_skills/skills/pdf/scripts/fill_pdf_form_with_annotations.py form.pdf fields.json annotated.pdf
```

## References

- `forms.md`: required JSON structures for form workflows.
- `reference.md`: additional PDF workflow guidance.
