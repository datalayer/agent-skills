---
name: pdf
description: Comprehensive PDF manipulation toolkit for extracting text and tables, creating new PDFs, merging/splitting documents, and handling forms. When Claude needs to fill in a PDF form or programmatically process, generate, or analyze PDF documents at scale.
license: Proprietary. LICENSE.txt has complete terms
---

# PDF Skill

## Environment

- None required.

## Script Inventory

### `scripts/check_fillable_fields.py`

- Purpose: detect whether a PDF contains fillable form fields.
- Required CLI params:
- positional `input_pdf`

### `scripts/extract_form_field_info.py`

- Method: `write_field_info(pdf_path: str, json_output_path: str)`
- Method: `get_field_info(reader: PdfReader)`
- Required CLI params:
- positional `input_pdf`
- positional `output_json`

### `scripts/fill_fillable_fields.py`

- Method: `fill_pdf_fields(input_pdf_path: str, fields_json_path: str, output_pdf_path: str)`
- Method: `validation_error_for_field_value(field_info, field_value)`
- Required CLI params:
- positional `input_pdf`
- positional `field_values_json`
- positional `output_pdf`
- Notes:
- Input JSON entries are expected to include `field_id`, `page`, and `value`.

### `scripts/convert_pdf_to_images.py`

- Method: `convert(pdf_path, output_dir, max_dim=1000)`
- Required CLI params:
- positional `input_pdf`
- positional `output_directory`

### `scripts/check_bounding_boxes.py`

- Method: `get_bounding_box_messages(fields_json_stream) -> list[str]`
- Required CLI params:
- positional `fields_json`
- Notes:
- Expects `form_fields` entries with `page_number`, `label_bounding_box`, and `entry_bounding_box`.

### `scripts/create_validation_image.py`

- Method: `create_validation_image(page_number, fields_json_path, input_path, output_path)`
- Required CLI params:
- positional `page_number`
- positional `fields_json`
- positional `input_image`
- positional `output_image`

### `scripts/fill_pdf_form_with_annotations.py`

- Method: `fill_pdf_form(input_pdf_path, fields_json_path, output_pdf_path)`
- Method: `transform_coordinates(bbox, image_width, image_height, pdf_width, pdf_height)`
- Required CLI params:
- positional `input_pdf`
- positional `fields_json`
- positional `output_pdf`
- Notes:
- Expects a `fields.json` object containing `pages` and `form_fields`.

### `scripts/check_bounding_boxes_test.py`

- Purpose: unittest module for `get_bounding_box_messages` behavior.
- Invocation: run as a Python unittest module.

## Usage Examples

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

- See `forms.md` for expected JSON structures used during form workflows.
- See `reference.md` for broader PDF processing notes outside these scripts.
