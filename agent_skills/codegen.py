# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Skill Code Generator - Generate callable Python skill files.

Based on the TypeScript POC pattern where skills are standalone CLI tools
that can be executed directly:
    python skills/analyze_file.py data.csv

Skills can also be imported and composed:
    from skills.analyze_file import analyze_file
    result = await analyze_file("/path/to/file.csv")

This module generates proper Python skill files with:
- Async function implementation
- CLI entrypoint for direct execution
- Import support for composition
- Access to generated tool bindings
"""

from pathlib import Path
from textwrap import dedent
from typing import Any, Optional


def generate_skill_file(
    name: str,
    description: str,
    code: str,
    parameters: Optional[list[dict[str, Any]]] = None,
    output_dir: Path = Path("./skills"),
) -> Path:
    """Generate a standalone Python skill file.
    
    Creates a skill file that can be:
    1. Run directly: `python skills/{name}.py [args]`
    2. Imported: `from skills.{name} import {name}`
    
    Args:
        name: Skill name (will be the filename and function name).
        description: Description of what the skill does.
        code: Python code implementing the skill (async function body).
        parameters: List of parameter definitions:
            [{"name": "file_path", "type": "str", "description": "...", "required": True}]
        output_dir: Directory to write the skill file.
    
    Returns:
        Path to the generated skill file.
    
    Example:
        generate_skill_file(
            name="analyze_csv",
            description="Analyze a CSV file and return statistics",
            code='''
                from generated.mcp.filesystem import read_file
                content = await read_file({"path": file_path})
                lines = content.split("\\n")
                return {"rows": len(lines), "columns": len(lines[0].split(","))}
            ''',
            parameters=[
                {"name": "file_path", "type": "str", "description": "Path to CSV file", "required": True}
            ]
        )
        
        # Generated file can be run:
        # python skills/analyze_csv.py data.csv
        # Or imported:
        # from skills.analyze_csv import analyze_csv
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build parameter list for function signature
    params = parameters or []
    param_strs = []
    required_params = []
    optional_params = []
    
    for p in params:
        type_hint = p.get("type", "Any")
        if p.get("required", True):
            required_params.append(f"{p['name']}: {type_hint}")
        else:
            default = p.get("default", "None")
            optional_params.append(f"{p['name']}: {type_hint} = {default!r}")
    
    all_params = required_params + optional_params
    param_signature = ", ".join(all_params) if all_params else ""
    
    # Build CLI argument parser
    cli_args_code = ""
    if params:
        cli_args_code = "    # Parse command line arguments\n"
        for i, p in enumerate(params):
            pname = p["name"]
            if p.get("required", True):
                cli_args_code += f"    {pname} = args[{i}] if len(args) > {i} else None\n"
                cli_args_code += f"    if {pname} is None:\n"
                cli_args_code += f'        print("Error: {pname} is required")\n'
                cli_args_code += f"        print_usage()\n"
                cli_args_code += f"        sys.exit(1)\n"
            else:
                default = p.get("default", "None")
                cli_args_code += f"    {pname} = args[{i}] if len(args) > {i} else {default!r}\n"
    
    # Build usage string
    usage_params = " ".join(
        f"<{p['name']}>" if p.get("required", True) else f"[{p['name']}]"
        for p in params
    )
    
    # Indent the user code
    code_lines = dedent(code).strip().split("\n")
    indented_code = "\n".join("    " + line for line in code_lines)
    
    # Generate the skill file
    skill_code = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Skill: {name}

{description}

Usage:
    python {name}.py {usage_params}

Or import:
    from skills.{name} import {name}
    result = await {name}(...)
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

# Add generated bindings to path
generated_path = Path(__file__).parent.parent / "generated"
if generated_path.exists() and str(generated_path) not in sys.path:
    sys.path.insert(0, str(generated_path))


async def {name}({param_signature}) -> Any:
    """{description}
    
    Args:
{chr(10).join(f"        {p['name']}: {p.get('description', '')}" for p in params) if params else "        None"}
    
    Returns:
        Skill result.
    """
{indented_code}


def print_usage():
    """Print usage information."""
    print(f"Usage: python {name}.py {usage_params}")
{chr(10).join(f'    print("  {p["name"]}: {p.get("description", "")}")' for p in params) if params else ""}


async def main():
    """CLI entrypoint."""
    args = sys.argv[1:]
    
{cli_args_code if cli_args_code else "    pass"}
    
    # Call the skill function
    result = await {name}({", ".join(p["name"] for p in params)})
    
    # Print the result
    if result is not None:
        import json
        if isinstance(result, (dict, list)):
            print(json.dumps(result, indent=2, default=str))
        else:
            print(result)


if __name__ == "__main__":
    asyncio.run(main())
'''
    
    # Write the file
    skill_path = output_dir / f"{name}.py"
    skill_path.write_text(skill_code)
    
    # Make it executable
    skill_path.chmod(0o755)
    
    return skill_path


def generate_skill_from_template(
    name: str,
    template: str,
    output_dir: Path = Path("./skills"),
    **variables,
) -> Path:
    """Generate a skill file from a template.
    
    Templates can include placeholders like {input_path} that get
    replaced with the provided variables.
    
    Args:
        name: Skill name.
        template: Template name or path.
        output_dir: Directory to write the skill file.
        **variables: Variables to substitute in the template.
    
    Returns:
        Path to the generated skill file.
    """
    # Built-in templates
    templates = {
        "file_processor": {
            "description": "Process files in a directory",
            "code": '''
from generated.mcp.filesystem import read_file, write_file, list_directory

entries = await list_directory({"path": input_dir})
results = []

for entry in entries.get("entries", []):
    if entry.endswith(file_extension):
        content = await read_file({"path": f"{input_dir}/{entry}"})
        # Process content here
        processed = content.upper()
        output_path = f"{output_dir}/{entry}"
        await write_file({"path": output_path, "content": processed})
        results.append({"file": entry, "status": "processed"})

return {"processed": len(results), "files": results}
''',
            "parameters": [
                {"name": "input_dir", "type": "str", "description": "Input directory", "required": True},
                {"name": "output_dir", "type": "str", "description": "Output directory", "required": True},
                {"name": "file_extension", "type": "str", "description": "File extension to process", "required": False, "default": ".txt"},
            ],
        },
        "api_fetcher": {
            "description": "Fetch data from an API with retry",
            "code": '''
from generated.mcp.bash import execute
import json

# Use curl to fetch data (bash tool)
result = await execute({"command": f"curl -s '{url}'"})
data = json.loads(result.get("stdout", "{}"))

# Filter or transform data
if filter_key and filter_value:
    data = [item for item in data if item.get(filter_key) == filter_value]

return data
''',
            "parameters": [
                {"name": "url", "type": "str", "description": "API URL to fetch", "required": True},
                {"name": "filter_key", "type": "str", "description": "Key to filter by", "required": False, "default": None},
                {"name": "filter_value", "type": "str", "description": "Value to filter", "required": False, "default": None},
            ],
        },
        "wait_for_condition": {
            "description": "Wait for a condition to become true",
            "code": '''
import asyncio
from datetime import datetime

start = datetime.now()
while True:
    # Check condition
    if check_command:
        from generated.mcp.bash import execute
        result = await execute({"command": check_command})
        if expected_output in result.get("stdout", ""):
            return {"success": True, "waited_seconds": (datetime.now() - start).seconds}
    
    elapsed = (datetime.now() - start).seconds
    if elapsed >= timeout:
        return {"success": False, "error": f"Timeout after {timeout}s"}
    
    await asyncio.sleep(interval)
''',
            "parameters": [
                {"name": "check_command", "type": "str", "description": "Command to check condition", "required": True},
                {"name": "expected_output", "type": "str", "description": "Expected output substring", "required": True},
                {"name": "timeout", "type": "int", "description": "Timeout in seconds", "required": False, "default": 60},
                {"name": "interval", "type": "int", "description": "Check interval in seconds", "required": False, "default": 5},
            ],
        },
    }
    
    if template in templates:
        tpl = templates[template]
        return generate_skill_file(
            name=name,
            description=tpl["description"].format(**variables) if variables else tpl["description"],
            code=tpl["code"],
            parameters=tpl["parameters"],
            output_dir=output_dir,
        )
    else:
        # Assume template is a path
        template_path = Path(template)
        if template_path.exists():
            template_content = template_path.read_text()
            # Simple variable substitution
            for key, value in variables.items():
                template_content = template_content.replace(f"{{{key}}}", str(value))
            return generate_skill_file(
                name=name,
                description=f"Skill generated from template: {template}",
                code=template_content,
                output_dir=output_dir,
            )
        else:
            raise ValueError(f"Unknown template: {template}")


__all__ = [
    "generate_skill_file",
    "generate_skill_from_template",
]
