# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Skill Files - Skills as importable Python code files.

This module implements the primary "skills as code" pattern where skills
are Python files that agents can:

- **Discover** by reading the skills directory
- **Create** by writing Python files
- **Execute** by importing and calling them
- **Compose** by importing multiple skills together

This is the recommended approach for building agent skills.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Optional
from dataclasses import dataclass, field


@dataclass
class SkillFile:
    """A skill represented as a Python file.
    
    Skills are Python files with async functions that compose MCP tools
    or other skills to accomplish tasks.
    
    Attributes:
        path: Path to the skill file.
        name: Skill name (derived from filename).
        module_name: Python module name for imports.
        description: Description from module docstring.
        functions: List of exported async functions.
    
    Example:
        skill = SkillFile.from_file(Path("skills/analyze_csv.py"), Path("skills"))
        func = skill.get_function()
        result = await func("/data/file.csv")
    """
    path: Path
    name: str
    module_name: str
    description: str = ""
    functions: list[str] = field(default_factory=list)
    
    @classmethod
    def from_file(cls, path: Path, skills_root: Path) -> "SkillFile":
        """Load skill metadata from a Python file.
        
        Args:
            path: Path to the skill file.
            skills_root: Root directory for skills (for module name calculation).
            
        Returns:
            SkillFile with metadata extracted from the file.
        """
        name = path.stem
        
        # Calculate module name relative to skills root
        try:
            relative = path.relative_to(skills_root)
            module_parts = list(relative.parent.parts) + [name]
            module_name = ".".join(["skills"] + list(module_parts))
        except ValueError:
            module_name = f"skills.{name}"
        
        # Parse the file to extract metadata
        content = path.read_text()
        description = ""
        functions = []
        
        try:
            tree = ast.parse(content)
            
            # Get module docstring
            if (tree.body and isinstance(tree.body[0], ast.Expr) and 
                isinstance(tree.body[0].value, ast.Constant)):
                description = tree.body[0].value.value
            
            # Find async functions (these are the skill's exported functions)
            for node in ast.walk(tree):
                if isinstance(node, ast.AsyncFunctionDef):
                    if not node.name.startswith("_"):
                        functions.append(node.name)
        except SyntaxError:
            pass
        
        return cls(
            path=path,
            name=name,
            module_name=module_name,
            description=description,
            functions=functions,
        )
    
    def load_module(self) -> Any:
        """Load the skill as a Python module.
        
        Returns:
            The loaded module.
        """
        spec = importlib.util.spec_from_file_location(self.module_name, self.path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load skill: {self.path}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[self.module_name] = module
        spec.loader.exec_module(module)
        
        return module
    
    def get_function(self, name: Optional[str] = None) -> Callable:
        """Get a function from the skill.
        
        Args:
            name: Function name. If None, returns the first public function
                  or a function matching the skill name.
        
        Returns:
            The async function.
        """
        module = self.load_module()
        
        if name:
            return getattr(module, name)
        
        # Try to find a function with the same name as the skill
        if hasattr(module, self.name):
            return getattr(module, self.name)
        
        # Return the first public async function
        if self.functions:
            return getattr(module, self.functions[0])
        
        raise AttributeError(f"No callable function found in skill: {self.name}")


class SkillDirectory:
    """Manager for skills stored as Python files in a directory.
    
    This implements the "skills as code" pattern:
    
    1. Skills are Python files with async functions
    2. Agents discover skills by listing the directory
    3. Agents create skills by writing Python files
    4. Agents execute skills by importing and calling them
    5. Agents compose skills by importing multiple skills together
    
    Example:
        # Initialize skills directory
        skills = SkillDirectory("./workspace/skills")
        
        # List available skills
        for skill in skills.list():
            print(f"{skill.name}: {skill.description}")
        
        # Get and execute a skill
        skill = skills.get("analyze_csv")
        func = skill.get_function()
        result = await func("/data/file.csv")
        
        # Create a new skill
        skills.create(
            name="process_batch",
            code='''
async def process_batch(input_dir: str, output_dir: str) -> dict:
    from skills.analyze_csv import analyze_csv
    from generated.mcp.filesystem import list_directory
    
    entries = await list_directory({"path": input_dir})
    results = []
    
    for entry in entries.get("entries", []):
        if entry.endswith(".csv"):
            result = await analyze_csv(f"{input_dir}/{entry}")
            results.append(result)
    
    return {"processed": len(results), "results": results}
''',
            description="Process files in batch",
        )
    """
    
    def __init__(self, path: str = "./skills"):
        """Initialize the skill directory.
        
        Args:
            path: Path to the skills directory.
        """
        self.path = Path(path)
        self._ensure_directory()
    
    def _ensure_directory(self) -> None:
        """Ensure the skills directory exists with proper structure."""
        self.path.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py for imports
        init_file = self.path / "__init__.py"
        if not init_file.exists():
            init_file.write_text('"""Skills directory - reusable code compositions."""\n')
        
        # Create README
        readme = self.path / "README.md"
        if not readme.exists():
            readme.write_text(self._get_readme_content())
    
    def _get_readme_content(self) -> str:
        """Get the README content for the skills directory."""
        return '''# Skills Directory

This directory contains reusable skill files - Python code that composes
multiple tools to accomplish specific tasks.

## What are Skills?

Skills are Python files with async functions that you can:
- **Discover** by reading this directory
- **Create** by writing Python files here
- **Execute** by importing and calling them in your code
- **Compose** by importing multiple skills together

## Creating a Skill

Create a Python file with an async function:

```python
#!/usr/bin/env python3
"""Analyze a CSV file and return statistics."""

async def analyze_csv(file_path: str) -> dict:
    """Analyze a CSV file.
    
    Args:
        file_path: Path to the CSV file.
    
    Returns:
        Statistics about the file.
    """
    from generated.mcp.filesystem import read_file
    
    content = await read_file({"path": file_path})
    lines = content.split("\\n")
    headers = lines[0].split(",") if lines else []
    
    return {
        "rows": len(lines) - 1,
        "columns": len(headers),
        "headers": headers,
    }


# Optional: CLI support for direct execution
if __name__ == "__main__":
    import asyncio
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_csv.py <file_path>")
        sys.exit(1)
    
    result = asyncio.run(analyze_csv(sys.argv[1]))
    import json
    print(json.dumps(result, indent=2))
```

## Using a Skill

In your executed code, import and call the skill:

```python
from skills.analyze_csv import analyze_csv

result = await analyze_csv("/data/sales.csv")
print(f"Found {result['rows']} rows")
```

## Composing Skills

Skills can import and use other skills:

```python
"""Process and analyze multiple files."""

async def batch_analyze(directory: str) -> list:
    from skills.analyze_csv import analyze_csv
    from generated.mcp.filesystem import list_directory
    
    entries = await list_directory({"path": directory})
    results = []
    
    for entry in entries.get("entries", []):
        if entry.endswith(".csv"):
            result = await analyze_csv(f"{directory}/{entry}")
            results.append({"file": entry, **result})
    
    return results
```

## Running Skills Directly

Skills with CLI support can be run directly:

```bash
python skills/analyze_csv.py /data/sales.csv
```
'''
    
    def list(self) -> list[SkillFile]:
        """List all skills in the directory.
        
        Returns:
            List of SkillFile objects.
        """
        skills = []
        
        for py_file in self.path.rglob("*.py"):
            # Skip __init__.py and private files
            if py_file.name.startswith("_"):
                continue
            
            try:
                skill = SkillFile.from_file(py_file, self.path)
                if skill.functions:  # Only include files with async functions
                    skills.append(skill)
            except Exception:
                pass
        
        return skills
    
    def get(self, name: str) -> Optional[SkillFile]:
        """Get a skill by name.
        
        Args:
            name: Skill name (filename without .py).
        
        Returns:
            SkillFile or None if not found.
        """
        # Try direct path first
        skill_path = self.path / f"{name}.py"
        if skill_path.exists():
            return SkillFile.from_file(skill_path, self.path)
        
        # Search in subdirectories
        for py_file in self.path.rglob(f"{name}.py"):
            if not py_file.name.startswith("_"):
                return SkillFile.from_file(py_file, self.path)
        
        return None
    
    def search(self, query: str, limit: int = 10) -> list[SkillFile]:
        """Search for skills matching a query.
        
        Args:
            query: Search query.
            limit: Maximum results.
        
        Returns:
            Matching skills sorted by relevance.
        """
        query_lower = query.lower()
        query_words = query_lower.split()
        
        scored = []
        for skill in self.list():
            text = f"{skill.name} {skill.description} {' '.join(skill.functions)}".lower()
            score = sum(1 for word in query_words if word in text)
            if score > 0:
                scored.append((score, skill))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [skill for _, skill in scored[:limit]]
    
    def create(
        self,
        name: str,
        code: str,
        description: str = "",
        make_executable: bool = True,
    ) -> SkillFile:
        """Create a new skill file.
        
        Args:
            name: Skill name (will be the filename).
            code: Python code (should define async functions).
            description: Description for the skill.
            make_executable: Whether to make the file executable.
        
        Returns:
            The created SkillFile.
        """
        # Build the file content
        content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""{description or f"Skill: {name}"}"""

{code.strip()}
'''
        
        # Write the file
        skill_path = self.path / f"{name}.py"
        skill_path.write_text(content)
        
        if make_executable:
            skill_path.chmod(0o755)
        
        return SkillFile.from_file(skill_path, self.path)
    
    def delete(self, name: str) -> bool:
        """Delete a skill.
        
        Args:
            name: Skill name.
        
        Returns:
            True if deleted, False if not found.
        """
        skill = self.get(name)
        if skill:
            skill.path.unlink()
            return True
        return False
    
    def add_to_sys_path(self) -> None:
        """Add the skills directory to sys.path for imports."""
        path_str = str(self.path.parent)  # Parent so "from skills.X import Y" works
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def setup_skills_directory(path: str = "./skills") -> SkillDirectory:
    """Set up a skills directory and add it to sys.path.
    
    This should be called during sandbox initialization to enable
    skill imports in executed code.
    
    Args:
        path: Path to the skills directory.
    
    Returns:
        Configured SkillDirectory.
    
    Example:
        # In sandbox setup
        skills = setup_skills_directory("./workspace/skills")
        
        # Now executed code can import skills
        # from skills.my_skill import my_function
    """
    skills = SkillDirectory(path)
    skills.add_to_sys_path()
    return skills
