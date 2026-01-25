# Simple Skills Examples

Comprehensive example demonstrating all agent-skills features.

## Run

```bash
python skills_example.py
```

## What This Example Demonstrates

### 1. Skill Creation (Programmatic)

```python
from agent_skills import SkillsManager, SkillContext

manager = SkillsManager("./skills")
skill = manager.create(
    name="data_analyzer",
    description="Analyze data from a file",
    content="# Data Analyzer\n...",
    python_code='print("Analyzing...")',
    allowed_tools=["filesystem__read_file"],
    tags=["data", "analysis"],
    context=SkillContext.FORK,
)
```

### 2. SKILL.md Format (Claude Code Compatible)

Parse and generate SKILL.md files with YAML frontmatter:

```python
from agent_skills import Skill

skill = Skill.from_skill_md("""---
name: web_scraper
description: Scrape and extract data from web pages
version: 1.0.0
allowed-tools:
  - http__fetch
  - html__parse
---

# Web Scraper Skill
...
""")
```

### 3. Skill Discovery

```python
# Discover skills from a directory
discovered = manager.discover()

# Search for skills
result = manager.search("data processing", limit=5)

# Filter by tags
data_skills = manager.list(tags=["data"])
```

### 4. Skill Execution

```python
skill = manager.get("data_analyzer")
execution = await manager.execute(
    skill,
    arguments={"file_path": "/tmp/data.txt"},
    timeout=10.0,
)
print(f"Success: {execution.success}")
```

### 5. Skill Versioning

```python
# Create versions
manager.create_version(skill.skill_id, "1.0.0")
manager.update(skill.skill_id, description="Updated description")
manager.create_version(skill.skill_id, "1.1.0")

# List versions
versions = manager.list_versions(skill.skill_id)
```

### 6. Skills as Code Files

```python
from agent_skills.simple import SimpleSkillsManager, SimpleSkill

manager = SimpleSkillsManager("./skills")
skill = SimpleSkill(
    name="greet_user",
    description="Return a greeting",
    code='async def greet_user(name: str) -> dict:\n    return {"message": f"Hello, {name}!"}'
)
manager.save_skill(skill)
```

### 7. MCP Server

```python
from agent_skills import skills_server, configure_server

configure_server(skills_path="./skills")
skills_server.run()
```

## Output

Running the example produces output for each demonstration, showing:
- Created skills with IDs and metadata
- Parsed SKILL.md content
- Discovery and search results
- Execution output and timing
- Version history
- MCP server tool descriptions
