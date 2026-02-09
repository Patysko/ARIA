"""
Skills Manager for ARIA.

Manages OpenClaw-compatible SKILL.md based skills.
The agent can create, load, list, and use skills autonomously.

Skill structure (OpenClaw/AgentSkills format):
    skill-name/
    â”œâ”€â”€ SKILL.md          (frontmatter + instructions)
    â”œâ”€â”€ scripts/          (executable code)
    â”œâ”€â”€ references/       (knowledge files)
    â””â”€â”€ assets/           (templates, data)
"""

import os
import re
import json
import time
import subprocess
from pathlib import Path
from typing import Optional


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from SKILL.md."""
    if not content.startswith("---"):
        return {}, content
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content

    frontmatter = {}
    for line in parts[1].strip().split("\n"):
        if ":" in line:
            key, _, val = line.partition(":")
            frontmatter[key.strip()] = val.strip()

    return frontmatter, parts[2].strip()


class Skill:
    def __init__(self, path: Path):
        self.path = path
        self.name = path.name
        self.skill_md = path / "SKILL.md"
        self.scripts_dir = path / "scripts"
        self.references_dir = path / "references"
        self.assets_dir = path / "assets"

        # Parse SKILL.md
        self.metadata = {}
        self.instructions = ""
        self.description = ""
        self.use_count = 0
        self.last_used = None
        self.created_at = None
        self.protected = False  # Protected skills cannot be modified by Thread 2
        self.error_log: list[dict] = []  # Recent errors, cleared after fix

        if self.skill_md.exists():
            content = self.skill_md.read_text()
            self.metadata, self.instructions = parse_frontmatter(content)
            self.name = self.metadata.get("name", self.name)
            self.description = self.metadata.get("description", "")
            # Protected flag from frontmatter: "protected: true"
            pv = self.metadata.get("protected", "false").lower().strip()
            self.protected = pv in ("true", "yes", "1")

        # Load stats if exist
        stats_file = path / ".stats.json"
        if stats_file.exists():
            try:
                stats = json.loads(stats_file.read_text())
                self.use_count = stats.get("use_count", 0)
                self.last_used = stats.get("last_used")
                self.created_at = stats.get("created_at")
                self.error_log = stats.get("error_log", [])
            except json.JSONDecodeError:
                pass

    def save_stats(self):
        stats_file = self.path / ".stats.json"
        stats_file.write_text(json.dumps({
            "use_count": self.use_count,
            "last_used": self.last_used,
            "created_at": self.created_at or time.time(),
            "error_log": self.error_log[-20:],  # keep last 20
        }))

    def use(self):
        """Mark skill as used."""
        self.use_count += 1
        self.last_used = time.time()
        self.save_stats()

    def get_scripts(self) -> list[Path]:
        if not self.scripts_dir.exists():
            return []
        return list(self.scripts_dir.glob("*"))

    def get_references(self) -> list[Path]:
        if not self.references_dir.exists():
            return []
        return list(self.references_dir.glob("*"))

    def run_script(self, script_name: str, args: list = None) -> dict:
        """Execute a script from this skill."""
        script_path = self.scripts_dir / script_name
        if not script_path.exists():
            return {"error": f"Skrypt nie znaleziony: {script_name}"}

        cmd = []
        if script_name.endswith(".py"):
            cmd = ["python3", str(script_path)]
        elif script_name.endswith(".sh"):
            cmd = ["bash", str(script_path)]
        else:
            cmd = [str(script_path)]

        if args:
            cmd.extend(args)

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30,
                cwd=str(self.path)
            )
            self.use()
            r = {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
            # Log errors
            if result.returncode != 0:
                self.log_error(script_name, result.stderr or result.stdout, args)
            return r
        except subprocess.TimeoutExpired:
            err = "Timeout (30s)"
            self.log_error(script_name, err, args)
            return {"error": err}
        except Exception as e:
            self.log_error(script_name, str(e), args)
            return {"error": str(e)}

    def log_error(self, script_name: str, error: str, args: list = None):
        """Record an error for this skill."""
        self.error_log.append({
            "script": script_name,
            "error": error[:500],
            "args": args or [],
            "timestamp": time.time(),
        })
        # Keep bounded
        if len(self.error_log) > 20:
            self.error_log = self.error_log[-20:]
        self.save_stats()

    def clear_errors(self):
        """Clear error log (called after successful fix cycle)."""
        self.error_log = []
        self.save_stats()

    def get_recent_errors(self, n: int = 5) -> list[dict]:
        """Get last N errors."""
        return self.error_log[-n:]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "path": str(self.path),
            "scripts": [s.name for s in self.get_scripts()],
            "references": [r.name for r in self.get_references()],
            "use_count": self.use_count,
            "last_used": self.last_used,
            "created_at": self.created_at,
            "protected": self.protected,
            "error_count": len(self.error_log),
            "recent_errors": self.error_log[-3:],
        }


class SkillsManager:
    def __init__(self, config):
        self.config = config
        self.skills_dir = config.SKILLS_DIR
        self.skills: dict[str, Skill] = {}
        self._load_all()

    def _load_all(self):
        """Scan skills directory and load all skills."""
        if not self.skills_dir.exists():
            return
        for item in self.skills_dir.iterdir():
            if item.is_dir() and (item / "SKILL.md").exists():
                skill = Skill(item)
                self.skills[skill.name] = skill

    def reload(self):
        """Reload all skills from disk."""
        self.skills.clear()
        self._load_all()

    def get(self, name: str) -> Optional[Skill]:
        return self.skills.get(name)

    def list_all(self) -> list[dict]:
        return [s.to_dict() for s in self.skills.values()]

    def list_names(self) -> list[str]:
        return list(self.skills.keys())

    def list_modifiable_names(self) -> list[str]:
        """List skill names that Thread 2 can modify."""
        return [n for n, s in self.skills.items() if not s.protected]

    def list_protected_names(self) -> list[str]:
        """List skill names that are protected from Thread 2 modifications."""
        return [n for n, s in self.skills.items() if s.protected]

    def is_protected(self, name: str) -> bool:
        """Check if a skill is protected."""
        skill = self.skills.get(name)
        return skill.protected if skill else False

    def find_relevant(self, query: str) -> list[Skill]:
        """Find skills relevant to a query based on description matching."""
        q = query.lower()
        scored = []
        for skill in self.skills.values():
            score = 0
            desc = skill.description.lower()
            name = skill.name.lower()
            if q in desc or q in name:
                score += 5
            tokens = set(q.split())
            score += sum(1 for t in tokens if t in desc or t in name)
            if score > 0:
                scored.append((score, skill))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored]

    def create_skill(self, name: str, description: str,
                     instructions: str, scripts: dict = None,
                     references: dict = None) -> Skill:
        """
        Create a new skill with OpenClaw-compatible format.

        Args:
            name: Skill identifier (kebab-case)
            description: When to trigger this skill
            instructions: Markdown body of SKILL.md
            scripts: dict of {filename: content} for scripts/
            references: dict of {filename: content} for references/
        """
        # Sanitize name
        safe_name = re.sub(r'[^a-z0-9-]', '-', name.lower().strip())
        skill_dir = self.skills_dir / safe_name

        # Create directories
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "scripts").mkdir(exist_ok=True)
        (skill_dir / "references").mkdir(exist_ok=True)
        (skill_dir / "assets").mkdir(exist_ok=True)

        # Write SKILL.md
        skill_md_content = f"""---
name: {safe_name}
description: {description}
---

{instructions}
"""
        (skill_dir / "SKILL.md").write_text(skill_md_content)

        # Write scripts
        if scripts:
            for filename, content in scripts.items():
                script_path = skill_dir / "scripts" / filename
                script_path.write_text(content)
                if filename.endswith((".sh", ".py")):
                    script_path.chmod(0o755)

        # Write references
        if references:
            for filename, content in references.items():
                (skill_dir / "references" / filename).write_text(content)

        # Write stats
        (skill_dir / ".stats.json").write_text(json.dumps({
            "use_count": 0,
            "last_used": None,
            "created_at": time.time(),
        }))

        # Load and register
        skill = Skill(skill_dir)
        self.skills[skill.name] = skill
        return skill

    def delete_skill(self, name: str) -> bool:
        """Delete a skill by name."""
        skill = self.skills.get(name)
        if not skill:
            return False
        import shutil
        shutil.rmtree(skill.path)
        del self.skills[name]
        return True

    def get_skills_prompt_section(self) -> str:
        """Generate a compact skills section for the system prompt (like OpenClaw does)."""
        if not self.skills:
            return "Brak zainstalowanych umiejÄ™tnoÅ›ci."

        lines = ["DostÄ™pne umiejÄ™tnoÅ›ci:"]
        for skill in self.skills.values():
            scripts = ", ".join(s.name for s in skill.get_scripts()) or "brak"
            prot = " [PROTECTED]" if skill.protected else ""
            lines.append(f"  â€¢ {skill.name}: {skill.description[:100]} [skrypty: {scripts}]{prot}")
        return "\n".join(lines)