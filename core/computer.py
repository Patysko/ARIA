"""
Computer Tools for ARIA.

Provides the agent with ability to interact with the local system:
- Execute shell commands
- Read/write/list files
- Run Python code
- Inspect system information
"""

import os
import subprocess
import platform
import shutil
from pathlib import Path
from typing import Optional


class ComputerTools:
    """System interaction layer for the agent."""

    def __init__(self, working_dir: str = None):
        self.working_dir = Path(working_dir or os.path.expanduser("~/aria-workspace"))
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.command_history: list[dict] = []

    def execute(self, command: str, timeout: int = 30, cwd: str = None) -> dict:
        """Execute a shell command."""
        work_dir = cwd or str(self.working_dir)
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=timeout, cwd=work_dir
            )
            entry = {
                "command": command,
                "stdout": result.stdout[-2000:] if result.stdout else "",
                "stderr": result.stderr[-1000:] if result.stderr else "",
                "returncode": result.returncode,
                "success": result.returncode == 0,
            }
            self.command_history.append(entry)
            return entry
        except subprocess.TimeoutExpired:
            return {"command": command, "error": f"Timeout ({timeout}s)", "success": False}
        except Exception as e:
            return {"command": command, "error": str(e), "success": False}

    def run_python(self, code: str, timeout: int = 30) -> dict:
        """Execute Python code and return output."""
        tmp_file = self.working_dir / "_aria_tmp_exec.py"
        tmp_file.write_text(code)
        result = self.execute(f"python3 {tmp_file}", timeout=timeout)
        tmp_file.unlink(missing_ok=True)
        return result

    def pip_install(self, packages: list[str], timeout: int = 120) -> dict:
        """Install Python packages via pip. Tries multiple methods."""
        if not packages:
            return {"success": True, "installed": []}
        safe = [p.strip() for p in packages if p.strip() and all(
            c.isalnum() or c in '-_.[]=<>!' for c in p.strip())]
        if not safe:
            return {"success": False, "error": "No valid package names"}

        pkg_str = ' '.join(safe)
        # Try multiple pip variants
        commands = [
            f"python3 -m pip install --break-system-packages -q {pkg_str}",
            f"pip3 install --break-system-packages -q {pkg_str}",
            f"pip install --break-system-packages -q {pkg_str}",
            f"python3 -m pip install --user -q {pkg_str}",
            f"pip3 install --user -q {pkg_str}",
        ]
        for cmd in commands:
            result = self.execute(cmd, timeout=timeout)
            if result.get("returncode", 1) == 0:
                return {
                    "success": True, "installed": safe,
                    "stdout": result.get("stdout", ""),
                    "stderr": result.get("stderr", ""),
                    "command": cmd,
                }
        # All failed â€” return last error
        return {
            "success": False, "installed": [],
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "error": result.get("error", result.get("stderr", "install failed")),
        }

    def extract_imports(self, code: str) -> list[str]:
        """Extract non-stdlib package names from Python code."""
        import re as _re
        STDLIB = {
            'abc','argparse','ast','asyncio','base64','bisect','calendar',
            'cgi','codecs','collections','colorsys','configparser','contextlib',
            'copy','csv','ctypes','dataclasses','datetime','decimal','difflib',
            'dis','email','enum','errno','faulthandler','fileinput','fnmatch',
            'fractions','ftplib','functools','gc','getpass','gettext','glob',
            'gzip','hashlib','heapq','hmac','html','http','imaplib','importlib',
            'inspect','io','ipaddress','itertools','json','keyword','linecache',
            'locale','logging','lzma','mailbox','math','mimetypes','mmap',
            'multiprocessing','numbers','operator','os','pathlib','pdb',
            'pickle','pkgutil','platform','plistlib','poplib','posixpath',
            'pprint','profile','pstats','py_compile','pydoc','queue',
            'quopri','random','re','readline','reprlib','resource','rlcompleter',
            'sched','secrets','select','selectors','shelve','shlex','shutil',
            'signal','site','smtplib','socket','socketserver','sqlite3','ssl',
            'stat','statistics','string','struct','subprocess','sys',
            'sysconfig','syslog','tabnanny','tarfile','tempfile','test',
            'textwrap','threading','time','timeit','token','tokenize',
            'trace','traceback','tracemalloc','tty','turtle','turtledemo',
            'types','typing','unicodedata','unittest','urllib','uuid',
            'venv','warnings','wave','weakref','webbrowser','winreg',
            'winsound','wsgiref','xdrlib','xml','xmlrpc','zipfile','zipimport',
            'zlib','_thread','__future__','builtins',
        }
        found = set()
        for match in _re.finditer(r'^\s*(?:import|from)\s+(\w+)', code, _re.MULTILINE):
            pkg = match.group(1)
            if pkg not in STDLIB:
                found.add(pkg)
        # Map common import names to pip names
        PIP_MAP = {
            'cv2': 'opencv-python', 'PIL': 'Pillow', 'bs4': 'beautifulsoup4',
            'sklearn': 'scikit-learn', 'yaml': 'pyyaml', 'dotenv': 'python-dotenv',
            'gi': 'PyGObject', 'attr': 'attrs', 'dateutil': 'python-dateutil',
        }
        return [PIP_MAP.get(p, p) for p in found]

    def read_file(self, path: str) -> dict:
        """Read a file's contents."""
        try:
            p = Path(path).expanduser()
            if not p.exists():
                return {"error": f"Plik nie istnieje: {path}"}
            if p.stat().st_size > 1_000_000:
                return {"error": "Plik za duÅ¼y (>1MB)", "size": p.stat().st_size}
            content = p.read_text(errors="replace")
            return {"content": content, "path": str(p), "size": len(content)}
        except Exception as e:
            return {"error": str(e)}

    def write_file(self, path: str, content: str) -> dict:
        """Write content to a file."""
        try:
            p = Path(path).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            return {"success": True, "path": str(p), "size": len(content)}
        except Exception as e:
            return {"error": str(e)}

    def list_dir(self, path: str = ".", max_depth: int = 2) -> dict:
        """List directory contents."""
        try:
            p = Path(path).expanduser()
            if not p.exists():
                return {"error": f"Katalog nie istnieje: {path}"}

            items = []
            for item in sorted(p.iterdir()):
                if item.name.startswith("."):
                    continue
                entry = {
                    "name": item.name,
                    "type": "dir" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None,
                }
                if item.is_dir() and max_depth > 1:
                    children = []
                    try:
                        for child in sorted(item.iterdir()):
                            if not child.name.startswith("."):
                                children.append({
                                    "name": child.name,
                                    "type": "dir" if child.is_dir() else "file",
                                })
                    except PermissionError:
                        pass
                    entry["children"] = children[:20]
                items.append(entry)

            return {"path": str(p), "items": items}
        except Exception as e:
            return {"error": str(e)}

    def system_info(self) -> dict:
        """Get system information."""
        return {
            "os": platform.system(),
            "os_version": platform.version(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "hostname": platform.node(),
            "working_dir": str(self.working_dir),
            "disk_free": shutil.disk_usage("/").free // (1024 * 1024),  # MB
            "home": str(Path.home()),
        }

    def file_exists(self, path: str) -> bool:
        return Path(path).expanduser().exists()

    def get_recent_commands(self, n: int = 5) -> list[dict]:
        return self.command_history[-n:]