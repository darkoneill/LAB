"""
Repository Map - AST-based project skeleton generator.

Generates a compressed, skeletal view of a Python project:
- Class names and their method signatures
- Top-level function signatures
- Module-level constants

This map is injected into Planner/Coder system prompts so the agent
has global awareness of the codebase without consuming full source tokens.
"""

import ast
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("openclaw.tools.repo_map")

# Maximum characters for the generated map (token budget guard)
_MAX_MAP_CHARS = 6000


def generate_repo_map(
    root: str,
    max_chars: int = _MAX_MAP_CHARS,
    include_globs: Optional[list[str]] = None,
    exclude_dirs: Optional[set[str]] = None,
) -> str:
    """
    Walk a project tree and produce a skeletal map using Python's ast module.

    Args:
        root: Root directory to scan.
        max_chars: Hard cap on output length (prevents token overflow).
        include_globs: If set, only match these patterns (default: all .py).
        exclude_dirs: Directory names to skip (default: common noise dirs).

    Returns:
        A compact multi-line string representing the project skeleton.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        return f"# repo_map: {root} is not a directory"

    exclude_dirs = exclude_dirs or {
        "__pycache__", ".git", ".venv", "venv", "node_modules",
        ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
        ".eggs", "*.egg-info",
    }

    py_files = sorted(root_path.rglob("*.py"))
    lines: list[str] = [f"# Repository Map: {root_path.name}/"]

    for py_file in py_files:
        # Skip excluded directories
        parts = py_file.relative_to(root_path).parts
        if any(p in exclude_dirs or p.endswith(".egg-info") for p in parts):
            continue

        rel = str(py_file.relative_to(root_path))
        file_lines = _parse_file(py_file)

        if file_lines:
            lines.append("")
            lines.append(f"## {rel}")
            lines.extend(file_lines)

        # Budget guard: stop early if we're over budget
        current_len = sum(len(l) + 1 for l in lines)
        if current_len > max_chars:
            lines.append("")
            lines.append(f"# ... truncated ({len(py_files)} files total)")
            break

    result = "\n".join(lines)
    if len(result) > max_chars:
        result = result[:max_chars] + "\n# ... truncated"

    logger.debug(f"repo_map generated: {len(result)} chars from {len(py_files)} files")
    return result


def _parse_file(filepath: Path) -> list[str]:
    """Parse a single Python file and extract its skeleton."""
    try:
        source = filepath.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError) as e:
        return [f"  # parse error: {e.__class__.__name__}"]

    lines: list[str] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            bases = ", ".join(_name_of(b) for b in node.bases) if node.bases else ""
            bases_str = f"({bases})" if bases else ""
            lines.append(f"  class {node.name}{bases_str}:")
            for item in ast.iter_child_nodes(node):
                if isinstance(item, ast.FunctionDef) or isinstance(item, ast.AsyncFunctionDef):
                    sig = _func_signature(item)
                    prefix = "async " if isinstance(item, ast.AsyncFunctionDef) else ""
                    lines.append(f"    {prefix}def {sig}")

        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            sig = _func_signature(node)
            prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
            lines.append(f"  {prefix}def {sig}")

        elif isinstance(node, ast.Assign):
            # Top-level constants (ALL_CAPS only)
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    lines.append(f"  {target.id} = ...")

    return lines


def _func_signature(node) -> str:
    """Extract function name and parameter names (no defaults/annotations for brevity)."""
    args = node.args
    params = []

    for arg in args.args:
        params.append(arg.arg)

    if args.vararg:
        params.append(f"*{args.vararg.arg}")

    for arg in args.kwonlyargs:
        params.append(arg.arg)

    if args.kwarg:
        params.append(f"**{args.kwarg.arg}")

    return_hint = ""
    if node.returns:
        return_hint = f" -> {_name_of(node.returns)}"

    return f"{node.name}({', '.join(params)}){return_hint}"


def _name_of(node) -> str:
    """Best-effort name extraction from an AST node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_name_of(node.value)}.{node.attr}"
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Subscript):
        return f"{_name_of(node.value)}[...]"
    return "..."
