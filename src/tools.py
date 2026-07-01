from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

SKILL_ALIASES = {
    "llm": "large language model",
    "rag": "retrieval augmented generation",
    "py": "python",
}


def load_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    file_path = Path(path)
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"JSON 文件不存在：{file_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 文件格式无效：{file_path}。{exc}") from exc
    except OSError as exc:
        raise OSError(f"读取 JSON 文件失败：{file_path}。{exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"JSON 文件内容必须是对象：{file_path}")
    return data


def load_text_file(path: str | Path) -> str:
    """Read a UTF-8 text file from disk."""
    file_path = Path(path)
    try:
        return file_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"文本文件不存在：{file_path}") from exc
    except OSError as exc:
        raise OSError(f"读取文本文件失败：{file_path}。{exc}") from exc


def list_jd_files(jd_dir: str | Path) -> list[Path]:
    """List .txt JD files in a folder."""
    folder = Path(jd_dir)
    if not folder.exists():
        raise FileNotFoundError(f"JD 文件夹不存在：{folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"JD 路径不是文件夹：{folder}")
    return sorted(folder.glob("*.txt"))


def load_user_profile(path: str = "data/user_profile.json") -> dict:
    """Load the candidate profile JSON file."""
    return load_json(path)


def load_jd_files(folder: str = "data/sample_jds") -> list[dict]:
    """Load all .txt job description files from a folder."""
    jd_files = []
    for file_path in list_jd_files(folder):
        jd_files.append(
            {
                "filename": file_path.name,
                "raw_text": load_text_file(file_path),
            }
        )
    return jd_files


def save_json(data: dict | list, path: str) -> None:
    """Save dict or list data as UTF-8 JSON."""
    if not isinstance(data, (dict, list)):
        raise TypeError("save_json 只支持保存 dict 或 list。")

    file_path = Path(path)
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        raise OSError(f"保存 JSON 文件失败：{file_path}。{exc}") from exc


def save_markdown(text: str, path: str) -> None:
    """Save Markdown text as UTF-8."""
    if not isinstance(text, str):
        raise TypeError("save_markdown 只支持保存 str 文本。")

    file_path = Path(path)
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise OSError(f"保存 Markdown 文件失败：{file_path}。{exc}") from exc


def generate_job_id(filename: str, index: int) -> str:
    """Generate a stable job_id from a filename and index."""
    if index < 0:
        raise ValueError("index 必须大于等于 0。")

    stem = Path(filename).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    if not slug:
        slug = f"{index:02d}"
    elif not re.search(r"_\d+$", slug):
        slug = f"{slug}_{index:02d}"

    if slug.startswith("job_"):
        return slug
    return f"job_{slug}"


def normalize_skill(skill: str) -> str:
    """Normalize a skill string for simple matching."""
    if not isinstance(skill, str):
        raise TypeError("skill 必须是 str。")

    normalized = " ".join(skill.strip().lower().split())
    return SKILL_ALIASES.get(normalized, normalized)


def simple_skill_overlap(profile_skills: list[str], job_skills: list[str]) -> list[str]:
    """Return normalized skill overlap, preserving profile skill order."""
    normalized_job_skills = {normalize_skill(skill) for skill in job_skills}
    overlap = []
    seen = set()

    for skill in profile_skills:
        normalized = normalize_skill(skill)
        if normalized in normalized_job_skills and normalized not in seen:
            overlap.append(normalized)
            seen.add(normalized)

    return overlap
