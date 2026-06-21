from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_RECORDS_PATH = Path("data/jobs_csv/job_records.jsonl")
DEFAULT_JD_DIR = Path("data/sample_jds")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strip_marker(line: str) -> str:
    return line.strip().lstrip("-*•0123456789.、) ").strip()


def _first_prefixed_value(lines: list[str], prefixes: tuple[str, ...]) -> str | None:
    for line in lines:
        lowered = line.casefold()
        if lowered.startswith(prefixes):
            return line.split(":", 1)[-1].split("：", 1)[-1].strip() or None
    return None


def _section_lines(lines: list[str], section_markers: tuple[str, ...]) -> list[str]:
    collected: list[str] = []
    active = False
    all_markers = (
        "responsibilities",
        "requirements",
        "required skills",
        "preferred",
        "preferred skills",
        "岗位职责",
        "工作职责",
        "职位描述",
        "岗位要求",
        "任职要求",
        "任职资格",
        "加分项",
        "优先",
    )

    for raw_line in lines:
        line = raw_line.strip()
        lowered = line.casefold().rstrip(":：")
        if any(lowered.startswith(marker) for marker in section_markers):
            active = True
            continue
        if active and any(lowered.startswith(marker) for marker in all_markers):
            break
        if active and line:
            collected.append(_strip_marker(line))
    return collected


def _slugify(text: str) -> str:
    text = text.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return re.sub(r"_+", "_", slug)


def _stable_job_id(title: str, company: str, raw_text: str, filename: str | None = None) -> str:
    filename_stem = Path(filename).stem if filename else ""
    base = _slugify(f"{filename_stem}_{company}_{title}") or "job"
    digest = hashlib.sha1(raw_text.encode("utf-8")).hexdigest()[:8]
    return f"job_{base}_{digest}" if not base.startswith("job_") else f"{base}_{digest}"


def _existing_job_ids(records_path: Path) -> set[str]:
    if not records_path.exists():
        return set()

    job_ids = set()
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        job_id = data.get("job_id")
        if isinstance(job_id, str):
            job_ids.add(job_id)
    return job_ids


def extract_job_record(raw_text: str, filename: str | None = None, source: str = "frontend") -> dict[str, Any]:
    """Extract a lightweight structured job record from JD text without calling an LLM."""
    text = raw_text.strip()
    if not text:
        raise ValueError("岗位文本不能为空。")

    lines = [_strip_marker(line) for line in text.splitlines() if _strip_marker(line)]
    title = _first_prefixed_value(lines, ("title:", "title：", "岗位:", "岗位：", "职位:", "职位："))
    if not title:
        title = lines[0] if lines else Path(filename or "job").stem

    company = _first_prefixed_value(lines, ("company:", "company：", "公司:", "公司：")) or "未知公司"
    location = _first_prefixed_value(lines, ("location:", "location：", "地点:", "地点：", "工作地点:", "工作地点："))
    salary = _first_prefixed_value(lines, ("salary:", "salary：", "薪资:", "薪资："))
    duration = _first_prefixed_value(lines, ("duration:", "duration：", "周期:", "周期：", "实习周期:", "实习周期："))
    education = _first_prefixed_value(lines, ("education:", "education：", "学历:", "学历："))

    responsibilities = _section_lines(lines, ("responsibilities", "岗位职责", "工作职责", "职位描述"))
    required_skills = _section_lines(lines, ("requirements", "required skills", "岗位要求", "任职要求", "任职资格"))
    preferred_skills = _section_lines(lines, ("preferred", "preferred skills", "加分项", "优先"))

    job_id = _stable_job_id(title, company, text, filename)
    return {
        "job_id": job_id,
        "title": title,
        "company": company,
        "location": location,
        "salary": salary,
        "duration": duration,
        "education": education,
        "responsibilities": responsibilities,
        "required_skills": required_skills,
        "preferred_skills": preferred_skills,
        "raw_text": text,
        "source": source,
        "source_filename": filename,
        "created_at": _utc_now(),
    }


def format_job_record_as_jd(record: dict[str, Any]) -> str:
    """Render a saved job record as the local JD text format used by JobPilot."""
    lines = [
        f"Title: {record.get('title') or '未命名岗位'}",
        f"Company: {record.get('company') or '未知公司'}",
    ]
    if record.get("location"):
        lines.append(f"Location: {record['location']}")
    if record.get("salary"):
        lines.append(f"Salary: {record['salary']}")
    if record.get("duration"):
        lines.append(f"Duration: {record['duration']}")
    if record.get("education"):
        lines.append(f"Education: {record['education']}")

    sections = [
        ("Responsibilities", record.get("responsibilities")),
        ("Required Skills", record.get("required_skills")),
        ("Preferred Skills", record.get("preferred_skills")),
    ]
    for heading, items in sections:
        lines.extend(["", f"{heading}:"])
        values = items if isinstance(items, list) else []
        if values:
            lines.extend(f"- {item}" for item in values)
        else:
            lines.append("- 暂无结构化提取结果，请参考 raw_text。")

    lines.extend(["", "Raw Text:", str(record.get("raw_text") or "")])
    return "\n".join(lines).rstrip() + "\n"


def save_job_record(
    raw_text: str,
    filename: str | None = None,
    source: str = "frontend",
    records_path: str | Path = DEFAULT_RECORDS_PATH,
    jd_dir: str | Path = DEFAULT_JD_DIR,
) -> dict[str, Any]:
    """Save one job into JSONL records and a local JD .txt file."""
    records_file = Path(records_path)
    jd_folder = Path(jd_dir)
    record = extract_job_record(raw_text=raw_text, filename=filename, source=source)

    records_file.parent.mkdir(parents=True, exist_ok=True)
    jd_folder.mkdir(parents=True, exist_ok=True)

    existing_ids = _existing_job_ids(records_file)
    already_exists = record["job_id"] in existing_ids
    if not already_exists:
        with records_file.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    jd_path = jd_folder / f"{record['job_id']}.txt"
    jd_path.write_text(format_job_record_as_jd(record), encoding="utf-8")

    return {
        **record,
        "already_exists": already_exists,
        "record_path": str(records_file),
        "jd_file_path": str(jd_path),
    }


def save_job_records(
    jd_texts: list[str],
    jd_filenames: list[str] | None = None,
    source: str = "frontend",
    records_path: str | Path = DEFAULT_RECORDS_PATH,
    jd_dir: str | Path = DEFAULT_JD_DIR,
) -> list[dict[str, Any]]:
    """Save multiple JD texts and skip empty inputs."""
    filenames = jd_filenames or []
    saved = []
    for index, raw_text in enumerate(jd_texts):
        text = raw_text.strip() if isinstance(raw_text, str) else ""
        if not text:
            continue
        filename = filenames[index] if index < len(filenames) else None
        saved.append(
            save_job_record(
                raw_text=text,
                filename=filename,
                source=source,
                records_path=records_path,
                jd_dir=jd_dir,
            )
        )
    return saved
