from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_RECORDS_PATH = Path("data/jobs_csv/job_records.jsonl")
DEFAULT_JD_DIR = Path("data/sample_jds")

_SECTION_MARKERS = (
    "responsibilities",
    "requirements",
    "required skills",
    "preferred",
    "preferred skills",
    "岗位职责",
    "工作职责",
    "职位描述",
    "工作内容",
    "岗位要求",
    "任职要求",
    "任职资格",
    "职位要求",
    "加分项",
    "加分项目",
    "优先条件",
)
_SALARY_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*[-~—至]\s*\d+(?:\.\d+)?\s*(?:元|k|K)(?:\s*/\s*(?:天|日|月))?")
_DURATION_PATTERN = re.compile(r"(?:持续|实习(?:期|周期|时长)?[:：]?\s*)?(\d+\s*(?:个)?月(?:及?以上)?)")
_EDUCATION_PATTERN = re.compile(r"(学历不限|大专|本科(?:及以上)?|硕士(?:及以上)?|研究生(?:及以上)?|博士(?:及以上)?)")
_CITY_PATTERN = re.compile(
    r"(北京|上海|深圳|广州|杭州|南京|苏州|成都|武汉|西安|天津|重庆|长沙|合肥|厦门|青岛|郑州|远程)"
)
_FOOTER_MARKERS = (
    "工作地址",
    "工作地点",
    "职位发布者",
    "招聘者",
    "立即沟通",
    "去app",
    "收藏",
    "举报",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strip_marker(line: str) -> str:
    cleaned = line.strip()
    cleaned = re.sub(r"^[一二三四五六七八九十]+[、.．]\s*", "", cleaned)
    return cleaned.lstrip("-*•0123456789.、) ").strip()


def _first_prefixed_value(lines: list[str], prefixes: tuple[str, ...]) -> str | None:
    for line in lines:
        lowered = line.casefold()
        if lowered.startswith(prefixes):
            return line.split(":", 1)[-1].split("：", 1)[-1].strip() or None
    return None


def _section_lines(lines: list[str], section_markers: tuple[str, ...]) -> list[str]:
    collected: list[str] = []
    active = False

    for raw_line in lines:
        line = raw_line.strip()
        lowered = line.casefold().rstrip(":：")
        matched_marker = next(
            (marker for marker in sorted(section_markers, key=len, reverse=True) if lowered.startswith(marker)),
            None,
        )
        if matched_marker:
            active = True
            remainder = re.sub(
                rf"^{re.escape(matched_marker)}\s*[:：]?\s*",
                "",
                line,
                count=1,
                flags=re.IGNORECASE,
            ).strip()
            if remainder:
                collected.append(_strip_marker(remainder))
            continue
        if active and any(lowered.startswith(marker) for marker in _SECTION_MARKERS):
            break
        if active and (
            any(marker in lowered for marker in _FOOTER_MARKERS)
            or (("·" in line or "・" in line) and any(role in lowered for role in ("hr", "招聘", "人力")))
        ):
            break
        if active and line:
            collected.append(_strip_marker(line))
    return list(dict.fromkeys(item for item in collected if item))


def _first_pattern_value(lines: list[str], pattern: re.Pattern[str], limit: int = 12) -> str | None:
    for line in lines[:limit]:
        match = pattern.search(line)
        if match:
            return match.group(1) if match.lastindex else match.group(0)
    return None


def _clean_title(value: str) -> str:
    title = _SALARY_PATTERN.sub("", value)
    title = re.sub(r"\s{2,}", " ", title).strip(" -—·|")
    return title or value.strip()


def _infer_company(lines: list[str]) -> str | None:
    for line in reversed(lines):
        if "·" not in line and "・" not in line:
            continue
        company = re.split(r"[·・]", line, maxsplit=1)[0].strip()
        if company and not any(word in company for word in ("先生", "女士", "招聘者", "HR")):
            return company
    return None


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
    title = _first_prefixed_value(
        lines,
        ("title:", "title：", "岗位名称:", "岗位名称：", "职位名称:", "职位名称：", "岗位:", "岗位：", "职位:", "职位："),
    )
    if not title:
        title = _clean_title(lines[0]) if lines else Path(filename or "job").stem
    else:
        title = _clean_title(title)

    company = (
        _first_prefixed_value(
            lines,
            ("company:", "company：", "公司全称:", "公司全称：", "公司名称:", "公司名称：", "公司:", "公司："),
        )
        or _infer_company(lines)
        or "未知公司"
    )
    location = _first_prefixed_value(lines, ("location:", "location：", "地点:", "地点：", "工作地点:", "工作地点："))
    location = location or _first_pattern_value(lines, _CITY_PATTERN, limit=8)
    salary = _first_prefixed_value(lines, ("salary:", "salary：", "薪资:", "薪资："))
    salary = salary or _first_pattern_value(lines, _SALARY_PATTERN, limit=8)
    duration = _first_prefixed_value(lines, ("duration:", "duration：", "周期:", "周期：", "实习周期:", "实习周期："))
    duration = duration or _first_pattern_value(lines, _DURATION_PATTERN)
    education = _first_prefixed_value(
        lines,
        ("education:", "education：", "学历要求:", "学历要求：", "学历:", "学历："),
    )
    education = education or _first_pattern_value(lines, _EDUCATION_PATTERN)

    responsibilities = _section_lines(
        lines,
        ("responsibilities", "岗位职责", "工作职责", "职位描述", "工作内容"),
    )
    required_skills = _section_lines(
        lines,
        ("requirements", "required skills", "岗位要求", "任职要求", "任职资格", "职位要求"),
    )
    preferred_skills = _section_lines(
        lines,
        ("preferred", "preferred skills", "加分项", "加分项目", "优先条件"),
    )

    record = {
        "job_id": "",
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
    record["organized_text"] = format_job_record_as_jd(record)
    record["job_id"] = _stable_job_id(title, company, record["organized_text"], filename)
    return record


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
            lines.append("- 未从原文中识别到明确内容")

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
