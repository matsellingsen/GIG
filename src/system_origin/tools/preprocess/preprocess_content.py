#!/usr/bin/env python3
"""
Preprocess Hugo markdown content for syntax-clean text chunking.

Outputs:
1) pages.jsonl: one cleaned canonical page object per markdown file
2) chunks.jsonl: cleaned chunks with provenance
3) report.json: preprocessing statistics and quality counters
"""

from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SHORTCODE_ATTR_PATTERN = r"((?:[^\"'>]|\"[^\"]*\"|'[^']*')*)"

PAIRED_SHORTCODE_PATTERN = re.compile(
    rf"\{{\{{<\s*([A-Za-z0-9_-]+)\s*{SHORTCODE_ATTR_PATTERN}\s*>\}}\}}(.*?)\{{\{{<\s*/\1\s*>\}}\}}",
    re.DOTALL,
)

SINGLE_SHORTCODE_PATTERN = re.compile(
    rf"\{{\{{<\s*([A-Za-z0-9_-]+)\s*{SHORTCODE_ATTR_PATTERN}\s*/?\s*>\}}\}}",
)

FRONT_MATTER_BLOCK_PATTERN = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
FRONT_MATTER_LINE_PATTERN = re.compile(r"^([A-Za-z_][\w-]*)\s*:\s*(.*)$")
ATTRIBUTE_PATTERN = re.compile(r"([A-Za-z_][\w-]*)\s*=\s*\"(.*?)\"")

ANCHOR_PATTERN = re.compile(r"<a\s+[^>]*href=\"([^\"]+)\"[^>]*>(.*?)</a>", re.IGNORECASE | re.DOTALL)
TAG_PATTERN = re.compile(r"<[^>]+>")
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$")
NUMBERED_LIST_PATTERN = re.compile(r"^\s*(\d+)[\.)]\s+(.*)$")
BULLET_LIST_PATTERN = re.compile(r"^\s*[-*]\s+(.*)$")
MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^\)]+)\)")


NOISE_PATTERNS = [
    re.compile(r"We are currently updating our support articles and some info may be outdated\.?", re.IGNORECASE),
    re.compile(r"Read release notes here\.?", re.IGNORECASE),
]


@dataclass
class ChunkContext:
    source_path: str
    page_id: str
    language: str
    page_type: str
    title: str
    date: str | None
    external_url: str | None


@dataclass
class Section:
    section_index: int
    section_title: str
    section_level: int | None
    text: str


@dataclass
class SectionGroup:
    sections: list[Section]
    level: int | None
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Hugo content and emit cleaned text chunks.")
    parser.add_argument(
        "--content-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "content",
        help="Path to top-level content directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "out" / "preprocessed",
        help="Directory for pages.jsonl, chunks.jsonl, and report.json.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Maximum tokens per chunk.",
    )
    return parser.parse_args()


def detect_language(path: Path) -> str:
    name = path.name
    if name.endswith(".en.md"):
        return "en"
    if name.endswith(".da.md"):
        return "da"
    return "unknown"


def derive_page_kind(path: Path) -> str:
    name = path.name
    if name.startswith("_index"):
        return "section_index"
    if name.startswith("index"):
        return "leaf_page"
    return "content_page"


def derive_page_type(path: Path, front_matter: dict[str, Any]) -> str:
    layout = str(front_matter.get("layout", "")).strip().lower()
    lower_path = str(path.as_posix()).lower()
    if "/news/" in lower_path:
        return "publication"
    if "/support/" in lower_path:
        return "support"
    if layout == "custom":
        return "marketing"
    if layout == "simple":
        return "support"
    return "general"


def safe_strip_quotes(value: str) -> str:
    v = value.strip()
    if len(v) >= 2 and ((v[0] == '"' and v[-1] == '"') or (v[0] == "'" and v[-1] == "'")):
        return v[1:-1]
    return v


def parse_front_matter(text: str) -> tuple[dict[str, Any], str]:
    match = FRONT_MATTER_BLOCK_PATTERN.search(text)
    if not match:
        return {}, text

    block = match.group(1)
    body = text[match.end() :]
    data: dict[str, Any] = {}
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        m = FRONT_MATTER_LINE_PATTERN.match(line)
        if not m:
            continue
        key = m.group(1)
        raw_value = m.group(2).strip()
        data[key] = parse_front_matter_value(raw_value)
    return data, body


def parse_front_matter_value(raw_value: str) -> Any:
    if raw_value == "":
        return ""
    lower = raw_value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if re.fullmatch(r"-?\d+", raw_value):
        try:
            return int(raw_value)
        except ValueError:
            return raw_value
    if re.fullmatch(r"-?\d+\.\d+", raw_value):
        try:
            return float(raw_value)
        except ValueError:
            return raw_value
    if raw_value.startswith("[") and raw_value.endswith("]"):
        inner = raw_value[1:-1].strip()
        if not inner:
            return []
        parts = [p.strip() for p in inner.split(",")]
        return [safe_strip_quotes(p) for p in parts]
    return safe_strip_quotes(raw_value)


def parse_shortcode_attributes(attr_text: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for k, v in ATTRIBUTE_PATTERN.findall(attr_text or ""):
        attrs[k] = html.unescape(v.strip())
    return attrs


def clean_inline_html(text: str) -> str:
    def _anchor_repl(match: re.Match[str]) -> str:
        href = match.group(1).strip()
        label = strip_tags(match.group(2).strip())
        label = normalize_whitespace(label)
        return f"[{label}]({href})" if label else href

    text = ANCHOR_PATTERN.sub(_anchor_repl, text)
    text = text.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    text = strip_tags(text)
    return html.unescape(text)


def strip_tags(text: str) -> str:
    return TAG_PATTERN.sub("", text)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    out_lines: list[str] = []
    blank_run = 0
    for ln in lines:
        if not ln:
            blank_run += 1
            if blank_run <= 1:
                out_lines.append("")
        else:
            blank_run = 0
            out_lines.append(ln)
    return "\n".join(out_lines).strip()


def sanitize_noise_lines(text: str) -> tuple[str, bool]:
    removed = False
    kept_lines: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            kept_lines.append("")
            continue

        cleaned = line
        for pattern in NOISE_PATTERNS:
            cleaned, subs = pattern.subn("", cleaned)
            if subs > 0:
                removed = True
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" .:-")
        if not cleaned:
            continue
        kept_lines.append(cleaned)
    return "\n".join(kept_lines), removed


def shortcode_to_text(name: str, attrs: dict[str, str], inner: str | None) -> str:
    lname = name.lower()
    is_paired_shortcode = inner is not None
    if lname == "img":
        alt = attrs.get("alt", "")
        src = attrs.get("src", "")
        if alt and src:
            return f"Image: {alt} (source: {src})"
        if alt:
            return f"Image: {alt}"
        if src:
            return f"Image source: {src}"
        return ""

    if lname == "team-member":
        pieces: list[str] = []
        name_part = " ".join([attrs.get("name", ""), attrs.get("name_line2", "")]).strip()
        if name_part:
            pieces.append(f"Team member: {name_part}")
        title = attrs.get("title", "")
        if title:
            pieces.append(f"role: {clean_inline_html(title)}")
        email = attrs.get("email", "")
        if email:
            pieces.append(f"email: {email}")
        linkedin = attrs.get("linkedin", "")
        if linkedin:
            pieces.append(f"linkedin: {linkedin}")
        return "; ".join(pieces)

    if lname == "code":
        lang = attrs.get("language", "text")
        title = attrs.get("title", "")
        header = f"Code example ({lang})"
        if title:
            header += f": {title}"
        body = (inner or "").strip("\n")
        if not body:
            return header
        return f"{header}\n```{lang}\n{body}\n```"

    if lname in {"alert", "lead"}:
        label = "Warning" if lname == "alert" else "Lead"
        payload = normalize_whitespace(clean_inline_html(inner or ""))
        if not payload:
            return ""
        return f"{label}: {payload}"

    key_fields = [
        "title",
        "subtitle",
        "text",
        "list",
        "vision_quote",
        "vision_author_name",
        "vision_author_title",
        "button_text",
        "label",
    ]
    parts: list[str] = []
    for field in key_fields:
        value = attrs.get(field, "").strip()
        if value:
            value = normalize_whitespace(clean_inline_html(value))
            if value:
                if is_paired_shortcode and field == "title":
                    parts.append(f"## {value}")
                if is_paired_shortcode and field == "subtitle":
                    parts.append(f"### {value}")
                parts.append(f"{field}: {value}")
    if inner:
        body = normalize_whitespace(inner)
        if body:
            parts.append(body)
    return "\n".join(parts)


def expand_shortcodes(text: str) -> str:
    current = text
    while True:
        changed = False

        def _paired_repl(match: re.Match[str]) -> str:
            nonlocal changed
            changed = True
            name = match.group(1)
            attrs = parse_shortcode_attributes(match.group(2))
            inner = match.group(3)
            return shortcode_to_text(name, attrs, inner)

        updated = PAIRED_SHORTCODE_PATTERN.sub(_paired_repl, current)
        current = updated
        if not changed:
            break

    def _single_repl(match: re.Match[str]) -> str:
        name = match.group(1)
        attrs = parse_shortcode_attributes(match.group(2))
        return shortcode_to_text(name, attrs, None)

    current = SINGLE_SHORTCODE_PATTERN.sub(_single_repl, current)
    return current


def normalize_markdown_lists(text: str) -> str:
    out_lines: list[str] = []
    for line in text.splitlines():
        m = NUMBERED_LIST_PATTERN.match(line)
        if m:
            out_lines.append(f"{m.group(1)}. {m.group(2).strip()}")
        else:
            out_lines.append(line)
    return "\n".join(out_lines)


def resolve_links(text: str, source_file: Path, content_root: Path) -> str:
    def _resolve(match: re.Match[str]) -> str:
        label = match.group(1).strip()
        target = match.group(2).strip()
        if target.startswith("http://") or target.startswith("https://") or target.startswith("mailto:"):
            return match.group(0)
        absolute = (source_file.parent / target).resolve()
        try:
            relative = absolute.relative_to(content_root.resolve())
            route = "/" + str(relative).replace("\\", "/")
            route = re.sub(r"(?:^|/)index\.(?:en|da)\.md$", "", route)
            route = re.sub(r"/_index\.(?:en|da)\.md$", "/", route)
            route = re.sub(r"\.(?:en|da)\.md$", "", route)
            route = route.rstrip("/") or "/"
            return f"[{label}]({route})"
        except ValueError:
            return match.group(0)

    return MARKDOWN_LINK_PATTERN.sub(_resolve, text)


def clean_body(raw_body: str, source_file: Path, content_root: Path) -> tuple[str, dict[str, int]]:
    stats = {
        "boilerplate_lines_removed": 0,
        "shortcode_residue": 0,
        "html_residue": 0,
    }

    body = raw_body
    body = expand_shortcodes(body)
    body = clean_inline_html(body)
    body = resolve_links(body, source_file=source_file, content_root=content_root)
    body = normalize_markdown_lists(body)
    body, removed_noise = sanitize_noise_lines(body)
    if removed_noise:
        stats["boilerplate_lines_removed"] += 1
    body = normalize_whitespace(body)

    if "{{<" in body or ">}}" in body:
        stats["shortcode_residue"] += 1
    if TAG_PATTERN.search(body):
        stats["html_residue"] += 1

    return body, stats


def estimate_token_count(text: str) -> int:
    # Approximate tokenizer-agnostic token count using whitespace segmentation.
    return len(re.findall(r"\S+", text))


def split_block_to_max_tokens(block: str, max_tokens: int) -> list[str]:
    words = re.findall(r"\S+", block)
    if len(words) <= max_tokens:
        return [block]

    segments: list[str] = []
    for i in range(0, len(words), max_tokens):
        part = " ".join(words[i : i + max_tokens]).strip()
        if part:
            segments.append(part)
    return segments


def split_into_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    current_lines: list[str] = []
    for line in text.splitlines():
        if line.strip() == "":
            if current_lines:
                blocks.append("\n".join(current_lines).strip())
                current_lines = []
            continue
        current_lines.append(line.rstrip())
    if current_lines:
        blocks.append("\n".join(current_lines).strip())
    return [b for b in blocks if b]


def split_into_sections(clean_text: str, fallback_title: str) -> list[Section]:
    lines = clean_text.splitlines()
    sections: list[Section] = []

    current_title = fallback_title.strip() or "Introduction"
    current_level: int | None = None
    current_lines: list[str] = []

    def flush_section() -> None:
        nonlocal current_lines
        text = normalize_whitespace("\n".join(current_lines))
        if text:
            sections.append(
                Section(
                    section_index=len(sections),
                    section_title=current_title,
                    section_level=current_level,
                    text=text,
                )
            )
        current_lines = []

    for raw_line in lines:
        line = raw_line.rstrip()
        heading = HEADING_PATTERN.match(line.strip())
        if heading:
            flush_section()
            current_title = normalize_whitespace(heading.group(2)) or "Untitled section"
            current_level = len(heading.group(1))
            current_lines = [line.strip()]
            continue
        current_lines.append(line)

    flush_section()

    if not sections and normalize_whitespace(clean_text):
        sections.append(
            Section(
                section_index=0,
                section_title=current_title,
                section_level=current_level,
                text=normalize_whitespace(clean_text),
            )
        )

    return sections


def split_section_into_chunks(section_text: str, max_tokens: int) -> list[str]:
    blocks = split_into_blocks(section_text)
    if not blocks:
        payload = normalize_whitespace(section_text)
        return [payload] if payload else []

    parts: list[str] = []
    current_part_blocks: list[str] = []
    current_tokens = 0

    def flush_part() -> None:
        nonlocal current_part_blocks, current_tokens
        if not current_part_blocks:
            return
        payload = normalize_whitespace("\n\n".join(current_part_blocks))
        if payload:
            parts.append(payload)
        current_part_blocks = []
        current_tokens = 0

    for block in blocks:
        block_tokens = estimate_token_count(block)
        if block_tokens == 0:
            continue

        if block_tokens > max_tokens:
            flush_part()
            for split_part in split_block_to_max_tokens(block, max_tokens=max_tokens):
                payload = normalize_whitespace(split_part)
                if payload:
                    parts.append(payload)
            continue

        if current_tokens + block_tokens <= max_tokens:
            current_part_blocks.append(block)
            current_tokens += block_tokens
        else:
            flush_part()
            current_part_blocks.append(block)
            current_tokens = block_tokens

    flush_part()
    return parts


def is_heading_only_section(section: Section) -> bool:
    if section.section_level is None:
        return False

    lines = [ln.strip() for ln in section.text.splitlines() if ln.strip()]
    if not lines:
        return False

    # Typical shortcode-derived heading-only sections look like:
    # "## Heading" + "title: Heading" (or subtitle variant), with no body text.
    if len(lines) <= 2 and lines[0].startswith("#"):
        for extra in lines[1:]:
            lower = extra.lower()
            if lower.startswith("title:") or lower.startswith("subtitle:"):
                return True
    return False


def merge_parent_heading_sections(sections: list[Section]) -> list[Section]:
    merged: list[Section] = []
    pending_heading_only: list[Section] = []

    for sec in sections:
        if is_heading_only_section(sec):
            pending_heading_only.append(sec)
            continue

        if pending_heading_only:
            mergeable: list[Section] = []
            non_mergeable: list[Section] = []

            if sec.section_level is not None:
                for parent in pending_heading_only:
                    if parent.section_level is not None and parent.section_level < sec.section_level:
                        mergeable.append(parent)
                    else:
                        non_mergeable.append(parent)
            else:
                non_mergeable = pending_heading_only[:]

            # Parents that are not hierarchical ancestors are kept as standalone sections.
            merged.extend(non_mergeable)

            if mergeable:
                parent_prefix = normalize_whitespace("\n\n".join(p.text for p in mergeable))
                sec.text = normalize_whitespace(f"{parent_prefix}\n\n{sec.text}")

            pending_heading_only = []

        merged.append(sec)

    if pending_heading_only:
        merged.extend(pending_heading_only)

    # Re-index after merge operations so section_index remains contiguous.
    for idx, section in enumerate(merged):
        section.section_index = idx

    return merged


def is_low_value_intro_section(section: Section, page_title: str, total_sections: int) -> bool:
    if total_sections <= 1:
        return False
    if section.section_index != 0:
        return False
    if section.section_level is not None:
        return False

    lines = [ln.strip() for ln in section.text.splitlines() if ln.strip()]
    if not lines:
        return False

    allowed_prefixes = ("title:", "subtitle:", "text:", "label:")
    lowered = [ln.lower() for ln in lines]
    if any(not ln.startswith(allowed_prefixes) for ln in lowered):
        return False

    title_lines = [ln for ln in lines if ln.lower().startswith("title:")]
    if not title_lines:
        return False

    title_value = title_lines[0].split(":", 1)[1].strip().casefold()
    if page_title.strip() and title_value and title_value != page_title.strip().casefold():
        return False

    # Keep drop logic conservative to avoid removing substantive introduction sections.
    if estimate_token_count(section.text) > 80:
        return False

    return True


def is_title_only_group(group: SectionGroup) -> bool:
    """
    Check if a section group contains essentially only a title (heading) with no meaningful content.
    This includes groups where the text is just markdown headings or very few actual content words.
    """
    if not group.sections or not group.text:
        return False

    # Count non-heading lines with actual content
    lines = group.text.splitlines()
    content_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not HEADING_PATTERN.match(stripped):
            content_lines.append(stripped)

    # If we have several content lines, it's not title-only
    if len(content_lines) > 3:
        return False

    # Check token count: if very few tokens, likely just a title
    tokens = estimate_token_count(group.text)
    if tokens > 60:  # Conservative threshold: more than ~60 tokens is probably real content
        return False

    return True


def group_sections_by_level(sections: list[Section], max_tokens: int) -> list[SectionGroup]:
    groups: list[SectionGroup] = []
    current_sections: list[Section] = []
    current_level: int | None = None
    current_tokens = 0

    def flush_group() -> None:
        nonlocal current_sections, current_level, current_tokens
        if not current_sections:
            return
        payload = normalize_whitespace("\n\n".join(s.text for s in current_sections))
        if payload:
            groups.append(SectionGroup(sections=current_sections[:], level=current_level, text=payload))
        current_sections = []
        current_level = None
        current_tokens = 0

    for section in sections:
        section_tokens = estimate_token_count(section.text)

        if not current_sections:
            current_sections = [section]
            current_level = section.section_level
            current_tokens = section_tokens
            continue

        same_level = section.section_level == current_level
        fits = current_tokens + section_tokens <= max_tokens

        # Keep same-level neighboring sections together unless adding one exceeds max tokens.
        if same_level and fits:
            current_sections.append(section)
            current_tokens += section_tokens
        else:
            flush_group()
            current_sections = [section]
            current_level = section.section_level
            current_tokens = section_tokens

    flush_group()
    return groups


def chunk_text(clean_text: str, context: ChunkContext, max_tokens: int) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []

    sections = split_into_sections(clean_text, fallback_title=context.title)
    sections = merge_parent_heading_sections(sections)
    if sections and is_low_value_intro_section(sections[0], context.title, len(sections)):
        sections = sections[1:]
        for idx, section in enumerate(sections):
            section.section_index = idx

    section_groups = group_sections_by_level(sections, max_tokens=max_tokens)

    # Identify which groups are title-only and should be merged into the next group
    title_only_indices = set()
    for i, group in enumerate(section_groups):
        if is_title_only_group(group) and i + 1 < len(section_groups):
            title_only_indices.add(i)

    # Merge title-only groups into the next group
    merged_groups: list[SectionGroup] = []
    i = 0
    while i < len(section_groups):
        if i in title_only_indices:
            # Skip this group, it will be merged into the next
            current_group = section_groups[i]
            # Find the next non-title-only group
            j = i + 1
            while j < len(section_groups) and j in title_only_indices:
                j += 1
            if j < len(section_groups):
                # Prepend this title-only group to the next group
                next_group = section_groups[j]
                combined_text = current_group.text + "\n\n" + next_group.text
                merged_group = SectionGroup(
                    sections=current_group.sections + next_group.sections,
                    level=next_group.level,  # Use the level of the content group
                    text=normalize_whitespace(combined_text),
                )
                # Replace the next group with merged
                section_groups[j] = merged_group
                # Mark all intermediate title-only groups as processed
                for k in range(i, j + 1):
                    title_only_indices.discard(k)
                # Process the merged group in the next iteration
                i = j
            else:
                # No next group to merge into; skip this orphaned title-only group
                i += 1
        else:
            merged_groups.append(section_groups[i])
            i += 1

    section_groups = merged_groups

    chunk_index = 0

    for i, group in enumerate(section_groups):
        prev_group = section_groups[i - 1] if i > 0 else None
        next_group = section_groups[i + 1] if i < len(section_groups) - 1 else None

        first_section = group.sections[0]
        last_section = group.sections[-1]

        group_parts = split_section_into_chunks(group.text, max_tokens=max_tokens)
        part_total = len(group_parts)

        if len(group.sections) == 1:
            section_title = first_section.section_title
        else:
            section_title = f"{first_section.section_title} (+{len(group.sections) - 1} more)"

        for part_idx, payload in enumerate(group_parts, start=1):
            chunks.append(
                {
                    "chunk_id": f"{context.page_id}::c{chunk_index:04d}",
                    "chunk_index": chunk_index,
                    "page_id": context.page_id,
                    "page_title": context.title,
                    "source_path": context.source_path,
                    "language": context.language,
                    "page_type": context.page_type,
                    "chunk_text_clean": payload,
                    "token_count": estimate_token_count(payload),
                    "section": {
                        "section_index": first_section.section_index,
                        "section_title": section_title,
                        "section_level": group.level,
                        "part_index": part_idx,
                        "part_total": part_total,
                        "part_label": f"part {part_idx}/{part_total}",
                        "prev_section_title": prev_group.sections[-1].section_title if prev_group else None,
                        "next_section_title": next_group.sections[0].section_title if next_group else None,
                        "group_size": len(group.sections),
                        "group_start_index": first_section.section_index,
                        "group_end_index": last_section.section_index,
                        "group_titles": [s.section_title for s in group.sections],
                    },
                    "provenance": {
                        "title": context.title,
                        "date": context.date,
                        "externalUrl": context.external_url,
                    },
                    "quality_flags": {
                        "has_html_residue": bool(TAG_PATTERN.search(payload)),
                        "has_shortcode_residue": "{{<" in payload or ">}}" in payload,
                    },
                }
            )
            chunk_index += 1

    return chunks


def page_id_from_path(path: Path, content_root: Path) -> str:
    rel = path.relative_to(content_root).as_posix()
    return re.sub(r"\.md$", "", rel)


def build_page_record(
    file_path: Path,
    content_root: Path,
    front_matter: dict[str, Any],
    clean_text: str,
    clean_stats: dict[str, int],
) -> dict[str, Any]:
    rel = file_path.relative_to(content_root).as_posix()
    page_id = page_id_from_path(file_path, content_root)
    language = detect_language(file_path)
    page_type = derive_page_type(file_path, front_matter)
    page_kind = derive_page_kind(file_path)

    return {
        "page_id": page_id,
        "source_path": rel,
        "language": language,
        "page_type": page_type,
        "page_kind": page_kind,
        "section_path": str(file_path.parent.relative_to(content_root).as_posix()),
        "slug": file_path.parent.name if file_path.name.startswith("index") else file_path.stem,
        "front_matter": front_matter,
        "clean_text": clean_text,
        "cleaning_flags": {
            "boilerplate_removed": clean_stats["boilerplate_lines_removed"] > 0,
            "shortcode_residue": clean_stats["shortcode_residue"] > 0,
            "html_residue": clean_stats["html_residue"] > 0,
        },
    }


def iter_markdown_files(content_root: Path) -> list[Path]:
    # English-only ingestion.
    return sorted(p for p in content_root.rglob("*.en.md") if p.is_file())


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def preprocess(content_root: Path, output_dir: Path, max_tokens: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    pages: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "content_root": str(content_root),
        "pages_total": 0,
        "chunks_total": 0,
        "languages": {},
        "page_types": {},
        "chunk_max_tokens": max_tokens,
        "quality": {
            "pages_with_shortcode_residue": 0,
            "pages_with_html_residue": 0,
            "pages_with_boilerplate_removed": 0,
        },
    }

    files = iter_markdown_files(content_root)
    for file_path in files:
        raw_text = file_path.read_text(encoding="utf-8")
        front_matter, body = parse_front_matter(raw_text)
        clean_text, clean_stats = clean_body(body, source_file=file_path, content_root=content_root)
        page = build_page_record(file_path, content_root, front_matter, clean_text, clean_stats)
        pages.append(page)

        lang = page["language"]
        report["languages"][lang] = report["languages"].get(lang, 0) + 1
        page_type = page["page_type"]
        report["page_types"][page_type] = report["page_types"].get(page_type, 0) + 1

        if page["cleaning_flags"]["shortcode_residue"]:
            report["quality"]["pages_with_shortcode_residue"] += 1
        if page["cleaning_flags"]["html_residue"]:
            report["quality"]["pages_with_html_residue"] += 1
        if page["cleaning_flags"]["boilerplate_removed"]:
            report["quality"]["pages_with_boilerplate_removed"] += 1

        page_id = page["page_id"]
        context = ChunkContext(
            source_path=page["source_path"],
            page_id=page_id,
            language=page["language"],
            page_type=page["page_type"],
            title=str(front_matter.get("title", "")),
            date=str(front_matter.get("date")) if front_matter.get("date") else None,
            external_url=str(front_matter.get("externalUrl")) if front_matter.get("externalUrl") else None,
        )
        page_chunks = chunk_text(clean_text, context=context, max_tokens=max_tokens)
        chunks.extend(page_chunks)

    report["pages_total"] = len(pages)
    report["chunks_total"] = len(chunks)

    write_jsonl(output_dir / "pages.jsonl", pages)
    write_jsonl(output_dir / "chunks.jsonl", chunks)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    content_root = args.content_root.resolve()
    output_dir = args.output_dir.resolve()

    if not content_root.exists() or not content_root.is_dir():
        raise SystemExit(f"Content root does not exist or is not a directory: {content_root}")

    report = preprocess(content_root=content_root, output_dir=output_dir, max_tokens=args.max_tokens)
    print("Preprocessing complete")
    print(f"Pages:  {report['pages_total']}")
    print(f"Chunks: {report['chunks_total']}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
