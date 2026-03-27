import re
from typing import List

# Regex patterns
SHORTCODE_OPEN = re.compile(r"{{<\s*([a-zA-Z0-9_-]+)(\s+[^>]*)?>}}")
SHORTCODE_CLOSE = re.compile(r"{{<\s*/\s*([a-zA-Z0-9_-]+)\s*>}}")
HEADING = re.compile(r"^#{1,6}\s+.*")

def extract_article_section(text: str) -> str:
    """Return only the <article>...</article> content."""
    start = text.find("<article")
    end = text.rfind("</article>")
    if start == -1 or end == -1:
        return ""
    return text[start:end + len("</article>")]

def has_closing_shortcode(lines: List[str], start_index: int, name: str) -> bool:
    """Check if a shortcode has a matching closing tag later."""
    close_pattern = re.compile(r"{{<\s*/\s*" + re.escape(name) + r"\s*>}}")
    for j in range(start_index + 1, len(lines)):
        if close_pattern.search(lines[j]):
            return True
    return False

def chunk_article(text: str) -> List[str]:
    """Chunk article content by top-level shortcodes AND markdown headings."""
    article = extract_article_section(text)
    if not article:
        return []

    lines = article.splitlines()
    chunks: List[str] = []
    current: List[str] = []

    i = 0
    depth = 0  # shortcode nesting depth

    while i < len(lines):
        line = lines[i]

        open_match = SHORTCODE_OPEN.match(line)
        close_match = SHORTCODE_CLOSE.match(line)
        heading_match = HEADING.match(line)

        # ─────────────────────────────────────────────
        # 1. Top-level Markdown heading → new chunk
        # ─────────────────────────────────────────────
        if depth == 0 and heading_match:
            if current:
                chunks.append("\n".join(current).strip())
                current = []
            current.append(line)
            i += 1
            continue

        # ─────────────────────────────────────────────
        # 2. Top-level shortcode open
        # ─────────────────────────────────────────────
        if depth == 0 and open_match:
            shortcode_name = open_match.group(1)

            # Paired shortcode?
            if has_closing_shortcode(lines, i, shortcode_name):
                if current:
                    chunks.append("\n".join(current).strip())
                    current = []

                block = [line]
                depth = 1
                i += 1

                while i < len(lines) and depth > 0:
                    block_line = lines[i]
                    block.append(block_line)

                    if SHORTCODE_OPEN.match(block_line):
                        depth += 1
                    if SHORTCODE_CLOSE.match(block_line):
                        depth -= 1

                    i += 1

                chunks.append("\n".join(block).strip())
                continue

            # Single shortcode
            else:
                if current:
                    chunks.append("\n".join(current).strip())
                    current = []
                chunks.append(line.strip())
                i += 1
                continue

        # ─────────────────────────────────────────────
        # 3. Normal line (inside or outside shortcode)
        # ─────────────────────────────────────────────
        current.append(line)
        i += 1

    # Final chunk
    if current:
        chunks.append("\n".join(current).strip())

    return [c for c in chunks if c.strip()]