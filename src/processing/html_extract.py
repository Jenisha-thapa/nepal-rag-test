"""
HTML -> ordered, typed content blocks.

Instead of flattening a page into one string, we walk the DOM in reading order and
keep its structure:

    ("heading", "प्राय सोधिने प्रश्नहरु", level=2)
    ("text",    "सामाजिक सुरक्षा भत्ता प्राप्त गर्न ...")
    ("item",    "नेपाली नागरिकता प्रमाणपत्र")                       # list item
    ("row",     "लाभान्वित वर्ग: जेष्ठ नागरिक; ...; मासिक रकम (रु.): ४०००")  # table row

Boilerplate (nav menus, header/footer, sidebars, cookie banners) is removed and
text repeated within a page (menus rendered several times) is dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import bs4
from bs4 import Comment, NavigableString, Tag

from src.utils import normalize_nepali_text

HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
BLOCK_TAGS = ["p", "div", "section", "article", "main", "ul", "ol", "li", "table", "dl", "dd", "dt",
              "blockquote", "pre", "figure", "form", "header", "footer", "nav", "aside", *HEADINGS]
DROP_TAGS = ["script", "style", "noscript", "svg", "iframe", "form", "button", "select", "input",
             "template", "nav", "header", "footer", "aside"]
# class / id tokens that mark navigation & chrome rather than content
BOILERPLATE = re.compile(
    r"^(nav|navbar|nav[-_].*|.*[-_]nav|menu|.*[-_]menu|dropdown.*|breadcrumbs?|footer.*|sidebar.*|social.*|"
    r"share.*|cookie.*|modal.*|offcanvas.*|skip.*|top-?bar|header-\d.*|logo.*|search.*|pagination)$",
    re.IGNORECASE,
)
# Notice tickers ("महत्त्वपूर्ण सूचना") often sit inside <header>; they are real content.
NOTICE_TICKER = re.compile(r"marquee|ticker|notice", re.IGNORECASE)
SERIAL_COLUMNS ={"क्र.सं.", "क्र.सं", "क्र.स.", "सि.नं.", "s.n.", "sn", "s.no.", "no."}
FAQ_MARKER = re.compile(r"^(?:[०-९0-9]+\s*[.)]\s*)?(?:प्रश्न|उत्तर|question|answer)\s*[:：]?", re.IGNORECASE)
BS_DATE = re.compile(
    r"[०-९0-9]{1,2}\s+(?:बैशाख|वैशाख|जेठ|जेष्ठ|असार|आषाढ|साउन|श्रावण|भदौ|भाद्र|असोज|आश्विन|कात्तिक|कार्तिक|"
    r"मंसिर|मङ्सिर|पुस|पुष|माघ|फागुन|फाल्गुन|चैत|चैत्र),?\s*[०-९0-9]{4}"
    r"|[२2][०0][६-९6-9][०-९0-9]\s?[/।.\-]\s?[०-९0-9]{1,2}\s?[/।.\-]\s?[०-९0-9]{1,2}"
)


@dataclass
class Block:
    kind: str  # heading | text | item | row
    text: str
    level: int = 0


@dataclass
class ExtractedPage:
    title: str
    date_bs: str
    blocks: list[Block]


def _norm(text: str) -> str:
    return normalize_nepali_text(text)


def _is_boilerplate(tag: Tag) -> bool:
    tokens = list(tag.get("class") or []) + ([tag.get("id")] if tag.get("id") else [])
    return any(BOILERPLATE.match(t) for t in tokens) or tag.get("role") in ("navigation", "banner", "contentinfo")


def _cell_text(cell: Tag) -> str:
    items = [_norm(li.get_text(" ", strip=True)) for li in cell.find_all("li")]
    return ", ".join(i for i in items if i) if items else _norm(cell.get_text(" ", strip=True))


def _table_rows(table: Tag) -> list[str]:
    """Render each row as 'header: value; header: value' so every row stands on its own."""
    rows = []
    for tr in table.find_all("tr"):
        if tr.find_parent("table") is not table:
            continue  # rows of nested tables are included in their parent cell's text
        cells = [_cell_text(c) for c in tr.find_all(["td", "th"], recursive=False)]
        if any(cells):
            rows.append(cells)
    if not rows:
        return []
    if len(rows) == 1:
        return [" | ".join(c for c in rows[0] if c)]

    header, out = rows[0], []
    for row in rows[1:]:
        if len(row) == len(header):
            parts = [f"{h}: {v}" if h else v for h, v in zip(header, row)
                     if v and h.lower() not in SERIAL_COLUMNS]
            line = "; ".join(parts)
        else:
            line = " | ".join(c for c in row if c)
        if line:
            out.append(line)
    return out


def _walk(node: Tag, out: list[Block]) -> None:
    for child in node.children:
        if isinstance(child, Comment):
            continue
        if isinstance(child, NavigableString):
            if text := _norm(str(child)):
                out.append(Block("text", text))
            continue
        if not isinstance(child, Tag) or _is_boilerplate(child):
            continue
        if child.name in HEADINGS:
            if text := _norm(child.get_text(" ", strip=True)):
                out.append(Block("heading", text, HEADINGS[child.name]))
        elif child.name == "table":
            out.extend(Block("row", r) for r in _table_rows(child))
        elif not child.find(BLOCK_TAGS):
            # leaf block or inline run: emit its whole text as one block
            if text := _norm(child.get_text(" ", strip=True)):
                out.append(Block("item" if child.name == "li" else "text", text))
        else:
            _walk(child, out)


def _keep(block: Block) -> bool:
    letters = len(re.findall(r"[^\W\d_]", block.text))
    digits = len(re.findall(r"[0-9०-९]", block.text))
    return letters >= 3 or digits >= 3 or FAQ_MARKER.match(block.text) is not None


def _extract_notice_tickers(soup: bs4.BeautifulSoup) -> list[Block]:
    """Pull notice tickers out of page chrome (<header>/<nav>) before the chrome is dropped."""
    blocks: list[Block] = []
    for chrome in soup.find_all(["header", "nav"]):
        for ticker in chrome.find_all(class_=lambda c: c and NOTICE_TICKER.search(" ".join(c) if isinstance(c, list) else c)):
            if ticker.find_parent(class_=lambda c: c and NOTICE_TICKER.search(" ".join(c) if isinstance(c, list) else c)):
                continue  # nested inside another ticker we already take
            _walk(ticker, blocks)
    return blocks


def extract_html(html: str, include_notices: bool = True) -> ExtractedPage:
    soup = bs4.BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "template"]):
        tag.decompose()
    notices = _extract_notice_tickers(soup) if include_notices else []
    # FAQ accordions put the question inside <h2 class="accordion__header"><button>…</button></h2>:
    # keep that text (unwrap the button) before buttons are dropped as UI chrome.
    for btn in soup.find_all("button"):
        in_heading = btn.find_parent(list(HEADINGS)) is not None
        accordion = any("accordion" in " ".join(p.get("class") or []) for p in [btn, *btn.parents][:3])
        if in_heading or accordion:
            btn.unwrap()
    for tag in soup(DROP_TAGS):
        tag.decompose()

    root = soup.find("main") or soup.find("article") or soup.body or soup
    h1 = root.find("h1")
    title = _norm(h1.get_text(" ", strip=True)) if h1 else ""
    if not title and soup.title:
        title = _norm(soup.title.get_text(strip=True).split("|")[0])

    date_el = root.find("time") or root.find(class_=re.compile(r"date", re.I))
    date_match = BS_DATE.search(_norm(date_el.get_text(" ", strip=True))) if date_el else None

    raw: list[Block] = []
    _walk(root, raw)
    raw.extend(notices)  # site-wide notices go after the page's own content

    blocks, seen = [], set()
    for b in raw:
        if not _keep(b):
            continue
        key = (b.kind, b.text)
        if key in seen and not FAQ_MARKER.match(b.text):  # repeated menus/notices; keep "उत्तर:" markers
            continue
        seen.add(key)
        blocks.append(b)

    return ExtractedPage(title=title, date_bs=date_match.group(0) if date_match else "", blocks=blocks)


def first_bs_date(text: str) -> str:
    m = BS_DATE.search(text)
    return m.group(0) if m else ""
