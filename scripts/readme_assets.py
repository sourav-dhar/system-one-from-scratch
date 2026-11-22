"""Generate the README graphics as self-contained SVGs, in light and dark variants.

Design rules:
* every number shown is from the survey (cited there) or measured by this project;
* text is shaped with HarfBuzz (kerning, ligatures) and drawn as glyph outlines, because GitHub serves
  README images under a content-security policy that blocks embedded fonts; nothing is fetched at view
  time, so the images render the same everywhere;
* typefaces: Instrument Serif (display) and IBM Plex Mono (labels, data), both SIL OFL 1.1
  (assets/fonts/OFL-*.txt);
* animations are CSS-only and purposeful; ``prefers-reduced-motion`` shows the final frame.

    uv run --extra assets python scripts/readme_assets.py
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parent.parent
FONTS = ROOT / "assets" / "fonts"
OUT = ROOT / "assets" / "readme"


# ---------------------------------------------------------------------------------------------
# theme
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Theme:
    name: str
    bg: str
    panel: str
    ink: str
    ink2: str
    ink3: str
    rule: str
    grid: str
    accent: str
    accent_soft: str
    s1: str  # categorical slots (dataviz reference palette, validated order)
    s2: str
    s3: str


LIGHT = Theme(
    "light",
    bg="#F6F4EF",
    panel="#FCFBF8",
    ink="#14161B",
    ink2="#555B67",
    ink3="#8C919A",
    rule="#DAD5CB",
    grid="#E5E0D6",
    accent="#E2501E",
    accent_soft="#F7E1D5",
    s1="#2A78D6",
    s2="#EB6834",
    s3="#1BAF7A",
)
DARK = Theme(
    "dark",
    bg="#0E1014",
    panel="#14171C",
    ink="#ECE8E0",
    ink2="#A7ADB7",
    ink3="#6E7480",
    rule="#2A2E36",
    grid="#1C2027",
    accent="#FF6B3D",
    accent_soft="#3A1F15",
    s1="#3987E5",
    s2="#D95926",
    s3="#199E70",
)


# ---------------------------------------------------------------------------------------------
# fonts: HarfBuzz shaping + glyph outlines
# ---------------------------------------------------------------------------------------------
class Font:
    def __init__(self, file: str, key: str) -> None:
        path = FONTS / file
        self.key = key
        self.tt = TTFont(path)
        self.order = self.tt.getGlyphOrder()
        self.glyphs = self.tt.getGlyphSet()
        self.upm = self.tt["head"].unitsPerEm
        self.hb = hb.Font(hb.Face(hb.Blob.from_file_path(str(path))))
        self._paths: dict[str, str] = {}

    def shape(self, text: str) -> list[tuple[str, int, int, int]]:
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(self.hb, buf, {"kern": True, "liga": True})
        return [
            (self.order[i.codepoint], p.x_advance, p.x_offset, p.y_offset)
            for i, p in zip(buf.glyph_infos, buf.glyph_positions, strict=True)
        ]

    def width(self, text: str, size: float, tracking: float = 0.0) -> float:
        g = self.shape(text)
        return (sum(a for _, a, _, _ in g) + tracking * self.upm * max(0, len(g) - 1)) * size / self.upm

    def path(self, name: str) -> str:
        if name not in self._paths:
            pen = SVGPathPen(self.glyphs, ntos=lambda v: str(round(v)))
            self.glyphs[name].draw(TransformPen(pen, (1, 0, 0, -1, 0, 0)))
            self._paths[name] = pen.getCommands()
        return self._paths[name]


SERIF = Font("InstrumentSerif-Regular.ttf", "s")
SERIF_IT = Font("InstrumentSerif-Italic.ttf", "i")
MONO = Font("IBMPlexMono-Regular.ttf", "m")
MONO_MD = Font("IBMPlexMono-Medium.ttf", "d")
MONO_SB = Font("IBMPlexMono-SemiBold.ttf", "b")


# ---------------------------------------------------------------------------------------------
# canvas
# ---------------------------------------------------------------------------------------------
class Canvas:
    def __init__(self, w: int, h: int, t: Theme, title: str, desc: str) -> None:
        self.w, self.h, self.t = w, h, t
        self.title, self.desc = title, desc
        self.body: list[str] = []
        self.defs: list[str] = []
        self.css: list[str] = []
        self._glyph_ids: dict[tuple[str, str], str | None] = {}

    def add(self, frag: str) -> None:
        self.body.append(frag)

    def _gid(self, font: Font, name: str) -> str | None:
        k = (font.key, name)
        if k not in self._glyph_ids:
            d = font.path(name)
            gid = f"{font.key}{len(self._glyph_ids):x}"
            self._glyph_ids[k] = gid if d else None
            if d:
                self.defs.append(f'<path id="{gid}" d="{d}"/>')
        return self._glyph_ids[k]

    def text(
        self,
        s: str,
        x: float,
        y: float,
        font: Font,
        size: float,
        fill: str,
        anchor: str = "start",
        tracking: float = 0.0,
        cls: str = "",
        opacity: float | None = None,
    ) -> float:
        """Draw ``s`` with its baseline at ``y``; returns the drawn width."""
        glyphs = font.shape(s)
        tu = tracking * font.upm
        total = sum(a for _, a, _, _ in glyphs) + tu * max(0, len(glyphs) - 1)
        scale = size / font.upm
        if anchor == "middle":
            x -= total * scale / 2
        elif anchor == "end":
            x -= total * scale
        pen, uses = 0.0, []
        for name, adv, xo, yo in glyphs:
            gid = self._gid(font, name)
            if gid:
                yattr = f' y="{-yo}"' if yo else ""
                uses.append(f'<use href="#{gid}" x="{pen + xo:.0f}"{yattr}/>')
            pen += adv + tu
        op = f' opacity="{opacity}"' if opacity is not None else ""
        run = f'<g transform="translate({x:.1f} {y:.1f}) scale({scale:.4f})" fill="{fill}"{op}>{"".join(uses)}</g>'
        # A CSS transform animation would *replace* the glyph group's transform attribute, so an animation
        # class always goes on a wrapper group that carries no transform of its own.
        self.add(f'<g class="{cls}">{run}</g>' if cls else run)
        return total * scale

    def lines(
        self,
        s: str,
        x: float,
        y: float,
        font: Font,
        size: float,
        fill: str,
        maxw: float,
        lh: float,
        **kw: object,
    ) -> int:
        """Word-wrap ``s`` to ``maxw``; returns the number of lines drawn."""
        out, cur = [], ""
        for word in s.split():
            trial = f"{cur} {word}".strip()
            if font.width(trial, size) <= maxw or not cur:
                cur = trial
            else:
                out.append(cur)
                cur = word
        if cur:
            out.append(cur)
        for i, line in enumerate(out):
            self.text(line, x, y + i * lh, font, size, fill, **kw)  # type: ignore[arg-type]
        return len(out)

    def svg(self) -> str:
        reduced = "@media (prefers-reduced-motion: reduce){*{animation:none!important}}"
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
            f'viewBox="0 0 {self.w} {self.h}" role="img" aria-labelledby="title desc">'
            f'<title id="title">{escape(self.title)}</title><desc id="desc">{escape(self.desc)}</desc>'
            f"<style>{''.join(self.css)}{reduced}</style><defs>{''.join(self.defs)}</defs>"
            f'<rect width="{self.w}" height="{self.h}" fill="{self.t.bg}"/>{"".join(self.body)}</svg>'
        )


def rect(
    x: float,
    y: float,
    w: float,
    h: float,
    fill: str = "none",
    stroke: str = "none",
    rx: float = 0,
    sw: float = 1,
    cls: str = "",
    extra: str = "",
) -> str:
    c = f' class="{cls}"' if cls else ""
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"{c}{extra}/>'
    )


def line(
    x1: float, y1: float, x2: float, y2: float, stroke: str, sw: float = 1, dash: str = "", cls: str = ""
) -> str:
    d = f' stroke-dasharray="{dash}"' if dash else ""
    c = f' class="{cls}"' if cls else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}"{d}{c}/>'


def crop_marks(c: Canvas, inset: float = 18, size: float = 12) -> None:
    w, h, s = c.w, c.h, c.t.ink3
    for x, y, dx, dy in (
        (inset, inset, 1, 1),
        (w - inset, inset, -1, 1),
        (inset, h - inset, 1, -1),
        (w - inset, h - inset, -1, -1),
    ):
        c.add(line(x, y, x + dx * size, y, s, 1) + line(x, y, x, y + dy * size, s, 1))


def label(c: Canvas, s: str, x: float, y: float, fill: str | None = None, size: float = 11) -> float:
    """Small-caps style mono label."""
    return c.text(s.upper(), x, y, MONO_MD, size, fill or c.t.ink3, tracking=0.12)


def dot(cx: float, cy: float, r: float, state: str, color: str, bg: str) -> str:
    """Filled (yes), half (partly) or empty (no) indicator."""
    if state == "yes":
        return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}"/>'
    if state == "part":
        return (
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{bg}" stroke="{color}" stroke-width="1.4"/>'
            f'<path d="M{cx} {cy - r} A{r} {r} 0 0 1 {cx} {cy + r} Z" fill="{color}"/>'
        )
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{bg}" stroke="{color}" stroke-width="1.4"/>'


FADE_UP = (
    ".rise{animation:rise .9s cubic-bezier(.2,.8,.2,1) both}"
    "@keyframes rise{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}"
)


def delays(prefix: str, n: int, start: float, step: float) -> str:
    return "".join(f".{prefix}{i}{{animation-delay:{start + i * step:.2f}s}}" for i in range(n))


# ---------------------------------------------------------------------------------------------
# 1. hero
# ---------------------------------------------------------------------------------------------
def hero(t: Theme) -> Canvas:
    c = Canvas(
        1280,
        460,
        t,
        "System One, from scratch",
        "Animated title card: a decision request with three options; probability bars fill in and "
        "the most likely option is highlighted. Illustrative example.",
    )
    c.defs.append(
        f'<pattern id="dots" width="22" height="22" patternUnits="userSpaceOnUse">'
        f'<circle cx="1.2" cy="1.2" r="1.1" fill="{t.grid}"/></pattern>'
    )
    c.add(rect(0, 0, 720, 460, fill="url(#dots)"))
    crop_marks(c)
    label(c, "System One  ·  from scratch  ·  open research", 56, 70, t.ink2, 12)
    c.add(line(56, 88, 700, 88, t.rule))
    c.text("System One", 52, 198, SERIF, 118, t.ink)
    c.text("from scratch.", 54, 300, SERIF_IT, 118, t.accent)
    c.text("Calibrated decisions in one forward pass. A rebuild, a field survey", 56, 350, MONO, 15, t.ink2)
    c.text("of ~30 systems, and an open model in the making.", 56, 374, MONO, 15, t.ink2)
    x: float = 56
    for chip in ("64 papers", "31 system notes", "14 failure modes", "5 figures"):
        w = MONO_MD.width(chip, 12) + 28
        c.add(rect(x, 396, w, 30, fill=t.panel, stroke=t.rule, rx=15))
        c.text(chip, x + w / 2, 416, MONO_MD, 12, t.ink2, anchor="middle")
        x += w + 10

    # decision card
    x0, y0, w, h = 760, 46, 464, 376
    c.add(rect(x0, y0, w, h, fill=t.panel, stroke=t.rule, rx=16))
    c.text("POST /v1/systemone", x0 + 24, y0 + 36, MONO_MD, 11.5, t.ink3, tracking=0.04)
    pill = "1 forward pass"
    pw = MONO_MD.width(pill, 11) + 22
    c.add(rect(x0 + w - 24 - pw, y0 + 18, pw, 26, fill=t.accent_soft, rx=13))
    c.text(pill, x0 + w - 24 - pw / 2, y0 + 35, MONO_MD, 11, t.accent, anchor="middle")
    c.add(line(x0, y0 + 56, x0 + w, y0 + 56, t.rule))
    label(c, "state", x0 + 24, y0 + 84, size=10.5)
    c.text("Order #4821 arrived 47 days ago.", x0 + 24, y0 + 108, MONO, 13.5, t.ink)
    c.text("The customer is asking for a refund.", x0 + 24, y0 + 128, MONO, 13.5, t.ink)
    c.add(rect(x0 + 24, y0 + 92, w - 48, 2, fill=t.accent, cls="scan", extra=' opacity="0"'))
    label(c, "question  ·  choice", x0 + 24, y0 + 166, size=10.5)
    c.text("Which refund policy applies?", x0 + 24, y0 + 198, SERIF_IT, 26, t.ink)
    opts = (
        ("Full refund  ·  within 30 days", 0.06),
        ("Store credit  ·  31–60 days", 0.89),
        ("Decline  ·  over 60 days", 0.05),
    )
    bw = w - 56 - 60
    for i, (name, p) in enumerate(opts):
        ry = y0 + 218 + i * 42
        win = p == max(q for _, q in opts)
        if win:
            c.add(rect(x0 + 12, ry, w - 24, 40, fill=t.accent_soft, rx=10, cls="hl"))
        c.text(name, x0 + 28, ry + 18, MONO, 13, t.ink)
        c.add(rect(x0 + 28, ry + 26, bw, 5, fill=t.rule, rx=2.5))
        c.add(rect(x0 + 28, ry + 26, bw * p, 5, fill=t.accent if win else t.ink3, rx=2.5, cls=f"bar b{i}"))
        c.text(
            f"{p:.2f}",
            x0 + w - 28,
            ry + 32,
            MONO_MD,
            13,
            t.accent if win else t.ink2,
            anchor="end",
            cls=f"num n{i}",
        )
    c.text("illustrative  ·  p(option | state, question)", x0 + 28, y0 + h - 16, MONO, 10.5, t.ink3)
    c.css.append(
        ".bar{transform-box:fill-box;transform-origin:0 50%;animation:grow 8s cubic-bezier(.2,.8,.2,1) infinite both}"
        ".num{animation:fade 8s ease infinite both}.hl{animation:hl 8s ease infinite both;animation-delay:.5s}"
        ".scan{animation:scan 8s ease-in-out infinite both}"
        "@keyframes grow{0%{transform:scaleX(0)}14%,90%{transform:scaleX(1);opacity:1}97%,100%{transform:scaleX(0);opacity:0}}"
        "@keyframes fade{0%,12%{opacity:0}20%,90%{opacity:1}97%,100%{opacity:0}}"
        "@keyframes hl{0%,16%{opacity:0}24%,90%{opacity:1}97%,100%{opacity:0}}"
        "@keyframes scan{0%{transform:translateY(0);opacity:0}2%{opacity:.9}9%,100%{transform:translateY(38px);opacity:0}}"
        + delays("b", 3, 0.5, 0.12)
        + delays("n", 3, 0.5, 0.12)
    )
    return c


# ---------------------------------------------------------------------------------------------
# 2. findings strip
# ---------------------------------------------------------------------------------------------
# System1-Arena Table 1 (arXiv 2610.21244), non-agentic: (name, group, p50 ms, accuracy %, Brier).
# group 0 = CLM, 1 = System-1-native models (incl. hosted APIs), 2 = open LLMs read zero-shot at the answer token.
S1A = [
    ("CLM-v0.1-8B", 0, 41, 41.4, 0.749),
    ("Decision-2.0-Vega-27B", 1, 228, 73.0, 0.343),
    ("Jev 1.13", 1, 321, 72.6, 0.349),
    ("Microsoft-Decision-1", 1, 304, 70.9, 0.369),
    ("OpenAI Decisions API", 1, 140, 68.6, 0.424),
    ("Decision-2.0-Lux-9B", 1, 104, 67.0, 0.397),
    ("Decision-2.0-Nox-4B", 1, 114, 65.1, 0.414),
    ("Decision-2.0-Kai-0.6B", 1, 72, 50.3, 0.537),
    ("Laya", 1, 136, 44.5, 0.660),
    ("Gemma-4-31B", 2, 60, 73.9, 0.474),
    ("gpt-oss-120b", 2, 284, 71.8, 0.534),
    ("Qwen3.8-27B", 2, 105, 70.2, 0.391),
    ("Gemma-4-26B-A4B", 2, 47, 69.9, 0.528),
    ("DiffusionGemma-26B-A4B", 2, 68, 69.7, 0.465),
    ("gpt-oss-20b", 2, 183, 68.7, 0.589),
    ("Qwen3.5-9B", 2, 77, 65.3, 0.461),
    ("Qwen3-32B", 2, 50, 65.0, 0.543),
    ("Llama-3.3-70B", 2, 46, 64.8, 0.640),
    ("Gemma-4-E4B", 2, 29, 62.8, 0.602),
    ("Qwen3-8B", 2, 30, 59.3, 0.715),
    ("Llama-3.1-8B", 2, 21, 53.1, 0.645),
    ("Qwen3-0.6B", 2, 27, 39.2, 0.937),
]
ECE = {"pre": (0.1598, 0.0154), "mid": (0.0648, 0.0131), "post": (0.0668, 0.0141)}  # this project


def findings(t: Theme) -> Canvas:
    c = Canvas(
        1280,
        330,
        t,
        "Findings so far",
        "Three result tiles: CLM-8B is the fastest of the System-1-native models on System1-Arena but the "
        "least accurate, below a zero-shot readout of its own backbone; one fitted temperature cuts the "
        "published CLM head's calibration error from 0.065 to 0.013; our evaluation reproduces the published "
        "heads to within 0.4 points.",
    )
    tiles = [
        (
            "The CLM paradox",
            "41.4%",
            " at 41 ms",
            "Fastest System-1-native model, least accurate: its own backbone read zero-shot scores 59.3%.",
            "System1-Arena, arXiv 2610.21244, Table 1",
        ),
        (
            "One scalar cuts the error",
            "0.065|0.013",
            "",
            "ECE of the published CLM mid head, shipped vs one temperature fitted in-distribution.",
            "this project · results/benchmark/heads.json",
        ),
        (
            "Reproduced within 0.4 pp",
            "69.16%",
            "",
            "Authors' mid head re-scored by our evaluation on 191,161 pairs. Published: 69.2% on ~100k.",
            "this project · Modal run on the full tricky-Q&A test",
        ),
    ]
    tw, gap, x0, y0, th = 386, 21, 40, 26, 278
    for i, (lab, big, small, cap, src) in enumerate(tiles):
        x = x0 + i * (tw + gap)
        c.add(f'<g class="rise r{i}">')
        c.add(rect(x, y0, tw, th, fill=t.panel, stroke=t.rule, rx=16))
        label(c, lab, x + 26, y0 + 38, size=10.5)
        if "|" in big:  # "before|after": the serif has no arrow glyph, so draw it in mono
            before, after = big.split("|")
            bw = c.text(before, x + 24, y0 + 104, SERIF, 60, t.ink)
            bw += c.text("→", x + 24 + bw + 10, y0 + 96, MONO, 30, t.ink3) + 20
            bw += c.text(after, x + 24 + bw, y0 + 104, SERIF, 60, t.accent)
        else:
            bw = c.text(big, x + 24, y0 + 104, SERIF, 60, t.ink)
        if small:
            c.text(small, x + 26 + bw, y0 + 104, MONO, 14, t.ink2)
        c.lines(cap, x + 26, y0 + 136, MONO, 12.5, t.ink2, tw - 52, 18)
        vx, vy, vw, vh = x + 26, y0 + 182, tw - 52, 58
        if i == 0:  # mini frontier: every System1-Arena row, CLM and its own backbone highlighted

            def to_x(ms: float, vx: float = vx, vw: float = vw) -> float:
                return vx + (math.log(ms) - math.log(19)) / (math.log(400) - math.log(19)) * vw

            def to_y(a: float, vy: float = vy, vh: float = vh) -> float:
                return vy + vh - (a - 36) / (77 - 36) * vh

            c.add(line(vx, vy + vh, vx + vw, vy + vh, t.rule))
            for name, _g, ms, acc, _b in S1A:
                hot = name in ("CLM-v0.1-8B", "Qwen3-8B")
                col = t.accent if name == "CLM-v0.1-8B" else (t.ink if hot else t.ink3)
                c.add(
                    f'<circle cx="{to_x(ms):.1f}" cy="{to_y(acc):.1f}" r="{5 if hot else 3.2}" fill="{col}" '
                    f'stroke="{t.panel}" stroke-width="1.2" opacity="{1 if hot else 0.7}"/>'
                )
            c.text("CLM", to_x(41) + 9, to_y(41.4) + 4, MONO_MD, 10.5, t.accent)
            c.text("its backbone", to_x(30) + 9, to_y(59.3) + 4, MONO_MD, 10.5, t.ink)
        elif i == 1:  # ECE before/after
            bar_w = vw / 3
            for j, (k, (own, fit)) in enumerate(ECE.items()):
                bx = vx + j * bar_w + 6
                for m, (v, col) in enumerate(((own, t.ink3), (fit, t.accent))):
                    hgt = v / 0.17 * (vh - 14)
                    c.add(rect(bx + m * 26, vy + vh - 14 - hgt, 20, hgt, fill=col, rx=3))
                c.text(k, bx + 23, vy + vh, MONO, 10, t.ink3, anchor="middle")
            c.text("shipped", vx + vw, vy + 8, MONO, 10, t.ink3, anchor="end")
            c.text("fitted", vx + vw, vy + 22, MONO, 10, t.accent, anchor="end")
        else:  # reproduction rows
            rows = (("pre", "51.73", "52.1"), ("mid", "69.16", "69.2"), ("post", "68.63", "68.5"))
            for j, (k, ours, ref) in enumerate(rows):
                ry = vy + 12 + j * 20
                c.text(k, vx, ry, MONO, 11.5, t.ink3)
                c.text(f"ours {ours}", vx + 52, ry, MONO_MD, 11.5, t.ink)
                c.text(f"published {ref}", vx + 168, ry, MONO, 11.5, t.ink2)
                c.add(
                    f'<path d="M{vx + vw - 16} {ry - 4} l4 4 l8 -9" fill="none" stroke="{t.s3}" '
                    f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
                )
        c.text(src, x + 26, y0 + th - 18, MONO, 10, t.ink3)
        c.add("</g>")
    c.css.append(FADE_UP + delays("r", 3, 0.1, 0.15))
    return c


# ---------------------------------------------------------------------------------------------
# 3. frontier chart
# ---------------------------------------------------------------------------------------------
def frontier(t: Theme) -> Canvas:
    c = Canvas(
        1280,
        660,
        t,
        "The System One frontier on System1-Arena",
        "Scatter of accuracy versus median latency for all 22 System-1 models and open LLMs in System1-Arena "
        "Table 1. Zero-shot answer-token readouts of open LLMs are the fastest accurate models but have worse "
        "Brier scores than the best System-1 models; CLM is the fastest System-1-native model and the least "
        "accurate.",
    )
    c.text("The System One frontier", 56, 74, SERIF, 40, t.ink)
    c.text(
        "System1-Arena, non-agentic: accuracy vs median latency per request  ·  arXiv 2610.21244, Table 1",
        56,
        104,
        MONO,
        13,
        t.ink2,
    )
    px, py, pw, ph = 110, 150, 1090, 400
    lo, hi = math.log(19), math.log(420)

    def to_x(ms: float) -> float:
        return px + (math.log(ms) - lo) / (hi - lo) * pw

    def to_y(a: float) -> float:
        return py + ph - (a - 35) / (80 - 35) * ph

    for a in range(40, 81, 10):
        c.add(line(px, to_y(a), px + pw, to_y(a), t.grid))
        c.text(f"{a}%", px - 14, to_y(a) + 4, MONO, 11.5, t.ink3, anchor="end")
    for ms in (20, 40, 60, 100, 200, 300):
        c.add(line(to_x(ms), py, to_x(ms), py + ph, t.grid))
        c.text(f"{ms} ms", to_x(ms), py + ph + 24, MONO, 11.5, t.ink3, anchor="middle")
    c.text("median latency per request (log scale)", px + pw, py + ph + 50, MONO, 11.5, t.ink2, anchor="end")
    # the fast-and-accurate region
    zx, zy = to_x(19), to_y(80)
    zw, zh = to_x(100) - zx, to_y(65) - zy
    c.add(
        rect(
            zx + 1,
            zy,
            zw,
            zh,
            fill=t.accent_soft,
            stroke=t.accent,
            rx=8,
            sw=1.2,
            cls="zone",
            extra=' stroke-dasharray="5 5"',
        )
    )
    c.text("fast and accurate", zx + 18, zy + 36, SERIF_IT, 25, t.accent)
    c.text("≥ 65% under 100 ms: only zero-shot readouts", zx + 18, zy + 58, MONO, 11.5, t.ink2)
    c.text("of open LLMs, with Brier 0.46–0.54 vs 0.34", zx + 18, zy + 75, MONO, 11.5, t.ink2)
    c.text("for the best System-1 models", zx + 18, zy + 92, MONO, 11.5, t.ink2)
    cols = (t.s1, t.s2, t.s3)
    labels = {  # name -> (dx, dy, anchor)
        "CLM-v0.1-8B": (12, 5, "start"),
        "Decision-2.0-Vega-27B": (-12, -12, "end"),
        "Jev 1.13": (12, -10, "start"),
        "Decision-2.0-Kai-0.6B": (12, 5, "start"),
        "Laya": (12, 18, "start"),
        "Gemma-4-31B": (12, -10, "start"),
        "Qwen3-8B": (12, 18, "start"),
    }
    order = sorted(range(len(S1A)), key=lambda k: S1A[k][2])
    for rank, k in enumerate(order):
        name, g, ms, acc, _brier = S1A[k]
        cx, cy = to_x(ms), to_y(acc)
        r = 8 if name in labels else 6
        c.add(
            f'<g class="pop p{rank}"><circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{cols[g]}" '
            f'stroke="{t.bg}" stroke-width="2.2"/></g>'
        )
        if name in labels:
            dx, dy, anc = labels[name]
            extra = "  (CLM's backbone, zero-shot)" if name == "Qwen3-8B" else ""
            c.text(
                f"{name}  {acc:.1f}%{extra}",
                cx + dx,
                cy + dy,
                MONO_MD if g == 0 else MONO,
                12,
                t.ink if g == 0 else t.ink2,
                anchor=anc,
                cls=f"rise q{rank}",
            )
    lx, ly = to_x(225), to_y(47.5)
    for j, name in enumerate(
        ("CLM (cached bi-encoder)", "other System-1 models", "open LLMs, answer-token readout")
    ):
        c.add(f'<circle cx="{lx + 8}" cy="{ly + j * 22 - 4}" r="6" fill="{cols[j]}"/>')
        c.text(name, lx + 22, ly + j * 22, MONO, 12, t.ink2)
    c.text(
        "Local models on an idle H100; hosted models include network time. Closed LLMs that reason score up "
        "to 81.1% (Claude Opus 5.5) at 2.3 s and sit off this chart.",
        56,
        636,
        MONO,
        11,
        t.ink3,
    )
    c.css.append(
        FADE_UP
        + ".pop{transform-box:fill-box;transform-origin:center;animation:pop .7s cubic-bezier(.2,.9,.3,1.4) both}"
        "@keyframes pop{from{transform:scale(0);opacity:0}to{transform:scale(1);opacity:1}}"
        ".zone{animation:march 6s linear infinite}@keyframes march{to{stroke-dashoffset:-60}}"
        + delays("p", len(S1A), 0.15, 0.06)
        + delays("q", len(S1A), 0.3, 0.06)
    )
    return c


# ---------------------------------------------------------------------------------------------
# 4. taxonomy
# ---------------------------------------------------------------------------------------------
FAMILIES = [  # code, name, score, examples, (options cached, state reused, order invariant), tag
    (
        "F1",
        "Bi-encoder",
        "scale · cos(f_s(state + question), f_a(option))",
        "CLM · sbert2s1",
        ("yes", "no", "yes"),
        "",
    ),
    (
        "F2",
        "Late interaction",
        "z_L · u_i  where  z_L = block(z_0; cached options)",
        "Jeva · NDM (set head, within a request)",
        ("yes", "no", "yes"),
        "milestone 1",
    ),
    (
        "F3",
        "Decoder readout",
        "pointer or label-token logits after one joint pass",
        "Kev · Jeff · Strands · decider · StartLux · Clef",
        ("no", "yes", "no"),
        "milestone 2",
    ),
    (
        "F4",
        "Encoder option-marker",
        "MLP over the hidden state at each option marker",
        "Laya · Von · Phocinae · GLiNER",
        ("no", "no", "part"),
        "",
    ),
    (
        "F5",
        "Encoder-decoder",
        "state encoded once, one decoder branch per question",
        "Bongard",
        ("no", "yes", "no"),
        "",
    ),
    ("F6", "Looped readout", "label logits after every loop, averaged", "SanSi", ("no", "part", "no"), ""),
    (
        "F7",
        "Frozen-reader head",
        "energy readout over tapped layers of a frozen LM",
        "vegaml / Vega",
        ("no", "part", "part"),
        "",
    ),
    (
        "F8",
        "Training-free",
        "label log-probs, rotation-averaged or shortlisted",
        "SemIf · AnyJev · Runoff",
        ("no", "yes", "part"),
        "",
    ),
]


def taxonomy(t: Theme) -> Canvas:
    c = Canvas(
        1280,
        620,
        t,
        "Eight ways to build a System One model",
        "Eight architecture families, each with its scoring rule, representative systems, and whether it "
        "caches options, reuses the state across questions, and is order invariant.",
    )
    c.text("Eight ways to build a System One model", 56, 74, SERIF, 40, t.ink)
    c.text(
        "what each family computes jointly, and what it can cache  ·  survey §3", 56, 104, MONO, 13, t.ink2
    )
    cw, ch, gap, x0, y0 = 278, 206, 18, 56, 138
    names = ("cache", "reuse", "order")
    for k, (code, name, score, ex, ind, tag) in enumerate(FAMILIES):
        x = x0 + (k % 4) * (cw + gap)
        y = y0 + (k // 4) * (ch + gap)
        hot = bool(tag)
        c.add(f'<g class="rise r{k}">')
        c.add(
            rect(x, y, cw, ch, fill=t.panel, stroke=t.accent if hot else t.rule, rx=14, sw=1.6 if hot else 1)
        )
        c.text(code, x + 22, y + 34, MONO_SB, 12, t.accent if hot else t.ink3, tracking=0.08)
        if tag:
            tw = MONO_MD.width(tag, 10.5) + 18
            c.add(rect(x + cw - 20 - tw, y + 18, tw, 22, fill=t.accent_soft, rx=11))
            c.text(tag, x + cw - 20 - tw / 2, y + 33, MONO_MD, 10.5, t.accent, anchor="middle")
        c.text(name, x + 22, y + 68, SERIF, 25, t.ink)
        n = c.lines(score, x + 22, y + 94, MONO, 11.5, t.ink2, cw - 44, 16)
        c.lines(ex, x + 22, y + 102 + n * 16, MONO, 11.5, t.ink3, cw - 44, 16)
        for j, (state, nm) in enumerate(zip(ind, names, strict=True)):
            ix = x + 22 + j * 76
            c.add(dot(ix + 5, y + ch - 26, 5, state, t.ink, t.panel))
            c.text(nm, ix + 16, y + ch - 22, MONO, 10.5, t.ink2)
        c.add("</g>")
    lx: float = 56
    for state, nm in (("yes", "yes"), ("part", "partly / depends"), ("no", "no")):
        c.add(dot(lx + 5, 590, 5, state, t.ink, t.bg))
        lx += c.text(nm, lx + 16, 594, MONO, 11, t.ink2) + 36
    c.text(
        "cache = option representations cached across requests  ·  reuse = state encoded once for many "
        "questions  ·  order = permutation-invariant by construction",
        56,
        612,
        MONO,
        10.5,
        t.ink3,
    )
    c.css.append(FADE_UP + delays("r", 8, 0.05, 0.07))
    return c


# ---------------------------------------------------------------------------------------------
# 5. architecture B
# ---------------------------------------------------------------------------------------------
def arrow(x1: float, y1: float, x2: float, y2: float, col: str, flow: bool = True, curve: float = 0.0) -> str:
    mx = (x1 + x2) / 2
    d = f"M{x1:.1f} {y1:.1f} C{mx + curve:.1f} {y1:.1f} {mx - curve:.1f} {y2:.1f} {x2:.1f} {y2:.1f}"
    cls = ' class="flow"' if flow else ""
    dash = ' stroke-dasharray="3 6"' if flow else ""
    return f'<path d="{d}" fill="none" stroke="{col}" stroke-width="1.6"{dash}{cls} marker-end="url(#ah)"/>'


def architecture(t: Theme) -> Canvas:
    c = Canvas(
        1280,
        700,
        t,
        "The locked design: one state, many questions, every option in its own lane",
        "Diagram: a causal backbone prefills the state once; each question forks from the exact cached "
        "state; every option is scored in its own lane, which reads the state but never another option; "
        "heads give choice probabilities, yes/no, score and a null outcome; per-type calibration fitted on "
        "held-out sources; the bar is the calibrated zero-shot readout of the same backbone; a cached-option "
        "path for agent loops follows in milestone 2 behind a go/no-go.",
    )
    c.defs.append(
        f'<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{t.ink3}"/></marker>'
    )
    c.text("One state, many questions, every option in its own lane", 56, 74, SERIF, 40, t.ink)
    c.text(
        "causal backbone  ·  exact state forking  ·  per-option lanes  ·  calibrated heads  ·  ADR 0003",
        56,
        104,
        MONO,
        13,
        t.ink2,
    )

    def box(
        x: float, y: float, w: float, h: float, lab: str, title: str, sub: str = "", hot: bool = False
    ) -> None:
        c.add(rect(x, y, w, h, fill=t.panel, stroke=t.accent if hot else t.rule, rx=12, sw=1.6 if hot else 1))
        label(c, lab, x + 16, y + 26, t.accent if hot else t.ink3, 10)
        c.text(title, x + 16, y + 52, SERIF, 21, t.ink)
        if sub:
            c.lines(sub, x + 16, y + 72, MONO, 10.5, t.ink2, w - 30, 15)

    # inputs
    box(56, 150, 180, 86, "state", "the situation", "text · JSON · chat")
    box(56, 252, 180, 86, "question", "what to decide", "choice · yes/no · score")
    box(56, 380, 180, 122, "options", "K candidates", "o_1 … o_K, each scored in its own lane")
    # backbone
    c.add(rect(280, 140, 430, 372, fill=t.panel, stroke=t.rule, rx=14))
    label(c, "backbone", 296, 166, size=10)
    c.text("Qwen3.5-4B + LoRA, causal", 296, 192, SERIF, 21, t.ink)
    c.add(rect(296, 210, 398, 36, fill=t.bg, stroke=t.rule, rx=8))
    c.text("state prefill  ·  once per state", 495, 233, MONO, 11.5, t.ink2, anchor="middle")
    for k in range(3):
        c.add(
            rect(
                312 + k * 6,
                258 + k * 8,
                230,
                58,
                fill=t.panel,
                stroke=t.accent if k == 2 else t.rule,
                rx=10,
                cls="mem" if k == 2 else "",
            )
        )
    label(c, "fork point", 340, 296, t.accent, 10)
    c.text("KV + DeltaNet state", 340, 318, MONO_MD, 12.5, t.ink)
    c.lines("exact: every question resumes from it", 568, 286, MONO, 10.5, t.ink2, 126, 15)
    # fork bus and per-option lanes
    c.add(line(304, 342, 304, 472, t.ink3, 1.4, "3 5", cls="flow"))
    for k, (o, s_name) in enumerate((("o_1", "s_1"), ("o_2", "s_2"), ("o_K", "s_K"))):
        y = 356 + k * 42
        c.add(rect(318, y, 376, 32, fill=t.bg, stroke=t.rule, rx=8))
        c.add(line(304, y + 16, 318, y + 16, t.ink3, 1.4))
        c.text(f"question + {o}", 332, y + 21, MONO, 11.5, t.ink2)
        c.text(f"→ {s_name}", 680, y + 21, MONO_MD, 11.5, t.ink, anchor="end")
    c.text("lanes never see each other: exact order invariance", 296, 498, MONO, 10, t.ink3)
    c.add(arrow(236, 193, 296, 228, t.ink3))
    c.add(arrow(236, 295, 300, 352, t.ink3))
    c.add(arrow(236, 456, 316, 456, t.ink3))
    # heads
    c.add(rect(750, 150, 228, 210, fill=t.panel, stroke=t.rule, rx=12))
    label(c, "heads", 766, 176, size=10)
    for k, (a, b) in enumerate(
        (
            ("choice", "softmax over s_i"),
            ("yes / no", "own head"),
            ("score", "expected level"),
            ("coverage", "null outcome"),
        )
    ):
        yy = 206 + k * 38
        c.text(a, 766, yy, MONO_MD, 12, t.ink)
        c.text(b, 962, yy, MONO, 11.5, t.ink2, anchor="end")
        if k < 3:
            c.add(line(766, yy + 14, 962, yy + 14, t.grid))
    c.add(arrow(694, 414, 750, 300, t.ink3))
    # calibration and output
    box(1010, 150, 214, 96, "calibration", "T per question type", "fitted on held-out sources")
    c.add(line(978, 198, 1010, 198, t.ink3, 1.6, "3 6", cls="flow"))
    for k, p in enumerate((0.07, 0.86, 0.04, 0.03)):
        yy = 272 + k * 14
        c.add(rect(1010, yy, 214, 6, fill=t.rule, rx=3))
        c.add(rect(1010, yy, 214 * p, 6, fill=t.accent if p > 0.5 else t.ink3, rx=3, cls=f"bar b{k}"))
    c.text("p(o_1 … o_K, none)", 1010, 342, MONO, 10.5, t.ink3)
    c.add(line(1117, 246, 1117, 268, t.ink3, 1.6, "3 6", cls="flow"))
    # the bar to beat
    box(
        1010,
        380,
        214,
        122,
        "the bar · milestone 0",
        "beat the backbone",
        "its own zero-shot readout, calibrated (Qwen3-8B: 59.3% on System1-Arena)",
        hot=True,
    )
    # milestone 2: cached path for agent loops
    c.add(
        rect(56, 586, 1168, 82, fill="none", stroke=t.accent, rx=14, sw=1.4, extra=' stroke-dasharray="6 6"')
    )
    label(c, "milestone 2  ·  agent loops", 76, 614, t.accent, 10)
    c.text("options cached once, without the state", 76, 646, SERIF, 23, t.ink)
    c.text(
        "a late block reads them from the forked state · kept only if, against the same head",
        560,
        628,
        MONO,
        11.5,
        t.ink2,
    )
    c.text(
        "on in-context options, it stays within 3 points and runs at least 5× faster",
        560,
        648,
        MONO,
        11.5,
        t.ink2,
    )
    c.add(line(640, 514, 640, 584, t.accent, 1.6, "3 6", cls="flow"))
    c.text("revisits · large catalogs", 652, 554, MONO, 10.5, t.accent)
    c.css.append(
        ".flow{animation:dash 1.1s linear infinite}@keyframes dash{to{stroke-dashoffset:-18}}"
        ".mem{animation:glow 3.2s ease-in-out infinite}@keyframes glow{0%,100%{stroke-opacity:1}50%{stroke-opacity:.25}}"
        ".bar{transform-box:fill-box;transform-origin:0 50%;animation:grow 6s cubic-bezier(.2,.8,.2,1) infinite both}"
        "@keyframes grow{0%{transform:scaleX(0)}16%,88%{transform:scaleX(1);opacity:1}96%,100%{transform:scaleX(0);opacity:0}}"
        + delays("b", 4, 0.3, 0.1)
    )
    return c


# ---------------------------------------------------------------------------------------------
# 6. roadmap
# ---------------------------------------------------------------------------------------------
STEPS = [  # title, two-line note, status
    ("Rebuild", ("CLM heads re-scored", "within 0.4 pts of published"), "done"),
    ("Field guide", ("375 papers · 31 systems", "4 independent reviews"), "done"),
    ("M0 · Harness", ("anchors 41.4% and 59.3%", "baselines + references"), "now"),
    ("M1 · Typed decisions", ("forked state · own lanes", "calibrated heads"), "next"),
    ("M2 · Cached path", ("options cached once", "go/no-go vs M1"), "later"),
    ("M3 · Release", ("model card · results", "v0.2.0"), "later"),
]


def roadmap(t: Theme) -> Canvas:
    c = Canvas(
        1280,
        270,
        t,
        "Roadmap",
        "Timeline: rebuild and field guide done; benchmark harness in progress; "
        "the typed-decision model next, then the cached path for agent loops and a release.",
    )
    c.text("Where we are", 56, 64, SERIF, 34, t.ink)
    x0, x1, y = 120, 1160, 150
    c.add(line(x0, y, x1, y, t.rule, 2))
    done_x = x0 + (x1 - x0) / 5 * 2
    c.add(line(x0, y, done_x, y, t.ink, 2))
    for k, (title, note, st) in enumerate(STEPS):
        x = x0 + (x1 - x0) / 5 * k
        if st == "done":
            c.add(
                f'<circle cx="{x}" cy="{y}" r="11" fill="{t.ink}"/>'
                f'<path d="M{x - 5} {y} l3.5 3.5 l6.5 -7" fill="none" stroke="{t.bg}" stroke-width="2.2" '
                f'stroke-linecap="round" stroke-linejoin="round"/>'
            )
        elif st == "now":
            c.add(
                f'<circle cx="{x}" cy="{y}" r="18" fill="none" stroke="{t.accent}" stroke-width="1.5" class="ping"/>'
                f'<circle cx="{x}" cy="{y}" r="9" fill="{t.accent}"/>'
            )
        else:
            c.add(f'<circle cx="{x}" cy="{y}" r="9" fill="{t.bg}" stroke="{t.ink3}" stroke-width="1.6"/>')
        col = t.accent if st == "now" else (t.ink if st == "done" else t.ink2)
        c.text(title, x, y - 30, SERIF, 21, col, anchor="middle")
        c.text(note[0], x, y + 40, MONO, 11, t.ink2, anchor="middle")
        c.text(note[1], x, y + 57, MONO, 11, t.ink3, anchor="middle")
    c.text("done", 56, 250, MONO_MD, 10.5, t.ink, tracking=0.1)
    c.text("in progress", 104, 250, MONO_MD, 10.5, t.accent, tracking=0.1)
    c.text("planned", 210, 250, MONO_MD, 10.5, t.ink3, tracking=0.1)
    c.css.append(
        ".ping{transform-box:fill-box;transform-origin:center;animation:ping 2.4s ease-out infinite}"
        "@keyframes ping{0%{transform:scale(.6);opacity:1}80%,100%{transform:scale(1.5);opacity:0}}"
    )
    return c


# ---------------------------------------------------------------------------------------------
# 7. failure modes
# ---------------------------------------------------------------------------------------------
FAILURES = [  # name, headline, context, source
    ("Option-order flips", "17%", "Jev answers change when options are reordered", "Jeva, Table 2"),
    ("Yes/no polarity", "50.5/100", "Jev flips when the yes/no identifiers swap", "Rafe & Das"),
    ("Large-K collapse", "0.85→0.02", "Laya accuracy, 20 → 50 options", "Rafe & Das"),
    ("Negation & wording", "−23 pp", "Jev on a paraphrased negation", "TypedBench"),
    ("Ordinal compression", "1 of 4", "Laya ladders that keep their level order", "TypedBench"),
    ("Long-context burying", "−22–43 pp", "Kev when evidence is buried in records", "Kev PLAN.md"),
    ("State collapse", "0.958", "CLM cosine between different states", "CLM issue #15"),
    ("Read right, act wrong", "73–78%", "of System One errors follow correct reading", "Reading Right"),
    ("Adversarial states", "46.3%", "attack success rate on Jev", "JevAdvBench"),
    ("Step vs. loop gap", "1.7%", "CLM loop success at 40.6% step accuracy", "System1-Arena"),
    ("Knowledge erosion", "−11 pp", "MMLU after Kev fine-tuning at 8B", "Kev PLAN.md"),
    ("Runtime drift", "0.16", "max probability shift, GGUF Q4 packing", "decider"),
    ("Out-of-scope accepted", "0.310", "share of out-of-scope inputs Jev accepts", "Rafe & Das"),
    ("Short-label blindness", "0.000", "CLM state sensitivity with short labels", "CLM issue #3"),
]


def failure_modes(t: Theme) -> Canvas:
    c = Canvas(
        1280,
        610,
        t,
        "Fourteen ways System One models fail",
        "Grid of fourteen failure modes, each with a headline measurement from an independent evaluation.",
    )
    c.text("Fourteen ways System One models fail", 56, 74, SERIF, 40, t.ink)
    c.text(
        "each measured by an independent evaluation or the system's own tracker  ·  survey §7",
        56,
        104,
        MONO,
        13,
        t.ink2,
    )
    cw, ch, gap, x0, y0 = 158, 222, 10, 56, 134
    for k, (name, big, ctx, src) in enumerate(FAILURES):
        x = x0 + (k % 7) * (cw + gap)
        y = y0 + (k // 7) * (ch + gap)
        c.add(f'<g class="rise r{k}">')
        c.add(rect(x, y, cw, ch, fill=t.panel, stroke=t.rule, rx=12))
        c.text(f"FM{k + 1}", x + 14, y + 26, MONO_SB, 10.5, t.ink3, tracking=0.08)
        n = c.lines(name, x + 14, y + 52, SERIF, 19, t.ink, cw - 26, 21)
        size = 26 if len(big) <= 6 else 21
        c.text(big, x + 14, y + 64 + n * 21 + 14, MONO_SB, size, t.accent)
        c.lines(ctx, x + 14, y + 64 + n * 21 + 40, MONO, 10.5, t.ink2, cw - 26, 14)
        c.text(src, x + 14, y + ch - 14, MONO, 9.5, t.ink3)
        c.add("</g>")
    c.css.append(FADE_UP + delays("r", 14, 0.05, 0.05))
    return c


GRAPHICS: dict[str, Callable[[Theme], Canvas]] = {
    "hero": hero,
    "findings": findings,
    "frontier": frontier,
    "taxonomy": taxonomy,
    "architecture": architecture,
    "roadmap": roadmap,
    "failure-modes": failure_modes,
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in GRAPHICS.items():
        for theme in (LIGHT, DARK):
            svg = fn(theme).svg()
            path = OUT / f"{name}-{theme.name}.svg"
            path.write_text(svg, encoding="utf-8")
            print(f"{path.relative_to(ROOT)}  {len(svg.encode()) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
