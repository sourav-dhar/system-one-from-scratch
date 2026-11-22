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
S1A = [  # (name, group, p50 ms, accuracy %) - System1-Arena Table 1 (arXiv 2610.21244)
    ("CLM-v0.1-8B", 0, 41, 41.4),
    ("Decision 2.0 Kai-0.6B", 1, 72, 50.3),
    ("Laya", 1, 136, 44.5),
    ("Decision 2.0 Vega-27B", 1, 228, 73.0),
    ("OpenAI Decisions", 2, 140, 68.6),
    ("Microsoft-Decision-1", 2, 304, 70.9),
    ("Jev 1.13", 2, 321, 72.6),
]
ECE = {"pre": (0.1598, 0.0154), "mid": (0.0648, 0.0131), "post": (0.0668, 0.0141)}  # this project


def findings(t: Theme) -> Canvas:
    c = Canvas(
        1280,
        330,
        t,
        "Findings so far",
        "Three result tiles: CLM-8B is the fastest but least accurate System-1 model on System1-Arena; "
        "one fitted temperature cuts the published CLM head's calibration error from 0.065 to 0.013; "
        "our evaluation reproduces the published heads exactly.",
    )
    group_colors = (t.s1, t.s2, t.s3)
    tiles = [
        (
            "The CLM paradox",
            "41.4%",
            " at 41 ms",
            "CLM-8B on System1-Arena: the fastest System-1 model, and the least accurate.",
            "System1-Arena, arXiv 2610.21244, Table 1",
        ),
        (
            "One scalar fixes calibration",
            "0.065|0.013",
            "",
            "Calibration error (ECE) of the published CLM mid head, shipped vs one fitted temperature.",
            "this project · results/benchmark/heads.json",
        ),
        (
            "Reproduced exactly",
            "69.16%",
            "",
            "Authors' mid head re-scored by our evaluation on 191,161 held-out pairs. Reference: 69.2%.",
            "this project · Modal run on the full held-out test",
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
        if i == 0:  # mini frontier

            def to_x(ms: float, vx: float = vx, vw: float = vw) -> float:
                return vx + (math.log(ms) - math.log(32)) / (math.log(400) - math.log(32)) * vw

            def to_y(a: float, vy: float = vy, vh: float = vh) -> float:
                return vy + vh - (a - 38) / (76 - 38) * vh

            c.add(line(vx, to_y(59.3), vx + vw, to_y(59.3), t.ink3, 1, "3 4"))
            c.add(line(vx, vy + vh, vx + vw, vy + vh, t.rule))
            for _name, g, ms, acc in S1A:
                r = 5.5 if g == 0 else 4
                col = t.accent if g == 0 else t.ink3
                c.add(
                    f'<circle cx="{to_x(ms):.1f}" cy="{to_y(acc):.1f}" r="{r}" fill="{col}" stroke="{t.panel}" stroke-width="1.5"/>'
                )
            c.text("CLM", to_x(41) + 9, to_y(41.4) + 4, MONO_MD, 10.5, t.accent)
            c.text("Jev", to_x(321) - 8, to_y(72.6) - 8, MONO, 10, t.ink3, anchor="end")
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
            rows = (("pre", "51.73", "51.7"), ("mid", "69.16", "69.2"), ("post", "68.63", "68.6"))
            for j, (k, ours, ref) in enumerate(rows):
                ry = vy + 12 + j * 20
                c.text(k, vx, ry, MONO, 11.5, t.ink3)
                c.text(f"ours {ours}", vx + 52, ry, MONO_MD, 11.5, t.ink)
                c.text(f"reference {ref}", vx + 168, ry, MONO, 11.5, t.ink2)
                c.add(
                    f'<path d="M{vx + vw - 16} {ry - 4} l4 4 l8 -9" fill="none" stroke="{t.s3}" '
                    f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
                )
        c.text(src, x + 26, y0 + th - 18, MONO, 10, t.ink3)
        c.add("</g>")
    del group_colors
    c.css.append(FADE_UP + delays("r", 3, 0.1, 0.15))
    return c


# ---------------------------------------------------------------------------------------------
# 3. frontier chart
# ---------------------------------------------------------------------------------------------
def frontier(t: Theme) -> Canvas:
    c = Canvas(
        1280,
        620,
        t,
        "The System One frontier on System1-Arena",
        "Scatter of accuracy versus p50 latency for seven System-1 models from System1-Arena Table 1. "
        "CLM is fastest at 41 ms but least accurate at 41.4%; no model reaches 65% under 100 ms.",
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
    px, py, pw, ph = 110, 150, 1090, 380
    lo, hi = math.log(32), math.log(420)

    def to_x(ms: float) -> float:
        return px + (math.log(ms) - lo) / (hi - lo) * pw

    def to_y(a: float) -> float:
        return py + ph - (a - 35) / (80 - 35) * ph

    for a in range(40, 81, 10):
        c.add(line(px, to_y(a), px + pw, to_y(a), t.grid))
        c.text(f"{a}%", px - 14, to_y(a) + 4, MONO, 11.5, t.ink3, anchor="end")
    for ms in (40, 60, 100, 200, 300):
        c.add(line(to_x(ms), py, to_x(ms), py + ph, t.grid))
        c.text(f"{ms} ms", to_x(ms), py + ph + 24, MONO, 11.5, t.ink3, anchor="middle")
    c.text("median latency per request (log scale)", px + pw, py + ph + 50, MONO, 11.5, t.ink2, anchor="end")
    # the empty corner
    zx, zy = to_x(32), to_y(80)
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
    c.text("the empty corner", zx + 18, zy + 38, SERIF_IT, 26, t.accent)
    c.text("≥ 65% accuracy under 100 ms:", zx + 18, zy + 62, MONO, 12, t.ink2)
    c.text("no System-1 model here yet", zx + 18, zy + 80, MONO, 12, t.ink2)
    # reference line
    c.add(line(px, to_y(59.3), px + pw, to_y(59.3), t.ink3, 1.2, "4 4"))
    c.text(
        "Qwen3-8B answer-token readout  ·  59.3%",
        px + pw - 4,
        to_y(59.3) - 9,
        MONO,
        11.5,
        t.ink2,
        anchor="end",
    )
    cols = (t.s1, t.s2, t.s3)
    offs = {
        "CLM-v0.1-8B": (12, 5, "start"),
        "Decision 2.0 Kai-0.6B": (12, 5, "start"),
        "Laya": (12, 18, "start"),
        "Decision 2.0 Vega-27B": (-12, -12, "end"),
        "OpenAI Decisions": (0, -16, "middle"),
        "Microsoft-Decision-1": (-12, 18, "end"),
        "Jev 1.13": (12, -10, "start"),
    }
    for i, (name, g, ms, acc) in enumerate(S1A):
        cx, cy = to_x(ms), to_y(acc)
        c.add(
            f'<g class="pop p{i}"><circle cx="{cx:.1f}" cy="{cy:.1f}" r="8" fill="{cols[g]}" '
            f'stroke="{t.bg}" stroke-width="2.5"/></g>'
        )
        dx, dy, anc = offs[name]
        c.text(
            f"{name}  {acc:.1f}%",
            cx + dx,
            cy + dy,
            MONO_MD if g == 0 else MONO,
            12,
            t.ink if g == 0 else t.ink2,
            anchor=anc,
            cls=f"rise q{i}",
        )
    # legend
    lx, ly = px + pw - 300, 140
    for j, name in enumerate(("cached bi-encoder", "open joint models", "hosted APIs")):
        c.add(f'<circle cx="{lx + 8}" cy="{ly + j * 22 - 4}" r="6" fill="{cols[j]}"/>')
        c.text(name, lx + 22, ly + j * 22, MONO, 12, t.ink2)
    c.text("Local models on an idle H100; hosted models include network time.", 56, 596, MONO, 11, t.ink3)
    c.css.append(
        FADE_UP
        + ".pop{transform-box:fill-box;transform-origin:center;animation:pop .7s cubic-bezier(.2,.9,.3,1.4) both}"
        "@keyframes pop{from{transform:scale(0);opacity:0}to{transform:scale(1);opacity:1}}"
        ".zone{animation:march 6s linear infinite}@keyframes march{to{stroke-dashoffset:-60}}"
        + delays("p", 7, 0.2, 0.12)
        + delays("q", 7, 0.35, 0.12)
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
        "Architecture B, built to grow into C",
        "Diagram: state and question are encoded separately; options are encoded once with their decision "
        "type and cached; one decision block lets the state read the cached options; heads give scores, a "
        "null probability, yes/no and score outputs; K-aware calibration; uncertain cases route to a "
        "milestone-2 joint slow path.",
    )
    c.defs.append(
        f'<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{t.ink3}"/></marker>'
    )
    c.text("Architecture B, built to grow into C", 56, 74, SERIF, 40, t.ink)
    c.text(
        "two towers  ·  late interaction over cached options  ·  a real null outcome  ·  K-aware calibration"
        "  ·  ADR 0001",
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
    box(
        56,
        380,
        180,
        122,
        "options",
        "K candidates",
        "o_1 … o_K, rendered with their type, never with the request",
    )
    # backbone
    c.add(rect(280, 140, 200, 372, fill=t.panel, stroke=t.rule, rx=14))
    label(c, "backbone", 296, 166, size=10)
    c.text("frozen or LoRA", 296, 192, SERIF, 21, t.ink)
    for y, nm in ((222, "state tower"), (300, "question tower"), (414, "option tower")):
        c.add(rect(296, y, 168, 40, fill=t.bg, stroke=t.rule, rx=8))
        c.text(nm, 380, y + 25, MONO, 11.5, t.ink2, anchor="middle")
    c.text("Qwen3-8B (anchor, taps)", 296, 474, MONO, 10, t.ink3)
    c.text("Qwen3.5-4B + LoRA", 296, 488, MONO, 10, t.ink3)
    c.text("Inkling 2 Mini (frozen)", 296, 502, MONO, 10, t.ink3)
    c.add(arrow(236, 193, 296, 242, t.ink3))
    c.add(arrow(236, 295, 296, 320, t.ink3))
    c.add(arrow(236, 441, 296, 434, t.ink3))
    # fuse
    c.add(f'<circle cx="560" cy="282" r="20" fill="{t.panel}" stroke="{t.rule}"/>')
    c.text("+", 560, 289, MONO_MD, 18, t.ink, anchor="middle")
    c.text("late fusion", 560, 322, MONO, 10.5, t.ink3, anchor="middle")
    c.add(arrow(464, 242, 540, 276, t.ink3))
    c.add(arrow(464, 320, 540, 288, t.ink3))
    # option memory (cache)
    for k in range(3):
        c.add(
            rect(
                520 + k * 6,
                404 + k * 8,
                200,
                74,
                fill=t.panel,
                stroke=t.accent if k == 2 else t.rule,
                rx=10,
                cls="mem" if k == 2 else "",
            )
        )
    label(c, "option memory", 548, 444, t.accent, 10)
    c.text("u_i · K_i · V_i", 548, 466, MONO_MD, 13, t.ink)
    c.text("cached across requests", 548, 484, MONO, 10.5, t.ink2)
    c.add(arrow(464, 434, 520, 440, t.ink3))
    # decision block
    box(
        760,
        214,
        210,
        136,
        "decision block",
        "the state reads",
        "z_L = z_0 + attn(z_0 → K, V). Options never see each other: exact order invariance.",
        hot=True,
    )
    c.add(arrow(580, 282, 760, 282, t.ink3))
    c.add(arrow(724, 430, 800, 350, t.accent, curve=-40))
    # heads
    c.add(rect(1010, 150, 214, 210, fill=t.panel, stroke=t.rule, rx=12))
    label(c, "heads", 1026, 176, size=10)
    for k, (a, b) in enumerate(
        (
            ("scores", "z_L · u_i"),
            ("coverage", "null outcome"),
            ("yes / no", "own head"),
            ("score", "expected level"),
        )
    ):
        yy = 206 + k * 38
        c.text(a, 1026, yy, MONO_MD, 12, t.ink)
        c.text(b, 1208, yy, MONO, 11.5, t.ink2, anchor="end")
        if k < 3:
            c.add(line(1026, yy + 14, 1208, yy + 14, t.grid))
    c.add(arrow(970, 282, 1010, 260, t.ink3))
    # calibration
    box(1010, 380, 214, 82, "calibration", "T(type, log K)")
    c.add(line(1117, 360, 1117, 380, t.ink3, 1.6, "3 6", cls="flow"))
    # output bars
    for k, p in enumerate((0.07, 0.86, 0.04, 0.03)):
        yy = 490 + k * 14
        c.add(rect(1010, yy, 214, 6, fill=t.rule, rx=3))
        c.add(rect(1010, yy, 214 * p, 6, fill=t.accent if p > 0.5 else t.ink3, rx=3, cls=f"bar b{k}"))
    c.text("p(o_1 … o_K, none)", 1010, 560, MONO, 10.5, t.ink3)
    c.add(line(1117, 462, 1117, 486, t.ink3, 1.6, "3 6", cls="flow"))
    # milestone 2 slow path
    c.add(
        rect(56, 586, 1168, 82, fill="none", stroke=t.accent, rx=14, sw=1.4, extra=' stroke-dasharray="6 6"')
    )
    label(c, "milestone 2  ·  architecture C", 76, 614, t.accent, 10)
    c.text("joint slow path on the same backbone", 76, 646, SERIF, 23, t.ink)
    c.text(
        "state, question and rendered options in one prefix_bidir pass · LoRA · reached only when",
        520,
        628,
        MONO,
        11.5,
        t.ink2,
    )
    c.text("the fast path is unsure (calibrated-confidence gate)", 520, 648, MONO, 11.5, t.ink2)
    c.add(
        f'<path d="M1010 421 C960 421 980 586 940 586" fill="none" stroke="{t.accent}" stroke-width="1.6" '
        f'stroke-dasharray="3 6" class="flow" marker-end="url(#ah)"/>'
    )
    c.text("low confidence", 905, 548, MONO, 10.5, t.accent, anchor="end")
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
    ("Rebuild", ("CLM heads re-scored", "51.73 / 69.16 / 68.63"), "done"),
    ("Field survey", ("64 papers · 31 systems", "14 failure modes"), "done"),
    ("M0 · Harness", ("reproduce S1-Arena 41.4%", "then run the references"), "now"),
    ("M1 · Architecture B", ("late interaction · null", "K-aware calibration"), "next"),
    ("M2 · Architecture C", ("joint slow path", "calibrated cascade"), "later"),
    ("M3 · Release", ("model card · results", "v0.2.0"), "later"),
]


def roadmap(t: Theme) -> Canvas:
    c = Canvas(
        1280,
        270,
        t,
        "Roadmap",
        "Timeline: rebuild and field survey done; benchmark harness in progress; "
        "architecture B next, then architecture C and a release.",
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
    ("Read right, act wrong", "73–85%", "of top systems' errors follow correct reading", "Reading Right"),
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
