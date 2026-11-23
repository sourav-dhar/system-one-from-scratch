"""Build the Gate A0 bake-off page: generate the axonometric architecture SVG and the cosine-gap
ruler from real data, then splice both into template.html -> clm-bakeoff.html.

Run from the repo:  uv run python <scratchpad>/gate-a0/build.py
"""

import gzip
import json
import math
import pathlib
import statistics as st

HERE = pathlib.Path(__file__).parent
REPO = pathlib.Path(__file__).resolve().parents[3]


def f(v: float) -> str:
    return f"{v:.1f}"


DX, DY = 24, 14


def card(x, cy, h, dx=DX, dy=DY):
    yt = cy - h / 2 + dy / 2
    return {"ft": (x, yt), "bt": (x + dx, yt - dy), "bb": (x + dx, yt - dy + h), "fb": (x, yt + h)}


def poly(c, cls):
    pts = [c["ft"], c["bt"], c["bb"], c["fb"]]
    return f'<polygon class="{cls}" points="{" ".join(f"{f(a)},{f(b)}" for a, b in pts)}"/>'


def line(p, q, cls, extra=""):
    return f'<line class="{cls}" x1="{f(p[0])}" y1="{f(p[1])}" x2="{f(q[0])}" y2="{f(q[1])}"{extra}/>'


def text(x, y, s, cls, anchor="start"):
    return f'<text class="{cls}" x="{f(x)}" y="{f(y)}" text-anchor="{anchor}">{s}</text>'


def funnel(a, b):
    return "".join(line(a[k], b[k], "conn") for k in ("ft", "fb", "bt", "bb"))


def sheet(x, cy, h, dx, dy, nlines):
    c = card(x, cy, h, dx, dy)
    out = [poly(c, "c-sheet")]
    x0, y0 = c["ft"]
    slope = dy / dx
    for i in range(nlines):
        yy = y0 + 13 + i * ((h - 24) / max(1, nlines - 1))
        ln = (dx - 14) * (0.55 if i % 3 == 2 else 0.85)
        sx, sy = x0 + 7, yy - 7 * slope
        out.append(line((sx, sy), (sx + ln, sy - ln * slope), "txtline"))
    return out, c


def vec(x, cy, h, cls, dx=9, dy=5, cells=True):
    c = card(x, cy, h, dx, dy)
    out = [poly(c, cls)]
    if cells:
        n = int(h // 7)
        for i in range(1, n):
            t = i / n * h
            out.append(line((c["ft"][0], c["ft"][1] + t), (c["bt"][0], c["bt"][1] + t), "cell"))
    return out, c


def stack(x0, cy, n=36, step=3.0, h=120):
    out = []
    for i in range(n):
        out.append(poly(card(x0 + i * step, cy, h), "c-layer c-frozen" if i == n - 1 else "c-layer"))
    return out, card(x0 + (n - 1) * step, cy, h)


def arrow(p, q, cls="flow"):
    return f'<path class="{cls}" d="M{f(p[0])},{f(p[1])} L{f(q[0])},{f(q[1])}" marker-end="url(#arr2)"/>'


def head(xs, cy, src):
    """Three trained layers 4096->1536->1536->512, plate heights proportional to width."""
    hs = [120 * 1536 / 4096, 120 * 1536 / 4096, 120 * 512 / 4096]
    out, prev, cards = [], src, []
    for x, h in zip(xs, hs):
        c = card(x, cy, h)
        out.append(funnel(prev, c))
        out.append(poly(c, "c-trained"))
        prev = c
        cards.append(c)
    return out, cards


def arch_svg() -> str:
    o = []
    o.append('<svg id="arch" class="dg" viewBox="0 0 1040 540" role="img" aria-labelledby="arch-t arch-d">')
    o.append('<title id="arch-t">CLM-v0.1-8B architecture: two towers over one frozen backbone</title>')
    o.append(
        '<desc id="arch-d">Top tower: the state and question, joined into one text, pass through frozen Qwen3-8B '
        "(36 decoder layers). The last token's 4096-dimensional hidden state goes through the trained state head "
        "(4096 to 1536 to 1536 to 512) and is normalised to the unit vector s. Bottom tower: each of the K option "
        "texts, without the state, passes through the same frozen weights and the trained action head. Both towers "
        "L2-normalise the 4096-dimensional embedding before their head. Option vectors, and state vectors too, are "
        "stored in a GPU vector arena on first use and read back on repeats. Each option's logit is its cosine with s "
        "times min(e to the lambda, 100); the probabilities are the softmax of the logits divided by T. Angles in the "
        "circle are exaggerated.</desc>"
    )
    o.append(
        '<defs><pattern id="ah" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
        '<line class="hl" x1="0" y1="0" x2="0" y2="5"/></pattern>'
        '<pattern id="ah2" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(-45)">'
        '<line class="hl" x1="0" y1="0" x2="0" y2="5"/></pattern>'
        '<marker id="arr2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" '
        'orient="auto-start-reverse"><path class="arrowhead" d="M0,1 L9,5 L0,9 Z"/></marker></defs>'
    )
    TOP, BOT = 170, 420

    # ---- top tower: state + question
    s_out, _ = sheet(30, TOP, 112, 64, 37, 7)
    o += s_out
    o.append(text(30, 84, "STATE + QUESTION", "tc"))
    o.append(arrow((100, TOP + 4), (154, TOP + 4)))
    st_out, last_top = stack(160, TOP)
    o += st_out
    o.append(text(225, 66, "Qwen3-8B · frozen", "tl", "middle"))
    o.append(text(225, 82, "36 decoder layers", "tm", "middle"))
    lt = (last_top["fb"][0] + 14, last_top["fb"][1] - 14)
    o.append(f'<circle class="dot" cx="{f(lt[0])}" cy="{f(lt[1])}" r="3.2"/>')
    o.append(text(225, 264, "last-token hidden state", "tm", "middle"))
    o.append(text(321, 248, "L2-NORM", "tc", "middle"))
    o.append(arrow((292, TOP), (312, TOP)))
    v_out, es = vec(316, TOP, 120, "c-vec")
    o += v_out
    o.append(
        text(321, 98, 'e<tspan class="sub" dy="3">s</tspan><tspan dy="-3"> · 4096</tspan>', "tm", "middle")
    )
    h_out, hc = head([378, 438, 498], TOP, es)
    o += h_out
    for c, dim, act in zip(hc, ["1536", "1536", "512"], ["GELU", "LN · GELU", "linear"]):
        o.append(text(c["ft"][0] + 12, c["bt"][1] - 8, dim, "tm", "middle"))
        o.append(text(c["ft"][0] + 12, 222, act, "tc", "middle"))
    o.append(
        text(
            450,
            100,
            'state head <tspan font-style="italic">f</tspan><tspan class="sub" dy="3">s</tspan>',
            "tl",
            "middle",
        )
    )
    o.append(text(450, 116, "9,443,840 params · trained", "tm", "middle"))
    o.append(arrow((526, TOP), (552, TOP)))
    so, s_card = vec(556, TOP, 15, "c-out", cells=False)
    o += so
    o.append(text(561, 150, "s", "tl it", "middle"))
    o.append(text(561, 200, "unit", "tc", "middle"))

    # ---- shared weights
    o.append(line((225, 268), (225, 346), "sharedl"))
    o.append(text(233, 312, "same weights", "tm"))

    # ---- bottom tower: K options
    for i in (3, 2, 1, 0):
        so, _ = sheet(30 + i * 9, BOT - i * 11, 60, 52, 30, 3)
        o += so
    o.append(text(30, 330, "K OPTION TEXTS · NO STATE", "tc"))
    o.append(arrow((116, BOT + 4), (154, BOT + 4)))
    sb_out, _ = stack(160, BOT)
    o += sb_out
    o.append(text(225, 506, "Qwen3-8B · same frozen weights", "tl", "middle"))
    o.append(arrow((292, BOT), (308, BOT)))
    ea = None
    for i in (3, 2, 1, 0):
        vo, c = vec(312 + i * 4, BOT, 120, "c-vec", cells=(i == 0))
        o += vo
        ea = c if i == 0 else ea
    o.append(
        text(
            330, 348, 'e<tspan class="sub" dy="3">a</tspan><tspan dy="-3"> · 4096 × K</tspan>', "tm", "middle"
        )
    )
    h_out, hc = head([378, 438, 498], BOT, ea)
    o += h_out
    for c, dim, act in zip(hc, ["1536", "1536", "512"], ["GELU", "LN · GELU", "linear"]):
        o.append(text(c["ft"][0] + 12, c["bt"][1] - 8, dim, "tm", "middle"))
        o.append(text(c["ft"][0] + 12, 472, act, "tc", "middle"))
    o.append(
        text(
            450,
            506,
            'action head <tspan font-style="italic">f</tspan><tspan class="sub" dy="3">a</tspan>',
            "tl",
            "middle",
        )
    )
    o.append(text(450, 522, "9,443,840 params · trained", "tm", "middle"))
    o.append(arrow((526, BOT), (546, BOT)))
    for i in (3, 2, 1, 0):
        vo, _ = vec(550 + i * 4, BOT, 15, "c-out", cells=False)
        o += vo
    o.append(text(563, 400, 'a<tspan class="sub" dy="3">1…K</tspan>', "tl it", "middle"))

    # ---- arena (isometric slab): front face hatched = cached
    fx0, fx1, fy0, fy1, ddx, ddy = 612, 744, 444, 486, 36, 21
    o.append(arrow((578, BOT + 2), (632, 436)))
    o.append(text(606, 454, "stored on first use", "tm", "end"))
    o.append(
        f'<polygon class="arena-top" points="{fx0},{fy0} {fx1},{fy0} {fx1 + ddx},{fy0 - ddy} {fx0 + ddx},{fy0 - ddy}"/>'
    )
    o.append(
        f'<polygon class="arena-side" points="{fx1},{fy0} {fx1 + ddx},{fy0 - ddy} {fx1 + ddx},{fy1 - ddy} {fx1},{fy1}"/>'
    )
    o.append(f'<rect class="arena-front" x="{fx0}" y="{fy0}" width="{fx1 - fx0}" height="{fy1 - fy0}"/>')
    for i in range(1, 8):
        t = i / 8
        o.append(line((fx0 + t * (fx1 - fx0), fy0), (fx0 + t * (fx1 - fx0) + ddx, fy0 - ddy), "grid-l"))
    for j in range(1, 3):
        t = j / 3
        o.append(line((fx0 + t * ddx, fy0 - t * ddy), (fx1 + t * ddx, fy0 - t * ddy), "grid-l"))
    o.append(text(fx0, 506, "GPU VECTOR ARENA", "tc"))
    o.append(text(fx0, 522, "512-d slots · LRU · 2% of memory by default", "tm"))

    # ---- the state vector is cached too
    o.append('<path class="flow cachel" d="M561,206 C561,300 690,330 700,418" marker-end="url(#arr2)"/>')
    o.append(text(596, 300, "s cached too, keyed by text", "tm", "end"))

    # ---- comparison: unit circle, angles exaggerated
    cx, cy, r = 905, 292, 84
    o.append(
        f'<path class="flow" d="M570,{TOP} L862,{TOP} C900,{TOP} {cx},{TOP + 8} {cx},{cy - r - 4}" marker-end="url(#arr2)"/>'
    )
    o.append(
        f'<path class="flow" d="M{fx1 + ddx + 2},{fy0 - ddy + 6} C820,410 832,380 {f(cx - r * 0.72)},{f(cy + r * 0.72)}" marker-end="url(#arr2)"/>'
    )
    o.append(text(800, 388, "read on each request", "tm", "end"))
    o.append(f'<circle class="circ" cx="{cx}" cy="{cy}" r="{r}"/>')
    o.append(line((cx - r, cy), (cx + r, cy), "axis-f"))
    o.append(line((cx, cy - r), (cx, cy + r), "axis-f"))

    def tip(deg, k=0.94):
        a = math.radians(deg)
        return (cx + r * k * math.cos(a), cy - r * k * math.sin(a))

    for deg in (63, 68, 78, 84):
        o.append(line((cx, cy), tip(deg), "avec"))
    o.append(line((cx, cy), tip(50), "svec"))
    a0, a1 = math.radians(50), math.radians(84)
    rr = 34
    o.append(
        f'<path class="arc" d="M{f(cx + rr * math.cos(a0))},{f(cy - rr * math.sin(a0))} '
        f'A{rr},{rr} 0 0 0 {f(cx + rr * math.cos(a1))},{f(cy - rr * math.sin(a1))}"/>'
    )
    o.append(text(cx + 30, cy - 46, 'θ<tspan class="sub" dy="3">i</tspan>', "tl it"))
    t = tip(50, 1.05)
    o.append(text(t[0] + 4, t[1] + 4, "s", "tl it"))
    t = tip(80, 1.06)
    o.append(text(t[0] - 6, t[1] - 2, 'a<tspan class="sub" dy="3">i</tspan>', "tl it", "middle"))
    o.append(
        text(
            cx,
            404,
            'ℓ<tspan class="sub" dy="3">i</tspan><tspan dy="-3"> = min(e</tspan><tspan class="sup" dy="-5">λ</tspan>'
            '<tspan dy="5">, 100) · cos θ</tspan><tspan class="sub" dy="3">i</tspan>',
            "tl",
            "middle",
        )
    )
    o.append(text(cx, 421, "λ trained; factor capped at 100", "tm", "middle"))
    o.append(text(cx, 436, "angles exaggerated, see §4", "tc", "middle"))
    o.append(arrow((cx, 442), (cx, 452)))
    o.append(f'<rect class="sm-box" x="{cx - 56}" y="454" width="112" height="40"/>')
    o.append(text(cx, 471, "softmax(ℓ / T)", "tl", "middle"))
    o.append(
        text(
            cx,
            487,
            'p<tspan class="sub" dy="3">1</tspan><tspan dy="-3"> … p</tspan><tspan class="sub" dy="3">K</tspan>',
            "tm",
            "middle",
        )
    )

    # ---- blueprint-only dimension + title block
    o.append('<g class="only-bp">')
    o.append(line((160, 288 - 10), (265, 288 - 10), "dim"))
    o.append("</g>")
    o.append("</svg>")
    return "\n".join(o)


def ruler_svg(gaps_item: dict, lows: list) -> str:
    W, x0, x1 = 640, 44, 604
    lo = -0.03

    def X(v):
        return x0 + (v - lo) / (0 - lo) * (x1 - x0)

    o = ['<svg id="ruler" class="dg" viewBox="0 0 640 214" role="img" aria-labelledby="rul-t rul-d">']
    o.append('<title id="rul-t">How far apart CLM puts the four options, in cosine</title>')
    o.append(
        '<desc id="rul-d">Each option\'s cosine with the state, relative to the top option, recovered from the '
        "recorded probabilities. On the example item: Mixtral 0, GPT-4 Turbo minus 0.0049, Claude 2 minus 0.0171, "
        f"GPT-3.5 Turbo minus 0.0214. Across all 250 RouterBench items, the lowest option sits "
        f"{-max(lows):.3f} to {-min(lows):.3f} below the top one, median {-st.median(lows):.3f}.</desc>"
    )
    for k in range(7):
        v = lo + k * 0.005
        o.append(line((X(v), 52), (X(v), 168), "rtick"))
        o.append(text(X(v), 186, f"{v:+.3f}".replace("+0.000", "0").replace("-", "−"), "tm", "middle"))
    o.append(text(x0, 204, "cosine relative to the top option", "tc"))
    # row 1: the example item
    o.append(line((x0, 72), (x1, 72), "raxis"))
    o.append(text(x0, 20, "ONE ITEM", "tc"))
    lab = {
        "gpt_3_5_turbo_1106": ("GPT-3.5 Turbo", "0.061", 44, "middle"),
        "claude_v2": ("Claude 2", "0.094", 20, "middle"),
        "gpt_4_1106_preview": ("GPT-4 Turbo", "0.322", 44, "middle"),
        "mixtral_8x7b_chat": ("Mixtral · gold", "0.523", 20, "end"),
    }
    for k, (name, p, y, anc) in lab.items():
        x = X(gaps_item[k])
        o.append(line((x, 66), (x, y + 6), "lead"))
        o.append(
            f'<circle class="rmark{" top" if k == "mixtral_8x7b_chat" else ""}" cx="{f(x)}" cy="72" r="5"/>'
        )
        tx = x + 6 if anc == "end" else x
        o.append(text(tx, y, f'{name} <tspan class="pv-n">p {p}</tspan>', "tm", anc))
    # row 2: all 250 items' lowest option
    for i, v in enumerate(lows):
        j = ((i * 2654435761) % 1000) / 1000 - 0.5
        o.append(f'<circle class="rdot" cx="{f(X(v))}" cy="{f(138 + j * 22)}" r="2.1"/>')
    med = st.median(lows)
    o.append(line((X(med), 118), (X(med), 158), "rmed"))
    o.append(text(X(med) + 5, 165, f"median {med:.3f}".replace("-", "−"), "tm"))
    o.append(text(x0, 108, "LOWEST OPTION ON EACH OF 250 ITEMS", "tc"))
    o.append("</svg>")
    return "\n".join(o)


def ruler_data():
    import sys

    sys.path.insert(0, str(REPO / "src"))
    from systemone.decisions import s1arena

    items = {it.id: it for it in s1arena.load(domains=["route"], tasks=["route/model_selection/routerbench"])}
    lows, ex = [], None
    path = REPO / "results/references/s1arena/Contrastive-LM__CLM-v0.1-8B/static/responses.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for row in fh:
            r = json.loads(row)
            it = items.get(r["item_id"])
            if not it:
                continue
            ((qk, _),) = it.request.questions.items()
            p = r["response"]["answers"][qk]["probs"]
            lp = {k: math.log(v) for k, v in p.items()}
            top = max(lp.values())
            gaps = {k: (v - top) / 100 for k, v in lp.items()}  # scale 100, T = 1 in our port
            lows.append(min(gaps.values()))
            if r["item_id"].endswith("4066bddf0c7e"):
                ex = gaps
    assert len(lows) == 250 and ex is not None
    return ex, lows


def main():
    ex, lows = ruler_data()
    page = (HERE / "template.html").read_text(encoding="utf-8")
    page = page.replace("<!--ARCH-->", arch_svg()).replace("<!--RULER-->", ruler_svg(ex, lows))
    (HERE / "clm-bakeoff.html").write_text(page, encoding="utf-8")
    print(
        "ok",
        {k: round(v, 5) for k, v in ex.items()},
        "lows",
        round(min(lows), 4),
        round(max(lows), 4),
        round(st.median(lows), 4),
    )


if __name__ == "__main__":
    main()
