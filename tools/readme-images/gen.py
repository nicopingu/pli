#!/usr/bin/env python3
"""README images of PLI: light and dark variants, in the style of the logo (engraved lines, hatching).
Run: python3 gen.py (writes out/; copy the SVG files to docs/assets/)."""
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Inter, Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

PAL = {
    "light": dict(bg="#FBFAF6", ink="#1C2541", mute="#5B6683", line="#C9CCD6", acc="#D9772B", acc2="#2F7F86",
                  card="#FFFFFF", soft="#F1EEE6"),
    "dark": dict(bg="#11151F", ink="#E8E3D6", mute="#9CA3B5", line="#3A4152", acc="#F0A050", acc2="#5FB7BD",
                 card="#181D29", soft="#1F2533"),
}


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def head(w, h, p, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" '
            f'aria-label="{esc(title)}" font-family="{FONT}">\n<title>{esc(title)}</title>\n<defs>\n'
            f'<pattern id="hatch" patternUnits="userSpaceOnUse" width="7" height="7" patternTransform="rotate(35)">'
            f'<line x1="0" y1="0" x2="0" y2="7" stroke="{p["acc"]}" stroke-width="1.6"/></pattern>\n'
            f'<pattern id="hatch2" patternUnits="userSpaceOnUse" width="7" height="7" patternTransform="rotate(35)">'
            f'<line x1="0" y1="0" x2="0" y2="7" stroke="{p["acc2"]}" stroke-width="1.6"/></pattern>\n'
            f'<pattern id="hatchm" patternUnits="userSpaceOnUse" width="9" height="9" patternTransform="rotate(35)">'
            f'<line x1="0" y1="0" x2="0" y2="9" stroke="{p["mute"]}" stroke-width="1.1"/></pattern>\n'
            f'<marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{p["mute"]}"/></marker>\n</defs>\n'
            f'<rect x="0.5" y="0.5" width="{w-1}" height="{h-1}" rx="14" fill="{p["bg"]}" stroke="{p["line"]}"/>\n')


def text(x, y, s, size, fill, weight=400, anchor="start", family=None, extra=""):
    f = f' font-family="{family}"' if family else ""
    return (f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{f}{extra}>'
            f'{esc(s)}</text>\n')


# ---------------------------------------------------------------- icons (line art, 24 x 24)
ICONS = {
    "text": '<path d="M4 5h16v10H9l-5 4z"/><path d="M8 9h8M8 12h5"/>',
    "voice": '<path d="M3 12h2M7 8v8M11 5v14M15 9v6M19 7v10M21 12h0"/>',
    "mic": '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3M8 21h8"/>',
    "image": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 17l5-5 4 4 3-3 6 6"/><circle cx="16" cy="9" r="1.6"/>',
    "video": '<rect x="3" y="5" width="13" height="14" rx="2"/><path d="M16 10l5-3v10l-5-3z"/>',
    "prune": '<circle cx="6" cy="7" r="2.5"/><circle cx="6" cy="17" r="2.5"/><path d="M8 8.5L20 17M8 15.5L20 7"/>',
    "swap": '<path d="M4 8h13l-3-3M20 16H7l3 3"/>',
    "tune": '<path d="M5 4v16M12 4v16M19 4v16"/><circle cx="5" cy="14" r="2" /><circle cx="12" cy="8" r="2"/><circle cx="19" cy="16" r="2"/>',
    "first": '<path d="M12 20V5M6 11l6-6 6 6"/><path d="M5 21h14"/>',
    "memory": '<rect x="5" y="5" width="14" height="14" rx="2"/><path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3"/><rect x="9" y="9" width="6" height="6"/>',
    "cancel": '<circle cx="12" cy="12" r="9"/><path d="M8.5 8.5l7 7M15.5 8.5l-7 7"/>',
    "guard": '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/><path d="M8.5 12l2.5 2.5 4.5-5"/>',
    "disk": '<ellipse cx="12" cy="6" rx="8" ry="3"/><path d="M4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
    "chip": '<rect x="4" y="4" width="16" height="16" rx="3"/><path d="M9 9h6v6H9z"/>',
    "stop": '<path d="M8 3h8l5 5v8l-5 5H8l-5-5V8z"/><path d="M8 12h8"/>',
    "bolt": '<path d="M13 2L5 14h6l-1 8 8-12h-6z"/>',
}


def icon(name, x, y, size, color, sw=1.8):
    k = size / 24
    return (f'<g transform="translate({x},{y}) scale({k})" fill="none" stroke="{color}" stroke-width="{sw / k * k}" '
            f'stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</g>\n')


def icons_files():
    """Table icons: one variant, in an accent color readable on light and dark backgrounds."""
    d = os.path.join(OUT, "icons"); os.makedirs(d, exist_ok=True)
    for n in ICONS:
        s = (f'<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" '
             f'stroke="#E0843A" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">{ICONS[n]}</svg>\n')
        open(os.path.join(d, f"{n}.svg"), "w").write(s)


# ---------------------------------------------------------------- 1. key numbers
def glance(mode):
    p = PAL[mode]; W, H = 900, 250
    s = head(W, H, p, "PLI at a glance: 111 tokens/s writing, 229k-token context, 0.12 s to resume a saved conversation, text, voice, images and video on one Mac")
    s += text(36, 50, "One Mac Studio. One server. Every model it needs.", 21, p["ink"], 650)
    s += text(36, 76, "Qwen3.8-27B 4-bit on an M1 Ultra 64 GB, measured on the validated machine", 13.5, p["mute"])
    tiles = [
        ("bolt", "111", "tok/s", "writing, short prompt", "(public suite, PLI 4.0.0)"),
        ("text", "229k", "tokens", "context window", "4 conversations at a time"),
        ("disk", "0.12", "s", "to resume a saved", "14k-token conversation"),
        ("chip", "5", "families", "text, voice, transcription,", "images, video with sound"),
    ]
    tw, gap, x0, y0 = 196, 16, 36, 98
    for i, (ic, big, unit, l1, l2) in enumerate(tiles):
        x = x0 + i * (tw + gap)
        s += f'<rect x="{x}" y="{y0}" width="{tw}" height="122" rx="10" fill="{p["card"]}" stroke="{p["line"]}"/>\n'
        s += f'<rect x="{x}" y="{y0}" width="6" height="122" rx="3" fill="url(#hatch)"/>\n'
        s += icon(ic, x + tw - 38, y0 + 14, 22, p["acc"])
        s += (f'<text x="{x + 20}" y="{y0 + 56}" fill="{p["ink"]}"><tspan font-size="36" font-weight="700">{big}</tspan>'
              f'<tspan font-size="15" font-weight="500" fill="{p["acc"]}" dx="6">{unit}</tspan></text>\n')
        s += text(x + 20, y0 + 84, l1, 13.5, p["ink"], 500)
        s += text(x + 20, y0 + 103, l2, 12.5, p["mute"])
    return s + "</svg>\n"


# ---------------------------------------------------------------- 2. architecture
def box(x, y, w, h, p, fill=None, stroke=None, rx=10, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill or p["card"]}" '
            f'stroke="{stroke or p["line"]}" stroke-width="1.3"{d}/>\n')


def arch(mode):
    p = PAL[mode]; W, H = 900, 470
    s = head(W, H, p, "PLI architecture: one server with an OpenAI and Anthropic API, the 27B with its drafts on GPU and Neural Engine, generation children started on demand, snapshots on disk, background tasks between requests")
    s += text(36, 46, "How a request flows", 19, p["ink"], 650)
    # clients
    s += box(36, 78, 168, 344, p, fill=p["soft"])
    s += text(120, 104, "Clients", 14, p["ink"], 600, "middle")
    for i, (a, b) in enumerate([("OpenAI API", "/v1/..."), ("Anthropic API", "/v1/messages"), ("agents, IDEs,", "chat apps"), ("voice assistant", "speech in, speech out")]):
        y = 124 + i * 72
        s += box(52, y, 136, 56, p)
        s += text(120, y + 24, a, 13, p["ink"], 500, "middle")
        s += text(120, y + 42, b, 11.5, p["mute"], 400, "middle", MONO if b.startswith("/") else None)
    # pli serve
    X, Y, Wb, Hb = 244, 78, 620, 344
    s += f'<rect x="{X}" y="{Y}" width="{Wb}" height="{Hb}" rx="12" fill="none" stroke="{p["ink"]}" stroke-width="2.2"/>\n'
    s += f'<rect x="{X + 16}" y="{Y - 11}" width="96" height="22" rx="4" fill="{p["bg"]}"/>\n'
    s += text(X + 24, Y + 5, "pli serve", 14, p["ink"], 700, family=MONO)
    s += f'<line x1="204" y1="250" x2="{X - 4}" y2="250" stroke="{p["mute"]}" stroke-width="1.6" marker-end="url(#arr)"/>\n'
    s += text(224, 240, "HTTP", 11, p["mute"], 500, "middle")
    # scheduler
    s += box(X + 20, Y + 26, 580, 44, p, fill=p["soft"])
    s += icon("first", X + 34, Y + 36, 22, p["acc"])
    s += text(X + 66, Y + 46, "Scheduler", 13.5, p["ink"], 600)
    s += text(X + 66, Y + 62, "small requests first, several conversations per round, the LLM before any other model", 11.5, p["mute"])
    # LLM
    lx, ly, lw, lh = X + 20, Y + 86, 284, 150
    s += box(lx, ly, lw, lh, p)
    s += f'<rect x="{lx}" y="{ly}" width="{lw}" height="8" rx="4" fill="url(#hatch)"/>\n'
    s += icon("text", lx + 14, ly + 20, 22, p["acc"])
    s += text(lx + 46, ly + 36, "Qwen3.8-27B, 4-bit", 14, p["ink"], 650)
    s += text(lx + 16, ly + 62, "draft: DFlash2 or the MTP head,", 12, p["ink"])
    s += text(lx + 16, ly + 78, "picked per conversation", 12, p["ink"])
    s += text(lx + 16, ly + 98, "same text as without a draft (temp. 0)", 12, p["mute"])
    for k, (t1, t2) in enumerate([("writing", "GPU"), ("prompt reading", "GPU + Neural Engine")]):
        bx = lx + 16 + k * 112; bw2 = 100 if k == 0 else 140
        if k == 1: bx = lx + 124
        s += box(bx, ly + 106, bw2, 36, p, fill=p["soft"], rx=6)
        s += text(bx + bw2 / 2, ly + 120, t1, 10.5, p["mute"], 400, "middle")
        s += text(bx + bw2 / 2, ly + 135, t2, 11.5, p["ink"], 600, "middle")
    # children
    cx, cy = X + 320, Y + 86
    s += box(cx, cy, 280, 150, p, dash="5 4")
    s += text(cx + 14, cy + 22, "Generation children, on demand", 12.5, p["ink"], 600)
    kids = [("voice", "Voice"), ("mic", "Transcription"), ("image", "Images"), ("video", "Video + sound")]
    for i, (ic, lab) in enumerate(kids):
        kx = cx + 12 + (i % 2) * 132; ky = cy + 36 + (i // 2) * 52
        s += box(kx, ky, 124, 42, p, fill=p["soft"], rx=7)
        s += icon(ic, kx + 9, ky + 10, 20, p["acc2"])
        s += text(kx + 36, ky + 26, lab, 12, p["ink"], 500)
    s += text(cx + 14, cy + 144, "started when asked, stopped when idle", 10.5, p["mute"])
    # bottom row
    by = Y + 256
    s += box(X + 20, by, 284, 70, p)
    s += icon("disk", X + 34, by + 14, 22, p["acc"])
    s += text(X + 66, by + 30, "Conversations on disk", 13, p["ink"], 600)
    s += text(X + 34, by + 54, "taken up again without being read again", 11.5, p["mute"])
    s += box(X + 320, by, 280, 70, p)
    s += icon("tune", X + 334, by + 14, 22, p["acc"])
    s += text(X + 366, by + 30, "Background tasks", 13, p["ink"], 600)
    s += text(X + 334, by + 54, "prune, swap and tune the drafter, idle only", 11.5, p["mute"])
    s += text(36, 452, "Memory is counted before a model starts; a request that does not fit gets a 503 instead of slowing the Mac.", 12, p["mute"])
    return s + "</svg>\n"


# ---------------------------------------------------------------- 3. speed
def speed(mode):
    p = PAL[mode]; W, H = 900, 400
    s = head(W, H, p, "PLI speed on an M1 Ultra 64 GB with Qwen3.8-27B 4-bit: writing 111.2, 45.8, 49.2 and 34.1 tokens per second and prompt reading 191, 323, 285 and 213 tokens per second for prompts of 62, 8k, 32k and 96k tokens")
    s += text(36, 46, "Speed by prompt size", 19, p["ink"], 650)
    s += text(36, 70, "Qwen3.8-27B 4-bit, one request at a time, M1 Ultra 64 GB (public suite, PLI 4.0.0)", 13, p["mute"])
    cols = [("62 tokens", 111.2, 191, "0.32 s"), ("8k", 45.8, 323, "24.8 s"), ("32k", 49.2, 285, "112 s"), ("96k", 34.1, 213, "450 s")]
    x0, y0, ch = 90, 104, 190; base = y0 + ch; vmax = 350
    for v in (0, 100, 200, 300):
        y = base - v / vmax * ch
        dash = "" if v == 0 else ' stroke-dasharray="3 4"'
        s += f'<line x1="{x0}" y1="{y:.1f}" x2="{W - 40}" y2="{y:.1f}" stroke="{p["line"]}" stroke-width="1"{dash}/>\n'
        s += text(x0 - 10, y + 4, str(v), 11, p["mute"], 400, "end")
    gw = (W - 40 - x0) / 4; bw = 58
    for i, (lab, wr, rd, tt) in enumerate(cols):
        gx = x0 + i * gw + gw / 2
        for j, (v, pat, col) in enumerate([(wr, "hatch", p["acc"]), (rd, "hatch2", p["acc2"])]):
            bx = gx - bw - 4 + j * (bw + 8); h = v / vmax * ch
            s += f'<rect x="{bx:.1f}" y="{base - h:.1f}" width="{bw}" height="{h:.1f}" fill="url(#{pat})" stroke="{col}" stroke-width="1.6"/>\n'
            s += text(bx + bw / 2, base - h - 7, f"{v:g}", 13, p["ink"], 650, "middle")
        s += text(gx, base + 22, lab, 13.5, p["ink"], 600, "middle")
        s += text(gx, base + 40, f"first token {tt}", 11.5, p["mute"], 400, "middle")
    ly = 372
    s += f'<rect x="{x0}" y="{ly - 11}" width="14" height="14" fill="url(#hatch)" stroke="{p["acc"]}" stroke-width="1.4"/>\n'
    s += text(x0 + 22, ly, "writing (tok/s)", 12.5, p["ink"])
    s += f'<rect x="{x0 + 150}" y="{ly - 11}" width="14" height="14" fill="url(#hatch2)" stroke="{p["acc2"]}" stroke-width="1.4"/>\n'
    s += text(x0 + 172, ly, "prompt reading (tok/s)", 12.5, p["ink"])
    s += text(W - 40, ly, "a saved conversation skips the reading: 0.12 s for 13,939 tokens", 12, p["mute"], 400, "end")
    return s + "</svg>\n"


# ---------------------------------------------------------------- 4. memory sharing
def sharing(mode):
    p = PAL[mode]; W, H = 900, 330
    s = head(W, H, p, "How one 64 GB Mac is shared: the 27B with its drafts and conversations takes about 31 GB, one generation model at a time fits in the rest, memory is counted before it starts")
    s += text(36, 46, "64 GB, counted before anything starts", 19, p["ink"], 650)
    s += text(36, 70, "memory measured on the validated M1 Ultra 64 GB, PLI 5.6.2", 13, p["mute"])
    x0, x1, y, h = 36, W - 36, 98, 58; gb = (x1 - x0) / 64
    halo = f' paint-order="stroke" stroke="{p["bg"]}" stroke-width="4" stroke-linejoin="round"'
    segs = [(16.1, "27B weights", "16.1 GB"), (4.1, "drafts", "4.1 GB"), (10.5, "conversations", "10.5 GB")]
    x = x0
    for g, lab, val in segs:
        w = g * gb
        s += f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{h}" fill="url(#hatch)" stroke="{p["acc"]}" stroke-width="1.5"/>\n'
        s += text(x + w / 2, y + h / 2 - 2, lab, 12.5 if w > 120 else 11.5, p["ink"], 600, "middle", extra=halo)
        s += text(x + w / 2, y + h / 2 + 15, val, 11.5, p["ink"], 500, "middle", extra=halo)
        x += w
    xl = x
    s += f'<rect x="{x:.1f}" y="{y}" width="{x1 - x:.1f}" height="{h}" fill="none" stroke="{p["line"]}" stroke-width="1.5" stroke-dasharray="5 4"/>\n'
    s += text((x + x1) / 2, y + h / 2 + 5, "33 GB left: macOS and one generation model", 12.5, p["mute"], 500, "middle")
    s += f'<path d="M{x0} {y + h + 10} v8 H{xl:.1f} v-8" fill="none" stroke="{p["acc"]}" stroke-width="1.5"/>\n'
    s += text((x0 + xl) / 2, y + h + 36, "the LLM: 30.7 GB, always loaded", 13, p["ink"], 600, "middle")
    s += f'<path d="M{xl + 2:.1f} {y + h + 10} v8 H{x1} v-8" fill="none" stroke="{p["acc2"]}" stroke-width="1.5"/>\n'
    s += text((xl + x1) / 2, y + h + 36, "one of these at a time, started on demand", 13, p["ink"], 600, "middle")
    kids = [("voice", "Voice, Drama 3 in 3 bits", 26.2), ("image", "Images, Krea 2 Turbo in 4/8 bits", 24.2), ("video", "Video with sound, MiniMax H3", 11)]
    ky = 210
    for i, (ic, lab, g) in enumerate(kids):
        yy = ky + i * 34; w = g * gb
        s += f'<rect x="{xl + 2:.1f}" y="{yy}" width="{w:.1f}" height="26" rx="4" fill="url(#hatch2)" stroke="{p["acc2"]}" stroke-width="1.3"/>\n'
        s += f'<rect x="{xl + 8:.1f}" y="{yy + 3}" width="22" height="20" rx="3" fill="{p["bg"]}"/>\n'
        s += icon(ic, xl + 9, yy + 4, 18, p["acc2"])
        if w > 250:
            s += text(xl + 38, yy + 18, lab, 12, p["ink"], 600, extra=halo)
            s += text(xl + 2 + w + 8, yy + 18, f"{g:g} GB", 12, p["ink"], 600)
        else:
            s += (f'<text x="{xl + 2 + w + 8:.1f}" y="{yy + 18}" font-size="12" fill="{p["ink"]}"><tspan font-weight="600">{g:g} GB</tspan>'
                  f'<tspan fill="{p["mute"]}" dx="8">{esc(lab)}</tspan></text>\n')
    # 5.6.3: the rules (counted memory, swap guard, cancellation) are in the README's table under the image, not repeated here
    s += text(36, ky + 18, "While one of them runs,", 12.5, p["ink"], 500)
    s += text(36, ky + 38, "the LLM is capped at what it holds:", 12.5, p["mute"])
    s += text(36, ky + 58, "its oldest conversations go to disk", 12.5, p["mute"])
    s += text(36, ky + 78, "instead of growing (PLI 5.6.3).", 12.5, p["mute"])
    return s + "</svg>\n"


for name, fn in (("pli-at-a-glance", glance), ("pli-architecture", arch), ("pli-speed", speed), ("pli-sharing", sharing)):
    for mode in ("light", "dark"):
        suffix = "" if mode == "light" else "-dark"
        open(os.path.join(OUT, f"{name}{suffix}.svg"), "w").write(fn(mode))
icons_files()
print("ok", sorted(os.listdir(OUT)))
