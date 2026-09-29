"""Generate the perimeter-board texture for the football pitch.

A strip of advertising hoardings: an off-white board with a repeating run of
panels along it. Written as a wide 2D image and tiled around the four boards.

## The brand names are invented, and that is not incidental

Every name here is made up. A rendered pitch carrying a real company's wordmark
is that company's advertising, placed by us, in a video we did not ask them
about -- and the mark is theirs whether the render is a product demo, a paper
figure or a clip on a slide. Invented names cost nothing and carry none of that.
Keep it that way if the list is extended.

## Why a texture rather than more geoms

The panels could be thin boxes laid against the boards, and that would put five
hundred extra geoms in a scene whose collision budget is already the thing that
limits `num_envs`. A texture is free at simulation time: the boards stay four
boxes and the picture is the renderer's problem.

## The aspect ratio is the thing to get right

A board is `BOARD_H` tall and runs the length of the pitch, so a texture meant to
tile along it has to be much wider than it is tall or the panels come out
stretched. `repeat` in the material is what sets how many times the strip goes
round; the image itself is one run of panels, drawn at the ratio a real hoarding
has (about 6:1 per panel).
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

#: One panel, in pixels. 6:1 is roughly a real pitch-side hoarding.
PANEL_W = 384
PANEL_H = 64

#: The board itself: near-white. Kept a shade off pure white rather than at
#: (1, 1, 1) because the material multiplies the texture by the geom's rgba and
#: the sun in this scene has diffuse 1.05 -- a pure-white board clips to flat
#: white on the lit faces and loses the top rail and the panel edges with it.
#: 0.95 leaves that headroom and still reads as white next to the turf.
BOARD_RGB = (0.95, 0.95, 0.96)

#: The reds. Two of them: the face, and a darker one behind it for the offset
#: shadow that gives the lettering its depth.
_RED = (198, 32, 38)
_RED_DARK = (122, 16, 22)

#: The panels: invented names on a near-white board, all in the same red.
#:
#: One colour rather than six was a choice. Hoardings at a real ground are a
#: jumble because six advertisers bought six panels; here they are scenery behind
#: the thing being looked at, and six saturated palettes pull the eye off the
#: robot. The same reasoning as `rough.py`'s `color_scheme="none"`, which
#: desaturated the stair terrain for the same reason.
#:
#: Panel backgrounds are a hair off the board colour so the panel edges read at
#: all -- at equal values the run becomes one continuous red-on-white band.
_PANELS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("NORTHWIND", (250, 250, 250)),
    ("KOMOREBI", (244, 244, 246)),
    ("ATLAS FEED", (250, 250, 250)),
    ("VELA LABS", (244, 244, 246)),
    ("SIX RIVERS", (250, 250, 250)),
    ("HALCYON", (244, 244, 246)),
)

#: Display faces to try, in order. Bold oblique reads as a hoarding rather than
#: as a caption -- the slant is most of what makes it look like signage.
#:
#: matplotlib bundles DejaVu and is in the dev extra, so the first candidate is
#: normally present. **This generator is not on any runtime path**: the PNG it
#: writes is committed, and a scene loads that file. So a missing font degrades
#: the next regeneration, not anybody's training run, which is why the fallback
#: chain ends at PIL's bitmap default rather than raising.
_FONT_CANDIDATES = (
    # Relative to matplotlib's package directory, which already ends in
    # `matplotlib/` -- naming it again here silently resolves to
    # `site-packages/matplotlib/matplotlib/...`, misses, and drops through to the
    # system font. Which is exactly what happened: the first render came out in
    # upright Bold because this path had the prefix doubled, and nothing said so.
    "mpl-data/fonts/ttf/DejaVuSans-BoldOblique.ttf",
    "mpl-data/fonts/ttf/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)


def _font(size: int):
    """The first of `_FONT_CANDIDATES` that loads, else PIL's bitmap default.

    Relative candidates are resolved against matplotlib's package directory,
    which is where the bundled DejaVu faces live. Absolute ones are tried as
    given.

    `load_default(size=...)` needs Pillow 10.1; older ones ignore the argument
    and return a fixed 11 px font. That draws legible-but-tiny text rather than
    failing, which is the right end for a fallback chain -- see the note on
    `_FONT_CANDIDATES` for why nothing here raises.
    """
    from pathlib import Path as _P

    roots = []
    try:
        import matplotlib

        roots.append(_P(matplotlib.__file__).parent)
    except ImportError:
        pass

    for cand in _FONT_CANDIDATES:
        paths = [_P(cand)] if _P(cand).is_absolute() else [r / cand for r in roots]
        for path in paths:
            if path.is_file():
                return ImageFont.truetype(str(path), size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def render_strip() -> Image.Image:
    """One run of every panel, side by side, as an RGB image."""
    w, h = PANEL_W * len(_PANELS), PANEL_H
    img = Image.new("RGB", (w, h), tuple(int(c * 255) for c in BOARD_RGB))
    draw = ImageDraw.Draw(img)
    font = _font(int(PANEL_H * 0.46))

    for i, (text, bg) in enumerate(_PANELS):
        x0 = i * PANEL_W
        # A hair of board left between panels, so the run reads as separate
        # hoardings rather than as one continuous printed band.
        draw.rectangle([x0 + 3, 3, x0 + PANEL_W - 4, h - 4], fill=bg)
        box = draw.textbbox((0, 0), text, font=font)
        x = x0 + (PANEL_W - (box[2] - box[0])) / 2 - box[0]
        y = (h - (box[3] - box[1])) / 2 - box[1]
        # Offset shadow first, face over it. Two draws is all the "display
        # lettering" here is: at 64 px tall the glyphs are a few pixels thick, so
        # an outline would close up the counters and a gradient would not survive
        # the texture filtering. An offset copy still reads at a distance.
        draw.text((x + 2, y + 2), text, font=font, fill=_RED_DARK)
        draw.text((x, y), text, font=font, fill=_RED)

    # A darker rail along the top edge. Every real hoarding has one, and without
    # it the boards read as a painted stripe on the grass rather than as an
    # object standing up from it.
    draw.rectangle([0, 0, w - 1, 2], fill=_RED_DARK)
    return img


def write(out_dir: Path, name: str = "hoardings") -> Path:
    """Render the strip and write it. Returns the path, for a `TextureCfg`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    render_strip().save(path)
    return path


__all__ = ["BOARD_RGB", "PANEL_H", "PANEL_W", "render_strip", "write"]
