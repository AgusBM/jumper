---
name: bundle-manual
description: Translate a built bundle's manual.en.json into Chinese as manual.zh.json and pack it into the .app. Use right after `python scripts/deploy.py` writes a bundle -- every bundle an agent builds ships with the Chinese manual as well as the English one -- and whenever a bundle is found with manual.en.json and no manual.zh.json, or when asked for "操作说明", "中文说明", "按键说明" or "manual" of a bundle/app.
---

# The bundle's manual, in Chinese

`scripts/deploy.py` writes `manual.en.json` (`kk-bundle-manual/1`) into every
bundle: each pad control and each key the bundle reads, what it does in each
mode, how each mode is reached and left, and every mode switch. It is generated
from the controller's own account of its controls (`padGuide`), so it cannot
disagree with the bindings. The pad and the keyboard are two paths since
2026-09-29 and the manual lists them apart: in each mode `controls` is the pad
and `keys` the keyboard, and each switch says which `device` it is on. The
Chinese version is a translation of it, and this skill is how it is made: the
build cannot write natural Chinese, and a person or an agent can.

## Steps

1. Find the bundle directory. `deploy.py` prints it
   (`[deploy] bundled <commit> -> …/out/bundle_<ts>/<name>`) and then prints the
   `--translate` command with that path already filled in. It holds
   `manual.en.json` and `bundle.json`.
2. Read `manual.en.json` in full.
3. Write `manual.zh.json` somewhere outside the bundle (a scratch file):
   - the same JSON, field for field, in the same order;
   - `"language": "zh"`;
   - translate **only** the text fields: `source`, `summary`, `enter`, `leave`,
     `keyboard`, `pad`, `does` -- and `does` wherever it is: in each mode's
     `controls` (the pad), in each mode's `keys` (the keyboard), and in
     `switches`. A `pad` that is `null` (a keyboard switch) stays `null`;
   - leave every other value exactly as it is -- `schema`, `bundle`, `mode`,
     `task`, `default`, `control`, `with`, `device`, `from`, the `key` of each
     entry in a mode's `keys` (`W`, `;`, `Space`, `↑`), and the `keys`
     lists of `switches` (`Space`, `Ctrl + 1`, `Num 1`). They are identifiers,
     and the check below refuses a translation that changes one.
4. Add it to the bundle, which checks it and packs the `.app` again:

   ```bash
   python scripts/deploy.py --translate out/bundle_<ts>/<name> --language zh --manual <scratch file>
   ```

   It refuses a file whose non-text fields differ from the English one, and
   holds the result to `deploy/app.schema`. Fix what it names and run it again.
5. If the `.app` is going to be committed (bundles are committed under
   `out/bundle_<ts>/<name>.app`), commit it after this step, not before: the
   translation changes the archive.

## How to write it

For someone holding a gamepad or sitting at a keyboard, in plain Chinese. Keep
mode names as identifiers where they name a mode (`进入 jump`), and say what the
mode is in the sentence when the English does. The manual lists each mode's
keys on their own; say what a key does, never which pad control it is -- the
English never says that either.

| English | 中文 |
|---|---|
| left stick / right stick | 左摇杆 / 右摇杆 |
| up and down / left and right | 上下 / 左右 |
| up: …; down: … | 上：……；下：…… |
| left trigger (LT) / right trigger (RT) | 左扳机（LT）/ 右扳机（RT） |
| left bumper (LB) / right bumper (RB) | 左肩键（LB）/ 右肩键（RB） |
| right stick click (R3) | 右摇杆按下（R3） |
| D-pad / D-pad up | 十字键 / 十字键上 |
| hold Menu, then D-pad up | 按住 Menu 再按十字键上 |
| Menu, pressed and let go on its own | 单独按下再松开 Menu |
| walk forward / walk back | 前进 / 后退 |
| step left / step right | 左移 / 右移 |
| turn left / turn right | 左转 / 右转 |
| twist left / twist right | 左扭 / 右扭 |
| nose down / nose up | 低头 / 抬头 |
| roll, right side down / left side down | 横滚，右侧低 / 横滚，左侧低 |
| stand taller / crouch lower | 升高 / 降低 |
| (moves it; let go, it stays) | （持续移动，松开即停留） |
| with right stick click (R3) held -- … | 按住右摇杆（R3）时——…… |
| hold for the other stick layer | 按住切换到摇杆的另一层 |
| tapped -- pressed and let go without the stick moving -- puts the moved axes back at rest | 轻点（按下后摇杆不动就松开）：被移动的量回到静止 |
| let go of everything: every control back to rest | 全部松开：所有操作回到静止 |
| switch to X; press again to switch back | 进入 X；再按一次退出 |
| (from X, Y only) | （仅在 X、Y 中有效） |
| leave X, Y -- whichever is on -- back to locomotion | 退出 X、Y 中正在进行的那一个，回到 locomotion |
| click A once / double-click A / triple-click A | 单击 A / 双击 A / 三连击 A |
| click A four times / click A five times | 四连击 A / 五连击 A |
| switch to X; double-click again to switch back | 进入 X；再双击一次退出 |

A task's own words -- `closes the left claw as far as it is held`, from its
`controls.yaml` -- are translated like any other text: `按多深，左爪就夹多紧`;
`holds the carried arm straight out, thumb up, while held` is
`按住时手臂向前伸直，拇指朝上`, and thumb-web up is `虎口朝上`. A claw mode lists
only its own side's claw (the left claw's lists `LT` and Space, not `RT`), so
translate what is there and add nothing.

## Why a separate step

The English is the build's and the Chinese is a translation of it, never the
other way round: a manual written in Chinese first would have no generator to
hold it to the bindings. `--translate` is what keeps the two the same manual in
two languages -- it compares every identifier -- and it is also why a bundle
built without an agent still ships the English one.
