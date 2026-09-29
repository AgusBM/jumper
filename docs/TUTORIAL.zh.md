<!-- tracks: TUTORIAL.md @ sha256:0cc9718280c37249 -->

# 教程 —— 三个技能，一台机器人，一路走到真机

一个完整的示例：训练 `jumper.tripod`、`jumper.jump`、`jumper.dance`，把三者装进**同一个
bundle**，在浏览器里驱动它，再放到机器人上跑。这里没有任何为教程编造的东西 —— 它就是写这
一页时本仓库交付的那个 bundle，下面每一条命令都是为写它而真的跑过的。

> **仓库交付的 `jumper` bundle 之后改过控制方式**（2026-09-26 和 2026-09-29）：`locomotion` 这一格
> 换成了 `jumper.posture`；`A` 或 Space 进入 jump，进入本身就是起跳的全部请求 —— 没有单独的 `go`；
> 四支舞是 Menu + 十字键（Ctrl + 1 2 3 4），四个固定动作是单按十字键（1 2 3 4）；`jumper.five_foot`
> 的爪子在 `LB`（左，或 `V`）和 `RB`（右，或 `B`）上；一个模式的开关每个设备一个，手柄的和键盘的
> 各是各的。当前的条目以 [`deploy/manifests.json`](../deploy/manifests.json) 为准，布局见
> [`CONTROLS.md`](CONTROLS.md) §5.11。下面的流程不变。

其它文档是参考手册：[`AGENT_SETUP.md`](AGENT_SETUP.md) 讲怎么把机器装起来，
[`USAGE.md`](USAGE.md) 讲任务与资产，[`../deploy/README.md`](../deploy/README.md) 是部署路线图，
[`DESIGN.md`](DESIGN.md) 讲为什么是这个形状。这一篇讲的是**顺序**，只做链接，不复述。

## 最后你会得到什么

```
out/bundle_<timestamp>/
├── jumper/       .rknn + runtime/board/controller (aarch64)       ← 机器人
│                 .onnx + runtime/web/controller.wasm              ← 浏览器
│                 .onnx + runtime/mjlab/<platform>/controller.so   ← play --app
└── jumper.app    同一个目录，打成 zip
```

**一个**状态机，一个 bundle，三种宿主，各取自己那一份。它装着三个策略，以及在它们之间切换的
`controller.toml`：

| 模式 | 任务 | 观测 → 动作 | 速率 | 怎么进去 |
|---|---|---|---|---|
| `locomotion` | `jumper.tripod` | 411 → 20 | 50 Hz | 级联最后那条 `always` 规则 |
| `jump` | `jumper.jump` | 167 → 20 | 200 Hz | `LB` 进入，再按 `A` 触发 |
| `dance` | `jumper.dance` | 209 → 22 | 50 Hz | `RB` |

`jump` 和 `dance` 是**参考引导**的：各自带着一段随策略同行的录制运动。这是整条路上**最容易
悄无声息地出错**的部分，所以本教程在它上面花的笔墨最多。

---

## 0. 先理解这条规则，后面才讲得通

**这条路上没有任何东西会大声失败**。关节顺序错了、QoS 配置没对上、训练时被门控的步态时钟在
部署后自由运行、摇杆被页面而不是被契约缩放 —— 每一个的结果都是**一台跑起来了但是错的机器人**，
而不是一个报错。

所以下面每一步的末尾都写了**输出里该读哪几行**。跳过它，就是让一个带着看不见的故障的 bundle
走到机器人身上。

---

## 1. 把机器装起来

照着 [`AGENT_SETUP.md`](AGENT_SETUP.md) 做。从干净克隆开始的简版：

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

`pip install -e .` 是**必需的，不是图方便**。`pyproject.toml` 声明了两个包根，`rl/` 下没有
`__init__.py`，没有这个可编辑安装它下面的东西一律导入不了 —— 这是刻意的，所以全仓库没有任何
一处 `sys.path` 操作。

验证装好了：

```bash
python scripts/train.py --list
python -m pytest tests/ -q
```

`--list` 必须在一台没装仿真依赖的机器上也能跑，而且跑完之后 `sys.modules` 里不应该有任何一个
重依赖。它把任务表打出来了，注册表就是通的。

macOS 上实时查看器需要 `.venv/bin/mjpython` 而不是 `python`；没有它运行会转为无头模式并明确
告知。

---

## 2. 训练三个任务

三次独立的训练。它们只共享一台机器人，别无其它。

```bash
python scripts/train.py --task jumper.tripod --num_envs 4096
python scripts/train.py --task jumper.jump   --num_envs 4096
python scripts/train.py --task jumper.dance  --num_envs 4096
```

日志落在 `logs/<model>/<task>/<timestamp>/`，检查点是 `model_<N>.pt`。`--resume` 从最新的那个
继续。`--no-tensorboard` 关掉单次运行的面板，不用去改 `.env`。

**`--num_envs` 要按你的后端选，不要照抄本页**。CUDA 加 warp 后端下越大越好，直到显存不答应为止。
原生 CPU 后端每个环境都有自己的一份 `MjModel` 和 `MjData`，内存随 `num_envs` 增长，所以一个按
GPU 定的数会把工作站压垮 —— 从小开始，再往上加。（线程数已经不跟着它走了：上限是 8，见
`--cpu_threads`。）

每个训练要跑多久，取决于你的机器和你的奖励调参，本文档不假装知道。它能告诉你的是**该盯什么**：

- `jumper.tripod` —— 速度跟踪任务，要盯的是指令跟踪奖励。它同时是其它步态的**对照组**：
  `flat`、`tripod`、`ripple`、`tetrapod` 之间只允许在步态、以及步态所隐含的速度上限上有差别，
  **其余一律不许不同**（`tests/test_task_parity.py` 会在任何其它差异上失败）。
- `jumper.jump` —— 参考引导。它的动作是叠加在一段录制运动
  （`tasks/jumper/jump/ref/high_jump_flat.npz`）上的**残差**，所以策略学的是修正量，不是轨迹。
  跳不离地通常意味着参考没被跟上，而不是奖励写错了。
- `jumper.dance` —— 以一段录制编舞为评分依据。编舞和脸部动画**已提交进仓库**
  （`tasks/jumper/dance/media/demo.{npz,mp4}`），所以干净克隆就能跑。音乐没有提交；训练用不到它，
  只有导出表演视频时才需要。第一次构建环境时会把
  片段转换好并缓存在 `media/.cache/`，缓存键是**源文件的内容** —— 换掉 `.npz` 下次就重新转换，
  没有会忘记的步骤，也没有陈旧缓存让你训练昨天的舞。
  参见 [`USAGE.md`](USAGE.md#a-task-that-needs-material-jumperdance)。

看其中一个跑起来：

```bash
python scripts/play.py --task jumper.dance
```

不给 `--checkpoint` 时，取 `logs/<model>/<task>/` 下最新的那个。

---

## 3. 导出每个策略

**一个导出目录就是一个策略**。这一步是契约被构建出来的地方，也是那些校验决定这个策略到底能不能
跑的地方。

```bash
python scripts/export.py --task jumper.tripod --checkpoint logs/<model>/jumper.tripod/<run>/model_<N>.pt
python scripts/export.py --task jumper.jump   --checkpoint logs/<model>/jumper.jump/<run>/model_<N>.pt
python scripts/export.py --task jumper.dance  --checkpoint logs/<model>/jumper.dance/<run>/model_<N>.pt
```

每次写出 `tasks/<task>/out/<date-time>/`，里面是 `actor.onnx`、`layout.json`、一份
`README.md` 和检查点的副本。重复导出是**新增**目录而不是替换；它来自哪个检查点记在里面，
bundler 会把它读回 `bundle.json` —— manifest 在 `policy` 下直接写导出目录本身。

`jumper.jump` 和 `jumper.dance` 各多写一个文件 —— `<name>.trajectory.json`，也就是它们契约的
`reference` 块所指名的那段录制（`jumper.dance` 还会渲染它的 `media/`，除非加 `--no-video`）。从此它就跟着策略走，而 bundler 宁可拒绝一个缺了录制的模式，
也不会发出一个跑不起来的策略。

**输出里要读这几行**：

| 行 | 含义 |
|---|---|
| `deployable terms` | 每个观测项都是部署端控制器建得出来的 |
| `every observed term is measurable` | 没有观测任何机器人测不到的量。一个拿 `base_lin_vel` 训出来的 actor 会在这里失败 —— 这是对的，因为机器人上没有传感器报告它 |
| ONNX 与 torch 的差值 | 导出的计算图与它来自的网络一致 |
| *（没有这一行）* `joint_limits` | 总会写出，导出本身就拒绝有缺口或上下限颠倒的情况。2026-09-20 之前的导出没有它，等板子的 controller 放进 bundle 之后（第 8 步），bundler 会拒绝那种导出 |

一次失败的导出，好过一次发出去的导出。这是整条路的精神所在。

---

## 4. 说清楚控制的含义

部署后控制器的手柄行为来自**两个文件，它们的和就是全部**：

| | 文件 | 说的是 |
|---|---|---|
| 模式之间 | `deploy/manifests.json` → `[[fsm.button]]` | 哪个控制**切换进**某个模式，在哪个设备上、用哪个手势、能从哪些模式按 |
| 模式内部 | `tasks/<task>/controls.yaml` | 该模式驱动时，摇杆和按键**做什么** |

两者都不许说对方那一半：manifest 不绑定摇杆到轴，任务不指名模式。**没有第三个地方**。

两者都从同一份封闭清单里选词：

```bash
python -m controller --vocabulary
```

十个按钮、四个方向键、六个轴、键盘上的键和它们的三个修饰键、九种手势。一个**控制**是一个手柄
按钮或一个键，加上一个手势 —— `rise`（按下去那一 tick）、`fall`（弹回来那一 tick）、`hold`
（按住的每一 tick）、`toggle`（按下直到再按一次），或者从 `single` 到 `quintuple` 的连击次数。
所以"按 A 预备、松 A 起跳"是**一个按钮绑了两次**，不是一个绑定干两件事。手柄和键盘是两条路：
一个键绑的是它要做的事，从不是一个手柄按钮。

**写这两个文件都用 `controls` skill**。它掌管那本字典，并且会拒绝不在里面的词。不要凭记忆重抄
那份清单。

本 bundle 在 `deploy/manifests.json` 里的条目是这样：

```json
"jumper": {
  "fsm": "jumper.controller.toml",
  "modes": {
    "locomotion": { "task": "jumper.tripod", "policy": "tasks/jumper/tripod/out/<dir>" },
    "jump":       { "task": "jumper.jump",   "policy": "tasks/jumper/jump/out/<dir>",
                    "pad": { "button": "LB", "on": "toggle" } },
    "dance":      { "task": "jumper.dance",  "policy": "tasks/jumper/dance/out/<dir>",
                    "pad": { "button": "RB", "on": "toggle" } }
  },
  "buttons": [ { "name": "jump_go", "pad": { "button": "A", "on": "fall" } } ]
}
```

这里按 2026-09-29 起 manifest 的写法来写：一个模式的开关每个设备一个，`pad` 和 `keys`，各有自己的
手势和能从哪些模式按（`from`）。写这一页的时候，模式自己带着 `button`、`key`、`on` 和 `with`；构建
出来的东西是一样的。

仓库自己的 `jumper` 条目曾经就是这一条，只是 `locomotion` 这一格换成了 `jumper.posture`：它走同一种
三足步态，额外接受一个姿态命令 —— 两个任务都用 W A S D 和 J L 驱动，姿态任务再加上 I K、U O、N M
和 H ;。在一格里换一个任务，只改两行。它后来挪了 jump 和 dance 的按键，加了爪子、另外三支舞
和四个固定动作（见开头的说明），它的级联也点名了所有这些模式，所以像这里这样的三模式 bundle 需要一份
自己的级联：`button:jump`、`button:dance`，再加上 `always`。

这里有三件事值得理解：

**`locomotion` 没有开关**。只有一个模式时，级联最后那条 `always` 规则就是你到达它的方式。
第二个模式必须有绑定，否则它不可达。

**这些路径是本机的**。导出目录带时间戳且被 git 忽略（随仓库交付的 manifest 所指的那几个是
手动加进 git 的），所以 manifest 指的是此处存在的目录。
换一台机器就先导出，把它打印出来的目录填进 `policy`。路径不存在时 bundler 会把这话原样说给你听。

**`jump_go` 在 `buttons` 里，不在 `modes` 里**。一个卡在某个瞬间的录制运动需要的是一个**事件**，
不是一次模式切换 —— `jumper.jump` 的蹬地窗口只有 40 ms 宽。它就是一条普通的 `[[fsm.button]]`；
任务的 `reference` 块里写着 `go_event: "jump_go"`，控制器在这个名字上启动运动的时钟。没有
`go_event` 的话，运动就从进入模式起**按定时器**跑 —— 那够在台架上看一眼，不是一个人能用的东西。
（`jumper.jump` 后来去掉了 `go_event`，改为 `starts_on_entry` —— 它的策略训练时都是从 go 之后起步，
从没在 go 帧上站着等过 —— 所以它的导出不再读 `jump_go`，还声明着它的 manifest 会被拒绝。这套机制
本身没变，留给确实要由人来挑时刻的运动。）

manifest 旁边的 `deploy/jumper.controller.toml` 装着级联和安全限位 —— 规则顺序、倾角上限、
各种超时。bundler 把两者合成起来。

**你不需要靠读来检查这两半**。`Bundle::open` 会拒绝冲突，所以 `scripts/deploy.py` 每次构建都会
拒绝：

```
'slow' switches on the pad's B, which mode 'walk''s controls use too: one press
would do both. Pick another control, or say with `from` that the switch is not
pressed in 'walk'
```

构建打印出了 bundle，就说明两半没有撞车。

---

## 5. 组装 bundle

```bash
python scripts/deploy.py
```

不带参数时，它构建 `deploy/manifests.json` 定义的那唯一一个 bundle（有多个时用
`--manifest jumper` 指名），并且在打包任何东西之前先把板子那一半做出来：在 docker 里交叉编译
控制器（第 8 步讲它是什么），并在 docker 里转换每一个没有最新 `.rknn` 的策略（第 7 步）。
第一次运行要构建两个镜像，会花一些时间；之后交叉编译是增量的，每个策略只转换一次。

```
[deploy] manifest jumper: dance <- jumper.dance, jump <- jumper.jump, locomotion <- jumper.tripod
[deploy] cross-building the board's controller (deploy/fsm/docker-build.sh)
[deploy] cross-building play's controller for Windows (deploy/fsm/docker-build.sh)
[deploy] cross-building play's controller for macOS (deploy/fsm/docker-build.sh)
[deploy] converting dance for the NPU (tasks/jumper/dance/out/<dir>)
[deploy] converting jump for the NPU (tasks/jumper/jump/out/<dir>)
[deploy] converting locomotion for the NPU (tasks/jumper/tripod/out/<dir>)
[deploy] bundled <commit> -> out/bundle_<timestamp>/jumper
           <size> MB  <n> files  -> jumper.app <size> MB
           board  rknn models, runtime/board/controller
           web    onnx models, runtime/web/controller.wasm
           mjlab  onnx models, runtime/mjlab/linux-x86_64/controller.so, runtime/mjlab/win-amd64/controller.pyd, runtime/mjlab/macosx-universal2/controller.so
             opens as web, mjlab, board
             dance            obs 209 -> act 22
             jump             obs 167 -> act 20
             locomotion       obs 411 -> act 20
             jump_go        A on fall
             dance          RB on toggle
             jump           LB on toggle
             22 joints on the wire, 3 mode(s), 6 rule(s)
             note: dance: its contract has no `controller` block, …
             note: jump: its contract has no `controller` block, …
             note: reference: 24 frames, …
[deploy] next: the Chinese manual. The bundle-manual skill writes it and adds it with
           python scripts/deploy.py --translate out/bundle_<timestamp>/jumper --language zh --manual <file>
```

bundle 写成**一个目录，外加旁边一个 `.app`**——换了个名字的 zip，任何 zip 工具都能直接打开。
目录是本机读的；归档是人拿去浏览器文件选择框、上传表单或机器人上的那一个。归档的内容放在它的根部，
因为消费者就是在那里找 `bundle.json`。如今完整的 `jumper` bundle —— 五个模式，已转换、已交叉
编译 —— 就是一个 10.9 MB 的 `.app`。

做完这些之后，如果 app 仍然缺板端控制器、缺某个 `.rknn` 或缺 reference 向量，就会被拒绝，
不会写出来。在没有 docker 的机器上，或者 bundle 只给浏览器或 `play` 加载时，
`--allow-incomplete` 会跳过这两个 docker 步骤，有什么用什么，并在 note 里写明缺了什么。

**要读那些 note**。它们是构建注意到、但没有拒绝的事：

- `its contract has no controller block, so a consumer falls back to its own key
  bindings` —— 该模式没带 `controls.yaml`。对一个没人驾驶的模式无害（`dance` 和 `jump` 都不
  被驾驶）；如果它本该有，就重新导出。
- `reference: N frames, M of them inferences` —— 用于比对的录制向量已写入。第 10 步会在板子上
  重放它们。

每个宿主读的都是 bundle 里唯一的那份 `controller.toml`、那一套契约和那份 `reference.json`，
所以两个宿主之间没有任何可以不一致的东西。

bundle 还带着 `manual.en.json`：每个按键和手柄按钮在每个模式下做什么，由控制器对自己手柄和按键的
描述生成。现在就给它配上中文译本 —— [`bundle-manual`](../.claude/skills/bundle-manual/SKILL.md)
skill 会写出 `manual.zh.json`，再用
`python scripts/deploy.py --translate out/bundle_<timestamp>/jumper --language zh --manual <file>`
把它加进去；这条命令会对照英文检查它，并重新打包 `.app`。

对于不值得写进 manifest 的一次性构建，`--mode name=<dir> --fsm <file>` 仍然可用。超过一个模式就
必须给 `--fsm`，因为优先级顺序是一个决定，任何默认值都会是某个人的无声选择。

---

## 6. 驱动它 —— 两个不需要硬件的宿主

### 在 `play` 里

```bash
python scripts/play.py --app out/bundle_<timestamp>/jumper.app
```

此时跑的是 **app 的**控制器 —— 它为本机平台带的那个扩展 —— 而不是某个任务自己的策略：和机器人
上跑的是同一份代码、包含全部模式，只是放在 mjlab 的物理里。app 自己的按钮和按键负责切模式（在
viewer 里，每个键的按下和松开都由 `mjrl.viewer.keys` 报上来），每个模式经由自己的控制定义读取按键和
手柄，执行器跟踪的是控制器给出的关节目标和增益 —— 所以切模式时的斜坡和真机上一样，任务的部署钩子
也会像在真机上那样驱动关节。

### 在浏览器里

把 `jumper.app` 上传到一个接受训练平台 bundle 的仿真场。它自带 `runtime/web/controller.wasm`，
所以跑的就是这次构建出来的控制器。

然后：机器人站在 `locomotion` 上。**`RB`** 切进 `dance`。**`LB`** 切进 `jump`，**按下再松开
`A`** 触发起跳 —— `go` 绑在 `fall` 也就是松手上，不是按下。两段录制运动播完都会自行释放，级联的
`always` 规则接住机器人，所以两者都不需要操作者手动退出来。（如今交付的 bundle 里：`A` 或 Space
是 jump，Menu + 十字键或 Ctrl + 1 2 3 4 是四支舞，单按十字键或 1 2 3 4 是四个固定动作，`LB` / `RB`
或 `V` / `B` 是爪子，单独按下再松开 Menu 或 Ctrl 提前离开一支舞或一个固定动作。）

有两件浏览器做不到的事，在你断定"坏了"之前值得先知道：

- **混合速率的 bundle 需要跟得上的宿主**。`jumper.jump` 是 200 Hz 契约，另外两个是 50 Hz。
  一个把控制器 tick 得比最快模式还慢的宿主，会把每个动作保持得过久，**同时**让录制运动的时钟推进
  得过慢 —— 跳跃会以慢动作播放。这不是错误，也没有任何地方报告它，所以先查宿主的控制速率，再去
  怪策略。
- **它从不跑 `.rknn`**。浏览器按设计取每个模式的 ONNX；NPU 对策略造成的影响在板子上测，见
  第 10 步。

---

## 7. 为 NPU 转换

只给 board 用，而且第 5 步已经做过了：manifest 里每个 `actor.rknn` 缺失、或由另一个
`actor.onnx` 转出来的导出，它都会转换。手动的话，一次一个导出：

```bash
bash deploy/convert/docker-convert.sh --bundle tasks/jumper/tripod/out/<dir>
bash deploy/convert/docker-convert.sh --bundle tasks/jumper/jump/out/<dir>
bash deploy/convert/docker-convert.sh --bundle tasks/jumper/dance/out/<dir>
```

Docker 在任何宿主上都能用，而且在非 Linux 上是**唯一**选择 —— rknn-toolkit2 没有发布 macOS 或
Windows 的 wheel。Linux 上可以改用 `deploy/convert/setup.sh` 建虚拟环境。

转换器会把结果与 onnxruntime 比对，并且**拒绝写出对不上的 `.rknn`**。fp16 大约落在 1.1e-3；
int8 实测 2.7e-1，相当于 3.9° 的关节误差，默认容差就卡在两者之间，所以 `--quantize` 不可能被
误用。它把 `actor.rknn` 写在导出目录旁边，bundler 就在那里找它；旁边还有一份
`actor.rknn.json`，记着它来自的那个 ONNX 的摘要，bundler 靠它判断两者是否仍然配套。

---

## 8. 交叉编译控制器

第 5 步每次打包之前都会跑这一步。手动的话：

```bash
bash deploy/fsm/docker-build.sh          # aarch64 release -> out/deploy/controller-aarch64
bash deploy/fsm/docker-build.sh test     # 宿主侧测试套件
```

是交叉编译而不是模拟，基于 `ubuntu:22.04`，因为板子的 glibc 是一个下限。这个 crate **就是**
控制器 —— 没有另一个程序去链接它。

每次都跑，而不是只在没有的时候跑：bundler 会把 `out/deploy/controller-aarch64` 里现有的那个
拷进 app，并记在当前源码的 commit 名下，而二进制本身说不出它来自哪份源码。上周留下的构建会被当成
本周的发出去。源码没变时，它只是容器自己的 target 卷里的一次增量 cargo 构建。

---

## 9. 碰板子之前先检查板子

```bash
python3 .claude/skills/deploy/scripts/check_board.py --host <user>@<board> \
    --bundle tasks/jumper/tripod/out/<dir>
```

`--host` 是板子的 ssh 目标；它没有默认值。
只读：不在板上运行任何二进制，也不发布任何东西。它会校验 glibc、librknnrt 版本、CycloneDDS 的
soname，以及 IDL 类型是否与本仓库生成的一致。退出码 0 全过，1 有差异，2 连不上。

它也能把板子的关节名和 bundle 的对一遍 —— **按名字比**，这是最要紧的那一项 —— 但前提是你告诉
它去哪里读：`--config <板子上的某个文件>`。没有默认值，因为除非有人放了一个，板子上并没有这种
文件。

> 注意它接收的是一个**导出目录**，不是 bundle。

---

## 10. 真机

把 bundle 拷到机器人上 —— `jumper/`，或者拷 `jumper.app` 过去在那里解压；它装着全部东西，包括
二进制。然后在它里面**按这个顺序**执行三条命令。前两条都不碰总线。

```bash
./runtime/board/controller --bundle . --dry-run
```

打印它将要用来驱动的那些数字是从哪里来的：摇杆的缩放与符号、DDS 域、QoS 文件，以及哪些模式跑在
桩上。什么都不发布。

```bash
./runtime/board/controller --bundle . --check-reference
```

重放 `reference.json` —— 构建 bundle 时录下的那些帧 —— 并报告这台宿主与构建它的那台机器之间的差
异。`observation` 和 `target` 应该是 `0`，而在板子上它们确实是：这个 tripod/jump/dance bundle
两项都是 `0.000e0`。

这就是整条路存在的意义所在的那次测量。它曾经显示 `joint_torque` 差了大约 2 N·m，后来查明是控制器
在用指令 PD 反推力矩，而不是读舵机真正报告的值。没有别的东西会发现这件事。

```bash
./runtime/board/controller --bundle . --machine /etc/mjrl/machine.toml
```

这一条会驱动。上面全部都是彩排。

---

## 出问题的时候

完整的表在 [`.claude/skills/deploy/SKILL.md`](../.claude/skills/deploy/SKILL.md)。以下是这个
bundle 上最容易绊住人的几条：

| 症状 | 可能的原因 |
|---|---|
| 模式进去了，机器人就是站着不动 | 录制运动的 `go` 从没触发。模式进去了、策略每 tick 都在推理、面板也这么显示 —— 只有运动自己的时钟能区分这两者 |
| 录制运动以慢动作播放 | 宿主 tick 控制器的速率低于该模式的 `control_hz`。运动的时钟是按控制步数走的 |
| 命令它站住，它却在原地踏步 | 步态时钟自由运行了；该策略训练时它在 `params.command_threshold` 以下是被门控关掉的 |
| 从第五个关节起动错了关节 | 关节顺序按下标配对，而不是按名字 |
| 摇杆毫无反应 | 用了字典里没有的名字。那是一份封闭清单，一个能解析通过的拼写错误就是一个什么都不驱动的绑定 |
| 机器人倒着走，而其它一切看起来都正常 | 符号错了。下游没有任何东西能抓住它，一只手放在手柄上能 |
| 按一下做了两件事 | 同一个控制出现在两个文件里。bundler 会把两处都指出来 —— 改其中一个，不要为了让它通过而放宽任何检查 |
| bundler 拒绝一个参考引导的模式 | 它的 `<name>.trajectory.json` 不在导出目录里 |
| 板子的 controller 放进去之后，bundler 拒绝这个 bundle | 某个契约里没有 `joint_limits`。重新导出 |
| 话题一直是空的，控制器保持位置不动 | IDL 或 QoS 不匹配。DDS 从来没有配对上端点，而这不是一个错误 |
