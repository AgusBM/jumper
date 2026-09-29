<!-- tracks: README.md @ sha256:fd116cb354cdbe04 -->

# deploy —— 从一个 checkpoint 到运行它的那一端

```
model_*.pt ──scripts/export.py──▶ actor.onnx ──convert/──▶ actor.rknn
                                  layout.json ──┐             │
                                                ▼             ▼
                          scripts/deploy.py [--manifest <name>]
                                                │
                                                ▼
                        controller   观测 → NPU → 关节目标 → dds/
```

一个 **export 目录**是一个策略。一个 **bundle** 是每个宿主都加载的东西：FSM 配置、每个 mode
一个策略（每种宿主要跑的模型格式各一份）、每个宿主各自的 controller 构建，以及一份手柄和键盘的
操作说明 —— 一个目录，外加它的一个 `.app`，板子、浏览器和 `play` 各自从里面取自己那一份。
[`BUNDLE.md`](BUNDLE.md) 逐个文件写明了它的格式，以及加载它的宿主必须做到什么。

参考轨迹引导的策略会多带一个文件。`jumper.jump` 的动作是在一段录制运动上的残差，`jumper.dance`
则是对着一段录制运动打分的，所以这段录制会跟着策略一起走，以 `<name>.trajectory.json` 的形
式放在 contract 旁边，contract 在一个 `reference` 块里点它的名；打包器会把它复制进来，并拒
绝那种 contract 点了名、文件却不在的 mode。没有它，每一种宿主在打开 bundle 时就拒绝这个
mode 并点出那个文件名，而不是先加载成功、到第一个 tick 才拒绝。读它的是
`fsm/src/trajectory.rs`。

| | 是什么 | 在哪里跑 |
|---|---|---|
| [`convert/`](convert/README.md) | 给 RK3576 NPU 做 ONNX → RKNN | 一台 PC（Docker，任何宿主） |
| [`dds/`](dds/README.md) | 机器人的 wire 接口：IDL、QoS、topic | 构建输入 |
| [`fsm/`](fsm/README.md) | controller：观测、动作解码、状态机 | 板子、浏览器，以及 `play` |

`fsm/` 是一个 crate，三个宿主。`controller` 就是机器人的整个程序 —— 没有另一个把它当库链接
进去的 controller，以前是有的。

每个目录都有自己的 README 讲细节；这一份是地图，以及不属于其中任何一个的那部分。

## 整条路径所依赖的那条原则

**机器人用到的每一个数字都来自策略自己的 `layout.json`，关节按名字配对**。下游不再重新声明
任何增益、任何缩放、任何关节顺序。

这条原则被写在这里的时间，远早于它对所有东西都成立的时间。有两个数字一直在悄悄违反它，都待
在一个二进制文件旁边的配置文件里，那里没有任何东西会拿它们去对照它们所描述的那台机器人：

- **关节位置钳位**，策略和电机之间的最后一道关卡，对全部 22 个关节都是 `[-3.30, 3.30]` ——
  这是 crate 单元测试里的占位值。22 个里有 20 个的两端限位都落在它里面，所以这道钳位每个
  tick 都在跑，却不可能触发。
- **摇杆缩放**：前进 0.80 m/s，而这个策略的命令范围是 0.50。推到底所要求的，是课程曾经下达
  过的任何指令的 1.6×，这看起来像是一个跑不快的策略，而不是一个被问了它从没见过的问题的策
  略。

两者现在都来自 contract（`joint_limits`、`command_ranges`）。教训是通用的那条：一个仿真器本
可以量出来的数字，不该待在宿主的配置里；而“它一直就是这个值”不是有人选过它的证据。

这不是为了整洁。在这台机器人上，wire 承载 22 个关节，而策略取的是它训练时的那个子集：运动
任务和 `jumper.jump` 观测并驱动 20 个，`jumper.dance` 是全部 22 个。运动策略不去碰的那两个，位
于 wire 下标 **4 和 9**，在中间：

```
obs_joint_order (20) → wire [0,1,2,3, 5,6,7,8, 10,11,…,21]
                                    ↑ 4      ↑ 9   这两个是手指
```

把它们按下标而不是按名字配对，第五个关节往后就全是错的，在一台真机上，而且什么都不会报出
来。已对着实机验证过：板子的 `[robot] joint_names` 和 contract 的 `wire_joint_order` 是同样
22 个名字、同样的顺序，并且 contract 里的每一个名字都能解析到。

## 检查了什么，在哪里检查

每一跳都拿前一跳来核对，因为这条路径上的每一种失败都是静默的 —— 是一个跑得起来但行为错误的
策略，不是一个崩掉的策略。

| 跳 | 检查 | 实测 |
|---|---|---|
| torch → ONNX | `scripts/export.py::_validate_onnx` | 3e-7 |
| ONNX → RKNN | `convert/onnx2rknn.py`，simulator 对 onnxruntime | 1.1e-3（fp16 下限） |
| contract → controller | `fsm/`：layout、观测 layout、FFI 结构体布局、QoS | 单元测试 |
| controller → robot | **板子**：运行时版本、IDL、关节顺序 | 见下 |
| host → host | 在每个宿主上回放 `reference.json`：`--check-reference` | controller 上是 0，含 `joint_torque` |

最后一行以前的结尾是“`joint_torque` 2.1 N·m，设计如此”，而那 2.1 N·m 并不是什么设计。舵机
是上报力矩的 —— `MotorControl_State` 里带着它，`dds.rs::take_state` 一直都在把它读进
`state.tau` —— 而“它不上报”这个说法是从 C++ controller 那里继承来的，那个说法对这台机器人是
错的。把这一项声明为重建出来的，就让 controller 去重新推导一个它本来就有的数字，而那个差距
是回放唯一对不上的东西。这一项现在是 `Measured`，回放把录下来的值喂给它，它像其他每一项一样
精确吻合。

此外，当 actor 观测了机器人测不出来的东西（`base_lin_vel` —— 没有状态估计器），或者观测了
机上 controller 构造不出来的项时，`scripts/export.py` 会干脆拒绝导出。宁可让导出失败，也不
要发出一个跑不起来的策略。

## 对着板子验证过的

经 ssh，只读：

| | 板子 | 本仓库 |
|---|---|---|
| OS / glibc | Ubuntu 22.04.5，glibc 2.35 | `fsm/Dockerfile` 在 22.04 上构建 |
| 内核 | 6.1.99-rt36（PREEMPT_RT） | — |
| librknnrt | 2.3.2 | `fsm/vendor/rknpu2/fetch.sh` 固定拉取 2.3.2，toolkit 2.3.2 |
| CycloneDDS | 11.0.1，`libddsc.so.11` | 容器构建出同样的 soname |
| libmbus | 2.4.0 | — |
| IDL 头文件 | 4 个，来自 libmbus | 与当时 `fsm/build.rs` 生成的完全一致；2026-09-29 刷新到 mbus `d9875988f065` 后 `MotorControl::State` 由 800 字节变为 832 字节，libmbus 2.4.0 早于这次变化——尚未在板上复查（见 [`dds/README.md`](dds/README.md)） |
| `joint_names` | 22 | 与 `wire_joint_order` 顺序相同 |

`.claude/skills/deploy/scripts/check_board.py` 会把这些全部重跑一遍。

**做那些检查的时候，有一个 controller 正作为服务在跑，而且是活跃的**。上面的一切之所以都是
只读的，原因就在这里。在有 controller 在跑的时候发布电机指令，意味着两个 controller 在驱动
同一批关节 —— 这也是 `controller` 有 `--dry-run` 的原因。

## 未决

- **`rknn_run` 在真实硬件上还没有验证过**。NPU 离开板子就没法跑，所以这部分算术只通过
  toolkit 的 simulator 检查过。
- **控制回路从来没有对着一条总线跑过**。`fsm/src/bin/controller.rs` 是有的，`--dry-run` /
  `--check-reference` 会走完打开总线之前的每一步，但这个回路的正确性几乎全在时序上 ——
  1 kHz 的发布对上每个 mode 各自的推理频率（`jumper.posture` 的行走和跳跃是 200 Hz，爪子
  的两个模式、各支舞和各个固定动作是 50）、ZOH 加上一个与频率无关的 EMA，以及 mode 切换时的斜坡 —— 这些
  离开板子一样都测不了。它也完全没有尝试实时调度：没有 `SCHED_FIFO`，没有 `mlockall`，
  没有抖动统计。
- **没有任何东西把一个 DDS domain 的两端绑在一起**。`controller` 两端都默认取 0，板子就是这
  么配的，`control-pod-svc` 也是在这个 domain 上发布的；它们今天是一致的，但不会自己保持一
  致。这一条以前带的那句警告说的是 `robot_control` 迁到了 domain 2 —— 那已经不再是这个
  controller 的问题了，因为它直接读手柄，而这提醒了一件事：一句过期的警告比没有警告更糟。
- **级联里没有夹爪状态**。`gripper_active` 在级联的词表里，每个宿主都传 `false`，用到它的
  配置会因此被拒绝：这个状态原本来自 `robot_control`，而这个控制器改读手柄了。夹爪走的是
  另一条路，由它自己的任务驱动——`jumper.five_foot` 的部署钩子按爪子那一侧的扳机（或 Space）
  开合，并在按住一个方向键（或 Shift、Alt、Ctrl）期间把手臂伸到预设姿态：这些都是该任务
  `controls.yaml` 按名字留给自己的控件（`task:`）——没有哪条规则看得到它们。
