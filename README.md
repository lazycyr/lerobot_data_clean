<div align="center">

# dclean

**LeRobot Dataset v3.0 质量检查与清洗工具**

只读检查 · 四类异常检测 · 手动删集 · 新副本重编号 · 统计量重算

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![LeRobot](https://img.shields.io/badge/LeRobot-v3.0-green)
![Tests](https://img.shields.io/badge/tests-48%20passed-brightgreen)

[快速开始](#快速开始) · [检测能力](#检测能力) · [数据清洗](#数据清洗) · [详细检测原理](README_DETECTION.md)

</div>

---

## 简介

`dclean` 用于检查并清理 **LeRobot Dataset v3.0** 格式的机器人数据集，覆盖缺文件、索引错位、时间戳漂移、关节跳变、轨迹离群、视频黑帧/冻结/模糊等问题。

项目最初面向 `Part_Sorting_300/selected`（walkerS2，300 episodes × 636 frames，30 fps，4 cameras），数据读取由 `meta/info.json` 和 `meta/episodes` 驱动，也适用于其他 v3.0 数据集。

- `dclean check` **只读**，只生成报告。
- `dclean clean` 生成**新的数据集副本**，不修改源数据。
- 报告使用 Markdown / CSV / JSON，可直接在 VS Code 查看。

---

## 快速开始

环境要求：Python ≥ 3.11，系统安装 `ffprobe` 和 `ffmpeg`。

```bash
cd data_clean

uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
```

检查数据集：

```bash
.venv/bin/dclean check \
  --dataset Part_Sorting_300/selected \
  --out reports/part_sorting_check \
  --config configs/part_sorting.toml
```

快速检查前 10 集：

```bash
.venv/bin/dclean check \
  --dataset Part_Sorting_300/selected \
  --out /tmp/dclean_smoke \
  --config configs/part_sorting.toml \
  --episodes 0-9
```

常用选项：

| 选项 | 说明 |
|---|---|
| `--episodes` | 只检查指定 episode，如 `0-9,15,20-22` |
| `--config` | 阈值配置，建议显式传入 |
| `--no-video` | 跳过所有视频检查 |
| `--no-video-quality` | 跳过全帧画面质量检查（更快） |
| `--full-report` | 额外输出全量明细表、校准 JSON 和 JSON 副本 |
| `--overwrite` | 允许写入非空输出目录 |

---

## 检测能力

| 类别 | 主要检查 |
|---|---|
| **完整性检查** | 文件和目录完整性、episode 引用、data 行数、`episode_index`/`index` 连续性、文件编号、孤儿文件 |
| **时序对齐** | 时间戳单调性、`dt ≈ 1/fps`、视频段边界与时长、MP4 packet PTS 与 action 逐帧对齐 |
| **轨迹异常** | NaN/Inf、恒定通道、关节跳变、action-state 残差、相对数据集中位轨迹的持续偏移 |
| **视频异常** | codec/分辨率/fps/帧数、黑帧、暗帧、冻结、模糊、过曝 |

补充说明：

- 视频质量默认**全帧检查**，不按时间抽样。
- 轨迹偏移使用数据集中位轨迹 + MAD 稳健尺度，能发现“整段动作走偏”，而不只是单帧尖峰。
- 检查只使用 action/state 的 dims 0–19；`state[20:48]` 物体位姿被视为冗余信息，不参与异常判断。
- `episode_quality.csv` 只输出有问题的 episode，并给出具体帧范围。

详细算法、公式和阈值见 [README_DETECTION.md](README_DETECTION.md)。

---

## 输出

```text
reports/part_sorting_check/
├── summary.md                # 四类问题汇总，优先看这个
└── episode_quality.csv       # 有问题的 episode
```

默认只写这两个主要文件。需要全量明细和阈值校准数据时加 `--full-report`，会额外输出：

```text
episode_quality.json
av_sync.csv/json
trajectory_scores.csv/json
trajectory_calibration.json
video_quality.csv/json
video_quality_calibration.json
```

`summary.md` 按四类输出；某一类没有 fail/warn 时写 **“未检出异常”**。

`reports/` 只保留源数据集检查和清洗结果检查两份报告；清洗后的完整数据集位于 `Part_Sorting_300/clean`。

---

## 数据清洗

`--episodes` 是要删除的 episode 列表。清洗写出新副本，并完成：

- data 过滤与重新编号：`episode_index`、全局 `index`、`task_index` 连续重映射；
- 每个保留 episode 写一个 data parquet；
- 视频按保留帧区间复制或重编码，默认 `libx264`，可用 `--vcodec libsvtav1`；
- 重建 `meta/episodes`、`info.json`、`tasks.parquet`；
- 低维 stats 精确重算，图像 stats 逐集复用并全局加权聚合；
- 修正重编号后过期的 `stats/index`、`stats/episode_index`、`stats/task_index`。

```bash
# 先看删除计划，不写文件
.venv/bin/dclean clean \
  --dataset Part_Sorting_300/selected \
  --out Part_Sorting_300/clean \
  --episodes 32,88,91,115,142,194,199,221,267 \
  --check-report reports/part_sorting_check/episode_quality.csv \
  --dry-run

# 确认后生成新数据集
.venv/bin/dclean clean \
  --dataset Part_Sorting_300/selected \
  --out Part_Sorting_300/clean \
  --episodes 32,88,91,115,142,194,199,221,267 \
  --check-report reports/part_sorting_check/episode_quality.csv

# 复检清洗结果（报告目录已存在时加 --overwrite）
.venv/bin/dclean check \
  --dataset Part_Sorting_300/clean \
  --out reports/part_sorting_clean_check \
  --config configs/part_sorting.toml \
  --no-video-quality \
  --overwrite
```

清洗输出包含 `cleaning_report.md`、`deleted_episodes.csv` 和 `index_mapping.csv`。

---

## 配置

配置文件为 `configs/part_sorting.toml`，每个参数都有中文注释。换数据集时必须重新校准：

- `fps.expected`：与 `meta/info.json` 的 fps 一致；
- `l2.joint_dims`：参与连续检查的关节维度；
- `l2.step_abs_floor`、`step_z`：瞬时冲击阈值，与关节单位有关；
- `l2.trajectory_z_warn`、`trajectory_z_fail`：运行 `dclean check --full-report` 后参考 `trajectory_calibration.json` 选择；
- `video_quality.blur_ratio_threshold`：运行 `dclean check --full-report` 后参考 `video_quality_calibration.json` 按相机调整。

---

## 示例结果

`Part_Sorting_300/selected` 实际运行结果：

| 项目 | 数值 |
|---|---:|
| 源 episodes / 帧数 | 300 / 190,800 |
| 删除 episodes | 9（轨迹偏移 fail） |
| 保留 episodes / 帧数 | 291 / 185,076 |
| 清洗后检查 | 291 clean / 0 warn / 0 fail |
| action/video PTS 对齐 | 1,164 行，0 mismatch |
| 低维 stats 对比直接计算 | 最大误差 0 |

删除的 9 集：`32, 88, 91, 115, 142, 194, 199, 221, 267`。

---

## 测试与限制

```bash
.venv/bin/pytest -q
# 48 passed
```

限制：

- 仅支持 LeRobot v3.0；视频清洗假设 CFR，不做重采样和音频处理。
- 图像 stats 复用逐集值并加权聚合，不是逐像素重算。
- 清洗保留源视频 `chunk/file` 编号；若某视频文件的所有 episode 都被删除，可能留下编号空洞。
- 不检测物体遮挡、标定错误或语义任务失败。
- 阈值与机器人、场景相关，不要直接套用其他数据集的配置。
