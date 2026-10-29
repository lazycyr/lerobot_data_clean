# dclean 四类检测逻辑详解

当前检查分四类，`summary.md` 固定按这四类输出；某类没有 fail/warn 就写 **“未检出异常”**，info 级说明单独列在 **“备注（info）”**。

> 本文只描述只读检查。删除 episode、重编号、重写视频和重算统计量的清洗流程见主 README 的 **数据清洗** 一节；清洗始终写入新数据集副本，不修改源数据。
>
> 默认检查只输出 `summary.md` 和 `episode_quality.csv`。本文提到的 `av_sync.csv`、`trajectory_scores.csv`、`video_quality.csv` 等明细表需要加 `--full-report`。

| 类别 | 主要数据源 | 目标 |
|---|---|---|
| 完整性检查 | `meta/info.json`、`meta/episodes`、`data`、`videos` | 数据集结构、文件、索引、行数是否自洽 |
| 时序对齐 | data `timestamp`/`frame_index`、episode 视频 `from/to`、MP4 packet PTS | 动作与视频的时间轴是否真正对齐 |
| 轨迹异常 | action/state dims 0–19 | 数值异常、瞬时冲击、持续轨迹偏移 |
| 视频异常 | 视频元数据 + 全帧灰度图 | codec/帧数问题，黑帧、冻结、模糊、过曝 |

---

# 一、完整性检查

原则：**LeRobot v3.0 的 episode 边界不在文件名里，而在 `meta/episodes` 里**。所有“文件、行号、时间偏移、视频位置”必须互相能对上。

## 1. 目录与文件完整性

检查：

- 必需路径：`meta/info.json`、`meta/tasks.parquet`、`meta/episodes/`、`data/`、`videos/`，以及每个视频相机的 `videos/<camera>/`。
- `meta/stats.json` 缺失只 warn，因为它是统计文件，不是数据本体。
- 每个 episode 引用的 data parquet、每个相机 mp4 是否存在、是否为空文件。
- 孤儿文件：`data/` 或 `videos/` 下存在但没有任何 episode 引用的文件。
- 命名与编号：`chunk-XXX`、`file-XXX` 是否三位零填充；file 编号是否连续。
- `meta/episodes` 的 `meta/episodes/chunk_index/file_index` 是否和它所在路径一致。

原理：v3.0 通过模板拼路径：

```text
data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet
videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4
```

只要路径、chunk/file 编号、metadata 引用三者有一处不一致，读取器就会静默读错文件或读不到数据。

## 2. 元数据自洽

检查：

- `episode_index` 从 0 到 N-1 连续、唯一。
- `len(episodes) == total_episodes`。
- `sum(length) == total_frames`。
- 第一集 `dataset_from_index == 0`，最后一集 `dataset_to_index == total_frames`。
- 相邻集满足 `前一个 dataset_to_index == 后一个 dataset_from_index`。
- 每集 `dataset_to_index - dataset_from_index == length`。
- `task_index` 能在 `meta/tasks.parquet` 中找到。

原理：`dataset_from_index` / `dataset_to_index` 是**全局行号区间，右端不含**。它们必须像一条连续的数轴一样覆盖整个数据集，否则按全局行号读 episode 会错位。

## 3. episode 与 parquet 结构

逐集检查：

- 实际行数是否等于 `length`。
- `episode_index` 列在切片内是否恒定。
- `frame_index` 是否等于 `0..length-1`。
- 全局 `index` 是否等于 `dataset_from_index + 0..length-1`。
- `task_index` 是否恒定且合法。
- 数据 schema：`action` / `observation.state` 是否为 `fixed_size_list`，维度是否为 20 / 48。
- 每个 data 文件的总行数是否等于该文件承载的所有 episode 的 `length` 之和。
- `meta/episodes` 里的 `stats/*/count` 是否等于 `length`（只对 count，不信任 stats 数值）。

原理：LeRobot 训练时按 `index` / `episode_index` / `frame_index` 取帧。任何一个索引断裂都会导致“取到别的 episode 的帧”或“视频与动作错位”。

当前数据这层全部通过。

---

# 二、时序对齐

原则：**帧号是唯一对齐键**。

- 动作：`timestamp = frame_index / fps`，每集从 0 开始。
- 视频：第 f 帧对应 `from_timestamp + f / fps`。
- `to_timestamp` 是半开区间右端：`to - from = length / fps`；最后一帧时间是 `(length-1)/fps`。
- 不要用 `to_timestamp` 和最后一帧 timestamp 做精确相等比较。

时序对齐分两层：

1. **元数据/容器自洽**：from/to、length、声明 fps、`nb_frames` 是否互相一致；
2. **实际 PTS 对齐**：从 MP4 读每帧 PTS，与动作 timestamp 逐帧比较。

## 1. 动作时间戳

对每集：

- `diff(timestamp) > 0`，不能倒退。
- `dt` 与 `1/fps` 比较，float32 容差默认 `1e-4 s`。
- `dt <= 0` 判重复时间戳。
- `dt > 1.5 × 1/fps` 判缺帧/gap。
- 其余明显偏离 `1/fps` 的判 dt outlier。

对应 issue：

```text
l1.timestamp_nonmonotonic
l1.timestamp_duplicate
l1.timestamp_gap
l1.timestamp_dt_outlier
```

## 2. 视频段元数据自洽

每集每相机独立检查：

- `from < to`，且在视频时长内。
- `(to - from) × video_fps ≈ length`，容差 1 帧。
- 数据最大 timestamp 不超过视频段结束。
- 同一相机、同一视频文件内，相邻 episode 满足 `前一集 to == 后一集 from`。
- 每个视频文件的 `nb_frames` 等于该文件承载的 episode 长度之和。
- 视频实际 fps 是否等于 `info.json` 的 fps。

对应 issue：

```text
l1.video_bounds
l1.video_segment_duration
l1.video_data_time_mismatch
l1.video_segments_not_contiguous
l1.video_frame_count
l1.video_fps_mismatch
```

原理：v3.0 中视频按相机独立切分。同一个 episode 在不同相机里的 `file_index` 和 `from_timestamp` 可能完全不同，所以每个相机必须独立解析、独立检查，不能复用偏移。

这一层是**算术和范围检查**：它证明 `from/to/length/fps/nb_frames` 自洽，但不读取每一帧的真实 PTS。

## 3. 实际 PTS 逐帧对齐

LeRobot 不存储每一帧视频的时间戳；视频帧的真实时间在 MP4 的 packet PTS 里。检查流程：

1. 每个视频文件只 demux 一次，读所有 packet PTS：

```bash
ffprobe -v error -select_streams v:0 -show_packets \
  -show_entries packet=pts_time -of csv=p=0 file.mp4
```

2. 对 episode 的视频段 `[from, to)` 取 PTS 序列 `video_pts`；
3. 与 data parquet 的 `timestamp` 归一化到首帧后逐帧比较：

```text
video_rel[f]  = video_pts[f] - video_pts[0]
action_rel[f] = timestamp[f] - timestamp[0]
max_drift     = max_f |video_rel[f] - action_rel[f]|
```

4. 同时检查：
   - `len(video_pts) == length`；
   - 第一帧 PTS 与 `from_timestamp` 的偏移；
   - action 起始 timestamp 是否约等于 0；
   - PTS 间隔是否约等于 `1/video_fps`（VFR 检测）。

默认阈值：

```text
warn_frames = 0.5
tol_frames  = 1.0
pts_gap_tol_frames = 0.5
```

结果：

- `max_drift > 1 帧`、帧数不一致、start offset > 1 帧 → fail；
- `max_drift > 0.5 帧` 或 PTS 间隔偏差 > 0.5 帧 → warn；
- 否则 ok。

issue id：

```text
l1.av_timestamp_mismatch
```

输出：

```text
av_sync.csv
av_sync.json
```

`av_sync.csv` 每集每相机一行：

```text
episode_index, camera, length, video_frames,
action_fps, video_fps,
action_start_s, video_first_pts, video_last_pts,
start_offset_s, max_drift_s, max_drift_frame,
pts_gap_max_dev_s, status, issue
```

为什么用 packet PTS：逐帧 `ffprobe -show_frames` 一个 1802 秒视频约 68 秒；packet PTS 只需约 0.8 秒。对 H.264/AV1 MP4，通常一个 packet 对应一帧；如果 packet 数与 `nb_frames` 不一致，会额外提示 `l1.av_pts_count`。

当前数据结果：

- 300 集 × 4 相机 = **1200 行**；
- **0 个 mismatch**；
- `action_fps ≈ 30.000029`，`video_fps ≈ 30.0003`（PTS 保留 6 位小数导致）；
- 每集 `length=636`，`video_frames=636`；
- `start_offset=0`；
- `max_drift ≈ 1e-6 s`；
- `pts_gap_max_dev ≈ 1e-6 s`。

说明：**元数据自洽和实际 PTS 两层都通过**。codec 声明不一致属于视频异常类，不在这里。

### 注意

- 实际 PTS 对齐不等于画面内容同步。相机 pipeline 可能有固定延迟，PTS 完全正确但画面内容仍滞后；这需要视觉-动作互相关，当前未实现。
- 如果视频是 VFR，`pts_gap_max_dev_s` 会变大并报 warn。
- 当前检查只读，不修改、不平移、不重采样。

---

# 三、轨迹异常

这层只看 **action 和 state 的 dims 0–19**；`state[20:48]`（物体位姿）视为冗余，不参与检查。夹爪 18/19 是离散 ±1，不进入连续 step/轨迹计算。

## 1. 数值有限性

- action 或 state 任意一帧出现 NaN/±Inf → `l2.nan_inf`，fail。
- 记录具体帧范围、NaN/Inf 数量。

## 2. 恒定通道

- 某维在 ≥95% 的 episode 中恒定 → 数据集级 info（当前是 dim14 `L_finger1`、dim18 `left_gripper`）。
- 某维在单集内恒定，但该维全局 std > 1e-3 → 该集 warn，说明这一段某个关节被冻结。
- 全局本来就静止的维不会逐集报 warn。

## 3. 瞬时冲击（joint step）

对每个连续维、每个相邻帧：

```text
Δ[t,j] = x[t+1,j] - x[t,j]
```

把所有 episode 的 Δ 按维汇总，计算中位数和 MAD：

```text
scale[j] = max(1.4826 × MAD[j], 0.01)
z = |Δ - median| / scale
```

同时满足 `z > 6` 且 `|Δ| > 0.1` 才算异常；默认 warn，超过 50 个事件 fail。

原理：绝对值防止微小噪声被 z 放大；scale floor 防止近静止关节 MAD=0 时 z 爆炸。离散夹爪不参与，否则 ±1 切换会被误判成巨大 step。

## 4. action–state 同帧残差

```text
residual[t,j] = |action[t,j] - state[t,j]|, j = 0..17
```

- p99 > 0.02 或 max > 0.1 → warn。
- 记录残差最大的帧和维度。
- 夹爪 18/19 完全相等，排除。
- 实测同帧残差均值约 `3e-4`，明显小于 `action[t]` vs `state[t+1]` 的 `5e-3`，所以用同帧比较。

## 5. 轨迹偏移离群（重点）

目标：不是看单帧毛刺，而是看“某一集在某段动作上整体走了不同路径”。

对 action dims 0–17：

```text
median[t,j] = 所有 episode 在帧 t、维度 j 的中位数
MAD[t,j]    = median(|x[e,t,j] - median[t,j]|)
scale[t,j]  = max(1.4826 × MAD[t,j], 0.01)
z[e,t,j]    = |x[e,t,j] - median[t,j]| / scale[t,j]
score[e,t]  = sqrt(mean_j(z²))
```

然后：

- 对 score 做 **15 帧滑动平均**；
- 找连续 ≥15 帧、平滑 score > 10 的区间；
- 区间内最大绝对偏移必须 ≥ 0.2；
- score ≥ 20 或最大偏移 ≥ 1.0 → fail，否则 warn。

为什么用 RMS 而不是最大单关节 z：单关节正常波动很容易到 z>10，max-z 会误报大量正常 episode；RMS 要求多个关节协同偏离，当前 300 集里正好只有 9 集异常。

当前结果：

- 9 集 fail：32, 88, 91, 115, 142, 194, 199, 221, 267；
- 帧范围约 110–146（3.7–4.9 s）；
- 主要偏移关节 `R_shoulder_pitch`，约 1.55–1.82 rad；
- 异常 score 31.7–34.6，正常集最高 8.72，阈值 10 落在空档里；
- 当轨迹偏移覆盖同一集的 joint step 警告时，step 行会被抑制，避免重复报告。

校准数据：

```text
trajectory_scores.csv
trajectory_calibration.json
```

换数据集时先看分数分布，不要把 10 当成通用阈值。

---

# 四、视频异常

视频异常分两层：**元数据层** 和 **全帧画面质量层**。

## 1. 视频元数据

用 ffprobe 检查：

- 实际 codec 与 `info.json` 声明是否一致；
- 分辨率、fps 是否一致；
- `nb_frames` 与时长 × fps 是否一致；
- 文件是否能被 ffprobe 读取。

当前数据：

- 4 个文件实际是 AV1，`info.json` 声明 h264；
- 影响 episode 238–299 的部分相机；
- 这是 info 级 metadata 备注，不直接判为画面异常。

## 2. 全帧画面质量

默认 **每一帧都检查**，不做 5 fps 时间抽样；空间上缩放到 160×120 灰度以控制开销。

对每帧计算：

```text
mean_luma       = 平均亮度
std_luma        = 亮度标准差
Laplacian       = -4I + 上下左右邻域
blur_score      = variance(Laplacian)
frame_diff      = mean(|I_t - I[t-1]|)
bright_fraction = 亮度 > 245 的像素比例
```

判定：

| 类型 | 判据 | 最短持续 |
|---|---|---|
| 黑帧 | `mean < 20` 且 `std < 5` | 0.3 s |
| 暗帧 | `mean < 40` 且 `std < 5`，排除黑帧 | 0.3 s |
| 过曝 | `bright_fraction > 0.5` | 0.3 s |
| 冻结 | `frame_diff < 0.5` | 0.8 s |
| 模糊 | `blur_score < 相机中位数 × 0.5`，且 `std ≥ 10`、非冻结 | 0.5 s |

原理：

- 黑帧/暗帧必须同时看均值和标准差，避免把正常暗场景当成故障；
- 清晰图像的 Laplacian 方差大，模糊图像小；但白墙/桌面本身方差也小，所以要排除低纹理和冻结帧；
- 冻结用相邻帧平均绝对差，静止场景会误报，所以只报 warn；
- 模糊阈值按相机独立校准：head 约 1245，wrist_left 约 820，wrist_right 约 583；如果用一个全局阈值，会大量误报 wrist_right。

事件通过：

```text
frame = round((t - from) / (to - from) × (length - 1))
```

映射回 episode 帧范围。

当前全帧结果：

- 1200 行 episode/camera 指标；
- 0 个画面质量异常；
- 之前 5 fps 抽样时的 6 段 wrist_right 模糊告警，在全帧下没有形成连续 ≥0.5 s 的事件，属于抽样造成的瞬时段；全帧结果更准确。

视频覆盖写进 summary：

```text
- 检查范围：300 集 × 4 相机；episode 帧 0..635；
  每相机 190800 帧；1200 行 episode/camera 指标
```

逐帧指标在 `video_quality.csv`，每相机阈值在 `video_quality_calibration.json`。

---

# 五、summary.md 输出格式

`summary.md` 固定四类：

```text
## 完整性检查
## 时序对齐
## 轨迹异常
## 视频异常
```

- 有 fail/warn：输出表格，包含 `Scope / Issue / Severity / Frames / Detail`；
- 没有 fail/warn：写 **“未检出异常。”**；
- info 级内容（codec 声明、全局恒定通道、校准基线）写入 **“备注（info）”**；
- 时序对齐和视频异常还会写检查范围，例如：

```text
- 检查范围：300 集 × 4 相机；每集 636 帧；
  1200 行 action/video PTS 对齐指标
```

```text
- 检查范围：300 集 × 4 相机；episode 帧 0..635；
  每相机 190800 帧；1200 行 episode/camera 指标
```

# 六、当前 300 集结果

| 类别 | 结果 |
|---|---|
| 完整性检查 | 未检出异常 |
| 时序对齐 | 未检出异常；1200 行 PTS 对齐，0 mismatch |
| 轨迹异常 | 9 集 fail，帧范围 110–146 / 112–145，`R_shoulder_pitch` 偏移约 1.55–1.82 rad |
| 视频异常 | 未检出异常；info 备注 4 个 AV1 文件 codec 声明不一致 |

总体：

```text
291 集干净
9 集 fail
0 集 warn
```

# 七、主要输出文件

默认只写主要报告：

```text
reports/part_sorting_check/
├── summary.md                      # 四类总结
└── episode_quality.csv             # 只含问题 episode，含帧范围
```

加 `--full-report` 后额外输出：

```text
episode_quality.json                # 问题列表的 JSON 副本
av_sync.csv/json                    # 全量 action/video PTS 对齐指标
video_quality.csv/json              # 全量视频画面指标
video_quality_calibration.json      # 每相机 blur 中位数/阈值
trajectory_scores.csv/json          # 每集轨迹分数（含干净集）
trajectory_calibration.json         # 轨迹阈值分布/建议
```

# 八、配置

完整逐项注释见：

```text
configs/part_sorting.toml
```

该文件已经为每个参数写明单位、默认值、调大/调小的效果和适用场景。

## 为下一个数据集配置的步骤

1. **复制配置**

```bash
cp configs/part_sorting.toml configs/my_dataset.toml
```

2. **先改数据集基本参数**

```toml
[fps]
expected = 30          # 必须等于 meta/info.json 的 fps

[l2]
joint_dims = [0, 17]   # 连续关节维度，按 features.action.names 改
discrete_dims = [18, 19]
trajectory_dims = [0, 17]
```

- `joint_dims` 和 `trajectory_dims` 是闭区间；
- 离散夹爪不要放进连续区间；
- 不同机器人的 action 维度不同，必须先看 `info.json` 的 `features.action.names`。

3. **再改绝对阈值**

这些阈值和单位强相关，换机器人/换单位必须重设：

```toml
step_abs_floor = 0.1              # action 单位，当前是 rad
action_state_max_warn = 0.1
trajectory_abs_floor = 0.2
trajectory_abs_fail = 1.0
black_luma = 20
dark_luma = 40
bright_luma = 245
freeze_diff = 0.5
```

- 如果 action 是米、归一化值或度，不能直接沿用 rad 的数值；
- 视频亮度阈值是 0–255 灰度，跟相机曝光有关。

4. **跑一次全量检查，不要先调 z 阈值**

```bash
.venv/bin/dclean check --dataset <new_dataset> --out reports/<new_dataset> --config configs/my_dataset.toml --full-report
```

然后看：

```text
reports/<new_dataset>/trajectory_calibration.json
reports/<new_dataset>/trajectory_scores.csv
reports/<new_dataset>/video_quality_calibration.json
reports/<new_dataset>/av_sync.csv
```

5. **用该数据集自己的分布选阈值**

- 轨迹：打开 `trajectory_scores.csv`，按 `max_smoothed_score` 排序；
  找正常簇和异常簇之间的空档，把 `trajectory_z_warn` 放在空档中间。
  没有明显空档时，用 p99/p99.5 作为起点，再结合 `trajectory_abs_floor`。
- 视频：看 `video_quality_calibration.json` 的每相机 blur 中位数；
  `blur_ratio_threshold` 默认 0.5，误报多就降到 0.4，漏报多就升到 0.6–0.7。
- PTS：先看 `av_sync.csv` 的 `max_drift_s` 分布；
  默认 `warn_frames=0.5`、`tol_frames=1.0`，如果相机 pipeline 有已知抖动再放宽。

6. **重跑并人工复核**

- 先看 summary 四类中的 fail/warn；
- 轨迹/视频异常都建议抽片段看视频；
- 如果某类全是正常任务变体，调高对应阈值，而不是删数据。

## 快速开关

```toml
[av_sync]
enabled = false       # 只跳过 PTS 对齐

[video_quality]
enabled = false       # 只跳过画面质量

[video_quality]
full_frame = false    # 改回时间抽样（采样率由 sample_fps 控制）
```

命令行等价开关：

```bash
--no-video             # 跳过 ffprobe、PTS、画面质量
--no-video-quality     # 保留元数据/PTS，只跳过画面质量
--full-report          # 额外输出全量明细表、校准 JSON 和 JSON 副本
```

## 常用段落

```toml
[av_sync]
enabled = true
tol_frames = 1.0
warn_frames = 0.5
pts_gap_tol_frames = 0.5

[video_quality]
enabled = true
full_frame = true
sample_fps = 5.0
width = 160
height = 120
black_luma = 20.0
black_std = 5.0
dark_luma = 40.0
bright_luma = 245.0
bright_fraction = 0.5
freeze_diff = 0.5
freeze_min_duration_s = 0.8
event_min_duration_s = 0.3
blur_ratio_threshold = 0.5
blur_min_texture_std = 10.0
blur_min_duration_s = 0.5

[l2]
trajectory_z_warn = 10.0
trajectory_z_fail = 20.0
trajectory_abs_floor = 0.2
trajectory_abs_fail = 1.0
trajectory_smooth_frames = 15
trajectory_min_segment_frames = 15
```

# 九、边界与限制

- 当前是**只读检查**，不删除、不裁剪、不重编码、不对齐、不重采样。
- 时序对齐含元数据自洽和实际 packet PTS 两层；但不检查画面内容滞后（相机 pipeline 延迟需要视觉-动作互相关）。
- 轨迹异常只检查 action/state dims 0–19；`state[20:48]` 物体位姿不检查。
- 轨迹检测要求 episode 等长且至少 10 条；否则写 `l2.trajectory_not_computed`。
- 视频全帧检查仍缩放到 160×120 灰度，不做原始 640×480 全分辨率；没有语义遮挡检测。
- 冻结和模糊可能对应正常的静止/运动模糊，因此只报 warn，需要人工复核。
- packet PTS 检查假设一个 packet 对应一帧；如果 packet 数与 `nb_frames` 不一致，会提示 `l1.av_pts_count`。
