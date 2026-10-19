# 发布内容与本地生成内容

此目录包含 PN→RP1 GMR/mink 桥接、Viser、必需的 Linux PN SDK、当前 RP1.3.2 模型、固定 GMR 最小运行源码及与当前工作区数值一致的默认标定。默认保留用户授权的人体派生骨长和解析标准 T-pose，旧中性尺寸版另存为可选标定。安装 Python/ROS 依赖并配置 PN 网络后，可以接收 PN、重定向并以 50 Hz 发布 reference。无需再取得外部模型或 GMR checkout，也无需先做真人标定。它不包含机器人 tracking 策略或部署代码。

## 可追踪的源码和参数

- `mocapapi_python/`：项目自有桥接、输入适配、IK、路径/外部SDK加载边界、模型导出和Viser代码。
- `mocapapi_python/mocap_api.py`、`mocap_main_base.py`：PN Python 接口与接收基类。
- `mocapapi_python/lib/librobotapi_x86-64.so`、`librobotapi_arm64.so`：按 CPU 架构加载的 Linux 原生 SDK；同一台机器只加载对应的一份。
- `assets/robots/rp1_description/mjcf/robot_28dof.teleop.xml` 及它引用的 `meshes/`：当前 RP1.3.2 自由根 28 关节模型、29 份 STL 和来源记录；不包含其它机器人或未使用的模型格式。
- `third_party/GMR/`：当前版本的原样 `motion_retarget.py`、最小包入口/空注册表、完整 MIT LICENSE 与来源记录；不包含 GMR 的其它机器人、人体数据工具、查看器和原 assets。
- `mocapapi_python/rp1/publish.yaml`：发布参数；固定高度偏移保留−0.04 m。
- `mocapapi_python/rp1/pn_axis_to_rp1_28dof.json`、`pn_axis_to_rp1_28dof_current.json`：相同的默认别名，保留当前工作区的人体标量骨长、人体标准根高度1.0095348389594085 m、解析标准 T-pose、pitch 映射、scale 0.78 和全部求解配置；仅去除来源录制路径、哈希、帧数并使用相对模型路径。
- `mocapapi_python/rp1/pn_axis_to_rp1_28dof_neutral.json`：旧中性发布标定的原样归档，人体标准根高度0.91 m、左右大腿/小腿均0.43 m；需用 `--calibration` 显式选择。
- `mocapapi_python/rp1/retarget.json`：reference的28关节顺序。GMR的限制来自生成标定，不自动使用这个文件中的其它SDK参数。
- `configs/presets/`：实时baseline与离线对照参数，不含人体标定、CAD几何或绝对模型路径。baseline已采用最后一版Viser对比参数：关闭关节加速度限制、IK前脚部支撑和pitch补偿，使用提高后的逐关节速度和根部四项限制。
- `tools/prepare_calibration.py`：`--template current|neutral` 默认 current，无参数精确保留 current 数值和求解参数；neutral 必须用独立 `--output`，不能覆盖默认别名或 neutral 归档。新模型或个人骨长可重建，姿态始终解析定义；只有显式 preset 才覆盖求解参数。
- `tools/make_demo_recording.py`：生成4秒合成世界骨架；没有真人动作。
- `tools/evaluate_recording.py`：离线IK/FK诊断，不打开SDK、ROS或机器人。
- `tools/check_release.py`：检查准备目录的内容边界。
- `tools/check_environment.py`：检查 SDK 可加载性、ROS 消息、模型/标定与子进程 Python 依赖，并完成一次标准 T-pose 离线 IK；不连接设备或启动发布。
- `tools/start_live.py`：定位仓库内模型、GMR 和两个 Python 环境，使用所选发布 YAML 的网络配置启动实时桥接；兼容旧网络环境变量回退，不自动加载 ROS、不自动进入人体跟随。
- `tests/`、`pytest.ini`与CI：运行合成/mock测试与附带 RP1/GMR 的模型检查，不连接设备或创建 ROS publisher。
- 文档、依赖清单和 `.env.example`。

最终交付的内容与哈希以打包工具生成的清单为准；这个说明不授权分发清单之外的第三方文件。

## 仅本地生成或自行取得

- 后续自选录制生成的人体尺寸标定、实验标定和自有模型配置；建议保存为独立文件，只有得到相应授权后才纳入发布。本次默认标定保留的个人派生骨长已获用户明确授权。
- `outputs/`：合成NPZ、IK回放、reference JSONL、诊断等运行输出。
- `.env`、虚拟环境和额外的外部 SDK。
- 首次 Viser 启动生成的简化模型网格缓存。
- 个人录制、历史调参数据和真实机器人的rosbag。

默认工具不读取个人录制文件，而是复用随包 current 模板中已有的派生骨长。只有显式选择 `--recording-lengths` 才读取新录制并统计全部帧父子距离中位数，不选择录制姿态。默认文件已删除来源录制路径、哈希和帧数，但保留个人尺寸数值，删除来源标识不构成匿名化承诺。真人原始录制仍不分发。

## 有意不包含的历史内容

此目录不包含旧备份 `librobotapi_x86-64.so.20260803`、Windows `MocapApi.dll`、厂商独立工具、G1/H1 样例、以前的最终 reference 约束 QP、个人分析目录、交互报告、截图、网格缓存、训练 policy 或真机 deployment 代码。当前 reference 只执行固定高度偏移、初始 XY/yaw 对齐和启动/回站过渡。

PN SDK、RP1 模型和 GMR 均保留第三方身份，不随项目自有代码统一授权；GMR 原 MIT LICENSE 随包提供，模型与 SDK 许可状态及版本见 [THIRD_PARTY](THIRD_PARTY.md)。

`replay_acc_off.json` 与当前实时baseline的求解参数一致；`replay_acc100.json` 仅开启100 rad/s²关节加速度限制，供离线对照。scale仍为0.78，发布高度偏移仍为−0.04 m。当前baseline尚未完成真实机器人动态验证；使用者应在自己的模型、跟踪接口和运行环境中验证。
