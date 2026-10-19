# 安装、模型准备与启动

以下步骤以 Ubuntu 22.04 / Linux x86_64、Python 3.10 和 ROS 2 Humble 为基准。项目可在 CPU 上求解 IK，不要求 GPU。ARM64 的 PN 库已提供，但完整依赖和实时流程尚未实测；Windows、macOS 和 32 位 ARM 不在当前支持范围。

所有命令在仓库根目录执行。示例使用 Bash；Zsh 用户加载 ROS 时改用 `setup.zsh`。不要把模型 XML 单独复制到一个新目录：其 `<include>`、`meshdir` 和网格引用的相对路径必须完整保留。

## 推荐：一键安装环境

ROS 2 Humble 已安装的 Ubuntu 22.04 上，运行：

```bash
./install.sh
```

这与 `./run.sh setup` 等价，按锁定的依赖清单创建两个本地环境：`.venv-ik` 用于 GMR/mink/Viser，`.venv-ros` 由系统 Python 创建并继承 ROS 系统包。默认安装完整显示依赖，会下载 Open3D 等较大的 Python 包；安装耗时主要取决于网络和软件源。SDK、RP1.3.2 模型、GMR 和默认标准 T-pose 标定已在仓库内，无需分别下载。

安装结束只做环境验证：依赖、模型与 SDK 可加载，ROS 模块可导入，并用标准 T-pose 完成一次离线 IK。安装不询问 PN 网络、不创建或修改 `.env` 或 YAML，不运行 demo、不连接设备、不启动 ROS publisher，不修改 retarget 参数或标定。

如果系统缺少 venv/ensurepip、编译工具或 ROS 消息包，先显式安装系统依赖：

```bash
./install.sh --system-deps
```

该选项需要 `sudo` 才能安装 Ubuntu/ROS 系统包；默认安装不调用 apt、不执行提权。ROS apt 源需先按[官方 Ubuntu 安装说明](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)配置；脚本不会自行添加 ROS 源或修改网卡地址。先补齐系统依赖再创建环境，可避免因缺 venv/ensurepip 留下不完整环境。

无显示需求只安装 IK 和 live 依赖：

```bash
./install.sh --no-viser
```

关闭显示的选择保存在本机 `.runtime/setup.json`，额外运行工具 `check`、`demo`、`start` 自动沿用。以后补装显示依赖并恢复显示时运行 `./install.sh --viser`；不带显示选项会沿用已有选择。运行工具上的 `--no-viser` 和 `--viser` 只对单次运行覆盖。

`--python /path/to/python3.10` 可指定创建 IK 环境的 Python；ROS 环境仍使用与 Humble ABI 匹配的系统 Python。`./install.sh --print-plan` 仅显示安装计划，不安装或写文件；也可组合 `--system-deps --print-plan` 先检查系统包计划。

## 安装后的可选检查、演示与网络配置

安装不会自动运行以下流程。需要时分别执行：

```bash
./run.sh check                      # 不连接设备的 live 环境检查
./run.sh check --offline            # 不要求 ROS 的离线环境检查
./run.sh demo                       # 合成回放并运行只读 Viser
./run.sh demo --no-viser            # 只完成离线回放验证
```

`demo` 自动生成 4 秒纯合成人体骨架，完成离线 IK 和 dry-run reference 回放，再运行只读 Viser。浏览器打开 [http://127.0.0.1:8890/](http://127.0.0.1:8890/)，终端按 **Ctrl-C** 退出。它不打开 PN，也不创建 ROS publisher。

准备接 PN 时，编辑 `mocapapi_python/rp1/publish.yaml` 中的 `pn_network` 即可设置电脑 PN 网口的本机 IP、PN-Link 设备 IP 和两个端口。也可选择运行 `./run.sh configure`：PN 四项保存到该 YAML，ROS domain 保存到本机 `.env`，保留其它设置；不安装依赖、不修改网卡。网络字段和优先级见下文手动步骤5。安装不依赖这些地址，但环境检查成功不代表 PN 网络已连通。

`run.sh` 读取 `.env` 中的字面赋值，不执行里面的 shell 命令；在 `/opt/ros/humble/setup.bash` 存在时受控加载该固定 ROS 文件。因此使用运行工具无需手动激活 Python 环境或加载 ROS。`.env` 已被 `.gitignore` 排除。其它 bridge 参数直接放在 `./run.sh start` 后；例如 `./run.sh start --viser-port 8892`。所有运行子命令支持 `--print-plan` 查看计划。

```bash
./run.sh start --dry-run            # 主动连接 PN，但不创建 ROS publisher
./run.sh start                      # 主动连接 PN，并正式发布 reference
```

正式启动及终端按键见下文。以下保留逐步安装操作，适合已有环境、排错或不使用一键安装的情况。

## 手动步骤 1：系统与 ROS

先按 [ROS 2 Humble 官方 Ubuntu 安装说明](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html) 配置 ROS apt 源并安装 Humble。然后补齐：

```bash
sudo apt update
sudo apt install git python3-venv python3-dev build-essential libstdc++6 libgl1 libegl1 libglib2.0-0 \
  ros-humble-rclpy ros-humble-std-msgs ros-humble-sensor-msgs \
  ros-humble-geometry-msgs ros-humble-visualization-msgs ros-humble-tf2-ros
source /opt/ros/humble/setup.bash
```

`rclpy` 必须与 ROS 发行版、操作系统 Python ABI 匹配。本例使用 `/usr/bin/python3` 创建 ROS 环境，再用独立 Python 运行 IK 和 Viser，避免把 ROS 的二进制模块装入不兼容的 Conda Python。

## 手动步骤 2：PN SDK 已随仓库提供

`mocapapi_python/mocap_api.py` 根据 `platform.machine()` 选择 `mocapapi_python/lib/` 下的库：

- `x86_64`：`librobotapi_x86-64.so`。
- `aarch64` / `arm64`：`librobotapi_arm64.so`。

Python 接口使用相对自身文件的路径找库，可以从仓库根目录启动。无需手动复制库到 `/usr/lib`，也无需使用 `.so.20260803` 旧备份或 Windows DLL。SDK 来源及许可状态见 [THIRD_PARTY](THIRD_PARTY.md)。

当前接收基类已支持显式网络参数和终端按键，不依赖桌面全局键盘监听。`requirements-live.txt` 中的 `docutils` 是厂商 wrapper 的导入依赖，NumPy 用于回放及录制；live 主进程不需要安装完整 GMR/Torch/Viser 依赖。

## 手动步骤 3：安装 IK / Viser 和 ROS Python 环境

RP1.3.2 模型已经包含在 `assets/robots/rp1_description/` 中：入口为 `mjcf/robot_28dof.teleop.xml`，保留它实际引用的 29 份 STL 网格及原目录结构。该入口包含自由根节点、28 个关节和 teleop 脚部定义；不要用基础仓库的 `robot.xml` 或其它 RP1 版本替换后继续使用旧标定。来源、版本与许可状态见 [THIRD_PARTY](THIRD_PARTY.md)。

固定 GMR 运行源码位于 `third_party/GMR/`，来自当前验证的 fork 提交 `d918bde83c5d72b0dbf979018793f6a74e529f10`。`motion_retarget.py` 保留原样，包入口和机器人注册表精简为 PN→RP1 运行所需内容，MIT LICENSE 随包保留。无需再 clone GMR，也不用 `pip install -e`；适配器直接加载仓库内源码。上游的其它机器人、查看器、人体数据工具与 assets 均未纳入。

```bash
cp .env.example .env
# 模型/GMR默认已指向仓库内；PN地址在mocapapi_python/rp1/publish.yaml中设置。
# 路径变量使用加载.env时的$PWD，因此请始终在仓库根目录加载。
set -a; . ./.env; set +a
export PYTHONNOUSERSITE=1

python3.10 -m venv .venv-ik
.venv-ik/bin/python -m pip install --upgrade pip
.venv-ik/bin/python -m pip install -r requirements.txt

/usr/bin/python3 -m venv --system-site-packages .venv-ros
.venv-ros/bin/python -m pip install -r requirements-live.txt
```

`.env` 中 `PN_RP1_SOLVER_PYTHON` 和 `PN_RP1_VISER_PYTHON` 默认指向根目录的 `.venv-ik/bin/python`。若使用已有环境，改成对应 Python 的绝对路径。`PN_RP1_MODEL` 和 `PN_RP1_GMR_ROOT` 默认指向仓库内模型和 GMR；保留这两个变量可以明确固定运行源码，避免导入其它环境中安装的 GMR。

最小 GMR 的依赖已列入 `requirements-ik.txt`，包含 Cython 和 rich。这里不需要 Torch、SMPL-X 人体模型或 GMR 的完整原版环境。Viser 与首次模型网格简化依赖列在 `requirements-viewer.txt`。`requirements.txt` 同时安装这两组；无显示需求可以只安装 IK 清单并在运行时加 `--no-viser`。

直接执行下文 Python 文件时，程序不会自动加载 `.env`；每次新终端都要进入仓库、加载 ROS 和 `.env`。使用 `./run.sh` 时这些步骤自动完成：

```bash
source /opt/ros/humble/setup.bash
set -a; . ./.env; set +a
export PYTHONNOUSERSITE=1
```

## 手动步骤 4：检查环境和离线回放

```bash
.venv-ik/bin/python tools/make_demo_recording.py
.venv-ros/bin/python tools/check_environment.py --mode offline
.venv-ros/bin/python tools/check_environment.py --mode live
```

仓库的 `mocapapi_python/rp1/pn_axis_to_rp1_28dof.json` 和 `pn_axis_to_rp1_28dof_current.json` 为内容相同的默认别名，与当前工作区保留相同的派生人体骨长、解析标准 T-pose、pitch 映射、scale 0.78 和求解配置。人体标准根高度为1.0095348389594085 m；仅移除来源录制路径、哈希、帧数并使用相对模型路径。用户已明确授权保留这些个人派生尺寸；原始录制不打包，也不从录制选择近似 T-pose。克隆后可以直接使用，安装脚本不会改动标定。默认约束开关和高度偏移见 [CONFIGURATION](CONFIGURATION.md)。

旧中性标定另存为 `mocapapi_python/rp1/pn_axis_to_rp1_28dof_neutral.json`，人体标准根高度为0.91 m、左右大腿/小腿均0.43 m。默认 `./run.sh start` 使用 current；要对比中性版：

```bash
./run.sh start --calibration mocapapi_python/rp1/pn_axis_to_rp1_28dof_neutral.json
```

只有改变模型、preset 或需要重建时才运行：

```bash
.venv-ik/bin/python tools/prepare_calibration.py
```

工具支持 `--template current|neutral`，默认 current。无参数重建会覆盖两个默认别名，但精确保留 current 全部数值、解析标准姿态和求解参数；不会重新变成中性骨长。只有显式 `--preset` 才覆盖参数。`--template neutral` 必须用 `--output` 写独立路径，禁止覆盖两个默认别名及随包 neutral 归档。新个人尺寸与实验标定请用 `--output outputs/my_calibration.json` 单独保存；`--recording-lengths` 仅计算骨段距离中位数，不选取录制姿态。

环境检查在独立子进程加载 SDK、ROS 消息与 Python 模块，并用配置的模型/标定做一次标准 T-pose 离线 IK；不创建 ROS 节点、不打开 PN 应用、不接收网络或控制机器人。`--skip-sdk-load` 仅检查 SDK 文件而不加载 native 库；`--no-viser` 不检查显示依赖；`--json-output outputs/environment.json` 可保存检查结果。检查成功证明所列依赖、模型和单帧求解可用，完整动作与时序仍要用离线回放验证。

接着运行 README 中的离线 evaluator 与 `--source replay --dry-run` 示例。确认能完成 IK、生成 reference，并可用 Viser 查看后，再连接 PN。

## 手动步骤 5：配置 PN 网络并发布 50 Hz reference

当前 GMR 实时入口和启动工具默认从已有 `mocapapi_python/rp1/publish.yaml` 读取网络配置。编辑该文件即可，无需新增配置文件或重新安装：

```yaml
pn_network:
  local_ip: 10.42.0.101   # 电脑连接 PN 的网口 IP
  local_port: 8002        # 本机 UDP 接收端口
  server_ip: 10.42.0.202  # PN-Link 设备 IP
  server_port: 8080       # PN-Link 命令端口
```

以上是随包默认值，须按自己的网络更换。`local_ip` 必须已分配给电脑连接 PN 的网口；`server_ip` 是 PN-Link 地址，不能填电脑自己的地址。四项分别按 **显式 CLI > YAML > 旧 PN 网络环境变量 > 旧默认值** 解析；`.env` 中已有的 `PN_LOCAL_IP/PN_LOCAL_PORT/PN_SERVER_IP/PN_SERVER_PORT` 不会覆盖 YAML 已配置项，只在相应字段缺失时回退使用。`.env` 继续保存 `ROS_DOMAIN_ID` 和运行路径。

这一网络 YAML 功能仅适用于 `mocap_rp1_gmr_reference.py`、`tools/start_live.py` 和 `run.sh start`，不改变旧 SDK 欧拉角入口的网络参数方式。`./run.sh configure` 可持久写入同一 YAML 的四个 PN 字段，将 domain 单独写入 `.env`。

```bash
ip -br addr
ip route get 10.42.0.202  # 换成 YAML 中实际的 server_ip
```

确认路由使用连接 PN 的网卡，源地址与 `local_ip` 一致。根据设备配置设置网卡同网段地址，允许相应 UDP 通信。程序会检查本机 UDP 绑定地址；路由请用上面的 `ip route get` 手工核对，不会自动修改网卡或猜测设备 IP。PN-Link 应已连接、穿戴及校准，且没有另一份客户端占用相同本地地址/端口。

保存 YAML 后退出已有发布进程并重新启动：

```bash
./run.sh start
# 临时覆盖 IP，不修改 YAML；端口也可用 --pn-local-port / --pn-server-port 覆盖。
./run.sh start --pn-local-ip 10.42.0.101 --pn-server-ip 10.42.0.202
# 选择另一份完整发布配置；路径可为绝对路径。
./run.sh start --publish-config /path/to/publish.yaml
```

若需手动使用底层启动工具：

```bash
python3 tools/start_live.py \
  --output outputs/live_reference.jsonl --metrics outputs/live_metrics.json
```

底层 `start_live.py` 定位仓库内 `.venv-ros/bin/python`、`.venv-ik/bin/python`、RP1 模型、GMR 和默认配置，按上述优先级解析网络。它与 GMR 直接入口和 `run.sh start` 均支持 `--publish-config`，缺省为仓库内的 `mocapapi_python/rp1/publish.yaml`。直接调用 Python 文件时需手动加载需要的 `.env` 运行变量和 ROS `setup.bash`；使用 `./run.sh start` 已自动处理。启动不会修改网卡或自动进入人体跟随。

需要直接使用原入口时，等价命令为：

```bash
.venv-ros/bin/python mocapapi_python/mocap_rp1_gmr_reference.py \
  --source live \
  --publish-config mocapapi_python/rp1/publish.yaml \
  --reference-topic /tracking/current_reference \
  --output outputs/live_reference.jsonl --metrics outputs/live_metrics.json
```

先使用 `--dry-run` 可以接收 PN、求解和显示，但不创建 ROS publisher。正常模式启动即以 50 Hz 发布待机站姿，按 **N** 开始采集，按 **A** 对齐并用 2 秒过渡进入跟随，按 **X** 回站，按 **Esc** 退出。不要在 live 示例中添加离线便利选项 `--autostart`。无图形需求可加 `--no-viser`。

集成 Viser 默认地址为 `http://127.0.0.1:8891/`；显示的机器人向量与发布向量相同。reference 为 `std_msgs/msg/Float32MultiArray`，35 个值依次是世界根位置 XYZ（米）、根四元数 WXYZ、28 个关节角（弧度）。接收方应使用同一 `ROS_DOMAIN_ID` 和关节顺序。

```bash
ros2 topic info /tracking/current_reference
ros2 topic hz /tracking/current_reference
```

目标发布周期为 20 ms。PN 新帧率、IK 求解频率和 reference 发布频率是三个指标；输入或求解赶不上时可能重复最近的 reference，50 Hz 不保证每条都是新姿态。`--metrics` 输出包含相应调度与延迟统计。

## 常见问题

- **`address not available`**：先用 `ip -br addr` 确认 YAML 的 `local_ip`（或本次 `--pn-local-ip` 覆盖值）真正属于本机，再检查 `ip route get`。不要通过反复启动 SDK 试错一个未分配的地址。
- **`cannot open shared object` / 架构错误**：检查 CPU 架构、对应 `.so` 和系统动态库。可用 `ldd mocapapi_python/lib/librobotapi_x86-64.so`（ARM64 换相应文件）寻找 `not found` 项。
- **找不到 `rclpy` 或消息包**：加载 ROS `setup.bash`，使用由 `/usr/bin/python3` 创建的 `--system-site-packages` 环境。
- **GMR 属性或函数参数不存在**：确认 `.env` 已加载，`PN_RP1_GMR_ROOT` 指向本仓库 `third_party/GMR`，不要导入另一个环境里的 GMR。恢复随包版本后运行环境检查。
- **模型网格丢失**：恢复 `assets/robots/rp1_description/` 内 XML 与 meshes 的相对目录结构，重新生成标定。改变 XML 后不要沿用已有模型哈希标定。
- **页面打不开**：确认启动输出中 Viser 子进程无依赖/模型错误；若端口已占用用 `--viser-port` 改端口；服务器和浏览器不在同一电脑时，`127.0.0.1` 指向各自本机。

发布的是运动学目标，仓库不包含真机 tracking 策略。当前参数与 −0.04 m 高度偏移的动态稳定性未做真实机器人验证；Viser 目标显示与真实跟踪效果应分别检查。
