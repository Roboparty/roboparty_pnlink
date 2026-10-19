# 配置与本地标定

参数 preset 是算法参数；标定包含模型的 CAD 派生几何、模型路径和人体骨长。仓库默认采用与当前工作区数值一致的标定，保留经用户明确授权的人体标量骨长、解析标准 T-pose、pitch 映射、scale 0.78 和全部求解配置。发布副本仅移除来源录制路径、录制哈希和帧数，并将模型路径改为仓库内相对路径；原始录制不打包。默认包含个人派生尺寸，删除来源标识不构成匿名化承诺。克隆后无需先生成。

`pn_axis_to_rp1_28dof.json` 和 `pn_axis_to_rp1_28dof_current.json` 是相同的默认别名，`source_calibration.source_root_reference_position_m[2]` 为1.0095348389594085 m。旧中性发布标定原样保留为 `pn_axis_to_rp1_28dof_neutral.json`，其人体标准根高度为0.91 m、左右大腿和小腿均为0.43 m。它们均位于 `mocapapi_python/rp1/`。不带标定参数启动时使用 current；中性版为显式可选项：

```bash
./run.sh start --calibration mocapapi_python/rp1/pn_axis_to_rp1_28dof_neutral.json
```

## 生成标定

```bash
.venv-ik/bin/python tools/prepare_calibration.py
```

工具的 `--template current|neutral` 默认是 `current`，无参数保留 current 人体尺寸、人体标准根位置和解析掌心向下 T-pose，不退回中性尺寸。所有 PN bind 四元数仍为 identity；RP1 标准姿态来自模型真实 FK 和脚底几何。工具不需要 PN SDK、不读取真人录制姿态。需要改用自己的模型时，设置 `PN_RP1_MODEL` 或传 `--model`，并重新生成标定。

没有 `--output` 时，同时写入两个默认配置文件：

- `mocapapi_python/rp1/pn_axis_to_rp1_28dof.json`
- `mocapapi_python/rp1/pn_axis_to_rp1_28dof_current.json`

随包的两个默认别名内容相同，模型路径使用相对于配置文件的仓库内路径，可搬迁。无参数重建精确保留 current 的全部数值和求解参数；显式 `--preset` 才覆盖相应参数。指定新模型或 `--recording-lengths` 重新生成几何时，也先继承所选模板的 limits、solver、contact 和 torso 配置，再应用可选 preset。显式 `--output` 只写指定文件，适合保存实验配置；输出文件与模型的相对目录关系应一并保留。修改模型 XML 后应重新生成标定；运行时会核对模型名称、SHA、自由度和关节顺序。

使用 `--template neutral` 必须指定独立 `--output`，不能覆盖两个默认别名或随包 neutral 归档。例如：

```bash
.venv-ik/bin/python tools/prepare_calibration.py \
  --template neutral --output outputs/my_neutral_calibration.json
```

确实需要适配个人尺寸时，可选择自己的录制：

```bash
.venv-ik/bin/python tools/prepare_calibration.py \
  --recording-lengths /path/to/your_pn_take.npz \
  --output outputs/my_calibration.json
```

这里仅统计全部帧父子骨段欧氏距离的中位数；标准姿态方向、bind旋转仍为解析定义，不选择某一录制帧作为 T-pose。新的个人尺寸或实验配置应保存在独立文件，只有明确授权后才纳入发布；当前随包默认尺寸属于本次已获授权的例外。

## 三个参数 preset

`configs/presets/baseline.json` 是实时缺省，采用最后一版 Viser 对比参数：

- 根位移 scale 为0.78，发布高度偏移仍由 YAML 设为−0.04 m。
- 髋/膝/腰关节速度限制为14.24192 rad/s，踝/上肢为21.88648 rad/s，与原 SDK 逐关节速度参数一致。
- `joint_acceleration_enabled=false`：关闭关节加速度限制及其派生的关节制动包络；数组中的100 rad/s²仅为未生效的名义值。
- 根线速度7.2 m/s、角速度14.4 rad/s、线加速度60 m/s²、角加速度96 rad/s²。
- `contact.enabled=false`：关闭IK前脚部支撑处理；`torso_compensation.enabled=false`：关闭腰 pitch 补偿。
- 关节角度、关节速度、根部限制与IK地板约束仍保留；发布仍为50 Hz，每周期一次 mink QP，没有发布阶段的第二次约束求解。

`replay_acc_off.json` 与 baseline 的求解参数一致，保留为显式离线对照入口。`replay_acc100.json` 仅开启100 rad/s²关节加速度限制，其余上述参数相同。当前缺省参数尚未完成真实机器人动态验证。

为实验生成独立标定，不覆盖实时默认：

```bash
.venv-ik/bin/python tools/prepare_calibration.py \
  --preset configs/presets/replay_acc100.json --output outputs/calibration_acc100.json
.venv-ik/bin/python tools/prepare_calibration.py \
  --preset configs/presets/replay_acc_off.json --output outputs/calibration_acc_off.json
```

用同一份 PN NPZ 对两者分别执行 evaluator 的 `--config` 和桥接的 `--calibration`，保存独立输出。两步必须选择相同标定；不能用一份 IK 与另一份 reference 混合对照。生成独立实验标定不会修改默认标定文件；不指定 `--preset` 时沿用所选模板的求解参数。

preset 顶层允许 `description`、`output_limits`、`solver`、`contact`、`torso_compensation_enabled`。工具把后三个参数段覆盖到生成配置，并把 torso 布尔值映射到 `torso_compensation.enabled`；不会从 preset 接受模型几何或真人姿态。

## 缩放、映射和关节限制

根位置目标为 `robot_T_root + 0.78 × (PN_Hips − PN_T_root)`，XYZ三个方向均使用0.78。四肢使用 PN 方向和实际 RP1 CAD 段长、关节中心残差重建，不是将所有点统一乘0.78。

当前任务映射：Hips→base、Spine2→腰部；UpLeg→hip_pitch、Leg→knee、Foot→ankle_roll；Arm→shoulder_pitch、ForeArm→elbow、Hand→wrist_pitch。左右各自对应。标准bind旋转偏移来自两者的解析/机械 T-pose。

pitch 补偿在开启时将 Spine2 相对 Hips 的矢状面倾角转移到虚拟骨盆，限幅±50°，并围绕双髋中点修正根位置。关闭时该附加角为零。脚部支撑开关控制IK前支撑检测、锁脚、压平、目标高度和任务权重；它与IK内部的地板约束是不同开关。

根部限制的核心实现使用每个切空间分量 `标量限制 / sqrt(3)` 的保守盒，保证整向量范数不超过该标量。关节速度/加速度以每个关节的数组值约束；这些是求解参数，不是经过硬件认证的能力。

## 发布 YAML

`mocapapi_python/rp1/publish.yaml` 控制 PN 网络、话题、QoS、固定50 Hz、100步/2秒过渡、源超时和 `root_z_offset_m`。当前保留值为−0.04 m；正值抬高、负值降低。每次从原IK复制后加一次Z，不随重复帧累积；待机站姿不加偏移，启动阶段随过渡混合。

高度偏移不重新求解IK，也不重新落脚；非零偏移后的脚高不由核心地板保证。原始PN、统一缩放PN和橙色IK任务点不加发布高度偏移。修改配置后重启入口生效。

### PN 网络

当前 GMR 实时入口、`./run.sh start` 和 `tools/start_live.py` 默认使用上述 YAML，也可用 `--publish-config /path/to/publish.yaml` 选择其它文件。编辑已有 `pn_network` 段即可更换网络地址：

```yaml
pn_network:
  local_ip: 10.42.0.101   # 电脑连接 PN 的网口 IP，须已分配给本机
  local_port: 8002        # 本机 UDP 接收端口
  server_ip: 10.42.0.202  # PN-Link 设备 IP，不能填电脑地址
  server_port: 8080       # PN-Link 命令端口
```

四项逐字段按 **显式 CLI > YAML > 旧 `PN_LOCAL_IP/PN_LOCAL_PORT/PN_SERVER_IP/PN_SERVER_PORT` 环境变量 > 旧默认值** 解析；`.env` 中旧 PN 值不会覆盖 YAML 已配置项。上面的地址是随包默认值，连接前须核对自己的网卡和设备。程序仅检查本机 UDP 绑定地址，不配置网卡、不猜设备 IP。该网络 YAML 支持仅适用于当前 GMR 入口及启动工具，旧 SDK 欧拉角入口仍沿用原网络参数方式。

保存 YAML 后退出旧发布进程并重启；CLI 覆盖只对本次启动有效，不改文件：

```bash
./run.sh start
./run.sh start --pn-local-ip 10.42.0.101 --pn-server-ip 10.42.0.202
./run.sh start --publish-config /path/to/publish.yaml
```

### 独立手指话题

`publish.yaml` 的 `hands` 段控制现有四个话题，默认开启、50 Hz、QoS `best_effort` / depth 1。GMR实时入口 `mocap_rp1_gmr_reference.py` 和旧欧拉角入口 `mocap_rp1_reference.py` 均只在这些数组中发布手指，不再包含手腕；没有新增 `/mocap/fingers/*` 话题。

- `/mocap/hand/left/poses`、`/mocap/hand/right/poses`：`geometry_msgs/PoseArray`，`frame_id=world`，位置单位为米，消息四元数字段为 `x/y/z/w`。
- `/mocap/hand/left/active`、`/mocap/hand/right/active`：`std_msgs/Bool`，仅表示该侧至少有一个手指位姿，不保证19点完整。

每侧最多19点，排除 `LeftHand` / `RightHand` 手腕。完整数据的顺序如下，名称均加 `Left` / `Right` 前缀：

1. `HandThumb1`、`HandThumb2`、`HandThumb3`。
2. 依次为 Index、Middle、Ring、Pinky；每根手指按 `InHand{Finger}`、`Hand{Finger}1`、`Hand{Finger}2`、`Hand{Finger}3` 排列。

缺失点按上述顺序省略，不填零，因此不完整数据的数组索引不能作为固定骨名使用；只有手腕、没有任何指骨时数组为空且 `active=false`。输入手腕仍保留供父子世界位姿计算和身体IK使用，只从发布的手指数组中排除。**使用旧20点数组的订阅端应删除首个手腕的假设并更新索引；修改后须重启发布程序生效。**

这些位姿来自PN原始指骨，经坐标轴转换、厘米到米和父子FK计算，不经过身体 scale、IK、根高度偏移、初始位置/朝向对齐或启动过渡；不是机器人手指电机角，也不是Pico的26点布局。身体 `/tracking/current_reference` 不包含手指。`--dry-run` 不创建ROS发布器。

```bash
ros2 topic echo /mocap/hand/left/poses --qos-reliability best_effort
ros2 topic echo /mocap/hand/left/active --qos-reliability best_effort
```

## 本地路径和运行环境

推荐用 `./install.sh` 一键安装环境，它与 `./run.sh setup` 等价。安装只创建 Python 环境、安装依赖并完成环境检查，不创建或修改 `.env`、YAML，不询问 IP，不修改 retarget 参数或标定。需要网络配置时编辑上述 `pn_network`，或单独运行 `./run.sh configure` 将 PN 地址/端口写入 YAML、将 ROS domain 写入 `.env`，无需重新安装依赖。`./run.sh check`、`start`、`demo` 都会从仓库根目录读取 `.env` 中的字面赋值，不执行其中的 shell 命令；仅在固定的 `/opt/ros/humble/setup.bash` 存在时受控加载 ROS。`.env` 不应提交到仓库。

`.env.example` 给出以下可选路径变量；直接运行底层 Python 文件时仍需由操作者显式加载 `.env`：

- `PN_RP1_MODEL`：默认指向仓库内 `assets/robots/rp1_description/mjcf/robot_28dof.teleop.xml`；生成标定时使用。
- `PN_RP1_GMR_ROOT`：默认指向仓库内 `third_party/GMR`，固定本项目验证的最小运行源码；不需要另行安装上游 GMR。更换成外部版本后应重新验证 API 和实际求解。
- `PN_RP1_SDK_DIR`：通常不必设置，仓库已带 Python 接口和 Linux SDK。仅当同目录接口文件不存在时，loader 才回退到此变量指定的 sample checkout（含 wrapper、base 和 `lib/`）；它不能覆盖优先加载的同目录文件。
- `PN_RP1_SOLVER_PYTHON`：IK worker的Python。
- `PN_RP1_VISER_PYTHON`：Viser子进程的Python。
- 当前所带接收基类直接读取终端，不需要 `pynput`。若使用另行取得、仍依赖 `pynput` 的旧 sample，可自行安装该包并在 headless 环境设 `PYNPUT_BACKEND=dummy`。

`.env` 保留 `ROS_DOMAIN_ID` 和 Python/模型/GMR 运行路径。四个旧 PN 网络变量在 `.env.example` 中仅作注释兼容示例，只有相应 YAML 字段缺省时才回退使用。应先用 `ip -br addr` 确认本机地址，并用 `ip route get 10.42.0.202`（换成 YAML 的 `server_ip`）核对连接 PN 的网卡，详见 [安装与排错](INSTALLATION.md)。

`./install.sh --no-viser` 只安装 IK 和 live 依赖，并将显示偏好写入被忽略的本地 `.runtime/setup.json`；后续 `check`、`start`、`demo` 自动沿用。命令上的 `--no-viser` 或 `--viser` 可单次覆盖；补齐显示依赖并恢复显示则运行 `./install.sh --viser`，不带选项会沿用已有选择。固定高度偏移仍只由 `mocapapi_python/rp1/publish.yaml` 决定，当前为−0.04 m，一键安装不会改为其它值。

安装不依赖 PN 地址，也不会自动开始回放或 live。安装和运行工具都支持 `--print-plan` 查看计划，`check --offline` 不要求 ROS。
