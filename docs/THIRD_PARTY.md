# 第三方内容与许可状态

项目自有代码许可证尚未选择。当前准备目录没有用一个统一的MIT许可证覆盖整个项目，也没有替使用者决定第三方内容的再分发权限。正式公开发布前，应由权利人确认项目自有代码许可及保留的第三方来源说明。

## 随仓库提供的 PN SDK

所带文件来自 [PN 官方 Python SDK sample 仓库](https://github.com/pnmocap/mocapapi_python)，原工作副本的基准提交为 `e2ac018`。仅保留本项目 Linux 实时输入所需内容：

- `mocapapi_python/mocap_api.py`：厂商 Python ctypes 接口。
- `mocapapi_python/mocap_main_base.py`：基于厂商 sample 的接收/命令处理基类，带本项目的网络配置检查、100 Hz 轮询及终端输入适配。
- `mocapapi_python/lib/librobotapi_x86-64.so`：Linux x86_64 原生 SDK。
- `mocapapi_python/lib/librobotapi_arm64.so`：Linux ARM64 原生 SDK。

原始动态库不在本项目中重新编译。Windows DLL、旧版本备份、其它架构和无关 sample 未纳入。发布清单保存实际文件哈希；Linux x86_64 的加载已验证，ARM64 整条运行链路未实测。

`PN_SDK_MANIFEST.json` 保存这四个文件的来源和 SHA-256。当前 x86_64 库版本为 `0.0.65.33a1385e`，ARM64 库为较早的 `0.0.37.ba779e34`，两者不是同一版；ARM64 使用者仍需在实际主机验证接口和输入。

目前未在原件中找到覆盖这些文件的明确再分发许可。将它们纳入运行包是本次仓库整理的范围，**不代表厂商已授权开源或本项目可替厂商授予许可**；使用及公开再分发时须遵守权利人的条款。这些文件不能被未来的项目自有代码许可证自动覆盖。项目未为 SDK 另行声明 MIT/Apache 许可。

正常运行优先加载仓库同目录 SDK；`PN_RP1_SDK_DIR` 用于同目录接口不存在时定位外部 sample，不改变外部 SDK 的许可。

## 随仓库提供的 RP1.3.2 模型

`assets/robots/rp1_description/` 仅包含当前运行入口 `mjcf/robot_28dof.teleop.xml` 和它实际引用的 29 份 STL。没有包含其它机器人、基础模型的无关 XML/URDF、ROS 构建目录或历史备份。模型文件和网格保持原件内容，按原相对目录保存。

模型入口来自 [Roboparty/roboparty_teleop](https://github.com/Roboparty/roboparty_teleop)，本地验证 checkout 为 `8fd1c118bd8f191c6f66fd89ed2abb0bc330e917`；基础 CAD 来源记录为 [Roboparty/rp1_description](https://github.com/Roboparty/rp1_description)，`UPSTREAM_COMMIT` 为 `84529a55649b5c9796d49093c266208c879f9f29`。该 teleop 入口包含额外脚部定义，不能以基础仓库 `robot.xml` 替代。文件来源、版本和 SHA-256 随发布清单记录。

目前没有在模型原件中找到明确覆盖这些资产的再分发许可。纳入本次运行包不代表项目作者可替模型权利人授予开源许可；SDK 与模型均不自动继承项目未来的自有代码许可证。

两个默认别名 `pn_axis_to_rp1_28dof.json` 和 `pn_axis_to_rp1_28dof_current.json` 与当前工作区数值一致：保留用户明确授权的个人派生骨长、人体标准根高度1.0095348389594085 m、解析标准 T-pose、pitch 映射、scale 0.78 和全部求解配置，仅移除来源录制路径、哈希和帧数，并改用相对模型路径。原始录制不随包提供；删除来源标识不意味着这些骨长已匿名化。旧中性标定原样保留为 `pn_axis_to_rp1_28dof_neutral.json`（人体标准根高度0.91 m、左右大腿/小腿均0.43 m），供显式切换。三份标定均包含模型派生几何，模型许可边界同样适用；实际模型哈希仍保留在配置中。

## 随仓库提供的固定 GMR 运行源码

[General Motion Retargeting / GMR](https://github.com/YanjieZe/GMR) 用于模型与任务栈构造。`third_party/GMR/` 来自当前使用的 [zerojuhao/GMR fork](https://github.com/zerojuhao/GMR)，基准提交为 `d918bde83c5d72b0dbf979018793f6a74e529f10`。

保留的 `general_motion_retargeting/motion_retarget.py` 与本地验证原件字节一致，SHA-256 为 `88e130553af9e6d0cdbdcb90f03412c54be97ce86904cec55115e49702841f76`。完整原 MIT LICENSE 保留在 `third_party/GMR/LICENSE`。本项目只精简包入口 `__init__.py` 和 `params.py` 注册表：仅导入该核心，机器人/配置表由 PN 适配器在运行时临时注册。不纳入 GMR 的其它机器人、assets、查看器、SMPL-X 或其它人体数据工具。目录内来源说明记录保留文件与适配范围；这是一份最小运行提取，不是完整 GMR 仓库。

本集成依赖 `ROBOT_XML_DICT`、`IK_CONFIG_DICT`、`GeneralMotionRetargeting(use_velocity_limit=...)` 以及 `model/configuration/tasks1` 属性。本项目的限制、单次 QP 调度和 reference 发布逻辑由 PN 适配源码定义，未调用 GMR 原版多次迭代 `retarget()` 循环。更换 GMR 版本后应重新验证实际求解；默认不需要更换或从远端取得该提交。

## 仍由包管理器提供的依赖

MuJoCo、mink、DAQP/qpsolvers、NumPy、SciPy、Cython、rich、Viser、Open3D、trimesh 等分别保留其发行许可。ROS 2 及消息包由相应系统包提供。本仓库不复制 Python 虚拟环境、ROS 运行库或整个依赖树。

`requirements-ik.txt` 包含最小 GMR 核心实际导入的 Cython 和 rich；不再因上游完整包入口导入 Torch、imageio、OpenCV 等无关工具。本项目的这条运动学路径不要求 GPU、SMPL-X 人体模型数据或 Torch。`requirements-live.txt` 仅安装 ROS 主进程与所带 SDK wrapper 所需的 Python 包，native SDK 已随仓库提供。

## 不随此目录分发的内容

个人原始录制、rosbag、真人原始骨架、其它未获授权的个人尺寸标定、历史调参输出、模型简化缓存、其它机器人模型、训练 policy、真机 tracking/deployment 代码均未纳入。本次默认标定明确包含获授权的派生人体骨长，不应再视为不含个人尺寸的中性模板。示例生成器仍只使用代码定义的中性解析人体骨架，与默认标定的人体尺寸来源不同。

发布内容清单见 [RELEASE_CONTENTS](RELEASE_CONTENTS.md)。第三方声明按各自文件边界适用，不能将任一依赖的许可推定为整个项目的许可。
