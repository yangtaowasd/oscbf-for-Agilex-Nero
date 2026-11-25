![oscbf_gif](https://github.com/user-attachments/assets/43b615a3-dcec-4fc4-bc7f-2ba674562323)

# Operational Space Control Barrier Functions

Code for *"Safe, Task-Consistent Manipulation with Operational Space Control Barrier Functions"* -- Daniel Morton and Marco Pavone

Accepted to IROS 2025, Hangzhou

[![Paper](http://img.shields.io/badge/arXiv-2503.06736-B31B1B.svg)](https://arxiv.org/abs/2503.06736)

## What is OSCBF?

This is a safe, high-performance, and easy-to-use controller / safety filter for robotic manipulators.

With OSCBF, you can...
- Operate at kilohertz speed even with over 400 active safety constraints
- Design safety constraints (barrier functions) easily via CBFpy
- Enforce safety on both torque-controlled and velocity-controlled robots
- Either apply a safety filter on top of your existing controller, or use our provided controller

In general, this will be especialy useful for enforcing safety during **teleoperation** or while executing **learned policies**

For more details and videos, check out the [project webpage](https://stanfordasl.github.io/oscbf/) as well as the [CBFpy documentation](https://danielpmorton.github.io/cbfpy/).


## AgileX Nero 七轴法兰版

在本机激活 `oscbf` 环境，从本仓库目录运行原项目的 `dynamic_motion` 示例，即可在 PyBullet 窗口观察 Nero。红球是法兰目标，绿色盒子是法兰位置的安全区域。Nero 图形窗口默认由你控制红球。拖动右侧 Parameters 面板里的 `Base X/Y/Z (mm)` 三个滑块，可以设置目标相对于底座原点的绝对坐标。滑块初值为启动时法兰的基座坐标。红球以最高 8 cm/s 移向设定位置，法兰朝向可随运动自动调整，机械臂受安全盒约束。窗口可用 Ctrl+C 结束。首次启动需要等待 JAX 编译控制器。

手动模式的绿色盒子覆盖基座坐标 X −470～470 mm、Y −250～250 mm、Z 150～570 mm，三个滑块均以底座原点为基准。当前 TCP 位于法兰安装面，要求其坐标模长 √(X²+Y²+Z²) < 700 mm。控制器对 TCP 采用 699 mm 球面边界，留出 1 mm 数值余量，TCP 的允许区域为盒子与球体的交集，控制器对第 3～7 连杆施加外形边界约束。J1～J5 使用按显示与碰撞网格拟合的胶囊体，额外余量为 0 mm。J6、J7 使用球体，基础半径分别由 36.58、40.36 mm 放大 10%，并各增加 6 mm 余量，实际约束半径分别为 46.238 mm 和 50.396 mm。包络体随关节姿态变化，肘部与末端外形均参与边界限制。红球可移到该区域外，机械臂由控制器限制在边界内。为便于操作越界目标，红球滑块在盒子各面外延伸 100 mm，范围为 X ±570 mm、Y ±350 mm、Z 50～670 mm。手动仿真以 TCP 约 (−400, −100, 400) mm 的屈肘姿态启动。法兰朝向允许自动调整，实际可达位置仍取决于关节构型与避障约束。

蓝色避障球位于底座坐标 (−300, 0, 250) mm，直径 70 mm。控制器对全部七个运动连杆的包络体施加避障约束，其中 J6、J7 预留 6 mm 间隙，J1～J5 的额外间隙为 0 mm。初始姿态已避开蓝球。离散仿真和 QP 软约束会产生数值误差，已运行的极限测试中出现过约 0.027 mm 的包络数值越界，对应实际碰撞网格与蓝球仍有间隙。

手动仿真采用位置优先控制，使用阻尼逆避免奇异附近的求逆放大。冗余关节根据边界间隙、关节限位和位置雅可比的可操作性主动调整姿态。关节上下限与位置雅可比的可操作性下限同时进入约束求解，蓝球和绿框限制参与同一控制循环。目标停止后若 TCP 持续停滞，控制器尝试在这些约束内重求关节姿态，并通过有限速的关节运动恢复跟踪。恢复过程允许 TCP 暂时偏离目标路径。终端输出 `Nero posture recovery found a feasible solution` 时，表示已找到候选恢复姿态，最终到达情况仍受约束和恢复路径影响。

```bash
conda activate oscbf
python -m oscbf.examples.dynamic_motion --robot nero --target manual --control_method torque
```

将 `torque` 改为 `velocity` 可以用相同场景比较速度控制。将 `--target manual` 改为 `--target trajectory` 可回放自动轨迹（X 方向振幅 80 mm，局部安全盒半宽为 X 40、Y 80、Z 80 mm）。自动轨迹模式下添加 `--headless --steps 3000` 可在无窗口模式下仿真 3 秒，输出目标和法兰的最大越界距离。仿真步长为 1 ms，无窗口模式默认选择自动轨迹。

以下命令运行回归测试和独立的七轴多约束位姿跟踪示例。

```bash
python -m unittest test.test_nero_manual_workspace test.test_nero_dynamics test.test_nero_control
python -m oscbf.examples.nero --steps 1000
```

`load_nero()` 从随包提供的 URDF 创建固定底座、竖直安装、空载法兰模型。任务坐标系为 `flange_link`。示例从屈肘初姿出发，保持法兰方向，沿基座 y 轴做 2 cm 往返运动，每步 2 ms。仿真保留法兰固定关节，以使用与动力学对照测试相同的惯性参数。

控制器用七轴总力矩跟踪位姿，用二次规划约束关节位置、速度、输入力矩、雅可比可操作度，以及七个运动连杆包络球的工作空间。默认工作空间在基座坐标下为 x/y 各 ±0.8 m、z 为 0 至 0.9 m。包络球用于工作空间边界，此配置未实现自碰撞或场景障碍物避让。安装夹爪、工具或负载后需更新惯性、碰撞几何和 TCP。

ROS 2 仿真与实机接口见同级工作区的 [Nero 运行说明](../oscbf_hardware_ws/README.md#agilex-nero)。

## How do I use this?

Check out the `examples/` folder for interactive demos in Pybullet! This is the best place to start

If you're applying this to a different robot, you'll need to provide a URDF -- we can parse the kinematics and dynamics from that. Some notes:
- I haven't written an MJCF parser yet, but it should be feasible
- All joints that shouldn't be controlled as part of the kinematic chain should be set to `fixed` (gripper joints, for instance). Since you probably want to still be able to use the gripper joints in simulation, the best way to handle this is to make a copy of the URDF: load the non-fixed one in sim, and parse the fixed one with OSCBF
- I manually defined the collision model for the Franka. There are probably better ways to parse or generate this data from meshes, but I haven't done it yet. For now, I'd recommend doing the same for your robot.

## FAQ

- "I already have a well-tuned operational space controller! I don't want to replace that"
  - You can use OSCBF as a safety filter on top of your existing controller!
- "I'm working with a mobile manipulator. Does this still work?"
  - Yes! We support prismatic and revolute joints, so adding a mobile base just adds 3DOF (PPR) to the beginning of the kinematic chain.
- "I'm controlling joint-space motions -- does this still apply?"
  - Depending on the task, you will likely want to modify the objective function. But, the robot dynamics and CBF formulation will still be useful!


## Installation

A virtual environment is optional, but highly recommended. For `pyenv` installation instructions, see [here](https://danielpmorton.github.io/cbfpy/pyenv).

```
git clone https://github.com/stanfordasl/oscbf
cd oscbf
pip install -e . scipy
```

Note: This code will work with most versions of `jax`, but there seems to have been a CPU slowdown introduced in version `0.4.32`. To avoid this, I use version `0.4.30`, which is the version indicated in this repo's `pyproject.toml` as well. However, feel free to use any version you like.

This has been tested on Python 3.10 and 3.11, on Ubuntu 22.04.


## Documentation

See the CBFpy documentation, available at [this link](https://danielpmorton.github.io/cbfpy)

Nero 手动仿真的入口、绿框范围、蓝球位置、TCP 距离限制与滑块参数位于演示模块。[dynamic_motion.py](oscbf/examples/dynamic_motion.py)

连杆胶囊体、球体参数及各连杆余量位于碰撞模型模块。控制、姿态调整和停滞恢复位于 Nero 控制模块。[nero_collision_model.py](oscbf/core/nero_collision_model.py)、[nero_control.py](oscbf/core/nero_control.py)

操作空间控制器与阻尼求逆实现位于控制器模块。手动仿真回归测试覆盖网格包络、初始约束、直臂姿态有限输出及恢复求解。[controllers.py](oscbf/core/controllers.py)、[test_nero_manual_workspace.py](test/test_nero_manual_workspace.py)


## Hardware / ROS2 

The code used to run the Franka Panda hardware experiments is available at [this link](https://github.com/StanfordASL/oscbf_hardware_ws)


## Citation
```
@inproceedings{morton2025oscbf,
  author={Morton, Daniel and Pavone, Marco},
  booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Safe, Task-Consistent Manipulation with Operational Space Control Barrier Functions}, 
  year={2025},
  pages={187-194},
  doi={10.1109/IROS60139.2025.11246389}
}
```

