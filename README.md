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

在本机激活 `oscbf` 环境，从本仓库目录运行原项目的 `dynamic_motion` 示例，即可在 PyBullet 窗口观察 Nero。红球是法兰目标，绿色盒子是法兰位置的安全区域。Nero 图形窗口默认由你控制红球。拖动右侧 Parameters 面板里的 `Target X/Y/Z (mm)` 三个滑块，可以相对初始法兰位置沿基座三轴移动目标。滑块归零即可返回起点。红球以最高 8 cm/s 移向设定位置，法兰保持初始朝向并受安全盒约束。窗口可用 Ctrl+C 结束。

```bash
conda activate oscbf
python -m oscbf.examples.dynamic_motion --robot nero --control_method torque
```

添加 `--target trajectory` 可回放原来的自动轨迹（x 方向振幅 8 cm、安全盒半宽 4 cm）。将 `torque` 改为 `velocity` 可以用相同场景比较速度控制。添加 `--headless --steps 3000` 可在无窗口模式下仿真 3 秒，输出目标和法兰的最大越界距离。该实验沿用原项目的末端安全盒约束。

另一个独立示例用于七轴多约束控制器的跟踪验证：

```bash
python -m unittest test.test_nero_dynamics test.test_nero_control
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
pip install -e .
```

Note: This code will work with most versions of `jax`, but there seems to have been a CPU slowdown introduced in version `0.4.32`. To avoid this, I use version `0.4.30`, which is the version indicated in this repo's `pyproject.toml` as well. However, feel free to use any version you like.

This has been tested on Python 3.10 and 3.11, on Ubuntu 22.04.


## Documentation

See the CBFpy documentation, available at [this link](https://danielpmorton.github.io/cbfpy)


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

