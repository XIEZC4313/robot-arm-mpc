"""
参考轨迹生成模块
生成末端执行器的笛卡尔轨迹，并通过逆运动学转换为关节空间轨迹
"""

import numpy as np


class TrajectoryGenerator:
    def __init__(self, arm_dynamics, center=None, radius=0.5):
        self.arm = arm_dynamics
        self.center = center if center is not None else [1.5, 0.5]
        self.radius = radius
        self.L = arm_dynamics.L

    def _ik_3dof(self, x_target, y_target, theta3=0.0):
        """
        三连杆平面机械臂逆运动学
        给定末端位置和第三个关节角（冗余自由度），求 θ1, θ2
        """
        L1, L2, L3 = self.L

        # 先计算第三个连杆末端（腕部）的位置
        wx = x_target - L3 * np.cos(theta3)
        wy = y_target - L3 * np.sin(theta3)

        # 二连杆逆运动学求 θ1, θ2
        dist_sq = wx**2 + wy**2
        cos_q2 = (dist_sq - L1**2 - L2**2) / (2 * L1 * L2)
        cos_q2 = np.clip(cos_q2, -1.0, 1.0)

        q2 = np.arccos(cos_q2)
        # 选择肘部向上的解

        k1 = L1 + L2 * np.cos(q2)
        k2 = L2 * np.sin(q2)
        q1 = np.arctan2(wy, wx) - np.arctan2(k2, k1)

        # θ3 是相对于第二个连杆的角度
        q3 = theta3 - q1 - q2

        return np.array([q1, q2, q3])

    def generate_circle(self, n_points, duration):
        """生成圆形轨迹"""
        t = np.linspace(0, duration, n_points, endpoint=False)
        cx, cy = self.center
        r = self.radius

        x_traj = cx + r * np.cos(2 * np.pi * t / duration)
        y_traj = cy + r * np.sin(2 * np.pi * t / duration)

        # 冗余自由度：θ3 缓慢变化
        theta3_traj = 0.3 * np.sin(2 * np.pi * t / duration)

        joint_traj = np.zeros((n_points, 3))
        for i in range(n_points):
            joint_traj[i] = self._ik_3dof(x_traj[i], y_traj[i], theta3_traj[i])

        return t, joint_traj, np.column_stack([x_traj, y_traj])

    def generate_figure_eight(self, n_points, duration):
        """生成 8 字形轨迹"""
        t = np.linspace(0, duration, n_points, endpoint=False)
        cx, cy = self.center
        r = self.radius

        x_traj = cx + r * np.sin(2 * np.pi * t / duration)
        y_traj = cy + r * np.sin(4 * np.pi * t / duration) / 2

        theta3_traj = 0.2 * np.cos(2 * np.pi * t / duration)

        joint_traj = np.zeros((n_points, 3))
        for i in range(n_points):
            joint_traj[i] = self._ik_3dof(x_traj[i], y_traj[i], theta3_traj[i])

        return t, joint_traj, np.column_stack([x_traj, y_traj])

    def generate_square(self, n_points, duration):
        """生成方形轨迹（带圆角平滑）"""
        t = np.linspace(0, duration, n_points, endpoint=False)
        cx, cy = self.center
        r = self.radius

        x_traj = np.zeros(n_points)
        y_traj = np.zeros(n_points)

        for i, ti in enumerate(t):
            phase = (ti / duration) % 1.0
            # 方形参数化：每条边占 1/4 周期
            if phase < 0.25:
                s = phase / 0.25
                x_traj[i] = cx + r * (1 - 2 * s)
                y_traj[i] = cy + r
            elif phase < 0.5:
                s = (phase - 0.25) / 0.25
                x_traj[i] = cx - r
                y_traj[i] = cy + r * (1 - 2 * s)
            elif phase < 0.75:
                s = (phase - 0.5) / 0.25
                x_traj[i] = cx - r * (1 - 2 * s)
                y_traj[i] = cy - r
            else:
                s = (phase - 0.75) / 0.25
                x_traj[i] = cx + r
                y_traj[i] = cy - r * (1 - 2 * s)

        theta3_traj = np.zeros(n_points)

        joint_traj = np.zeros((n_points, 3))
        for i in range(n_points):
            joint_traj[i] = self._ik_3dof(x_traj[i], y_traj[i], theta3_traj[i])

        return t, joint_traj, np.column_stack([x_traj, y_traj])

    def generate(self, trajectory_type='circle', n_points=500, duration=10.0):
        """统一接口生成轨迹"""
        generators = {
            'circle': self.generate_circle,
            'figure_eight': self.generate_figure_eight,
            'square': self.generate_square,
        }
        if trajectory_type not in generators:
            raise ValueError(f"未知轨迹类型: {trajectory_type}，可选: {list(generators.keys())}")

        return generators[trajectory_type](n_points, duration)
