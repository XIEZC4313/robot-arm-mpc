"""
仿真主循环
运行 MPC 控制的机械臂轨迹跟踪仿真
"""

import numpy as np


class Simulator:
    def __init__(self, arm_dynamics, mpc_controller, dt=None):
        self.arm = arm_dynamics
        self.mpc = mpc_controller
        self.dt = dt if dt is not None else arm_dynamics.dt

    def run(self, x0, ref_joint_traj, ref_cartesian_traj, dt_ref):
        """
        运行仿真

        Args:
            x0: 初始状态 [θ1, θ2, θ3, dθ1, dθ2, dθ3]
            ref_joint_traj: 参考关节轨迹 (n_points, 3)
            ref_cartesian_traj: 参考笛卡尔轨迹 (n_points, 2)
            dt_ref: 参考轨迹的时间步长

        Returns:
            dict: 包含所有仿真数据
        """
        n_ref = len(ref_joint_traj)
        Np = self.mpc.Np
        dt = self.dt

        # 预计算参考角速度和角加速度
        ref_dtheta = np.zeros_like(ref_joint_traj)
        ref_ddtheta = np.zeros_like(ref_joint_traj)
        for i in range(n_ref - 1):
            ref_dtheta[i] = (ref_joint_traj[i + 1] - ref_joint_traj[i]) / dt_ref
        for i in range(1, n_ref - 1):
            ref_ddtheta[i] = (ref_dtheta[i] - ref_dtheta[i - 1]) / dt_ref

        # 状态记录
        x_history = [x0.copy()]
        u_history = []
        end_effector_history = []
        ref_ee_history = []
        solve_success_count = 0
        total_count = 0

        x_current = x0.copy()

        # 主仿真循环
        step = 0
        while step < n_ref - Np - 1:
            # 构建参考轨迹段 (Np+1, 6)
            ref_segment = np.zeros((Np + 1, 6))
            for k in range(Np + 1):
                idx = min(step + k, n_ref - 1)
                ref_segment[k, :3] = ref_joint_traj[idx]
                ref_segment[k, 3:] = ref_dtheta[idx]

            # MPC 求解
            u_opt, solved = self.mpc.solve(x_current, ref_segment)
            if solved:
                solve_success_count += 1
            total_count += 1

            # 应用控制，推进动力学
            x_next = self.arm.step(x_current, u_opt)

            # 记录数据
            x_history.append(x_next.copy())
            u_history.append(u_opt.copy())

            # 末端位置
            ee_pos = self.arm.get_end_effector(x_current[:3])
            end_effector_history.append(ee_pos)
            ref_ee_history.append(ref_cartesian_traj[step])

            x_current = x_next
            step += 1

        # 性能指标
        end_effector_history = np.array(end_effector_history)
        ref_ee_history = np.array(ref_ee_history)
        tracking_error = np.linalg.norm(end_effector_history - ref_ee_history, axis=1)

        results = {
            'x_history': np.array(x_history),
            'u_history': np.array(u_history),
            'end_effector': end_effector_history,
            'ref_end_effector': ref_ee_history,
            'tracking_error': tracking_error,
            'max_error': np.max(tracking_error),
            'rmse': np.sqrt(np.mean(tracking_error**2)),
            'mean_error': np.mean(tracking_error),
            'solve_rate': solve_success_count / max(total_count, 1),
            'n_steps': step,
        }

        return results
