"""
可视化模块
使用 matplotlib FuncAnimation 实现机械臂运动动画
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle


class Visualizer:
    def __init__(self, arm_dynamics, results, ref_cartesian_traj, dt=0.02):
        self.arm = arm_dynamics
        self.results = results
        self.ref_traj = ref_cartesian_traj
        self.dt = dt
        self.n_steps = results['n_steps']

    def animate(self, interval=20, save_path=None):
        """
        创建动画

        Args:
            interval: 帧间隔 (ms)
            save_path: 保存路径（.mp4 或 .gif），None 则只显示
        """
        fig = plt.figure(figsize=(14, 6))

        # 左侧：机械臂动画
        ax1 = fig.add_subplot(121)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('3-DOF Robot Arm MPC Tracking')

        # 确定绘图范围
        all_x = np.concatenate([self.ref_traj[:, 0], self.results['end_effector'][:, 0]])
        all_y = np.concatenate([self.ref_traj[:, 1], self.results['end_effector'][:, 1]])
        margin = 0.5
        ax1.set_xlim(all_x.min() - margin, all_x.max() + margin)
        ax1.set_ylim(all_y.min() - margin, all_y.max() + margin)

        # 参考轨迹
        ax1.plot(self.ref_traj[:, 0], self.ref_traj[:, 1],
                 'b--', alpha=0.4, label='Reference', linewidth=1.5)

        # 绘图元素
        arm_line, = ax1.plot([], [], 'o-', color='#2196F3', linewidth=3,
                             markersize=8, markerfacecolor='#FF5722',
                             markeredgecolor='black', markeredgewidth=1.5, label='Arm')
        actual_traj, = ax1.plot([], [], 'r-', alpha=0.7, linewidth=1.5,
                                label='Actual')
        end_dot, = ax1.plot([], [], 'go', markersize=6, label='End-effector')
        time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=10,
                             verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax1.legend(loc='upper right', fontsize=8)

        # 右侧：关节角度和力矩
        ax2 = fig.add_subplot(222)
        ax2.set_title('Joint Angles')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Angle (rad)')
        ax2.grid(True, alpha=0.3)

        n = self.n_steps
        t = np.arange(n) * self.dt
        colors = ['#e74c3c', '#2ecc71', '#3498db']
        angle_lines = []
        for i in range(3):
            line, = ax2.plot(t, self.results['x_history'][:n, i],
                             color=colors[i], label=f'θ{i + 1}', linewidth=1.2)
            angle_lines.append(line)
        ax2.legend(fontsize=8)

        ax3 = fig.add_subplot(224)
        ax3.set_title('Joint Torques')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Torque (Nm)')
        ax3.grid(True, alpha=0.3)

        torque_lines = []
        for i in range(3):
            line, = ax3.plot(t[:len(self.results['u_history'])],
                             self.results['u_history'][:, i],
                             color=colors[i], label=f'τ{i + 1}', linewidth=1.2)
            torque_lines.append(line)
        ax3.legend(fontsize=8)

        # 时间指示线
        time_line1 = ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        time_line2 = ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

        fig.tight_layout()

        def init():
            arm_line.set_data([], [])
            actual_traj.set_data([], [])
            end_dot.set_data([], [])
            time_text.set_text('')
            time_line1.set_xdata([0])
            time_line2.set_xdata([0])
            return arm_line, actual_traj, end_dot, time_text, time_line1, time_line2

        def update(frame):
            # 更新机械臂
            theta = self.results['x_history'][frame, :3]
            positions = self.arm.get_joint_positions(theta)
            arm_line.set_data(positions[0], positions[1])

            # 更新实际轨迹
            ee = self.results['end_effector'][:frame + 1]
            actual_traj.set_data(ee[:, 0], ee[:, 1])
            end_dot.set_data([ee[-1, 0]], [ee[-1, 1]])

            # 更新时间
            current_t = frame * self.dt
            time_text.set_text(f't = {current_t:.2f}s')

            # 更新时间指示线
            time_line1.set_xdata([current_t, current_t])
            time_line2.set_xdata([current_t, current_t])

            return arm_line, actual_traj, end_dot, time_text, time_line1, time_line2

        # 降低帧数以加快动画
        step = max(1, self.n_steps // 200)
        frames = list(range(0, self.n_steps, step))

        anim = animation.FuncAnimation(fig, update, frames=frames,
                                       init_func=init, interval=interval,
                                       blit=False)

        if save_path:
            print(f"Saving animation to {save_path}...")
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=30)
            else:
                anim.save(save_path, writer='ffmpeg', fps=30)
            print("Animation saved.")
        else:
            plt.show()

        return anim

    def plot_summary(self, save_path=None):
        """绘制静态汇总图"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        n = self.n_steps
        t = np.arange(n) * self.dt
        colors = ['#e74c3c', '#2ecc71', '#3498db']

        # 1. 轨迹对比
        ax = axes[0, 0]
        ax.plot(self.ref_traj[:, 0], self.ref_traj[:, 1],
                'b--', linewidth=2, label='Reference')
        ee = self.results['end_effector']
        ax.plot(ee[:, 0], ee[:, 1], 'r-', linewidth=1.5, label='Actual')
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('End-effector Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. 跟踪误差
        ax = axes[0, 1]
        ax.plot(t, self.results['tracking_error'] * 1000, 'r-', linewidth=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Error (mm)')
        ax.set_title(f'Tracking Error (RMSE={self.results["rmse"] * 1000:.1f}mm)')
        ax.grid(True, alpha=0.3)

        # 3. 关节角度
        ax = axes[1, 0]
        for i in range(3):
            ax.plot(t, self.results['x_history'][:n, i],
                    color=colors[i], label=f'θ{i + 1}', linewidth=1.2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (rad)')
        ax.set_title('Joint Angles')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. 控制力矩
        ax = axes[1, 1]
        for i in range(3):
            ax.plot(t[:len(self.results['u_history'])],
                    self.results['u_history'][:, i],
                    color=colors[i], label=f'τ{i + 1}', linewidth=1.2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Torque (Nm)')
        ax.set_title('Joint Torques')
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.suptitle('MPC 3-DOF Robot Arm - Simulation Summary', fontsize=14)
        fig.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        plt.close(fig)

        fig.suptitle('MPC 3-DOF Robot Arm - Simulation Summary', fontsize=14)
        fig.tight_layout()
        plt.show()
