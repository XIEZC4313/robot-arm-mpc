"""
三轴机械臂 MPC 轨迹跟踪控制 - 入口脚本
"""

import argparse
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from arm_dynamics import ArmDynamics
from trajectory import TrajectoryGenerator
from mpc_controller import MPCController
from simulator import Simulator
from visualizer import Visualizer


def main():
    parser = argparse.ArgumentParser(description='3-DOF Robot Arm MPC Control')
    parser.add_argument('--trajectory', type=str, default='circle',
                        choices=['circle', 'figure_eight', 'square'],
                        help='Reference trajectory type')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Simulation duration (seconds)')
    parser.add_argument('--dt', type=float, default=0.02,
                        help='Control time step (seconds)')
    parser.add_argument('--Np', type=int, default=20,
                        help='Prediction horizon')
    parser.add_argument('--Nc', type=int, default=10,
                        help='Control horizon')
    parser.add_argument('--save', type=str, default=None,
                        help='Save animation to file (.mp4 or .gif)')
    parser.add_argument('--no-animate', action='store_true',
                        help='Skip animation, show summary plot only')
    parser.add_argument('--save-plot', type=str, default='summary.png',
                        help='Save summary plot to file')
    parser.add_argument('--radius', type=float, default=0.5,
                        help='Trajectory radius (m)')

    args = parser.parse_args()

    print("=" * 50)
    print("3-DOF Robot Arm MPC Trajectory Tracking")
    print("=" * 50)
    print(f"Trajectory: {args.trajectory}")
    print(f"Duration: {args.duration}s, dt: {args.dt}s")
    print(f"MPC: Np={args.Np}, Nc={args.Nc}")
    print()

    # 1. 创建机械臂模型
    print("[1/5] Building arm dynamics model...")
    arm = ArmDynamics(link_lengths=[1.0, 0.8, 0.6],
                      link_masses=[2.0, 1.5, 1.0],
                      dt=args.dt)

    # 2. 生成参考轨迹
    print("[2/5] Generating reference trajectory...")
    traj_gen = TrajectoryGenerator(arm, center=[1.5, 0.5], radius=args.radius)
    n_points = int(args.duration / args.dt)
    t_ref, ref_joint, ref_cartesian = traj_gen.generate(
        args.trajectory, n_points, args.duration)
    dt_ref = t_ref[1] - t_ref[0]
    print(f"  Trajectory points: {n_points}")

    # 3. 创建 MPC 控制器
    print("[3/5] Building MPC controller...")
    mpc = MPCController(arm, Np=args.Np, Nc=args.Nc)

    # 4. 运行仿真
    print("[4/5] Running simulation...")
    sim = Simulator(arm, mpc)

    # 初始状态：使用参考轨迹的起始关节角，角速度为零
    x0 = np.zeros(6)
    x0[:3] = ref_joint[0]

    start_time = time.time()
    results = sim.run(x0, ref_joint, ref_cartesian, dt_ref)
    sim_time = time.time() - start_time

    print(f"\n  Simulation completed in {sim_time:.1f}s")
    print(f"  Steps: {results['n_steps']}")
    print(f"  MPC solve rate: {results['solve_rate'] * 100:.1f}%")
    print(f"  Max tracking error: {results['max_error'] * 1000:.2f} mm")
    print(f"  Mean tracking error: {results['mean_error'] * 1000:.2f} mm")
    print(f"  RMSE: {results['rmse'] * 1000:.2f} mm")

    # 5. 可视化
    print("\n[5/5] Visualizing...")
    viz = Visualizer(arm, results, ref_cartesian, args.dt)

    if args.no_animate:
        if args.save_plot:
            viz.plot_summary(save_path=args.save_plot)
        else:
            viz.plot_summary()
    else:
        viz.animate(save_path=args.save)

    print("\nDone!")


if __name__ == '__main__':
    main()
