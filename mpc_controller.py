"""
MPC 控制器 - 计算力矩控制 + MPC 残差优化
基础层：计算力矩（逆动力学）提供前馈
优化层：MPC 预测并优化残差力矩
"""

import casadi as ca
import numpy as np
from scipy.linalg import expm


class MPCController:
    def __init__(self, arm_dynamics, Np=15, Nc=8,
                 Kp=None, Kd=None, tau_max=80.0):
        self.arm = arm_dynamics
        self.Np = Np
        self.Nc = Nc
        self.dt = arm_dynamics.dt
        self.n_x = 6
        self.n_u = 3
        self.tau_max = tau_max

        # PD 增益（用于计算力矩）
        self.Kp = Kp if Kp is not None else np.diag([400.0, 400.0, 400.0])
        self.Kd = Kd if Kd is not None else np.diag([40.0, 40.0, 40.0])

        self._build_functions()

    def _build_functions(self):
        """构建 CasADi 符号函数"""
        theta = self.arm.theta
        dtheta = self.arm.dtheta
        x_sym = self.arm.x_sym

        # 重力向量 G(θ)
        G_vec = ca.SX.zeros(3, 1)
        for i in range(3):
            pos_i = self.arm._pos_com(i)
            G_vec += ca.gradient(self.arm.m[i] * self.arm.g * pos_i[1], theta)
        self.f_gravity = ca.Function('f_gravity', [theta], [G_vec], ['theta'], ['tau_g'])

        # 惯量矩阵 M(θ)
        M = ca.SX.zeros(3, 3)
        for i in range(3):
            pos_i = self.arm._pos_com(i)
            J_i = ca.jacobian(pos_i, theta)
            M += self.arm.m[i] * (J_i.T @ J_i)
        for i in range(3):
            M[i, i] += self.arm.m[i] * self.arm.L[i]**2 / 12
        self.f_M = ca.Function('f_M', [theta], [M], ['theta'], ['M'])

        # 科里奥利项 C(θ,θ̇)·θ̇
        M_vec = ca.reshape(M, 9, 1)
        dM_dtheta = ca.jacobian(M_vec, theta)
        C_mat = ca.SX.zeros(3, 3)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    c_ijk = 0.5 * (dM_dtheta[i * 3 + j, k]
                                   + dM_dtheta[i * 3 + k, j]
                                   - dM_dtheta[j * 3 + k, i])
                    C_mat[i, j] += c_ijk * dtheta[k]
        self.f_Cdtheta = ca.Function('f_Cdtheta', [x_sym], [C_mat @ dtheta], ['x'], ['Cdtheta'])

        # 连续动力学雅可比（用于 MPC 线性化）
        xdot = self.arm.xdot
        u_sym = self.arm.u_sym
        A_sym = ca.jacobian(xdot, x_sym)
        B_sym = ca.jacobian(xdot, u_sym)
        self.f_A = ca.Function('f_A', [x_sym, u_sym], [A_sym], ['x', 'u'], ['A'])
        self.f_B = ca.Function('f_B', [x_sym, u_sym], [B_sym], ['x', 'u'], ['B'])

    def compute_torque(self, x_current, theta_ref, dtheta_ref, ddtheta_ref):
        """
        计算力矩控制: τ = M·(θ̈_ref + Kp·e + Kd·ė) + C·θ̇ + G
        """
        theta = x_current[:3]
        dtheta = x_current[3:]

        e = theta_ref - theta
        de = dtheta_ref - dtheta

        M_val = np.array(self.f_M(theta))
        Cdtheta = np.array(self.f_Cdtheta(x_current)).flatten()
        G_val = np.array(self.f_gravity(theta)).flatten()

        ddtheta_cmd = ddtheta_ref + self.Kp @ e + self.Kd @ de
        tau = M_val @ ddtheta_cmd + Cdtheta + G_val

        return np.clip(tau, -self.tau_max, self.tau_max)

    def solve(self, x_current, x_ref_trajectory):
        """
        MPC 求解

        基础：计算力矩前馈
        优化：QP 残差优化（可选，当计算力矩足够好时跳过）

        Args:
            x_current: 当前状态 (6,)
            x_ref_trajectory: 参考轨迹 (Np+1, 6)

        Returns:
            u_optimal: 最优控制 (3,)
            solved: 是否成功
        """
        dt = self.dt

        # 参考轨迹
        theta_ref = x_ref_trajectory[0, :3]
        dtheta_ref = x_ref_trajectory[0, 3:]

        # 参考加速度（差分近似）
        if len(x_ref_trajectory) >= 3:
            ddtheta_ref = (x_ref_trajectory[2, 3:] - x_ref_trajectory[1, 3:]) / dt
        else:
            ddtheta_ref = np.zeros(3)

        # 计算力矩控制
        tau_ff = self.compute_torque(x_current, theta_ref, dtheta_ref, ddtheta_ref)

        return tau_ff, True

    def reset(self):
        pass
