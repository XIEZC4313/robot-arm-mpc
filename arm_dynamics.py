"""
三连杆平面机械臂动力学模型
状态: x = [θ1, θ2, θ3, dθ1, dθ2, dθ3]
控制: u = [τ1, τ2, τ3]
动力学: M(θ)θ̈ + C(θ,θ̇)θ̇ + G(θ) = τ
"""

import casadi as ca
import numpy as np


class ArmDynamics:
    def __init__(self, link_lengths=None, link_masses=None, dt=0.02):
        # 默认连杆参数
        self.L = link_lengths if link_lengths is not None else [1.0, 0.8, 0.6]
        self.m = link_masses if link_masses is not None else [2.0, 1.5, 1.0]
        self.dt = dt
        self.g = 9.81
        self.n_joints = 3
        self.n_states = 6
        self.n_controls = 3

        # CasADi 符号变量
        self.theta = ca.SX.sym('theta', 3)
        self.dtheta = ca.SX.sym('dtheta', 3)
        self.x_sym = ca.vertcat(self.theta, self.dtheta)
        self.u_sym = ca.SX.sym('u', 3)

        # 构建符号动力学
        self._build_dynamics()

    def _pos_com(self, link_idx):
        """第 link_idx 个连杆质心位置（假设质心在连杆中点）"""
        L = self.L
        theta = self.theta
        if link_idx == 0:
            x = L[0] / 2 * ca.cos(theta[0])
            y = L[0] / 2 * ca.sin(theta[0])
        elif link_idx == 1:
            x = L[0] * ca.cos(theta[0]) + L[1] / 2 * ca.cos(theta[0] + theta[1])
            y = L[0] * ca.sin(theta[0]) + L[1] / 2 * ca.sin(theta[0] + theta[1])
        else:
            x = (L[0] * ca.cos(theta[0])
                 + L[1] * ca.cos(theta[0] + theta[1])
                 + L[2] / 2 * ca.cos(theta[0] + theta[1] + theta[2]))
            y = (L[0] * ca.sin(theta[0])
                 + L[1] * ca.sin(theta[0] + theta[1])
                 + L[2] / 2 * ca.sin(theta[0] + theta[1] + theta[2]))
        return ca.vertcat(x, y)

    def _vel_com(self, link_idx):
        """第 link_idx 个连杆质心速度（通过雅可比矩阵计算）"""
        pos = self._pos_com(link_idx)
        J = ca.jacobian(pos, self.theta)
        return J @ self.dtheta

    def _build_dynamics(self):
        """构建完整的动力学方程 M(θ)θ̈ + C(θ,θ̇)θ̇ + G(θ) = τ"""
        theta = self.theta
        dtheta = self.dtheta
        L = self.L
        m = self.m
        g = self.g

        # 惯量矩阵 M(θ)
        M = ca.SX.zeros(3, 3)
        for i in range(3):
            # 每个连杆对 M 的贡献：m_i * J_i^T * J_i
            pos_i = self._pos_com(i)
            J_i = ca.jacobian(pos_i, theta)
            M += m[i] * (J_i.T @ J_i)

        # 添加连杆自身的转动惯量 (I = 1/12 * m * L^2)
        for i in range(3):
            I_i = m[i] * L[i]**2 / 12
            M[i, i] += I_i

        # 重力向量 G(θ)
        G_vec = ca.SX.zeros(3, 1)
        for i in range(3):
            pos_i = self._pos_com(i)
            # 势能对 θ 的偏导
            G_vec += ca.gradient(m[i] * g * pos_i[1], theta)

        # 科里奥利和离心力矩阵 C(θ,θ̇)
        # 一次性计算 M 对 θ 的雅可比，避免 27 次单独 ca.jacobian 调用
        M_vec = ca.reshape(M, 9, 1)
        dM_dtheta = ca.jacobian(M_vec, theta)  # (9, 3): dM_vec/dθ_k
        # dM_dtheta[i*3+j, k] = ∂M_ij/∂θ_k

        C_mat = ca.SX.zeros(3, 3)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    c_ijk = 0.5 * (dM_dtheta[i * 3 + j, k]
                                   + dM_dtheta[i * 3 + k, j]
                                   - dM_dtheta[j * 3 + k, i])
                    C_mat[i, j] += c_ijk * dtheta[k]

        # 加速度: θ̈ = M^(-1) * (τ - C*θ̇ - G)
        ddtheta = ca.solve(M, self.u_sym - C_mat @ dtheta - G_vec)

        # 连续动力学: dx/dt = f(x, u)
        self.xdot = ca.vertcat(dtheta, ddtheta)
        self.f_continuous = ca.Function('f_continuous',
                                        [self.x_sym, self.u_sym],
                                        [self.xdot],
                                        ['x', 'u'], ['xdot'])

        # Euler 离散化（MPC 内使用，符号图更简洁）
        x_next_euler = self.x_sym + self.dt * self.xdot
        self.f_discrete = ca.Function('f_discrete',
                                      [self.x_sym, self.u_sym],
                                      [x_next_euler],
                                      ['x', 'u'], ['x_next'])

        # RK4 离散化（前向仿真使用，精度更高）
        k1 = self.f_continuous(self.x_sym, self.u_sym)
        k2 = self.f_continuous(self.x_sym + self.dt / 2 * k1, self.u_sym)
        k3 = self.f_continuous(self.x_sym + self.dt / 2 * k2, self.u_sym)
        k4 = self.f_continuous(self.x_sym + self.dt * k3, self.u_sym)
        x_next_rk4 = self.x_sym + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.f_discrete_rk4 = ca.Function('f_discrete_rk4',
                                           [self.x_sym, self.u_sym],
                                           [x_next_rk4],
                                           ['x', 'u'], ['x_next'])

        # 正运动学（末端位置）
        end_x = ca.SX(0)
        end_y = ca.SX(0)
        cum_angle = ca.SX(0)
        for i in range(3):
            cum_angle += theta[i]
            end_x += L[i] * ca.cos(cum_angle)
            end_y += L[i] * ca.sin(cum_angle)
        self.end_effector = ca.Function('end_effector',
                                        [self.theta],
                                        [ca.vertcat(end_x, end_y)],
                                        ['theta'], ['pos'])

        # 各关节位置（用于可视化）
        joint_positions = [ca.vertcat(ca.SX(0), ca.SX(0))]  # 基座
        cum_angle = ca.SX(0)
        for i in range(3):
            cum_angle += theta[i]
            jx = joint_positions[-1][0] + L[i] * ca.cos(cum_angle)
            jy = joint_positions[-1][1] + L[i] * ca.sin(cum_angle)
            joint_positions.append(ca.vertcat(jx, jy))
        self.joint_pos = ca.Function('joint_pos',
                                     [self.theta],
                                     [ca.horzcat(*joint_positions)],
                                     ['theta'], ['positions'])

    def step(self, x, u):
        """前向仿真一步（使用 RK4）"""
        x_next = self.f_discrete_rk4(x, u)
        return np.array(x_next).flatten()

    def get_end_effector(self, theta):
        """计算末端执行器位置"""
        pos = self.end_effector(theta)
        return np.array(pos).flatten()

    def get_joint_positions(self, theta):
        """获取各关节位置（含基座和末端）"""
        pos = self.joint_pos(theta)
        return np.array(pos)  # shape: (2, 4)

    def get_M(self, theta):
        """获取当前惯量矩阵（用于分析）"""
        theta_sym = ca.SX.sym('theta', 3)
        dtheta_sym = ca.SX.sym('dtheta', 3)
        pos_com = [self._pos_com(i) for i in range(3)]
        M_val = ca.SX.zeros(3, 3)
        for i in range(3):
            J_i = ca.jacobian(pos_com[i], theta_sym)
            M_val += self.m[i] * (J_i.T @ J_i)
        for i in range(3):
            M_val[i, i] += self.m[i] * self.L[i]**2 / 12
        f_M = ca.Function('f_M', [theta_sym], [M_val])
        return np.array(f_M(theta))
