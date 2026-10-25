from math import factorial

import numpy as np

from pydrake.systems.framework import (
    LeafSystem,
    BasicVector,
    Context,
    ContinuousState,
)


# Planar piecewise constant curvature (PCC) kinematics, written with complex numbers z = x + i*y.
# A segment of length L bent by an angle q has the point at arc length s = sigma*L in
#     z = L * sigma * g(sigma*q),     g(a) = int_0^1 exp(i*a*t) dt = sin(a)/a + i*(1 - cos(a))/a
# in the frame of the segment base. g and its derivatives are entire functions: close to a = 0
# (straight segment) they are evaluated with their series instead of the closed form.

_SERIES_SWITCH = 0.5
_N_SERIES = 16
_n = np.arange(_N_SERIES)
_FACTORIAL = np.array([float(factorial(k)) for k in range(_N_SERIES)])


def g_and_derivatives(alpha):
    """g(alpha), g'(alpha), g''(alpha) for an array of angles, as complex arrays."""
    alpha = np.asarray(alpha, dtype=float)
    small = np.abs(alpha) < _SERIES_SWITCH
    a = np.where(small, 1.0, alpha)  # np.where evaluates both branches: keep the closed form finite

    # I_k = int_0^1 t^k exp(i a t) dt, integrating by parts: I_k = (exp(i a) - k I_{k-1}) / (i a)
    ia = 1j * a
    e = np.exp(ia)
    I0 = (e - 1) / ia
    I1 = (e - I0) / ia
    I2 = (e - 2 * I1) / ia

    # series: I_k = sum_n (i a)^n / (n! (n + k + 1))
    p = (1j * alpha[..., None]) ** _n / _FACTORIAL
    I0 = np.where(small, (p / (_n + 1)).sum(-1), I0)
    I1 = np.where(small, (p / (_n + 2)).sum(-1), I1)
    I2 = np.where(small, (p / (_n + 3)).sum(-1), I2)

    # g^(k) = i^k I_k
    return I0, 1j * I1, -I2


def pcc_kinematics(q, dq, L, seg, sigma):
    """Points along the backbone of a planar PCC arm with its base at the origin, pointing +x.

    Point k lies in segment seg[k], at the fraction sigma[k] of its length (sigma = 1 is the tip
    of the segment). Returns
        z (N,)          positions, complex x + i*y
        Jz (N, n)       dz/dq
        bias (N,)       Jz_dot @ dq: acceleration of the points when q_ddot = 0
        phi (N,)        orientation of the backbone at the points
        Jphi (N, n)     dphi/dq (constant)
    """
    q, dq, L = np.asarray(q, float), np.asarray(dq, float), np.asarray(L, float)
    seg, sigma = np.asarray(seg, int), np.asarray(sigma, float)
    n, N = len(q), len(seg)

    Phi = np.concatenate(([0.0], np.cumsum(q)))   # Phi[j]: orientation of the base of segment j
    W = np.concatenate(([0.0], np.cumsum(dq)))    # W[j]: its angular velocity
    rot = np.exp(1j * Phi[:n])

    # whole segments: T_j is the tip of segment j relative to its base
    g0, g1, g2 = g_and_derivatives(q)
    T = L * rot * g0
    dT = np.zeros((n, n), dtype=complex)
    for j in range(n):
        dT[j, :j] = 1j * T[j]
        dT[j, j] = L[j] * rot[j] * g1[j]
    bT = L * rot * (-(W[:n]**2) * g0 + 2j * W[:n] * dq * g1 + dq**2 * g2)

    # bases of the segments: B_j = sum_{l<j} T_l
    B = np.concatenate(([0j], np.cumsum(T)))
    dB = np.vstack((np.zeros((1, n), dtype=complex), np.cumsum(dT, axis=0)))
    bB = np.concatenate(([0j], np.cumsum(bT)))

    # the points: base of their segment + partial segment
    rows = np.arange(N)
    before = np.arange(n)[None, :] < seg[:, None]
    Ls, rs, Ws, dqs = L[seg], rot[seg], W[seg], dq[seg]
    h0, h1, h2 = g_and_derivatives(sigma * q[seg])
    P = Ls * sigma * rs * h0

    z = B[seg] + P
    Jz = dB[seg] + np.where(before, 1j * P[:, None], 0)
    Jz[rows, seg] += Ls * sigma**2 * rs * h1
    bias = bB[seg] + Ls * sigma * rs * (-(Ws**2) * h0 + 2j * Ws * sigma * dqs * h1 + sigma**2 * dqs**2 * h2)

    phi = Phi[seg] + sigma * q[seg]
    Jphi = np.where(before, 1.0, 0.0)
    Jphi[rows, seg] = sigma

    return z, Jz, bias, phi, Jphi


class PlanarSoftPCC(LeafSystem):

    """Planar soft arm with PCC segments, driven by a force at the tip: dynamic in q.

    q_i is the bending angle of segment i. The mass along the backbone is lumped at Gauss-Legendre
    nodes, so M(q) and C(q, q_dot) are the exact Lagrangian terms of that discretised arm.
    """

    def __init__(self, n_segments=3, stiffness=1.0, damping=0.05, gravity=0.0, q_rest=None,
                 n_nodes=6, n_backbone=10):

        LeafSystem.__init__(self)

        self.n = n_segments
        self.n_nodes = n_nodes
        self.n_backbone = n_backbone  # backbone points per segment, for the "backbone" output
        self.init_dynamic_model_parameters(stiffness, damping, gravity, q_rest)

        self.M = np.zeros((self.n, self.n))
        self.C = np.zeros((self.n))
        self.passive_forces = np.zeros((self.n))   # elasticity + damping + gravity
        self.ActuationMatrix = np.zeros((self.n, 2))
        self.Jac = np.zeros((2, self.n))
        self.Jac_dot_qdot = np.zeros((2))

        group_index = self.DeclareContinuousState(2 * self.n)

        self.DeclareVectorInputPort("F", BasicVector(2))
        self.DeclareStateOutputPort("state", group_index)
        self.DeclareVectorOutputPort("EE_state",
                                     BasicVector(4),
                                     self.CalcEEState,
                                     prerequisites_of_calc={self.all_state_ticket()})
        self.DeclareVectorOutputPort("backbone",
                                     BasicVector(2 * self.num_backbone_points()),
                                     self.CalcBackbone,
                                     prerequisites_of_calc={self.all_state_ticket()})

    def init_dynamic_model_parameters(self, stiffness, damping, gravity, q_rest):

        n = self.n
        self.L = np.full(n, 1.0 / n)        # total length 1 m
        self.m = np.full(n, 1.0 / n)        # total mass 1 kg
        self.r = 0.02                       # radius of the cross-section
        self.K = np.broadcast_to(np.asarray(stiffness, float), (n,)).copy()   # bending stiffness EI/L
        self.D = np.broadcast_to(np.asarray(damping, float), (n,)).copy()
        self.g = gravity                    # along -y; 0 = horizontal plane
        self.q_rest = np.zeros(n) if q_rest is None else np.asarray(q_rest, float)

        # Gauss-Legendre nodes on each segment
        x, w = np.polynomial.legendre.leggauss(self.n_nodes)
        self.node_seg = np.repeat(np.arange(n), self.n_nodes)
        self.node_sigma = np.tile((x + 1) / 2, n)
        self.node_mass = self.m[self.node_seg] * np.tile(w / 2, n)
        # rotational inertia of a disc cross-section, per unit length: rho * r^2 / 4
        self.node_inertia = self.node_mass * self.r**2 / 4

    def DoCalcTimeDerivatives(self, context: Context, derivatives_out: ContinuousState):

        F = self.get_input_port(0).Eval(context)
        state = context.get_continuous_state_vector().CopyToVector()

        q_dot = state[self.n:]
        self.eval_dyn_model(state_vect=state)
        q_dot_dot = np.linalg.solve(self.M, (self.ActuationMatrix @ F) - self.C - self.passive_forces)
        derivatives = np.concatenate((q_dot, q_dot_dot))
        derivatives_out.get_mutable_vector().SetFromVector(derivatives)

    def eval_dyn_model(self, state_vect):
        self._eval_kinematics(state_vect)
        self._eval_M_of_q()
        self._eval_C_of_q()
        self._eval_passive_forces(state_vect)
        self._eval_jac_ee_xy()
        self._eval_actuation_matrix()

    def _eval_kinematics(self, state_vect):

        q, dq = state_vect[:self.n], state_vect[self.n:]
        seg = np.append(self.node_seg, self.n - 1)          # nodes + tip
        sigma = np.append(self.node_sigma, 1.0)
        z, Jz, bias, _, Jphi = pcc_kinematics(q, dq, self.L, seg, sigma)
        self.nodes = (z[:-1], Jz[:-1], bias[:-1], Jphi[:-1])
        self.tip = (z[-1], Jz[-1], bias[-1])

    def _eval_M_of_q(self):

        _, Jz, _, Jphi = self.nodes
        # sum_k m_k J_k^T J_k, with J_k the real 2 x n Jacobian: Re(Jz^H Jz)
        self.M = (Jz.conj().T * self.node_mass) @ Jz
        self.M = self.M.real + (Jphi.T * self.node_inertia) @ Jphi

    def _eval_C_of_q(self):

        # Coriolis and centrifugal terms: sum_k m_k J_k^T (J_dot_k q_dot); Jphi is constant
        _, Jz, bias, _ = self.nodes
        self.C = ((Jz.conj().T * self.node_mass) @ bias).real

    def _eval_passive_forces(self, state_vect):

        q, dq = state_vect[:self.n], state_vect[self.n:]
        _, Jz, _, _ = self.nodes
        gravity = self.g * (Jz.imag.T @ self.node_mass)
        self.passive_forces = self.K * (q - self.q_rest) + self.D * dq + gravity

    def _eval_jac_ee_xy(self):

        _, Jz, bias = self.tip
        self.Jac = np.vstack((Jz.real, Jz.imag))
        self.Jac_dot_qdot = np.array([bias.real, bias.imag])

    def _eval_actuation_matrix(self):

        # force applied at the EE: by kineto-static duality the actuation matrix is J^T(q)
        self.ActuationMatrix = self.Jac.T.copy()

    def forward_kin(self, state_vect):
        z, _, _, _, _ = pcc_kinematics(state_vect[:self.n], np.zeros(self.n), self.L, [self.n - 1], [1.0])
        return z[0].real, z[0].imag

    def num_backbone_points(self):
        return self.n * self.n_backbone + 1

    def backbone(self, q):
        """(N, 2) points along the backbone, from the base to the tip."""
        seg = np.concatenate(([0], np.repeat(np.arange(self.n), self.n_backbone)))
        sigma = np.concatenate(([0.0], np.tile(np.arange(1, self.n_backbone + 1) / self.n_backbone, self.n)))
        z, _, _, _, _ = pcc_kinematics(q, np.zeros(self.n), self.L, seg, sigma)
        return np.column_stack((z.real, z.imag))

    def energy(self, state_vect):
        """Kinetic + elastic + gravitational energy."""
        q, dq = state_vect[:self.n], state_vect[self.n:]
        self._eval_kinematics(state_vect)
        self._eval_M_of_q()
        z = self.nodes[0]
        return (0.5 * dq @ self.M @ dq + 0.5 * self.K @ (q - self.q_rest)**2
                + self.g * self.node_mass @ z.imag)

    def CalcEEState(self, context, EE_state):

        state = context.get_continuous_state_vector().CopyToVector()
        z, Jz, _, _, _ = pcc_kinematics(state[:self.n], state[self.n:], self.L, [self.n - 1], [1.0])
        v = Jz[0] @ state[self.n:]
        EE_state.SetFromVector(np.array([z[0].real, z[0].imag, v.real, v.imag]))

    def CalcBackbone(self, context, backbone):
        state = context.get_continuous_state_vector().CopyToVector()
        backbone.SetFromVector(self.backbone(state[:self.n]).ravel())
