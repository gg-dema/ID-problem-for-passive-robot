import numpy as np
import pytest
import sympy as sp

from pydrake.systems.analysis import ResetIntegratorFromFlags, Simulator

from src.mapping.soft_PCC_theta import PlanarSoftPCC_Theta
from src.plant.soft_PCC import PlanarSoftPCC, g_and_derivatives, pcc_kinematics
from src.simulation.soft_PCC_sim import simulate


rng = np.random.default_rng(0)
L3 = np.full(3, 1 / 3)


def random_states(k, low=-2.0, high=2.0):
    return [np.concatenate((rng.uniform(low, high, 3), rng.uniform(-2, 2, 3))) for _ in range(k)]


def tip(q, dq=np.zeros(3)):
    z, Jz, bias, _, _ = pcc_kinematics(q, dq, L3, [2], [1.0])
    return z[0], Jz[0], bias[0]


# --- kinematics -------------------------------------------------------------------------------

def test_g_matches_sympy_and_is_continuous_at_the_series_switch():
    a = sp.symbols('a', real=True)
    g = (sp.exp(sp.I * a) - 1) / (sp.I * a)
    derivatives = [g, sp.diff(g, a), sp.diff(g, a, 2)]
    for alpha in [-2.7, -0.6, -0.2, 1e-4, 0.3, 0.7, 3.0]:
        got = g_and_derivatives(np.array([alpha]))
        for k in range(3):
            ref = complex(derivatives[k].subs(a, sp.Float(alpha, 30)).evalf(30))
            assert abs(got[k][0] - ref) < 1e-14
    below, above = g_and_derivatives(np.array([0.5 - 1e-12, 0.5 + 1e-12]))[2]
    assert abs(below - above) < 1e-11
    assert np.allclose([x[0] for x in g_and_derivatives(np.zeros(1))], [1, 0.5j, -1 / 3], atol=1e-16)


def test_kinematics_matches_sympy():
    q, dq = sp.symbols('q1:4', real=True), sp.symbols('dq1:4', real=True)
    L = [sp.Rational(1, 3)] * 3

    def g(x):
        return (sp.exp(sp.I * x) - 1) / (sp.I * x)

    def z_sym(i, s):
        z, Phi = 0, 0
        for j in range(i):
            z += L[j] * sp.exp(sp.I * Phi) * g(q[j])
            Phi += q[j]
        return z + L[i] * s * sp.exp(sp.I * Phi) * g(s * q[i])

    for state in random_states(3):
        subs = dict(zip(q + dq, state))
        for i, s in [(0, sp.Rational(3, 10)), (1, sp.Rational(7, 10)), (2, sp.Integer(1))]:
            zs = z_sym(i, s)
            z, Jz, bias, _, _ = pcc_kinematics(state[:3], state[3:], L3, [i], [float(s)])
            assert abs(z[0] - complex(zs.subs(subs).evalf())) < 1e-14
            for m in range(3):
                assert abs(Jz[0, m] - complex(sp.diff(zs, q[m]).subs(subs).evalf())) < 1e-14
            b = sum(sp.diff(zs, q[m], q[l]) * dq[m] * dq[l] for m in range(3) for l in range(3))
            assert abs(bias[0] - complex(b.subs(subs).evalf())) < 1e-12


def test_tip_jacobian_and_bias_match_finite_differences():
    eps = 1e-6
    for state in random_states(10):
        q, dq = state[:3], state[3:]
        _, J, bias = tip(q, dq)
        J_fd = [(tip(q + eps * e)[0] - tip(q - eps * e)[0]) / (2 * eps) for e in np.eye(3)]
        assert np.allclose(J, J_fd, atol=1e-8)
        # J_dot q_dot = d/dt (J q_dot) along q_dot
        Jdq_fd = (tip(q + eps * dq)[1] @ dq - tip(q - eps * dq)[1] @ dq) / (2 * eps)
        assert abs(bias - Jdq_fd) < 1e-7


# --- dynamics ---------------------------------------------------------------------------------

def mass_matrix(plant, q):
    plant._eval_kinematics(np.concatenate((q, np.zeros(plant.n))))
    plant._eval_M_of_q()
    return plant.M.copy()


def test_mass_matrix_is_symmetric_positive_definite_and_converged():
    plant, fine = PlanarSoftPCC(), PlanarSoftPCC(n_nodes=24)
    for state in random_states(10):
        M = mass_matrix(plant, state[:3])
        assert np.allclose(M, M.T, atol=1e-15)
        assert np.linalg.eigvalsh(M).min() > 0
        assert np.allclose(M, mass_matrix(fine, state[:3]), rtol=1e-10, atol=1e-12)


def test_coriolis_terms_are_the_christoffel_terms_of_M():
    plant, eps = PlanarSoftPCC(), 1e-6
    for state in random_states(10):
        q, dq = state[:3], state[3:]
        plant.eval_dyn_model(state)
        dM = [(mass_matrix(plant, q + eps * e) - mass_matrix(plant, q - eps * e)) / (2 * eps) for e in np.eye(3)]
        # C_i = sum_jk (dM_ij/dq_k - 1/2 dM_jk/dq_i) dq_j dq_k
        C_ref = np.array([sum((dM[k][i, j] - 0.5 * dM[i][j, k]) * dq[j] * dq[k]
                              for j in range(3) for k in range(3)) for i in range(3)])
        assert np.allclose(plant.C, C_ref, atol=1e-7)
        # energy consistency: dq^T (C - 1/2 M_dot dq) = 0
        M_dot = sum(dM[k] * dq[k] for k in range(3))
        assert abs(dq @ (plant.C - 0.5 * M_dot @ dq)) < 1e-7


def test_gravity_is_the_gradient_of_the_potential():
    plant, eps = PlanarSoftPCC(gravity=9.81, stiffness=0.0, damping=0.0), 1e-6
    for state in random_states(5):
        q = state[:3]
        plant.eval_dyn_model(np.concatenate((q, np.zeros(3))))
        grad = [(plant.energy(np.concatenate((q + eps * e, np.zeros(3))))
                 - plant.energy(np.concatenate((q - eps * e, np.zeros(3))))) / (2 * eps) for e in np.eye(3)]
        plant.eval_dyn_model(np.concatenate((q, np.zeros(3))))
        assert np.allclose(plant.passive_forces, grad, atol=1e-7)


def test_model_is_continuous_through_straight_segments():
    plant = PlanarSoftPCC()
    base = np.array([0.0, 0.4, -0.7, 0.3, -0.5, 1.1])
    terms = []
    for q1 in (-1e-9, 0.0, 1e-9):
        s = base.copy()
        s[0] = q1
        plant.eval_dyn_model(s)
        terms.append(np.concatenate((plant.M.ravel(), plant.C, plant.Jac.ravel())))
    assert np.all(np.isfinite(terms))
    assert np.allclose(terms[0], terms[1], atol=1e-8) and np.allclose(terms[2], terms[1], atol=1e-8)


# --- collocated coordinates -------------------------------------------------------------------

def test_collocated_mapping_round_trip_and_accelerations():
    joint, theta = PlanarSoftPCC(gravity=9.81), PlanarSoftPCC_Theta(gravity=9.81)
    for state in random_states(20, low=0.2, high=1.2):   # C-shapes, inside the chart
        F = rng.uniform(-3, 3, 2)
        sign = np.sign(np.linalg.det(theta.jac_h(state[:3])))
        back = theta.inverse_mapping(theta.forward_mapping(state), state[:2] + rng.normal(0, 0.05, 2), sign)
        assert np.allclose(back, state, atol=1e-10)

        joint.eval_dyn_model(state)
        q_dot_dot = np.linalg.solve(joint.M, joint.ActuationMatrix @ F - joint.C - joint.passive_forces)
        theta.eval_dyn_model(state)
        theta_dot_dot = np.linalg.solve(theta.M_of_theta, theta.ActuationMatrix @ F - theta.C_of_theta)
        assert np.allclose(theta_dot_dot, theta.Jac_h @ q_dot_dot + theta.Jac_h_dot_qdot, atol=1e-9)


def test_inverse_mapping_detects_leaving_the_chart():
    theta = PlanarSoftPCC_Theta()
    state = np.array([0.6, 0.6, 0.6, 0, 0, 0])
    sign = np.sign(np.linalg.det(theta.jac_h(state[:3])))
    with pytest.raises(RuntimeError, match="chart"):
        theta.inverse_mapping(theta.forward_mapping(state), state[:2], -sign)


# --- simulations in Drake ---------------------------------------------------------------------

def free_energy_drift(damping):
    plant = PlanarSoftPCC(damping=damping)
    context = plant.CreateDefaultContext()
    context.SetContinuousState([0.6, -0.3, 0.9, 1.0, -2.0, 1.5])
    plant.get_input_port(0).FixValue(context, np.zeros(2))
    simulator = Simulator(plant, context)
    ResetIntegratorFromFlags(simulator, "runge_kutta5", 0.1)
    simulator.get_mutable_integrator().set_target_accuracy(1e-10)
    energies = []
    for t in np.linspace(0.1, 3.0, 30):
        simulator.AdvanceTo(t)
        energies.append(plant.energy(context.get_continuous_state_vector().CopyToVector()))
    return np.array(energies)


def test_energy_is_conserved_without_damping_and_decreases_with_it():
    E = free_energy_drift(damping=0.0)
    assert np.abs(E - E[0]).max() < 1e-6 * E[0]
    E = free_energy_drift(damping=0.05)
    assert np.all(np.diff(E) < 1e-9) and E[-1] < 0.9 * E[0]


def test_joint_and_collocated_closed_loops_agree():
    logs_q, _ = simulate(collocated=False, sim_time=3.0, accuracy=1e-9)
    logs_theta, _ = simulate(collocated=True, sim_time=3.0, accuracy=1e-9)
    q, q_from_theta = logs_q["q"].data(), logs_theta["q"].data()
    n = min(q.shape[1], q_from_theta.shape[1])
    assert np.abs(q[:, :n] - q_from_theta[:, :n]).max() < 1e-6
