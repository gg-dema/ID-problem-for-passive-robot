import numpy as np

from pydrake.common.value import AbstractValue
from pydrake.systems.framework import BasicVector, EventStatus, ValueProducer
from overrides import override

from src.plant.soft_PCC import PlanarSoftPCC, pcc_kinematics


class PlanarSoftPCC_Theta(PlanarSoftPCC):

    """Same soft arm, integrated in collocated coordinates theta = (x_ee, y_ee, q3, ..., qn).

    theta -> q has no closed form: (q1, q2) come from a Newton solve warm started at the solution
    of the previous step. The seed and the sign of det(Jac_h), which picks the branch (chart) of
    the inverse, live in the discrete state of the system, so the integrator's trial evaluations
    and rejected steps cannot change the result. Use initialize() to set the state from q.
    """

    def __init__(self, **kwargs):

        PlanarSoftPCC.__init__(self, **kwargs)

        self.M_of_theta = np.zeros((self.n, self.n))
        self.C_of_theta = np.zeros((self.n))
        self.Jac_h = np.zeros((self.n, self.n))
        self.Jac_h_dot_qdot = np.zeros((self.n))
        # constant in collocated coordinates: F acts on x_ee and y_ee only
        self.ActuationMatrix = np.zeros((self.n, 2))
        self.ActuationMatrix[:2, :2] = np.eye(2)

        # [seed for q1, seed for q2, sign of det(Jac_h)]
        self.seed_index = self.DeclareDiscreteState(np.array([0.6, 0.6, 1.0]))
        self.DeclarePerStepUnrestrictedUpdateEvent(self._update_seed)

        # one Newton solve per state, shared by the derivatives and the output ports
        self.q_cache = self.DeclareCacheEntry(
            "q_state",
            ValueProducer(allocate=lambda: AbstractValue.Make(np.zeros(2 * self.n)), calc=self._calc_q_state),
            {self.all_state_ticket()})
        self.DeclareVectorOutputPort('q_vect',
                                     BasicVector(2 * self.n),
                                     self.CalcQFromTheta,
                                     prerequisites_of_calc={self.all_state_ticket()})

    def initialize(self, context, state_q):
        """Set the state of the system from a state in q, with the seed and chart that go with it."""
        state_q = np.asarray(state_q, float)
        sign = np.sign(np.linalg.det(self.jac_h(state_q[:self.n])))
        if sign == 0:
            raise ValueError("the initial configuration is singular for the collocated coordinates")
        context.SetContinuousState(self.forward_mapping(state_q))
        context.SetDiscreteState(self.seed_index, np.array([state_q[0], state_q[1], sign]))

    @override
    def eval_dyn_model(self, state_vect):
        # state_vect is in q: M, C, passive forces and Jac of the EE, then the change of coordinates
        PlanarSoftPCC.eval_dyn_model(self, state_vect)
        self._eval_jac_h()
        self._eval_jac_h_dot()
        self._eval_M_of_theta()
        self._eval_C_of_theta()

    @override
    def _eval_actuation_matrix(self):
        pass  # constant, set in __init__

    def forward_mapping(self, state_q):
        q, dq = state_q[:self.n], state_q[self.n:]
        x, y = self.forward_kin(q)
        theta = np.concatenate(([x, y], q[2:]))
        theta_dot = self.jac_h(q) @ dq
        return np.concatenate((theta, theta_dot))

    def inverse_mapping(self, state_theta, seed, chart_sign):

        theta, theta_dot = state_theta[:self.n], state_theta[self.n:]
        target = theta[0] + 1j * theta[1]

        # Newton on (q1, q2), with q3 ... qn = theta3 ... thetan
        q = np.concatenate((seed, theta[2:]))
        for _ in range(50):
            z, Jz, _, _, _ = pcc_kinematics(q, np.zeros(self.n), self.L, [self.n - 1], [1.0])
            err = z[0] - target
            if abs(err) < 1e-13:
                break
            J12 = np.array([[Jz[0, 0].real, Jz[0, 1].real],
                            [Jz[0, 0].imag, Jz[0, 1].imag]])
            q[:2] -= np.linalg.solve(J12, [err.real, err.imag])
        else:
            raise RuntimeError(f"theta -> q: Newton did not converge at theta = {theta}")

        jac_h = self.jac_h(q)
        if np.sign(np.linalg.det(jac_h)) != chart_sign:
            raise RuntimeError(f"the arm left the collocated chart (det(Jac_h) changed sign) at q = {q}")

        q_dot = np.linalg.solve(jac_h, theta_dot)
        return np.concatenate((q, q_dot))

    def jac_h(self, q):

        _, Jz, _, _, _ = pcc_kinematics(q[:self.n], np.zeros(self.n), self.L, [self.n - 1], [1.0])
        jac_h = np.zeros((self.n, self.n))
        jac_h[0, :] = Jz[0].real
        jac_h[1, :] = Jz[0].imag
        jac_h[2:, 2:] = np.eye(self.n - 2)
        return jac_h

    def _eval_jac_h(self):

        self.Jac_h = np.zeros((self.n, self.n))
        self.Jac_h[:2, :] = self.Jac
        self.Jac_h[2:, 2:] = np.eye(self.n - 2)

    def _eval_jac_h_dot(self):

        # Jac_h_dot q_dot: only the EE rows of Jac_h depend on q
        self.Jac_h_dot_qdot = np.concatenate((self.Jac_dot_qdot, np.zeros(self.n - 2)))

    def _eval_M_of_theta(self):

        inv_jac_h = np.linalg.inv(self.Jac_h)
        self.M_of_theta = inv_jac_h.T @ self.M @ inv_jac_h

    def _eval_C_of_theta(self):

        # M(theta) theta_dot_dot + C(theta) = A F, with
        # C(theta) = Jac_h^-T (C + passive forces) - M(theta) Jac_h_dot q_dot
        inv_jac_h = np.linalg.inv(self.Jac_h)
        self.C_of_theta = inv_jac_h.T @ (self.C + self.passive_forces) - self.M_of_theta @ self.Jac_h_dot_qdot

    def DoCalcTimeDerivatives(self, context, derivatives_theta):

        F = self.GetInputPort("F").Eval(context)
        state_theta = context.get_continuous_state_vector().CopyToVector()

        self.eval_dyn_model(self.q_state(context))

        theta_dot = state_theta[self.n:]
        theta_dot_dot = np.linalg.solve(self.M_of_theta, (self.ActuationMatrix @ F) - self.C_of_theta)
        derivatives = np.concatenate((theta_dot, theta_dot_dot))
        derivatives_theta.get_mutable_vector().SetFromVector(derivatives)

    def q_state(self, context):
        return self.q_cache.Eval(context)

    def _calc_q_state(self, context, value):
        seed = context.get_discrete_state(self.seed_index).CopyToVector()
        state_theta = context.get_continuous_state_vector().CopyToVector()
        value.set_value(self.inverse_mapping(state_theta, seed[:2], seed[2]))

    def _update_seed(self, context, state):
        q = self.q_state(context)
        sign = context.get_discrete_state(self.seed_index).CopyToVector()[2]
        state.get_mutable_discrete_state().get_mutable_vector(int(self.seed_index)).SetFromVector(
            np.array([q[0], q[1], sign]))
        return EventStatus.Succeeded()

    def CalcQFromTheta(self, context, Q_vect):
        Q_vect.SetFromVector(self.q_state(context))

    @override
    def CalcEEState(self, context, EE_state):
        state_theta = context.get_continuous_state_vector().CopyToVector()
        EE_state.SetFromVector(np.array([state_theta[0], state_theta[1],
                                         state_theta[self.n], state_theta[self.n + 1]]))

    @override
    def CalcBackbone(self, context, backbone):
        backbone.SetFromVector(self.backbone(self.q_state(context)[:self.n]).ravel())
