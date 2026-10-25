
import numpy as np
from math import atan2, asin, hypot
from numpy import cos, sin

from src.plant.PRR import PlanarPRR
from pydrake.systems.framework import BasicVector
from overrides import override

class PlanarPRR_Theta(PlanarPRR):

    def __init__(self, friction=0.0):

        PlanarPRR.__init__(self, friction)
        
        self.M_of_theta = np.zeros((3, 3))
        self.C_of_theta = np.zeros((3))
        self.ActuationMatrix = np.array([[1, 0], [0, 1], [0, 0]])
        self.DeclareVectorOutputPort('q_vect', BasicVector(6), self.CalcQFromTheta)
        # in theory it's possible to define the same actuation matrix of before and get the same numerical result
        # but, considering that the new actuation matrix will be constant, I prefer to fix it here 

        self.Jac_h = np.zeros((3, 3))
        self.Jac_h_dot = np.zeros((3, 3))
   
    @override
    def eval_dyn_model(self, state_vect): 
        state_q = self.inverse_mapping(state_vect)
        self._eval_M_of_q(state_q)
        self._eval_C_of_q(state_q)
        self._eval_jac_h(state_q)
        self._eval_jac_h_dot(state_q)
        self._eval_M_of_theta(state_q)
        self._eval_C_of_theta(state_q)

    def inverse_mapping(self, state_theta) -> np.array:

        theta_dot = np.array([state_theta[3], state_theta[4], state_theta[5]])
        q3 = state_theta[2]
        # y_ee = R*sin(q2 + phi) does not depend on q1, so solve it for q2 first.
        # Of the two solutions keep the one with l1*c2 + l2*c23 > 0, where Jac_h is invertible
        # (the arm reaches forward of the carriage, as in the initial configuration q = 0)
        R = hypot(self.l[1] + self.l[2]*cos(q3), self.l[2]*sin(q3))
        phi = atan2(self.l[2]*sin(q3), self.l[1] + self.l[2]*cos(q3))
        q2 = asin(state_theta[1] / R) - phi
        q1 = state_theta[0] - self.l[1]*cos(q2) - self.l[2]*cos(q2+q3)
        q_vect = np.array([q1, q2, q3])

        q_vect_dot = np.linalg.solve(self.jac_h(q_vect), theta_dot)

        return np.concatenate( (q_vect, q_vect_dot) )
    
    def forward_mapping(self, state_q): 
        q1, q2, q3, = state_q[0], state_q[1], state_q[2]
        dq1, dq2, dq3 = state_q[3], state_q[4], state_q[5]

        theta1, theta2 = super().forward_kin(state_q)
        theta3 = state_q[2]

        # theta_dot = Jac_h(q) q_dot. A fresh jacobian, so the state of the class is not modified
        theta_dot = self.jac_h(state_q) @ np.array([dq1, dq2, dq3])

        return np.concatenate((np.array([theta1, theta2, theta3]), theta_dot))

    def _eval_M_of_theta(self, state_in_q):

        super()._eval_M_of_q(state_in_q)
        inv_jac_h = np.linalg.inv(self.Jac_h)
        self.M_of_theta = inv_jac_h.T @ self.M @ inv_jac_h

    def _eval_C_of_theta(self, state_in_q):

        # M(theta) theta_dot_dot + C(theta) = A F, with C(theta) = Jac_h^-T n(q, q_dot) - M(theta) Jac_h_dot q_dot
        q_dot = np.array([state_in_q[3], state_in_q[4], state_in_q[5]])
        n = self.C + self.friction * q_dot
        self.C_of_theta = np.linalg.inv(self.Jac_h).T @ n - self.M_of_theta @ self.Jac_h_dot @ q_dot

    def jac_h(self, state_in_q):

        q2, q3 = state_in_q[1], state_in_q[2]

        jac_h = np.zeros((3, 3))
        jac_h[0, :] = [1, -self.l[1]*sin(q2) - self.l[2]*sin(q2+q3), -self.l[2]*sin(q2+q3)]
        jac_h[1, :] = [0, self.l[1]*cos(q2) + self.l[2]*cos(q2+q3), self.l[2]*cos(q2+q3)]
        jac_h[2, :] = [0, 0, 1]
        return jac_h

    def _eval_jac_h(self, state_in_q):

        self.Jac_h = self.jac_h(state_in_q)

    def _eval_jac_h_dot(self, state_in_q):

        q2, q3 = state_in_q[1], state_in_q[2]
        q2dot, q3dot = state_in_q[4], state_in_q[5]

        self.Jac_h_dot[0, 0] = 0
        self.Jac_h_dot[0, 1] = -self.l[1]*cos(q2)*q2dot - self.l[2]*cos(q2+q3)*(q2dot + q3dot)
        self.Jac_h_dot[0, 2] = - self.l[2]*cos(q2+q3)*(q2dot + q3dot)

        self.Jac_h_dot[1, 0] = 0
        self.Jac_h_dot[1, 1] = -self.l[1]*sin(q2)*q2dot - self.l[2]*sin(q2+q3)*(q2dot + q3dot)
        self.Jac_h_dot[1, 2] = - self.l[2]*sin(q2+q3)*(q2dot + q3dot)

        self.Jac_h_dot[2, 0] = 0
        self.Jac_h_dot[2, 1] = 0
        self.Jac_h_dot[2, 2] = 0

    def DoCalcTimeDerivatives(self, context, derivatives_theta):

            fx, fy = self.GetInputPort("F").Eval(context)

            F = np.array([fx, fy])

            state_theta = context.get_continuous_state_vector()
            theta_dot_1, theta_dot_2, theta_dot_3 = state_theta[3], state_theta[4], state_theta[5]

            self.eval_dyn_model(state_theta)

            theta_dot = np.array([theta_dot_1, theta_dot_2, theta_dot_3])
            theta_dot_dot = np.linalg.inv(self.M_of_theta) @ ((self.ActuationMatrix @ F) - self.C_of_theta)
            # output 
            derivatives = np.concatenate( (theta_dot, theta_dot_dot) )
            derivatives_theta.get_mutable_vector().SetFromVector(derivatives)

    def CalcQFromTheta(self, context, Q_vect): 
        state_theta = context.get_continuous_state_vector()
        state_q = self.inverse_mapping(state_theta)
        Q_vect.SetFromVector(state_q)

    @override
    def CalcEEState(self, context, EE_state):
        state_theta = context.get_continuous_state_vector()
        EE_state.SetFromVector(np.array([state_theta[0], state_theta[1],state_theta[3], state_theta[4]]))


    @override
    def forward_kin(self, state_vect):
        raise(NotImplementedError)

        