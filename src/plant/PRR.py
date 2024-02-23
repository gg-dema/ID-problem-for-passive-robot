from pydrake.systems.framework import (
    LeafSystem, 
    BasicVector,
    Context,
    ContinuousState, 
)

import numpy as np
from numpy import cos, sin
from math import atan2 


class PlanarPRR(LeafSystem):

    def __init__(self):

        LeafSystem.__init__(self)

        self.M = np.zeros((3, 3))
        self.C = np.zeros((3))
        self.ActuationMatrix = np.zeros((3, 2))
        self.Jac = np.zeros((2, 3))

        group_index = self.DeclareContinuousState(6)

        self.DeclareVectorInputPort("F", BasicVector(2))
        self.DeclareStateOutputPort("state", group_index)
        self.DeclareVectorOutputPort("EE_state", 
                                     BasicVector(4),
                                     self.CalcEEState, 
                                     prerequisites_of_calc={self.all_state_ticket()},
)
        
        self.init_dynamic_model_parameters()

    def DoCalcTimeDerivatives(self, context:Context, derivatives_out:ContinuousState):

        fx, fy = self.get_input_port(0).Eval(context)
        F = np.array([fx, fy])
        state = context.get_continuous_state_vector()

        dq1, dq2, dq3 = state[3], state[4], state[5]
        self.eval_dyn_model(state_vect=state)
        q_dot = np.array([dq1, dq2, dq3])
        q_dot_dot = np.linalg.inv(self.M) @ ((self.ActuationMatrix @ F) - self.C)
        derivatives = np.concatenate( (q_dot, q_dot_dot) )
        derivatives_out.get_mutable_vector().SetFromVector(derivatives)

    def init_dynamic_model_parameters(self):

        self.m = [1, 1, 1]
        self.I = [1, 1, 1]
        self.d = [None, 0.25, 0.25]
        self.l = [None, 0.5, 0.5]  # the first is none because is q1
        
        self.a1 = self.m[0] + self.m[1] + self.m[2]
        self.a2 = self.I[1] + self.m[1] * (self.d[1]**2) + self.m[2]*(self.l[2]**2)
        self.a3 = self.I[2] + self.m[2] * (self.d[2]**2)
        self.a4 = self.m[2] * self.d[2]
        self.a5 = self.m[1] * self.d[1] + self.m[2] * self.l[1]

    def eval_dyn_model(self, state_vect):
        self._eval_M_of_q(state_vect)
        self._eval_C_of_q(state_vect)
        self._eval_actuation_matrix(state_vect)

    def _eval_M_of_q(self, state_vect):

        q1, q2, q3 = state_vect[0], state_vect[1], state_vect[2]
        
        self.M[0, 0] =  self.a1
        self.M[0, 1] = -self.a5 * sin(q2)  - self.a4 * sin(q2 + q3)
        self.M[0, 2] = -self.a4 * sin(q2 + q3)

        self.M[1, 0] = -self.a5 * sin(q2) - self.a4 * sin(q2 + q3)
        self.M[1, 1] = self.a2 + self.a3 + 2*self.a4 * self.l[1] * cos(q3)
        self.M[1, 2] = self.a3 + self.a4 * self.l[1] * cos(q3)

        self.M[2, 0] = -self.a4 * sin(q2 + q3)
        self.M[2, 1] = self.a3 + self.a4 * self.l[1] * cos(q3)
        self.M[2, 2] = self.a3

    def _eval_C_of_q(self, state_vect):
    
        q1, q2, q3 = state_vect[0], state_vect[1], state_vect[2]
        dq1, dq2, dq3 = state_vect[3], state_vect[4], state_vect[5]

        self.C[0] = (-self.a5*cos(q2) + self.a4*cos(q2+q3))*(dq2**2) - self.a4*cos(q2+q3)*dq3*(dq2 + dq3)
        self.C[1] = -self.a4 * self.l[1] * sin(q3) * dq3 * (2*dq2 + dq3)
        self.C[2] = self.a4 * self.l[2] * sin(q3) * (dq2**2)

    def _eval_actuation_matrix(self, state_vect):

        q1, q2, q3 = state_vect[0], state_vect[1], state_vect[2]

        self.ActuationMatrix[0, 0]  = 1
        self.ActuationMatrix[0, 1]  = 0 
        
        self.ActuationMatrix[1, 0] = -self.l[1] * sin(q2) - self.l[2] * sin(q2 + q3)
        self.ActuationMatrix[1, 1] = -self.l[1] * cos(q2) - self.l[2] * cos(q2 + q3)
        
        self.ActuationMatrix[2, 0] = -self.l[1] * sin(q2) 
        self.ActuationMatrix[2, 1] = -self.l[1] * cos(q2)

    def _eval_jac_ee_xy(self, state_vect):
        
        q2, q3 = state_vect[1], state_vect[2]
        self.Jac[0, :] = [1, -self.l[1]*sin(q2) - self.l[2]*sin(q2+q3), -self.l[2]*sin(q2+q3)]
        self.Jac[1, :] = [0, self.l[1]*cos(q2) + self.l[2]*cos(q2+q3), self.l[2]*cos(q3)]

    def forward_kin(self, state_vect):
        q1, q2, q3 = state_vect[0], state_vect[1], state_vect[2]
        x = q1 + self.l[1]*cos(q2) + self.l[2]*cos(q2+q3)
        y =  self.l[1]*sin(q2) + self.l[2]*sin(q2+q3)
        return x, y 


    def CalcEEState(self, context, EE_state):

        state = context.get_continuous_state_vector()
        q1, q2, q3 = state[0], state[1], state[2]
        dq1, dq2, dq3 = state[3], state[4], state[5]
        self._eval_jac_ee_xy(state)
        EE_x, EE_y = self.forward_kin([q1, q2, q3])
        EE_x_dot, EE_y_dot = self.Jac @ np.array([dq1, dq2, dq3])
        EE_state.SetFromVector(np.array([EE_x, EE_y, EE_x_dot, EE_y_dot]))
