import numpy as np
from numpy import sin, cos


import robot_header.header_plant as header_plant
from pydrake.systems.framework import BasicVector, Context, ContinuousState   

class PlanarPRR(header_plant):

    def __init__(self):

        super.__init__(self)

        self.M = np.zeros((3, 3))
        self.C = np.zeros((3))
        self.ActuationMatrix = np.zeros((3, 2))
        self.Jac = np.zeros((3, 3))

        # def joint limit: 
        self.limit_q1 = 12
        self.limit_q2 = np.pi
        self.limit_q3 = np.pi

        group_index = self.DeclareContinuousState(6)

        self.DeclareVectorInputPort("F", BasicVector(2))
        self.DeclareStateOutputPort("state", group_index)
        
        self.init_dynamic_model_parameters()

    def DoCalcTimeDerivatives(self, context:Context, derivatives_out:ContinuousState):

        fx, fy = self.get_input_port(0).Eval(context)
        F = np.array([fx, fy])
        state = context.get_continuous_state_vector()

        # test on fixing joint limit. Still to do: this next lines doesn't work
        #dq1 = min(max(state[3], -self.limit_q1), self.limit_q1)
        #dq2 = min(max(state[4], -self.limit_q2), self.limit_q2)
        #dq3 = min(max(state[5], -self.limit_q3), self.limit_q3)

        dq1, dq2, dq3 = state[3], state[4], state[5]
        self.eval_dyn_model(state_vect=state)
        q_dot = np.array([dq1, dq2, dq3])
        q_dot_dot = np.linalg.inv(self.M) @ ((self.ActuationMatrix @ F) - self.C)
        derivatives = np.concatenate( (q_dot, q_dot_dot) )
        derivatives_out.get_mutable_vector().SetFromVector(derivatives)

    def init_dynamic_model_parameters(self):

        self.m = [1, 1, 1]
        self.I = [1, 1, 1]
        self.d = [None, 0.5, 0.5]
        self.l = [None, 1, 1]  # the first is none because is q1
        
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
