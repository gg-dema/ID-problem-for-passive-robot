import numpy as np
from numpy import sin, cos

from pydrake.systems.framework import LeafSystem, BasicVector, Context, ContinuousState

import sys
sys.path.append('./')
from PPR import PlanarPPR

class PlanarPPR_gravity(PlanarPPR): 

    def __init__(self):
        super().__init__()
        self.G = np.array([-4.9, -4.9, 0])

    def DoCalcTimeDerivatives(self, context:Context, derivatives_out:ContinuousState):

        fx, fy = self.get_input_port(0).Eval(context)
        F = np.array([fx, fy])
        state = context.get_continuous_state_vector()

        dq1, dq2, dq3 = state[3], state[4], state[5]
        self.eval_dyn_model(state_vect=state)
        
        q_dot = np.array([dq1, dq2, dq3])
        #q_dot_dot = np.linalg.inv(self.M) @ ((self.ActuationMatrix @ F) - self.C - self.Fv*q_dot - self.Fc@np.sign(q_dot))
        q_dot_dot = np.linalg.inv(self.M) @ ((self.ActuationMatrix @ F) - self.C )
        derivatives = np.concatenate( (q_dot, q_dot_dot) )
        derivatives_out.get_mutable_vector().SetFromVector(derivatives)
    

    def _eval_C_of_q(self, state_vect):
      
        q1, q2, q3 = state_vect[0], state_vect[1], state_vect[2]
        dq1, dq2, dq3 = state_vect[3], state_vect[4], state_vect[5]

        self.C[0] = -self.a4*(dq3**2)*sin(q3) + self.G[0]
        self.C[1] = -self.a4*(dq3**2)*cos(q3)+ self.G[0]
        self.C[2] =  self.G[0]



if __name__=="__main__":
    P = PlanarPPR_gravity()