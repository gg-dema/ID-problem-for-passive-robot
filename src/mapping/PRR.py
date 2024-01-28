from pydrake.systems.framework import (
    BasicVector,
    Context,
    ContinuousState,
    LeafSystem
)

from src.plant.PRR import PlanarPRR
from math import atan2
from numpy import cos, sin
import numpy as np

class PlanarPRR_inTheta(PlanarPRR):

    def __init__(self):

        PlanarPRR.__init__(self)
        
        self.M_of_theta = np.zeros((3, 3))
        self.C_of_theta = np.zeros((3))
        self.ActuationMatrix = np.array([[1, 0], [0, 1], [0, 0]])
        self.DeclareVectorInputPort('q_vect', BasicVector(6))
        # in theory it's possible to define the same actuation matrix of before and get the same numerical result
        # but, considering that the new actuation matrix will be constant, I prefer to fix it here 

        self.Jac_h = np.zeros((3, 3))
        self.Jac_h_dot = np.zeros((3, 3))
    

    def get_F_vect_input_port(self):
        return self.get_input_port(0)
    
    def get_q_vect_input_port(self):
        return self.get_input_port(1)
    
    def eval_dyn_model(self, state_q): 
        #state_q = self.inverse_mapping(state_theta)
        self._eval_M_of_q(state_q)
        self._eval_C_of_q(state_q)
        self._eval_jac_h(state_q)
        self._eval_jac_h_dot(state_q)
        self._eval_M_of_theta(state_q)

    def inverse_mapping(self, state_theta) -> np.array:
        
        theta_dot = np.array([state_theta[3], state_theta[4], state_theta[5]])
        q3 = state_theta[2]
        q2 = atan2(state_theta[1], state_theta[0]) - atan2(self.l[2]*sin(q3), self.l[1] + self.l[2]*cos(q3))
        q1 = state_theta[0] - self.l[1]*cos(q2) - self.l[2]*cos(q2+q3)
        q_vect = np.array([q1, q2, q3])
       
        if not (theta_dot == np.zeros(3) ).all():
          q_vect_dot = np.linalg.inv(self.Jac_h) @ theta_dot
        else:
            q_vect_dot = np.zeros(3)

        return np.concatenate( (q_vect, q_vect_dot) )
    

    def _eval_M_of_theta(self, state_in_q):

        super()._eval_M_of_q(state_in_q)
        inv_jac_h = np.linalg.inv(self.Jac_h)
        self.M_of_theta = inv_jac_h.T @ self.M @ inv_jac_h

    def _eval_jac_h(self, state_in_q):

        q2, q3 = state_in_q[1], state_in_q[2]
        
        self.Jac_h[0, 0] = 1
        self.Jac_h[0, 1] = -self.l[1]*sin(q2) - self.l[2]*sin(q2+q3)
        self.Jac_h[0, 2] = -self.l[2]*sin(q2+q3)
        self.Jac_h[1, 0] = 0
        self.Jac_h[1, 1] = self.l[1]*cos(q2) + self.l[2]*cos(q2+q3)
        self.Jac_h[1, 2] = self.l[2]*cos(q2+q3)
        self.Jac_h[2, 0] = 0
        self.Jac_h[2, 1] = 0
        self.Jac_h[2, 2] = 1

    def _eval_jac_h_dot(self, state_in_q): 

        q2, q3 = state_in_q[1], state_in_q[2]
        q2dot, q3dot = state_in_q[4], state_in_q[5]

        self.Jac_h_dot[0, 0] = 0
        self.Jac_h_dot[0, 1] = -self.l[1]*cos(q2)*q2dot - self.l[2]*cos(q2+q3)*q2dot*q3dot
        self.Jac_h_dot[0, 2] = - self.l[2]*cos(q2+q3)*q2dot*q3dot

        self.Jac_h_dot[1, 0] = 0
        self.Jac_h_dot[1, 1] = -self.l[1]*sin(q2)*q2dot - self.l[2]*sin(q2+q3)*q2dot*q3dot
        self.Jac_h_dot[1, 2] = - self.l[2]*sin(q2+q3)*q2dot*q3dot

        self.Jac_h_dot[2, 0] = 0
        self.Jac_h_dot[2, 1] = 0
        self.Jac_h_dot[2, 2] = 0

    def DoCalcTimeDerivatives(self, context:Context, derivatives_theta:ContinuousState):

            fx, fy = self.get_input_port(0).Eval(context)
            state_q = self.get_input_port(1).Eval(context)

            F = np.array([fx, fy])

            state_theta = context.get_continuous_state_vector()
            theta_dot_1, theta_dot_2, theta_dot_3 = state_theta[3], state_theta[4], state_theta[5]

            self.eval_dyn_model(state_q)
            # forma di stato ?? 
            theta_dot = np.array([theta_dot_1, theta_dot_2, theta_dot_3])
            theta_dot_dot = (np.linalg.inv(self.M_of_theta) @ self.ActuationMatrix @ F ) + (self.Jac_h_dot @ np.linalg.inv(self.Jac_h) @ theta_dot) + self.C
            # output 
            derivatives = np.concatenate( (theta_dot, theta_dot_dot) )
            derivatives_theta.get_mutable_vector().SetFromVector(derivatives)

class PlanarPRR_theta_extractor(LeafSystem):
    
    def __init__(self):
        LeafSystem.__init__(self)
        self.DeclareVectorInputPort('theta_state', BasicVector(6))
        self.DeclareVectorOutputPort('EndEffector_state', BasicVector(4), self.extract_ee_state)

    def extract_ee_state(self, context, ee_state):
        theta_state = self.get_input_port().Eval(context)
        print(theta_state[0], theta_state[1])
        ee_state.SetFromVector( np.array([theta_state[0], theta_state[1], theta_state[3], theta_state[4] ]) )

    
