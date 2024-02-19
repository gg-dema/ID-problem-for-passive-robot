import numpy as np
from numpy import sin, cos 

from src.plant.PPR import PlanarPPR


class PlanarPPR_theta(PlanarPPR):

    def __init__(self):
        
        PlanarPPR.__init__(self)

        self.M_of_theta = np.zeros((3, 3))
        self.ActuationMatrix = np.array([[1, 0], 
                                         [0, 1],
                                         [0, 0]])

        self.Jac_h = np.zeros((3, 3))
        self.Jac_h_dot = np.zeros((3, 3))
        self.Friction_term = np.zeros((3))

    def DoCalcTimeDerivatives(self, context, derivatives_theta):
        
        fx, fy = self.GetInputPort("F").Eval(context)
        F = np.array([fx, fy])
        state_theta = context.get_continuous_state_vector()
        theta_dot_1, theta_dot_2, theta_dot_3 = state_theta[3], state_theta[4], state_theta[5]

        self.eval_dyn_model(state_theta)
        theta_dot = np.array([theta_dot_1, theta_dot_2, theta_dot_3])
        
        theta_dot_dot = (np.linalg.inv(self.M_of_theta) @ self.ActuationMatrix @ F.T) +(self.Jac_h_dot @ np.linalg.inv(self.Jac_h) @ theta_dot) - self.C - self.Friction_term
        derivatives = np.concatenate( (theta_dot, theta_dot_dot) )
        derivatives_theta.get_mutable_vector().SetFromVector(derivatives)
    
    def eval_dyn_model(self, state_theta):
        state_q = self.inverse_mapping(state_theta)
        self._eval_M_of_q(state_q)
        self._eval_friction_term(state_q)
        self._eval_C_of_q(state_q)
        self._eval_Jac_h(state_q)
        self._eval_Jac_h_dot(state_q)
        self._eval_M_of_theta()

    
    def inverse_mapping(self, state_theta) -> np.array:
        theta_dot = np.array([state_theta[3], state_theta[4], state_theta[5]])
        
        q3 = state_theta[2]
        q1 = state_theta[1] - self.l[2]*sin(q3)
        q2 = state_theta[0] - self.l[2]*cos(q3)
        q_vect = np.array([q1, q2, q3])
       
        if not (self.Jac_h == np.zeros((3,3)) ).all(): # at the first run Jac_h is all zeros
            q_vect_dot = np.linalg.inv(self.Jac_h) @ theta_dot
        else:
            q_vect_dot = np.zeros(3)

        return np.concatenate( (q_vect, q_vect_dot) )
        
    def _eval_friction_term(self, state_q):
        q_dot = np.array([state_q[3], state_q[4], state_q[5]])
        self.Friction_term = self.Fv*q_dot + self.Fc*np.sign(q_dot)

    def _eval_M_of_theta(self):
        inv_Jac_h = np.linalg.inv(self.Jac_h)
        self.M_of_theta = inv_Jac_h.T @ self.M @ inv_Jac_h

    def _eval_Jac_h(self, state_in_q):
        q3 = state_in_q[2]
        self.Jac_h[0, 0] = 0
        self.Jac_h[0, 1] = 1
        self.Jac_h[0, 2] = -self.l[2]*sin(q3)

        self.Jac_h[1, 0] = 1
        self.Jac_h[1, 1] = 0
        self.Jac_h[1, 2] = self.l[2]*cos(q3)

        self.Jac_h[2, 0] = 0
        self.Jac_h[2, 1] = 0
        self.Jac_h[2, 2] = 1
        

    def _eval_Jac_h_dot(self, state_in_q): 

        q3, q3dot = state_in_q[2], state_in_q[5]
        self.Jac_h_dot[0, 0] = 0 
        self.Jac_h_dot[0, 1] = 0
        self.Jac_h_dot[0, 2] = -self.l[2]*cos(q3)*q3dot

        self.Jac_h_dot[1, 0] = 0
        self.Jac_h_dot[1, 1] = 0
        self.Jac_h_dot[1, 2] = -self.l[2]*sin(q3)*q3dot

        self.Jac_h_dot[2, 0] = 0
        self.Jac_h_dot[2, 1] = 0
        self.Jac_h_dot[2, 2] = 0



