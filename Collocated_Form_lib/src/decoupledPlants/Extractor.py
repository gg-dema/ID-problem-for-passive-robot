
import numpy as np
from numpy import cos, sin 


from pydrake.systems.framework import LeafSystem, BasicVector



class PlanarPRR_theta_extractor(LeafSystem):
    
    def __init__(self):
        LeafSystem.__init__(self)
        self.DeclareVectorInputPort('theta_state', BasicVector(6))
        self.DeclareVectorOutputPort('EndEffector_state', BasicVector(4), self.extract_ee_state)

    def extract_ee_state(self, context, ee_state):
        theta_state = self.get_input_port().Eval(context)
        ee_state.SetFromVector( np.array([theta_state[0], theta_state[1], theta_state[3], theta_state[4] ]) )


class PlanarPRR_forward_kin(LeafSystem):

    def __init__(self):
        LeafSystem.__init__(self)

        self.Jac = np.zeros((3, 3))
        self.l = np.array([None, 1, 1])

        self.DeclareVectorInputPort('q_state', BasicVector(6))
        self.DeclareVectorOutputPort('EndEffector_state', BasicVector(4), self.CalcForwardAndDiffKinematic)

    def CalcForwardAndDiffKinematic(self, context, theta_state):

        q_state = self.get_input_port(0).Eval(context)
        q = np.array([q_state[0], q_state[1], q_state[2]])
        q_dot = np.array([q_state[3], q_state[4], q_state[5]])

        self.eval_jac(q)
        theta_1, theta_2 = self.forward_kin(q)
        theta_1_dot, theta_2_dot, _ = self.Jac @ q_dot
        theta_state.SetFromVector([theta_1, theta_2, theta_1_dot, theta_2_dot])
        print(f"x_ee: {theta_1}, y_ee: {theta_2}")


    def eval_jac(self, q): 
        
        q1, q2, q3 = q[0], q[1], q[2]

        self.Jac[0] = [1, -self.l[1]*sin(q2) - self.l[2]*sin(q2+q3), - self.l[2]*sin(q2+q3)]
        self.Jac[1] = [0, self.l[1]*cos(q2) + self.l[2]*cos(q2+q3), self.l[2]*cos(q2+q3)]
        self.Jac[2] = [0, 1, 1]
    
    def forward_kin(self, q):
        q1, q2, q3 = q[0], q[1], q[2]

        theta_1 = q1 + self.l[1]*cos(q2) + self.l[2]*cos(q2+q3)
        theta_2 = self.l[1]*sin(q2) + self.l[2]*sin(q2+q3)

        return theta_1, theta_2

