import numpy as np
from numpy import sin, cos

from pydrake.systems.framework import LeafSystem, BasicVector, Context, ContinuousState


class PlanarPPR(LeafSystem):

    """PPR robot: dynamic in q"""
    
    def __init__(self):

        LeafSystem.__init__(self)

        self.M = np.zeros((3, 3))
        self.C = np.zeros((3))
        self.ActuationMatrix = np.zeros((3, 2))
        self.Jac = np.zeros((3, 3))
        
        self.Fc = np.array([0.2, 0.2, 0.2])
        self.Fv = np.array([0.1, 0.1, 0.1])

        group_index = self.DeclareContinuousState(6)

        self.DeclareVectorInputPort("F", BasicVector(2))
        self.DeclareStateOutputPort("state", group_index)
        
        self.init_dynamic_model_parameters()

    def DoCalcTimeDerivatives(self, context:Context, derivatives_out:ContinuousState):

        fx, fy = self.get_input_port(0).Eval(context)
        F = np.array([fx, fy])
        state = context.get_continuous_state_vector()

        dq1, dq2, dq3 = state[3], state[4], state[5]
        self.eval_dyn_model(state_vect=state)
        
        q_dot = np.array([dq1, dq2, dq3])
        q_dot_dot = np.linalg.inv(self.M) @ ((self.ActuationMatrix @ F) - self.C - self.Fv*q_dot - self.Fc@np.sign(q_dot))
        
        derivatives = np.concatenate( (q_dot, q_dot_dot) )
        derivatives_out.get_mutable_vector().SetFromVector(derivatives)

    def init_dynamic_model_parameters(self):

        self.m = [1, 1, 1]
        self.I = [1, 1, 1]
        self.d = [None, None, 0.5]
        self.l = [None, None, 1]  
        
        self.a1 = self.m[0] + self.m[1] + self.m[2]
        self.a2 = self.m[1] + self.m[2] 
        self.a3 = self.I[2] + self.m[2] * (self.d[2]**2)
        self.a4 = self.m[2] * self.d[2]

    def eval_dyn_model(self, state_vect):
        self._eval_M_of_q(state_vect)
        self._eval_C_of_q(state_vect)
        self._eval_actuation_matrix(state_vect)

    def _eval_M_of_q(self, state_vect):

        q1, q2, q3 = state_vect[0], state_vect[1], state_vect[2]
        
        self.M[0, 0] =  self.a1
        self.M[0, 1] = 0
        self.M[0, 2] = 0

        self.M[1, 0] = 0
        self.M[1, 1] = self.a2
        self.M[1, 2] = -self.a4 * sin(q3) 

        self.M[2, 0] = 0
        self.M[2, 1] = -self.a4 * sin(q3)
        self.M[2, 2] = self.a3

    def _eval_C_of_q(self, state_vect):
      
        q1, q2, q3 = state_vect[0], state_vect[1], state_vect[2]
        dq1, dq2, dq3 = state_vect[3], state_vect[4], state_vect[5]

        self.C[0] = -self.a4*(dq3**2)*sin(q3)
        self.C[1] = -self.a4*(dq3**2)*cos(q3)
        self.C[2] = 0

    def _eval_actuation_matrix(self, state_vect):
        q1, q2, q3 = state_vect[0], state_vect[1], state_vect[2]

        self.ActuationMatrix[0, :] = [0, 1]
        self.ActuationMatrix[1, :] = [1, 0]
        self.ActuationMatrix[2, :] = [-self.l[2]*sin(q3), self.l[2] * cos(q3)]




class ForwardKinPPR(LeafSystem):

    def __init__(self):
        LeafSystem.__init__(self)
        self.DeclareVectorInputPort('q', BasicVector(6))
        self.DeclareVectorOutputPort('EE_state', BasicVector(4), self.extract_ee_state)


    def extract_ee_state(self, context, ee_state):
        x = self.get_input_port().Eval(context)
        q1, q2, q3 = x[0], x[1], x[2]
        dq1, dq2, dq3 = x[3], x[4], x[5]
        x = q2 + 1*cos(q3)
        y = q1 + 1*sin(q3)

        jac = np.array( [[0, 1, -1*sin(q3)], 
                         [1, 0, 1*cos(q3)], 
                         ] )
        
        x_dot, y_dot = jac @ np.array([dq1, dq2, dq3])
        ee_state.SetFromVector( np.array([x, y, x_dot, y_dot]) )

    @staticmethod
    def forward(q):
        #print(q)
        x = q[1] + 1*cos(q[2])
        y = q[0] + 1*sin(q[2])

        #print(f"x : {x}, y: {y}")

        jac = np.array( [[0, 1, -1*sin(q[2])], 
                         [1, 0, 1*cos(q[2])], 
                         ] )
        #print(f' jac \n{jac}')
        x_dot, y_dot = jac @ np.array([q[3], q[4], q[5]])
        #print(f"x_dot {x_dot}, y_dot {y_dot}")
        return  np.array([x, y, x_dot, y_dot ]) 

if __name__=="__main__":

    PlanarPPR()
    