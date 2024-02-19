import numpy as np
from numpy import sin, cos

from pydrake.systems.framework import LeafSystem, BasicVector, Context, ContinuousState


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
    print(ForwardKinPPR())

