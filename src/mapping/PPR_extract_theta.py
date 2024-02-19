import numpy as np
from pydrake.systems.framework import LeafSystem, BasicVector

class ThetaExtractor(LeafSystem):
    
    def __init__(self):
        LeafSystem.__init__(self)
        self.DeclareVectorInputPort('theta_state', BasicVector(6))
        self.DeclareVectorOutputPort('EE_state', BasicVector(4), self.extract_ee_state)

    def extract_ee_state(self, context, ee_state):
        theta_state = self.get_input_port().Eval(context)
        ee_state.SetFromVector( np.array([theta_state[0], theta_state[1], theta_state[3], theta_state[4] ]) )
