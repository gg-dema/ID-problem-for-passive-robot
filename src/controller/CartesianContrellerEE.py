from pydrake.systems.framework import (
    LeafSystem, 
    BasicVector, 
)
from math import atan2

from numpy import cos, sin 
import numpy as np


class CartesianController(LeafSystem):

    def __init__(self, kx=20, ky=20, kx_dot=50, ky_dot=50):
        
        LeafSystem.__init__(self)
        self.DeclareNumericParameter(BasicVector([kx, ky, kx_dot, ky_dot]))
        
        self.DeclareVectorInputPort('actual_state', BasicVector(4))
        self.DeclareVectorInputPort('desired_state', BasicVector(4))

        self.DeclareVectorOutputPort('F', BasicVector(2), self.CalcCartesiaForce)

    def CalcCartesiaForce(self, context, F): 
        actual_state = self.GetInputPort('actual_state').Eval(context)
        desider_state = self.GetInputPort('desired_state').Eval(context)

        K_vect = context.get_numeric_parameter(0).get_value()
        error = desider_state - actual_state
        
        fx = K_vect[0]*error[0] + K_vect[2]*error[2]
        fy = K_vect[1]*error[1] + K_vect[3]*error[3]

        F.SetFromVector(np.array([fx, fy]))
    