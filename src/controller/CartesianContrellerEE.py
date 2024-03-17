from pydrake.systems.framework import (
    LeafSystem, 
    BasicVector, 
)
from math import atan2

from numpy import cos, sin 
import numpy as np


class CartesianController(LeafSystem):

    def __init__(self, kp=20, kd=50):
        
        LeafSystem.__init__(self)
        self.DeclareNumericParameter(BasicVector([kp, kd]))
        
        self.DeclareVectorInputPort('actual_state', BasicVector(4))
        self.DeclareVectorInputPort('desired_state', BasicVector(4))

        self.DeclareVectorOutputPort('F', BasicVector(2), self.CalcCartesiaForce)

    def CalcCartesiaForce(self, context, F): 
        actual_state = self.GetInputPort('actual_state').Eval(context)
        desider_state = self.GetInputPort('desired_state').Eval(context)

        K_vect = context.get_numeric_parameter(0).get_value()
        kp, kd = K_vect[0], K_vect[1]
        error = desider_state - actual_state
        
        fx = kp*error[0] + kd*error[2]
        fy = kp*error[1] + kd*error[3]

        F.SetFromVector(np.array([fx, fy]))



class PD_gravity_compensation(CartesianController):

    def __init__(self, Q_final=None, kp=20, kd=50):
        #raise(NotImplementedError)
        super().__init__(kp, kd)
        self.Q = np.zeros(3)
        self.calc_gravity_compensation(Q_final)

    
    def calc_gravity_compensation(self, q_vect):
        self.Q = np.array([2.5, 2.5])
        return 0
    

    def CalcCartesiaForce(self, context, F): 
        actual_state = self.GetInputPort('actual_state').Eval(context)
        desider_state = self.GetInputPort('desired_state').Eval(context)

        K_vect = context.get_numeric_parameter(0).get_value()
        kp, kd = K_vect[0], K_vect[1]
        error = desider_state - actual_state
        
        fx = kp*error[0] + kd*error[2] - self.Q[1]
        fy = kp*error[1] + kd*error[3] - self.Q[0]

        F.SetFromVector(np.array([fx, fy]))
    