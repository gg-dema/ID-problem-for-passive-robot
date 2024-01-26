from abc import ABC, abstractmethod

from pydrake.systems.framework import LeafSystem, BasicVector
from pydrake.systems.framework import Context, ContinuousState, SystemOutput

import numpy as np

class Planar3DofRobot(ABC, LeafSystem):

    
    def __init__(self):
        LeafSystem.__init__(self)
        self.M = np.zeros((3,3))
        self.J = np.zeros((3,3))
        self.init_dynamic_model_parameters()

    @abstractmethod
    def init_dynamic_model_parameters():
        pass
        
    @abstractmethod
    def _eval_M(state_vect):
        pass

class Planar3DofExtractor(ABC, LeafSystem):

    def __init__(self):
        LeafSystem.__init__(self)
        