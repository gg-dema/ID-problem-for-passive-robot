import numpy as np 
from Collocated_Form_lib.src.controllers import PD
from Collocated_Form_lib.src.decoupledPlants.PRR_system import PRR


def simulate(robot, controller, log_level, simulation_parameters):
    pass



if __name__ == "__main__":

    robot = PRR()
    controller = PD()

    logger_level = ... #funct() from argparser : struct that specify wich data want to save: pydot, plt, meshcat
    simulation_parameters = ... # funct() return struct from argparse

    simulate(robot, controller, logger_level, simulation_parameters)