from src.mapping.PRR import PlanarPRR_inTheta, PlanarPRR_theta_extractor
from src.controller.CartesianContrellerEE import CartesianController

from pydrake.systems.framework import DiagramBuilder


def PlanarPRR_Controller():

    builder = DiagramBuilder()
    
      
    plant = builder.AddSystem(PlanarPRR_inTheta()) 
    state_extractor = builder.AddSystem(PlanarPRR_theta_extractor())
    controller = builder.AddSystem(CartesianController())

    plant.set_name("PlanarPRR theta")
    state_extractor.set_name("state extractor")
    controller.set_name("cartesian controller")

    builder.Connect(plant.get_output_port(), state_extractor.get_input_port(0))
    builder.Connect(state_extractor.get_output_port(), controller.get_input_port_estimated_state())
    
    builder.Connect(controller.get_output_port(), plant.get_F_vect_input_port())


    builder.ExportInput(plant.get_q_vect_input_port())
    builder.ExportOutput(controller.get_output_port())
    builder.ExportOutput(state_extractor.get_output_port())
    
    diagram = builder.Build()
    diagram.set_name("diagram")
    
    context = diagram.CreateDefaultContext()


    return diagram, {'PlanarPRR_theta':plant, "state extractor":state_extractor, "cartesian controller":controller}, context
