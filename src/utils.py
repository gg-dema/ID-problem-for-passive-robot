""" insert here the abstraction of the different plotting function """
from time import sleep 

from pydot import graph_from_dot_data


from pydrake.systems.framework import DiagramBuilder
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.geometry import MeshcatVisualizer
from pydrake.multibody.parsing import Parser

from manipulation.scenarios import AddMultibodyTriad




def save_diagram_svg(file_name, diagram):
    svg = graph_from_dot_data(diagram.GetGraphvizString())[0].create_svg()
    with open(f'../log/{file_name}.svg', 'wb') as f:
        f.write(svg)



def visualize(meshcat, robot_sdf_path, q_state_vect, rate):
    meshcat.Delete()

    builder = DiagramBuilder()

    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-3)
    Parser(plant, scene_graph).AddModels(robot_sdf_path)
    plant.Finalize()



    BODY_NAMES = ['l1', 'l2', 'l3']
    for body_name in BODY_NAMES:
        AddMultibodyTriad(plant.GetFrameByName(body_name), scene_graph)

    visualizer = MeshcatVisualizer.AddToBuilder(
        builder,
        scene_graph.get_query_output_port(),
        meshcat
        )
        
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    for q_values in q_state_vect:
        plant_context.SetDiscreteState(q_values)
        diagram.ForcedPublish(context)
        sleep(rate)
