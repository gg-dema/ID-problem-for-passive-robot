"""Closed loop of the planar soft arm with the Cartesian PD at its tip.

    python -m src.simulation.soft_PCC_sim

runs the default scenario in collocated coordinates and serves the animation in Meshcat.
"""
import time

import numpy as np

from pydrake.geometry import Meshcat, MeshcatAnimation
from pydrake.systems.analysis import ResetIntegratorFromFlags, Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import LogVectorOutput

from src.controller.CartesianContrellerEE import CartesianController
from src.mapping.soft_PCC_theta import PlanarSoftPCC_Theta
from src.plant.soft_PCC import PlanarSoftPCC
from src.visualization.soft_arm_meshcat import SoftArmVisualizer, draw_path, draw_target


# default scenario: from a C-shape to the tip position of a more bent C-shape,
# both well inside the chart of the collocated coordinates
Q0 = np.full(3, 0.6)
TARGET = np.array([0.28, 0.72])
KP, KD = 100.0, 9.0


def simulate(collocated, q0=Q0, target=TARGET, sim_time=10.0, kp=KP, kd=KD, plant_kwargs=None,
             meshcat=None, log_period=0.01, accuracy=1e-8):
    """Simulate the closed loop. Returns the logs ("q", "state", "EE", "F") and the plant.

    "q" is always the state in joint coordinates [q, q_dot]; "state" is the integrated state
    (theta for the collocated model).
    """
    builder = DiagramBuilder()

    plant_kwargs = plant_kwargs or {}
    plant = builder.AddSystem(PlanarSoftPCC_Theta(**plant_kwargs) if collocated else PlanarSoftPCC(**plant_kwargs))
    controller = builder.AddSystem(CartesianController(kp=kp, kd=kd))

    builder.Connect(plant.GetOutputPort("EE_state"), controller.GetInputPort("actual_state"))
    builder.Connect(controller.GetOutputPort("F"), plant.GetInputPort("F"))

    loggers = {
        "q": LogVectorOutput(plant.GetOutputPort("q_vect" if collocated else "state"), builder, log_period),
        "state": LogVectorOutput(plant.GetOutputPort("state"), builder, log_period),
        "EE": LogVectorOutput(plant.GetOutputPort("EE_state"), builder, log_period),
        "F": LogVectorOutput(controller.GetOutputPort("F"), builder, log_period),
    }

    if meshcat is not None:
        visualizer = builder.AddSystem(SoftArmVisualizer(
            meshcat, plant.num_backbone_points(), plant.n_backbone, plant.L[0] / plant.n_backbone, plant.r))
        builder.Connect(plant.GetOutputPort("backbone"), visualizer.get_input_port())

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)
    controller_context = controller.GetMyContextFromRoot(context)

    state_q0 = np.concatenate((q0, np.zeros(plant.n)))
    if collocated:
        plant.initialize(plant_context, state_q0)
    else:
        plant_context.SetContinuousState(state_q0)
    controller.GetInputPort("desired_state").FixValue(controller_context, np.array([target[0], target[1], 0, 0]))

    simulator = Simulator(diagram, context)
    # 5th order: far fewer (Python) derivative evaluations than the default RK3 at tight accuracy
    ResetIntegratorFromFlags(simulator, "runge_kutta5", 0.1)
    simulator.get_mutable_integrator().set_target_accuracy(accuracy)

    if meshcat is not None:
        draw_target(meshcat, target)
        meshcat.StartRecording(frames_per_second=30)
    simulator.AdvanceTo(sim_time)

    logs = {name: logger.FindLog(context) for name, logger in loggers.items()}
    if meshcat is not None:
        meshcat.StopRecording()
        meshcat.get_mutable_recording().set_loop_mode(MeshcatAnimation.LoopMode.kLoopRepeat)
        meshcat.PublishRecording()
        draw_path(meshcat, logs["EE"].data()[:2])

    return logs, plant


def summary(logs, plant, target=TARGET):
    """Final tip error, final q and the range of cond(Jac_h) along the run."""
    q = logs["q"].data()
    ee = logs["EE"].data()
    tip_error = np.hypot(target[0] - ee[0, -1], target[1] - ee[1, -1])
    jac_h = PlanarSoftPCC_Theta.jac_h
    cond = [np.linalg.cond(jac_h(plant, q[:plant.n, k])) for k in range(q.shape[1])]
    return {"tip_error": tip_error, "q_end": q[:plant.n, -1], "cond_jac_h": (min(cond), max(cond))}


if __name__ == "__main__":
    meshcat = Meshcat()
    logs, plant = simulate(collocated=True, meshcat=meshcat)
    s = summary(logs, plant)
    print(f"tip error at the end: {s['tip_error']:.2e} m, q at the end: {np.round(s['q_end'], 3)}, "
          f"cond(Jac_h) in [{s['cond_jac_h'][0]:.1f}, {s['cond_jac_h'][1]:.1f}]")
    print(f"Meshcat: {meshcat.web_url()}  (Open Controls > Animations to replay)", flush=True)
    while True:
        time.sleep(60)
