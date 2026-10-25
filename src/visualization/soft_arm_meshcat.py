import numpy as np

from pydrake.geometry import Box, Cylinder, Rgba, Sphere
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.systems.framework import BasicVector, EventStatus, LeafSystem


SEGMENT_COLORS = [Rgba(0.15, 0.45, 0.85, 1), Rgba(0.95, 0.55, 0.15, 1),
                  Rgba(0.20, 0.70, 0.35, 1), Rgba(0.75, 0.30, 0.60, 1)]


class SoftArmVisualizer(LeafSystem):

    """Draws the backbone of a planar soft arm in Meshcat, as a chain of thin cylinders.

    Input "backbone": (x0, y0, x1, y1, ...) from the base to the tip, as given by the plant. The
    geometry is created here, before any recording starts: publishing only moves it, passing the
    context time as time_in_recording so that a Meshcat recording replays the motion.
    """

    def __init__(self, meshcat, n_points, points_per_segment, piece_length, radius=0.02,
                 period=1/30, prefix="soft_arm"):

        LeafSystem.__init__(self)

        self.meshcat = meshcat
        self.prefix = prefix
        self.n_pieces = n_points - 1

        self.DeclareVectorInputPort("backbone", BasicVector(2 * n_points))
        self.DeclareInitializationPublishEvent(self._publish)
        self.DeclarePeriodicPublishEvent(period, 0.0, self._publish)

        meshcat.SetObject(f"{prefix}/base", Box(0.04, 0.12, 0.08), Rgba(0.4, 0.4, 0.4, 1))
        meshcat.SetTransform(f"{prefix}/base", RigidTransform([-0.02, 0.0, 0.0]))
        for k in range(self.n_pieces):
            color = SEGMENT_COLORS[(k // points_per_segment) % len(SEGMENT_COLORS)]
            # slightly longer than the piece, so the chain looks continuous where it bends
            meshcat.SetObject(f"{prefix}/piece_{k}", Cylinder(radius, 1.05 * piece_length), color)
        meshcat.SetObject(f"{prefix}/tip", Sphere(1.4 * radius), Rgba(0.2, 0.2, 0.2, 1))

    def _publish(self, context):

        p = self.get_input_port().Eval(context).reshape(-1, 2)
        t = context.get_time()
        d = p[1:] - p[:-1]
        mid = (p[1:] + p[:-1]) / 2
        phi = np.arctan2(d[:, 1], d[:, 0])
        for k in range(self.n_pieces):
            # Cylinder axis is +z: turn it onto x, then rotate it in the plane
            R = RotationMatrix.MakeZRotation(phi[k]) @ RotationMatrix.MakeYRotation(np.pi / 2)
            self.meshcat.SetTransform(f"{self.prefix}/piece_{k}", RigidTransform(R, [mid[k, 0], mid[k, 1], 0.0]), t)
        self.meshcat.SetTransform(f"{self.prefix}/tip", RigidTransform([p[-1, 0], p[-1, 1], 0.0]), t)
        return EventStatus.Succeeded()


def draw_target(meshcat, xy, path="target", rgba=Rgba(0.85, 0.15, 0.15, 1)):
    meshcat.SetObject(path, Sphere(0.025), rgba)
    meshcat.SetTransform(path, RigidTransform([xy[0], xy[1], 0.0]))


def draw_path(meshcat, xy, path="tip_path", rgba=Rgba(0.1, 0.1, 0.1, 1)):
    """Static line through the points xy (2 x N), e.g. the logged tip positions."""
    xy = np.asarray(xy, float)
    meshcat.SetLine(path, np.vstack((xy, np.zeros((1, xy.shape[1])))), 2.0, rgba)
