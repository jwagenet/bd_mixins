from math import radians, cos, sin, sqrt
import copy as copy_module

from build123d import *
from build123d.build_common import WorkplaneList, flatten_sequence, validate_inputs
from build123d.objects_curve import BaseEdgeObject
from build123d.geometry import Axis, Vector, VectorLike, TOLERANCE
from build123d.topology import Edge, Face, Wire, Curve

from scipy.optimize import minimize
import sympy

from ocp_vscode import *


class PointArcTangentLine(BaseEdgeObject):
    """Line Object: Point Arc Tangent Line

    Create a straight, tangent line from a point to a circular arc.

    Args:
        point (VectorLike): intersection point for tangent
        arc (Curve | Edge | Wire): circular arc to tangent, must be GeomType.CIRCLE
        side (Side, optional): side of arcs to place tangent arc center, LEFT or RIGHT. 
            Defaults to Side.LEFT
        mode (Mode, optional): combination mode. Defaults to Mode.ADD
    """

    def __init__(
        self,
        point: VectorLike,
        arc: Curve | Edge | Wire,
        side: Side = Side.LEFT,
        mode: Mode = Mode.ADD,
        ):

        side_sign = {
            Side.LEFT: -1,
            Side.RIGHT: 1,
        }

        context: BuildLine | None = BuildLine._get_context(self)
        validate_inputs(context, self)

        tangent_point  = WorkplaneList.localize(point)
        if context is None:
            # Making the plane validates points and arc are coplanar
            workplane = Edge.make_line(tangent_point, arc.arc_center).common_plane(
                *arc.edges()
            )
            if workplane is None:
                raise ValueError("PointArcTangentLine only works on a single plane.")
        else:
            workplane = copy_module.copy(
                WorkplaneList._get_context().workplanes[0]
            )

        arc_center = arc.arc_center
        radius = arc.radius
        midline = tangent_point - arc_center

        if midline.length < radius:
            raise ValueError("Cannot find tangent for point inside arc.")

        # Find angle phi between midline and x
        # and angle theta between midplane length and radius
        # add the resulting angles with a sign on theta to pick a direction
        # This angle is the tangent location around the circle from x
        phi = midline.get_signed_angle(workplane.x_dir)
        other_leg = sqrt(midline.length ** 2 - radius ** 2)
        theta = WorkplaneList.localize((radius, other_leg)).get_signed_angle(workplane.x_dir)
        angle = side_sign[side] * theta + phi
        intersect = WorkplaneList.localize((
            radius * cos(radians(angle)),
            radius * sin(radians(angle)))
            ) + arc_center

        tangent = Edge.make_line(intersect, tangent_point)
        super().__init__(tangent, mode)


class PointArcTangentArc(BaseEdgeObject):
    """Line Object: Point Arc Tangent Arc

    Create an arc defined by a point/tangent pair and another line which the other end
    is tangent to.

    Contains a solver.

    Args:
        pnt (VectorLike): starting point of tangent arc
        tangent (VectorLike): direction at starting point of tangent arc
        other (Union[Curve, Edge, Wire]): reference line
        side (Side, optional): select which arc to keep Defaults to Side.LEFT.
        mode (Mode, optional): combination mode. Defaults to Mode.ADD.

    Raises:
        RuntimeError: Point is already tangent to other!
        RuntimeError: No tangent arc found.
    """

    _applies_to = [BuildLine._tag]

    def __init__(
        self,
        point: VectorLike,
        direction: VectorLike,
        arc: Curve | Edge | Wire,
        side: Side = Side.LEFT,
        mode: Mode = Mode.ADD,
        old_method=False
    ):
        context: BuildLine | None = BuildLine._get_context(self)
        validate_inputs(context, self)

        arc_point = WorkplaneList.localize(point)
        wp_tangent = WorkplaneList.localize(direction).normalized()

        if context is None:
            # Making the plane validates pnt, tangent, and other are coplanar
            workplane = Edge.make_line(arc_point, arc_point + wp_tangent).common_plane(
                *arc.edges()
            )
            if workplane is None:
                raise ValueError("PointArcTangentArc only works on a single plane")
        else:
            workplane = copy_module.copy(
                WorkplaneList._get_context().workplanes[0]
            )

        arc_tangent = Vector(direction).transform(
            workplane.reverse_transform, is_direction=True
        ).normalized()

        # Determine where arc_point is located relative to other
        # ref forms a bisecting line parallel to arc tangent with same distance from other
        # center as arc point in direction of arc tangent
        tangent_perp = arc_tangent.cross(workplane.z_dir)
        ref_scale = (arc.arc_center - arc_point).dot(-arc_tangent)
        ref = ref_scale * arc_tangent + arc.arc_center
        ref_to_point = (arc_point - ref).dot(tangent_perp)

        print(ref)
        print(ref_scale, ref_to_point, (arc_tangent + arc.arc_center).normalized(), arc_tangent)
        # show_object([Line(arc.arc_center, ref)])

        side_sign = -1 if side == Side.LEFT else 1
        # Tangent radius to infinity (and beyond)
        if side_sign * ref_to_point == arc.radius:
            raise RuntimeError("Point is already tangent to other!")

        if old_method:
            rotation_axis = Axis(workplane.origin, workplane.z_dir)

            # Use magnitude and sign of ref to arc point along with keep to determine
            #   which "side" angle the arc center will be on
            # - the arc center is the same side if the point is further from ref than other radius
            # - minimize type determines near or far side arc to minimize to
            side_sign = 1 if ref_to_point < 0 else -1
            minimize_type = 1
            if abs(ref_to_point) < arc.radius:
                # point/tangent pointing inside other, both arcs near
                if side == Side.LEFT:
                    angle = 90
                else:
                    angle = -90
            else:
                # point/tangent pointing outside other, one near arc one far
                angle = side_sign * -90
                if side == Side.LEFT:
                    minimize_type = side_sign * -minimize_type
                else:
                    minimize_type = side_sign * minimize_type

            # Protect against massive circles that are effectively straight lines
            max_size = 10 * arc.bounding_box().add(arc_point).diagonal

            # Function to be minimized - note radius is a numpy array
            def func(radius, perpendicular_bisector, minimize_type):
                center = arc_point + perpendicular_bisector * radius[0]
                separation = arc.distance_to(center)

                if minimize_type == 1:
                    # near side arc
                    target = abs(separation - radius)
                elif minimize_type == -1:
                    # far side arc
                    target = abs(separation - radius + arc.radius * 2)

                return target

            # Find arc center by minimizing func result
            perpendicular_bisector = arc_tangent.rotate(rotation_axis, angle)
            result = minimize(
                func,
                x0=0,
                args=(perpendicular_bisector, minimize_type),
                method="Nelder-Mead",
                bounds=[(0.0, max_size)],
                tol=TOLERANCE,
            )
            tangent_radius = result.x[0]
            tangent_center = arc_point + perpendicular_bisector * tangent_radius

            # dir needs to be flipped for far arc
            tangent_normal = (arc.arc_center - tangent_center).normalized()
            tangent_dir = minimize_type * tangent_normal.cross(workplane.z_dir)
            tangent_point = tangent_radius * tangent_normal + tangent_center

            # Confirm tangent point is on other
            if abs(arc.radius - (tangent_point - arc.arc_center).length) > TOLERANCE:
                # If we find the point on other where the tangent is parallel to arc tangent
                # 1. form a line 1 following that tangent
                # 2. form a line 2 from arc point to that point
                # as the distance between the line 1 and arc_point goes to 0 and
                # the angle between line 2 and arc tangent approaches 0 or 180
                # the minimize will fail at max_size
                #
                # distance = ref_to_point + minimize_type * (angle / 90) * other.radius
                # angle = (other.arc_center - side_sign * normal * other.radius - arc_pt).get_angle(arc_tangent)
                #
                # This should be the only way this error arises
                raise RuntimeError("No tangent arc found, no tangent point found")

            # Confirm new tangent point is colinear with point tangent on other
            other_dir = arc.tangent_at(tangent_point)
            print(other_dir, tangent_dir, tangent_dir.get_angle(other_dir))
            if tangent_dir.get_angle(other_dir) > TOLERANCE:
                raise RuntimeError("No tangent arc found, found tangent out of tolerance")

        else:
            # Method:
            # - Draw line perpendicular to direction with length of arc radius away from arc
            # - Draw line from this point (ref_perp) to arc center, find angle with ref_perp
            # - Find length of segment along this line from ref_perp to direction intercept
            # - This segment is + or - from length ref_prep to arc center to find ref_radius
            # - Find intersections arcs with ref_radius from ref_center and arc center
            # - The intercept of this line with perpendicular is the tangent arc center
            # Side.LEFT is always the arc further ccw per right hand rule

            # ref_radius and ref_center determined by table below
            # Position    Arc    Ref_radius Ref_center
            # outside     near   -seg       +perp
            # outside     far    +seg       -perp
            # inside to   near   +seg       +perp
            # inside from near   +seg       -perp

            pos_sign = 1 if round(ref_to_point, 6) < 0 else -1
            if abs(ref_to_point) <= arc.radius:
                arc_type = -1
                if ref_scale > 1:
                    # point/tangent pointing from inside other, two near arcs
                    other_sign = pos_sign * side_sign
                else:
                    # point/tangent pointing to inside other, two near arcs
                    other_sign = -pos_sign * side_sign
            else:
                # point/tangent pointing outside other, one near arc one far
                other_sign = 1
                arc_type = side_sign * pos_sign

            # Find perpendicular and located it to ref_perp and ref_center
            perpendicular = -pos_sign * direction.cross(Plane.XY.z_dir).normalized() * arc.radius
            ref_perp = perpendicular + arc_point
            ref_center = other_sign * arc_type * perpendicular + arc_point

            # Find ref_radius
            angle = perpendicular.get_angle(ref_perp - arc.arc_center)
            center_dist = (ref_perp - arc.arc_center).length
            segment = arc.radius / cos(radians(angle))
            if arc_type == 1:
                ref_radius = center_dist - segment
            elif arc_type == -1:
                ref_radius = center_dist + segment

            print(ref_radius)
            a = CenterArc(arc.arc_center, ref_radius, 0, 360)
            b = CenterArc(ref_center, ref_radius, 0, 360)
            # show_object()
            # show_object([ a,b])

            local = [workplane.to_local_coords(p) for p in [ref_center, arc.arc_center]]
            ref_circles = [sympy.Circle(sympy.Point2D(local[i].X, local[i].Y), ref_radius) for i in range(2)]
            ref_intersections = sympy.intersection(*ref_circles)

            line1 = sympy.Line(sympy.Point(arc_point.X, arc_point.Y), sympy.Point(ref_center.X, ref_center.Y))
            line2 = sympy.Line(*ref_intersections)
            intercept = line1.intersect(line2)

            tangent_center = workplane.from_local_coords(Vector(float(sympy.N(intercept.args[0].x)), float(sympy.N(intercept.args[0].y))))
            tangent_radius = (tangent_center - arc_point).length

            # dir needs to be flipped for far arc
            tangent_normal = (arc.arc_center - tangent_center).normalized()
            tangent_dir = arc_type * tangent_normal.cross(workplane.z_dir)
            tangent_point = tangent_radius * tangent_normal + tangent_center

            show_object([tangent_center])
            print(abs(arc.radius - (tangent_point - arc.arc_center).length))

            # Confirm tangent point is on arc
            if abs(arc.radius - (tangent_point - arc.arc_center).length) > TOLERANCE:
                raise RuntimeError("No tangent arc found, no tangent point found")

            # Confirm new tangent point is colinear with point tangent on arc
            other_dir = arc.tangent_at(tangent_point)
            print(other_dir, tangent_dir, tangent_dir.get_angle(other_dir))
            # if tangent_dir.get_angle(other_dir) > TOLERANCE:
            #     raise RuntimeError("No tangent arc found, found tangent out of tolerance")

        arc = TangentArc(arc_point, tangent_point, tangent=arc_tangent)
        super().__init__(arc.edge(), mode=mode)


class ArcArcTangentLine(BaseEdgeObject):
    """Line Object: Arc Arc Tangent Line

    Create a straight line tangent to two arcs.

    Args:
        start_arc (Curve | Edge | Wire): starting arc, must be GeomType.CIRCLE
        end_arc (Curve | Edge | Wire): ending arc, must be GeomType.CIRCLE
        side (Side): side of arcs to place tangent arc center, LEFT or RIGHT. 
            Defaults to Side.LEFT
        keep (Keep): which tangent arc to keep, INSIDE or OUTSIDE. 
            Defaults to Keep.INSIDE
        mode (Mode, optional): combination mode. Defaults to Mode.ADD
    """

    def __init__(
        self,
        start_arc: Curve | Edge | Wire,
        end_arc: Curve | Edge | Wire,
        side=Side.LEFT,
        keep=Keep.INSIDE,
        mode=Mode.ADD,
        ):

        context: BuildLine | None = BuildLine._get_context(self)
        validate_inputs(context, self)

        if context is None:
            # Making the plane validates start arc and end arc are coplanar
            workplane = start_arc.edge().common_plane(
                *end_arc.edges()
            )
            if workplane is None:
                raise ValueError("ArcArcTangentLine only works on a single plane.")
        else:
            workplane = copy_module.copy(
                WorkplaneList._get_context().workplanes[0]
            )

        side_sign = 1 if side == Side.LEFT else -1
        arcs = [start_arc, end_arc]
        points = [arc.arc_center for arc in arcs]
        radii = [arc.radius for arc in arcs]
        midline = points[1] - points[0]

        if midline.length == 0:
            raise ValueError("Cannot find tangent for concentric arcs.")

        if (keep == Keep.INSIDE or keep == Keep.BOTH):
            if midline.length < sum(radii):
                raise ValueError("Cannot find INSIDE tangent for overlapping arcs.")

            if midline.length == sum(radii):
                raise ValueError("Cannot find INSIDE tangent for tangent arcs.")

        # Method:
        # https://en.wikipedia.org/wiki/Tangent_lines_to_circles#Tangent_lines_to_two_circles
        # - angle to point on circle of tangent incidence is theta + phi
        # - phi is angle between x axis and midline
        # - OUTSIDE theta is angle formed by triangle legs (midline.length) and (r0 - r1)
        # - INSIDE theta is angle formed by triangle legs (midline.length) and (r0 + r1)
        # - INSIDE theta for arc1 is 180 from theta for arc0

        phi = midline.get_signed_angle(workplane.x_dir)
        radius = radii[0] + radii[1] if keep == Keep.INSIDE else radii[0] - radii[1]
        other_leg = sqrt(midline.length ** 2 - radius ** 2)
        theta = WorkplaneList.localize((radius, other_leg)).get_signed_angle(workplane.x_dir)
        angle = side_sign * theta + phi

        intersect = []
        for i in range(len(arcs)):
            angle = i * 180 + angle if keep == Keep.INSIDE else angle
            intersect.append(WorkplaneList.localize((
                radii[i] * cos(radians(angle)),
                radii[i] * sin(radians(angle)))
                ) + points[i])

        tangent = Edge.make_line(intersect[0], intersect[1])
        super().__init__(tangent, mode)


class ArcArcTangentArc(BaseEdgeObject):
    """Line Object: Arc Arc Tangent Arc

    Create an arc tangent to two arcs and a radius.

    Args:
        start_arc (Curve | Edge | Wire): starting arc, must be GeomType.CIRCLE
        end_arc (Curve | Edge | Wire): ending arc, must be GeomType.CIRCLE
        radius (float): radius of tangent arc
        side (Side): side of arcs to place tangent arc center, LEFT or RIGHT. 
            Defaults to Side.LEFT
        keep (Keep): which tangent arc to keep, INSIDE or OUTSIDE. 
            Defaults to Keep.INSIDE
        mode (Mode, optional): combination mode. Defaults to Mode.ADD.
    """

    def __init__(
        self,
        start_arc: Curve | Edge | Wire,
        end_arc: Curve | Edge | Wire,
        radius: float,
        side=Side.LEFT,
        keep=Keep.INSIDE,
        mode=Mode.ADD,
        ):

        context: BuildLine | None = BuildLine._get_context(self)
        validate_inputs(context, self)

        if context is None:
            # Making the plane validates start arc and end arc are coplanar
            workplane = start_arc.edge().common_plane(end_arc.edge())
            if workplane is None:
                raise ValueError("ArcArcTangentArc only works on a single plane.")

            # I dont know why, but workplane.z_dir is flipped from expected
            if workplane.z_dir != start_arc.normal():
                workplane = -workplane
        else:
            workplane = copy_module.copy(
                WorkplaneList._get_context().workplanes[0]
            )

        side_sign = 1 if side == Side.LEFT else -1
        keep_sign = 1 if keep == Keep.INSIDE else -1
        arcs = [start_arc, end_arc]
        points = [arc.arc_center for arc in arcs]
        radii = [arc.radius for arc in arcs]

        # make a normal vector for sorting intersections
        midline = points[1] - points[0]
        normal = side_sign * midline.cross(workplane.z_dir)

        # The range midline.length / 2 < tangent radius < math.inf should be valid
        # Sometimes fails if min_radius == radius, so using >=
        min_radius = (midline.length - keep_sign * (radii[0] + radii[1])) / 2
        if min_radius >= radius:
            raise ValueError(f"The arc radius is too small. Should be greater than {min_radius}.")

        old_method = False
        if old_method:
            net_radius = radius + keep_sign * (radii[0] + radii[1]) / 2

            # Technically the range midline.length / 2 < radius < math.inf should be valid
            if net_radius <= midline.length / 2:
                raise ValueError(f"The arc radius is too small. Should be greater than {(midline.length - keep_sign * (radii[0] + radii[1])) / 2} (and probably larger).")

            # Current intersection method doesn't work out to expected range and may return 0
            # Workaround to catch error midline.length / net_radius needs to be less than 1.888 or greater than .666 from testing
            max_ratio = 1.888
            min_ratio = .666
            if midline.length / net_radius > max_ratio:
                raise ValueError(f"The arc radius is too small. Should be greater than {midline.length / max_ratio - keep_sign * (radii[0] + radii[1]) / 2}.")

            if midline.length / net_radius < min_ratio:
                raise ValueError(f"The arc radius is too large. Should be less than {midline.length / min_ratio - keep_sign * (radii[0] + radii[1]) / 2}.")

        # Method:
        # https://www.youtube.com/watch?v=-STj2SSv6TU
        # - the centerpoint of the inner arc is found by the intersection of the
        #   arcs made by adding the inner radius to the point radii
        # - the centerpoint of the outer arc is found by the intersection of the
        #   arcs made by subtracting the outer radius from the point radii
        # - then it's a matter of finding the points where the connecting lines
        #   intersect the point circles

        if old_method:
            ref_arcs = [CenterArc(points[i], keep_sign * radii[i] + radius, start_angle=0, arc_size=360) for i in range(len(arcs))]
            ref_intersections = ref_arcs[0].edge().intersect(ref_arcs[1].edge())

            try:
                arc_center = ref_intersections.sort_by(Axis(points[0], normal))[0]
            except AttributeError as exception:
                raise RuntimeError("Arc radius thought to be okay, but is too big or small to find intersection.")

        else:
            local = [workplane.to_local_coords(p) for p in points]
            ref_circles = [sympy.Circle(sympy.Point2D(local[i].X, local[i].Y), keep_sign * radii[i] + radius) for i in range(len(arcs))]
            ref_intersections = ShapeList([workplane.from_local_coords(Vector(float(sympy.N(p.x)), float(sympy.N(p.y)))) for p in sympy.intersection(*ref_circles)])
            arc_center = ref_intersections.sort_by(Axis(points[0], normal))[0]

        intersect = [points[i] + keep_sign * radii[i] * (Vector(arc_center) - points[i]).normalized() for i in range(len(arcs))]

        if side == Side.LEFT:
            intersect.reverse()

        arc = RadiusArc(*intersect, radius=radius)
        super().__init__(arc, mode)


class AubergineSlot(BaseSketchObject):
    """Sketch Object: Aubergine Slot

    Add a curved slot of varying width defined by center length, and start, end, inner, and outer radii.

    Args:
        float (float): center spacing of slot
        start_radius (float): radius of start circle
        end_radius (float): radius of end circle
        inner_radius (float): radius of inner arc
        outer_radius (float): radius of outer arc
        side (Side): side of slot to place arc centers. Defaults to Side.LEFT
        rotation (float, optional): angles to rotate objects. Defaults to 0.
        mode (Mode, optional): combination mode. Defaults to Mode.ADD.
    """

    def __init__(self,
                 length,
                 start_radius,
                 end_radius,
                 inner_radius,
                 outer_radius,
                 side = Side.LEFT,
                 rotation = 0,
                 mode = Mode.ADD):

        start_point = Vector(0, 0)
        end_point = Vector(length, 0)

        center_side = 1 if side == Side.LEFT else -1

        c1 = CenterArc(start_point, start_radius, start_angle=0, arc_size=360)
        c2 = CenterArc(end_point, end_radius, start_angle=0, arc_size=360)

        a1 = ArcArcTangentArc(c1, c2, inner_radius, side=side, keep=Keep.INSIDE)
        a2 = ArcArcTangentArc(c1, c2, outer_radius, side=side, keep=Keep.OUTSIDE)
        a3 = TangentArc([a1 @ 0, a2 @ 0], tangent=center_side * (a1 % 0))
        a4 = TangentArc([a1 @ 1, a2 @ 1], tangent=center_side * -(a1 % 1))

        face = Face(Wire.combine([a1, a2, a3, a4]))

        super().__init__(obj=face, rotation=rotation, mode=mode)


class AubergineArcSlot(BaseSketchObject):
    """Sketch Object: Aubergine Arc Slot

    Add a curved slot of varying width defined by center length, and start, end, inner, and outer radii.

    Args:
        arc (Curve | Edge | Wire): reference arc
        start_height (float): height of start circle
        end_height (float): height of end circle
        width_factor (float): 
        mode (Mode, optional): combination mode. Defaults to Mode.ADD.
    """

    def __init__(self,
                 arc: Curve | Edge | Wire,
                 start_height: float,
                 end_height: float,
                 width_factor: float,
                 mode = Mode.ADD):

        start_arc = CenterArc(arc @ 0, start_height / 2, start_angle=0, arc_size=360)
        end_arc = CenterArc(arc @ 1, end_height / 2, start_angle=0, arc_size=360)
        midline = arc @ 1 - arc @ 0

        # factor = width_factor * (1.888 - .666) + .666
        # inner_radius = midline.length / factor - (start_height + end_height) / 4
        # outer_radius = midline.length / factor + (start_height + end_height) / 4

        inner_radius = arc.radius - width_factor
        outer_radius = arc.radius + width_factor

        inner_arc = ArcArcTangentArc(start_arc, end_arc, inner_radius, side=Side.LEFT, keep=Keep.INSIDE)
        outer_arc = ArcArcTangentArc(start_arc, end_arc, outer_radius, side=Side.LEFT, keep=Keep.OUTSIDE)
        start_arc = TangentArc([inner_arc @ 0, outer_arc @ 0], tangent=-(inner_arc % 0))
        end_arc = TangentArc([inner_arc @ 1, outer_arc @ 1], tangent=inner_arc % 1)

        face = make_face([start_arc, end_arc, inner_arc, outer_arc])
        super().__init__(obj=face, mode=mode)