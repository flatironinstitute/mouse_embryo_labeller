"""
Tools for fitting 3d point collections to ellipsoid containers.
"""

import numpy as np
import scipy.optimize as opt
from numpy.linalg import norm

def vv(*args):
    "Array creation convenience."
    return np.array(args, dtype=np.float)

# 3d Affine transformation matrices:

def apply_affine_transform(transform_matrix, points3d):
    points1 = np.ones((len(points3d), 4), dtype=np.float)
    points1[:, :3] = points3d
    transformed = transform_matrix.dot(points1.T).T[:, :3]
    return transformed

def rotate_xy(theta):
    "Rotation around the z axis."
    cs = np.cos(theta)
    sn = np.sin(theta)
    return vv(
        [cs, sn, 0, 0],
        [-sn, cs, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    )

def rotate_yz(theta):
    "Rotation around the x axis."
    cs = np.cos(theta)
    sn = np.sin(theta)
    return vv(
        [1, 0, 0, 0],
        [0, cs, sn, 0],
        [0, -sn, cs, 0],
        [0, 0, 0, 1],
    )

def rotate_xz(theta):
    "Rotation around the y axis."
    cs = np.cos(theta)
    sn = np.sin(theta)
    return vv(
        [cs, 0, sn, 0],
        [0, 1, 0, 0],
        [-sn, 0, cs, 0],
        [0, 0, 0, 1],
    )

def translate3d(t):
    return vv(
        [1, 0, 0, t[0]],
        [0, 1, 0, t[1]],
        [0, 0, 1, t[2]],
        [0, 0, 0, 1]
    )
        
def scale3d(s):
    return vv(
        [s[0], 0, 0, 0],
        [0, s[1], 0, 0],
        [0, 0, s[2], 0],
        [0, 0, 0, 1]
    )

def box_border_polys(center, v_up, v_right, v_back, shrink=0.8):
    "For drawing reference frames -- Polygons for a box using side vectors with sides shrunk towards the side center."
    [center, v_up, v_right, v_back] = [vv(*x) for x in [center, v_up, v_right, v_back]]
    polys = []
    for (origin, up, right, back) in [
        (center - v_up - v_right - v_back, v_up, v_right, v_back),
        (center + v_up + v_right + v_back, - v_up, - v_right, - v_back),
    ]:
        for (u, r) in [(up, right), (up, back), (right, back)]:
            poly = rect_points(origin, 2*u, 2*r, shrink)
            polys.append(poly)
    return vv(*polys)
    
def rect_points(start, v_up, v_right, shrink):
    "Rectangle polygon points shrunk towards center.  For drawing reference frames."
    [start, v_up, v_right] = [vv(*x) for x in [start, v_up, v_right]]
    corners = vv(start, start + v_up, start + v_up + v_right, start + v_right)
    center = corners.mean(axis=0)
    shrunk_corners = (shrink * corners) + ((1.0 - shrink) * center.reshape(1, 3))
    return shrunk_corners

# For drawing: circles that outline a unit sphere offset by X, Y, and Z.
circles1 = []
for i in range(3):
    phi = (i + 0.5) * np.pi / 6
    z = np.sin(phi)
    cphi = np.cos(phi)
    c = (8 - i) * 4
    P = np.zeros((c, 3), dtype=np.float)
    P[:, 2]= z
    thetas = (2 * np.pi / c) * np.arange(c)
    P[:, 0] = np.sin(thetas) * cphi
    P[:, 1] = np.cos(thetas) * cphi
    circles1.append(P)
    
circles2 = []
for P in circles1:
    circles2.append(P)
    Pm = -P
    Pm[:, 0] = P[:, 0]
    Pm[:, 1] = P[:, 1]
    circles2.append(Pm)
    
circles = []
for P in circles2:
    circles.append(P)
    Pc = P.copy()
    #print(P.shape, Pc.shape)
    Pc[:, 2] = P[:, 0]
    Pc[:, 0] = P[:, 2]
    circles.append(Pc)
    Pc = P.copy()
    Pc[:, 2] = P[:, 1]
    Pc[:, 1] = P[:, 2]
    circles.append(Pc)

sphere_boundary_points = np.vstack(circles)

HSIDE = 5
SIDE = HSIDE * 2
sphere_points = np.zeros((SIDE * SIDE * SIDE, 3), dtype=np.float)
count = 0
scale = 1.0 / HSIDE
for I in range(-HSIDE, HSIDE):
    x = I * scale
    for J in range(-HSIDE, HSIDE):
        y = J * scale
        for K in range(-HSIDE, HSIDE):
            z = K * scale
            if ((x * x) + (y * y) + (z * z)) <= 1.0:
                sphere_points[count] = (x, y, z)
                count += 1
sphere_points = sphere_points[:count]

class EllipsoidFitter:
        
    def __init__(self, points, penalty_factor=10.0):
        """
        Fit an ellipsoid around the XYZ points.
        The ellipse is penalized for "violations" which measure the points exceeding the ellipse boundary,
        and for "size" which measures the volume of the ellipse.
        A higher penalty_factor weights the violations more strongly than the size.
        A lower penalty factor prefers a smaller ellipsoid and allows larger violations (more outliers).
        """
        #self.difference_factor = difference_factor
        self.penalty_factor = penalty_factor
        P = self.points = np.array(points, dtype=np.float)
        (self.npoints, three) = P.shape
        assert three == 3, "Only 3d points supported " + repr(P.shape)
        self.maxes = maxes = P.max(axis=0)
        self.mins = mins = P.min(axis=0)
        self.mid = 0.5 * (maxes + mins)
        self.diff = diff = maxes - mins
        md = self.mdiff = diff.max()
        #self.current_polygon = None
        self.text = None
        # Rotations (one extra to allow flexibility).
        self.xy_alpha = 0
        self.xz_alpha = 0
        self.yz_alpha = 0
        # Relative scale (as exponents)
        self.xy_scale_exponent = 0
        self.xz_scale_exponent = 0
        # Initial size
        self.size = self.mdiff
        #self.size = 4   # DEBUG
        self.c = self.mid #self.mid
        self.initial_guess = self.guess_list()
        self.last_guess = self.initial_guess
        self._widget = None

    def guess_list(self):
        "Geometry parameters packed as a sequence (for use in scipy.optimize)."
        result = list(self.c) + [
            self.size,
            self.xy_scale_exponent,
            self.xz_scale_exponent,
            self.xy_alpha,
            self.xz_alpha,
            self.yz_alpha,
        ]
        #print("guess", result)
        return result
    
    def unpack_guess(self, guess):
        "Geometry parameters unpacked from a sequence (as returned by scipy.optimize)."
        self.c = vv(*guess[:3])
        [
            self.size,
            self.xy_scale_exponent,
            self.xz_scale_exponent,
            self.xy_alpha,
            self.xz_alpha,
            self.yz_alpha,
        ] = guess[3:]

    def transform_matrix(self):
        """
        Affine transformation which transforms XYZ points into a space where the fitted ellipsoid
        is a unit sphere centered at the origin.
        """
        scale_factor1 = np.exp(self.xy_scale_exponent)
        scale_factor2 = np.exp(self.xz_scale_exponent)
        scale_factor0 = 1.0 / (scale_factor1 * scale_factor2)
        scale_vector = (1.0 / self.size) * vv(scale_factor0, scale_factor1, scale_factor2)
        scale = scale3d(scale_vector)
        xy_rotation = rotate_xy(self.xy_alpha)
        yz_rotation = rotate_yz(self.yz_alpha)
        xz_rotation = rotate_xz(self.xz_alpha)
        rotate = xy_rotation.dot(yz_rotation.dot(xz_rotation))
        translate = translate3d(- self.c)
        return scale.dot(rotate).dot(translate)

    def transform_points(self, points, M=None):
        "Apply affine transformation to XYZ points."
        if M is None:
            M = self.transform_matrix()
        return apply_affine_transform(M, points)

    def penalty(self, guess=None):
        """
        Compute the penalty for this guessed transformation.
        """
        if guess is None:
            guess = self.last_guess
        self.unpack_guess(guess)
        points = self.points
        # Transform the points to the unit sphere space
        transformed = self.transform_points(points)
        # Find the violations for the points outside of the sphere, scaled by the penalty factor.
        norms = norm(transformed, axis=1)
        violations = np.maximum(0, norms - 1).sum() ** 2 * self.penalty_factor
        avg_violation = violations / len(norms)
        # Compute the penalty for the size.
        size_penalty = (self.size / self.mdiff) ** 2
        penalty = size_penalty + avg_violation
        #print("penalty", penalty, "size penalty", size_penalty, "violations", violations)
        return penalty 

    def optimize(self, draw=True, method='BFGS'):
        """
        Attempt to minimize the penalty function using a method of scipy.optimize.minimize.
        """
        callback = None
        if draw:
            callback = self.draw_guess
        self.optimized = opt.minimize(self.penalty, x0=self.initial_guess, callback=callback, method=method)
        if draw:
            self.sphere3d.orbit_all(radius=3, center3d=[0,0,0])
            self.ellipsoid3d.orbit_all(radius=2*self.mdiff, center3d=self.mid)
        return self.optimized

    def draw_guess(self, guess=None, delay=0.1, reset=True):
        import time
        assert self._widget != None, "Cannot draw -- no widget target."
        if guess is None:
            guess = self.last_guess
        if delay:
            time.sleep(delay)
        sf = self.sphere3d
        ef = self.ellipsoid3d
        M = self.transform_matrix()
        info = EllipsoidInfo(M)
        #invM = np.linalg.inv(M)
        invM = info.Minv
        d2 = 0.5 * self.diff
        with self.ellipsoid_canvas.delay_redraw():
            if reset:
                ef.reset()
            self.draw_circles(self.ellipsoid3d, invM)
            self.draw_box(self.ellipsoid3d, self.mid, d2)
            #self.draw_points(self.ellipsoid3d)
            info.annotate_points(self.ellipsoid3d, self.points, "green", "red")
        with self.sphere_canvas.delay_redraw():
            if reset:
                sf.reset()
            self.draw_circles(self.sphere3d)
            self.draw_points(self.sphere3d, M)
            self.draw_box(self.sphere3d, self.mid, d2, None, M)

    def draw_circles(self, f3d, M=None):
        for circle in circles:
            self.draw_points(f3d, points=circle, M=M, color="cyan", polygon=True)

    def draw_points(self, f3d, M=None, points=None, color="blue", polygon=False):
        if points is None:
            points = self.points
        if M is not None:
            points = self.transform_points(points, M)
        if polygon:
            f3d.polygon(points, color=color, fill=False)
        else:
            for p in points:
                f3d.circle(p, color=color, r=2, fill=False)

    def widget(self, side=500):
        from jp_doodle import dual_canvas, nd_frame
        import ipywidgets as widgets
        # ellipsoid canvas and frames:
        ellipsoid_canvas = self.ellipsoid_canvas = dual_canvas.DualCanvasWidget(width=side, height=side)
        mdiff = self.mdiff
        offset = mdiff * 0.7
        (minx, miny, minz) = self.mid - offset
        (maxx, maxy, maxz) = self.mid + offset
        self.ellipsoid2d = ellipsoid_canvas.frame_region(
            0, 0, side, side,
            minx, miny, maxx, maxy,
        )
        self.ellipsoid3d = nd_frame.ND_Frame(self.ellipsoid_canvas, self.ellipsoid2d)
        self.ellipsoid3d.orbit(center3d=self.mid, radius=3*offset, shift2d=(offset/2, offset/3))
        self.draw_box(self.ellipsoid3d, self.mid, 0.5 * self.diff, "pink")
        ellipsoid_canvas.fit()
        # sphere canvas and frames
        sphere_canvas = self.sphere_canvas = dual_canvas.DualCanvasWidget(width=side, height=side)
        self.sphere2d = sphere_canvas.frame_region(
            0, 0, side, side,
            -2, -2, 2, 2
        )
        self.sphere3d = nd_frame.ND_Frame(self.sphere_canvas, self.sphere2d)
        self.sphere3d.orbit(center3d=[0,0,0], radius=3, shift2d=(0.5, 0.3))
        self.draw_box(self.sphere3d, [0,0,0], [1,1,1], "pink")
        sphere_canvas.fit()
        # combo widget
        widget = self._widget = widgets.HBox(children=[ellipsoid_canvas, sphere_canvas])
        self.draw_guess(delay=None, reset=False)
        return widget

    box_colors = "red brown magenta goldenrod salmon olive".split()

    def draw_box(self, frame3d, center, offset, color=None, M=None):
        if color is None:
            colors = list(self.box_colors)
        else:
            colors = [color] * 6
        sides = box_border_polys(center, [offset[0], 0, 0], [0, offset[1], 0], [0, 0, offset[2]])
        for poly in sides:
            color = colors.pop()
            if M is not None:
                poly = self.transform_points(poly, M)
            frame3d.polygon(poly, color=color, fill=False)

    def get_info(self):
        """
        Get the ellipsoid info corresponding to the last guess.
        """
        return EllipsoidInfo(self.transform_matrix())

class EllipsoidInfo:

    def __init__(self, transform_matrix):
        """
        Calculations for a 3d ellipsoid.
        A point is on the ellipsoid if the transformed point lies on the unit sphere.
        """
        self.M = transform_matrix
        self.Minv = np.linalg.inv(self.M)
        [self.center] = apply_affine_transform(self.Minv, vv([0,0,0]))

    def axes(self):
        """
        Return inverse transform of X Y Z offsets for the sphere space.
        If the transformation was generated using EllipsoidFit above then these
        axes will correspond to the major axes of the ellipsoid.
        """
        axis_vertices = vv([1,0,0], [0,1,0], [0,0,1])
        translated_axis_vertices = apply_affine_transform(self.Minv, axis_vertices)
        return translated_axis_vertices - self.center.reshape((1, 3))

    def axis_lengths(self):
        return [norm(x) for x in self.axes()]

    def volume(self):
        """
        Volume computed from axis lengths assuming the matrix was generated by EllipsoidFit.
        """
        [a, b, c] = self.axis_lengths()
        # https://en.wikipedia.org/wiki/Ellipsoid
        return (4 * np.pi / 3.0) * a * b * c

    def surface_area(self):
        """
        Approximate surface area computed from axis lengths assuming the matrix was generated by EllipsoidFit.
        From https://en.wikipedia.org/wiki/Ellipsoid.
        """
        p = 1.6075
        [a, b, c] = self.axis_lengths()
        ap = a ** p 
        bp = b ** p 
        cp = c ** p 
        sm = ap * bp + ap * cp + bp * cp
        rt = np.sqrt(sm / 3.0)
        return 4 * np.pi * rt

    def sphericity(self):
        # https://en.wikipedia.org/wiki/Sphericity
        V = self.volume()
        A = self.surface_area()
        pithird = np.pi ** (1.0/3.0)
        numerator = pithird * ((6.0 * V) ** (2.0/3.0))
        return numerator / A

    def offset_to_ellipse(self, point3d, epsilon=1e-10):
        """
        return distances d and c and normalized vector n
        
        >>> (d, c, n) = info.offseet_to_ellipse(poing3d)

        such that point3d + c * n is the center of the ellipsoid and
        point3d + d * n lies on the ellipsoid on the line segment between point3d and the center.
        The distance c will always be positive or zero but d will be negative if point3d
        lies inside the ellipsoid and positive if outside.
        """
        point3d = np.array(point3d, dtype=np.float)
        [tp] = apply_affine_transform(self.M, vv(point3d))
        to_origin = - tp
        td = norm(to_origin)
        if td < epsilon:
            # pick an arbitrary direction
            td = 0.0
            tn = vv(1, 0, 0)
        else:
            tn = to_origin / td
        [center, proj_on_ellipsoid] = apply_affine_transform(self.Minv, vv([0,0,0], tn))
        to_center = proj_on_ellipsoid - center
        proj_offset = norm(to_center)
        n = to_center / proj_offset
        c = norm(point3d - center)
        d = c - proj_offset
        return (d, c, n)

    def offsets_to_ellipse(self, points3d, epsilon=1e-10):
        points3d = np.array(points3d, dtype=np.float)
        tps = apply_affine_transform(self.M, points3d)
        (N, three) = tps.shape
        assert three == three, "bad shape: " + repr(tps.shape)
        to_origin = - tps
        tds = norm(to_origin, axis=1)
        assert tds.shape == (N,), repr(tds.shape)
        too_small = tds < epsilon
        tds = np.where(too_small, 1.0, tds)  # remove too small divisors to avoid errors
        tns = to_origin / (tds.reshape((N, 1)))
        tns = np.where(too_small.reshape((N, 1)), vv([1, 0, 0]), tns)  # arbitrary normal where too small
        proj_on_ellipsoid = apply_affine_transform(self.Minv, tns)
        center = self.center
        reshaped_center = center.reshape((1,3))
        to_center = proj_on_ellipsoid - reshaped_center
        proj_offset = norm(to_center, axis=1)
        too_small = proj_offset < epsilon
        a = self.axis_lengths()[0]
        proj_offset = np.where(too_small, a, proj_offset)  # avoid errors
        ns = to_center / (proj_offset.reshape((N, 1)))
        ns = np.where(too_small.reshape((N, 1)), vv(1,0,0), ns)
        cs = norm(points3d - reshaped_center, axis=1)
        ds = cs - proj_offset
        return (ds, cs, ns)

    def other_projection(self, other_info, sphere_points=sphere_boundary_points):
        """
        Project the points of this ellipse into the sphere space for the other ellipse.
        Used for testing inclusion.
        """
        ellipse_points = apply_affine_transform(self.Minv, sphere_points)
        other_sphere_points = apply_affine_transform(other_info.M, ellipse_points)
        return other_sphere_points

    def is_inside(self, other_info, epsilon=1e-10):
        "Quickly test whether self is almost completely contained in other"
        boundary_in_other_sphere = self.other_projection(other_info)
        norms = norm(boundary_in_other_sphere, axis=1)
        assert norms.shape == (len(boundary_in_other_sphere),)
        return np.all(boundary_in_other_sphere < (1.0 + epsilon))

    def proportion_inside_of(self, other_info, epsilon=1e-10):
        """
        Return approximate proportion of this ellipse inside the other ellipse by relative volume.
        Returns 1 if completely inside and 0 if there is (nearly) no intersection.
        """
        points_in_other_sphere = self.other_projection(other_info, sphere_points)
        norms = norm(points_in_other_sphere, axis=1)
        assert norms.shape == (len(points_in_other_sphere),)
        inside = (norms < (1.0 + epsilon))
        (total,) = inside.shape
        count = inside.astype(np.int).sum()
        if count == total:
            return 1
        return count * (1.0 / total)

    def relative_offset_to_center(self, point3d):
        """
        Linear measure of whether the point is near the center of the ellipse or the boundary.
        Returns 0 at the center, 1 at the boundary of the ellipse and > 1 outside the ellipse.
        """
        projection = apply_affine_transform(self.M, [point3d])
        return norm(projection[0])

    def relative_offset_to_center_of(self, other_info):
        """
        Linear measure of how near the center of this ellipse is to the center of the other
        ellipse.  Returns 0 at the center, 1 at the boundary of the ellipse and > 1 outside the ellipse.
        """
        return other_info.relative_offset_to_center(self.center)

    def annotate_points(self, f3d, points, inside_color, outside_color):
        for p in points:
            (d, c, n) = self.offset_to_ellipse(p)
            if d <= 0:
                color = inside_color
                fill = False
            else:
                color = outside_color
                fill = True
            f3d.circle(p, r=2, color=color, fill=fill)