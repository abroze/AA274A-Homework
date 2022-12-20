import numpy as np

EPSILON_OMEGA = 1e-3

def compute_Gx(xvec, u, dt):
    """
    Inputs:
                     xvec: np.array[3,] - Turtlebot state (x, y, theta).
                        u: np.array[2,] - Turtlebot controls (V, omega).
    Outputs:
        Gx: np.array[3,3] - Jacobian of g with respect to xvec.
    """
    ########## Code starts here ##########
    # TODO: Compute Gx
    # HINT: Since theta is changing with time, try integrating x, y wrt d(theta) instead of dt by introducing om
    # HINT: When abs(om) < EPSILON_OMEGA, assume that the theta stays approximately constant ONLY for calculating the next x, y
    #       New theta should not be equal to theta. Jacobian with respect to om is not 0.
    x_prev = xvec[0]
    y_prev = xvec[1]
    theta_prev = xvec[2]
    V = u[0]
    om = u[1]

    if abs(om) < EPSILON_OMEGA:
        Gx = np.array([[1, 0, -V * dt * np.sin(theta_prev)],
                      [0, 1, V * dt * np.cos(theta_prev)],
                      [0, 0, 1]])
    else:
        Gx = np.array([[1, 0, (V/om) * (np.cos(theta_prev + om * dt) - np.cos(theta_prev))],
            [0, 1, (V/om) * (np.sin(theta_prev + om * dt) - np.sin(theta_prev))],
                      [0, 0, 1]])

    ########## Code ends here ##########
    return Gx
    

def compute_Gu(xvec, u, dt):
    """
    Inputs:
                     xvec: np.array[3,] - Turtlebot state (x, y, theta).
                        u: np.array[2,] - Turtlebot controls (V, omega).
    Outputs:
        Gu: np.array[3,2] - Jacobian of g with respect to u.
    """
    ########## Code starts here ##########
    # TODO: Compute Gu
    # HINT: Since theta is changing with time, try integrating x, y wrt d(theta) instead of dt by introducing om
    # HINT: When abs(om) < EPSILON_OMEGA, assume that the theta stays approximately constant ONLY for calculating the next x, y
    #       New theta should not be equal to theta. Jacobian with respect to om is not 0.
    x_prev = xvec[0]
    y_prev = xvec[1]
    theta_prev = xvec[2]
    V = u[0]
    om = u[1]

    if abs(om) < EPSILON_OMEGA:
        dx_dV = dt * np.cos(theta_prev)
        dx_dom = -(1/2) * V * (dt**2) * np.sin(theta_prev)
        dy_dV = dt * np.sin(theta_prev)
        dy_dom = -(1/2) * V * (dt**2) * np.cos(theta_prev)
    else:
        dx_dV = (1/om) * (np.sin(theta_prev + om * dt) - np.sin(theta_prev))
        dx_dom = (V * dt / om) * np.cos(theta_prev + om * dt) - (V/om**2) * (np.sin(theta_prev + om * dt) - np.sin(theta_prev))
        dy_dV = (-1/om) * (np.cos(theta_prev + om * dt) - np.cos(theta_prev))
        dy_dom = (V * dt / om) * (np.sin(theta_prev + om * dt)) + (V/om**2) * (np.cos(theta_prev + om * dt) - np.cos(theta_prev))

    dtheta_dV = 0
    dtheta_dom = dt

    Gu = np.array([[dx_dV, dx_dom], [dy_dV, dy_dom], [dtheta_dV, dtheta_dom]])

    ########## Code ends here ##########
    return Gu


def compute_dynamics(xvec, u, dt, compute_jacobians=True):
    """
    Compute Turtlebot dynamics (unicycle model).

    Inputs:
                     xvec: np.array[3,] - Turtlebot state (x, y, theta).
                        u: np.array[2,] - Turtlebot controls (V, omega).
        compute_jacobians: bool         - compute Jacobians Gx, Gu if true.
    Outputs:
         g: np.array[3,]  - New state after applying u for dt seconds.
        Gx: np.array[3,3] - Jacobian of g with respect to xvec.
        Gu: np.array[3,2] - Jacobian of g with respect to u.
    """
    ########## Code starts here ##########
    # TODO: Compute g, Gx, Gu
    # HINT: To compute the new state g, you will need to integrate the dynamics of x, y, theta
    # HINT: Since theta is changing with time, try integrating x, y wrt d(theta) instead of dt by introducing om
    # HINT: When abs(om) < EPSILON_OMEGA, assume that the theta stays approximately constant ONLY for calculating the next x, y
    #       New theta should not be equal to theta. Jacobian with respect to om is not 0.

    x_prev = xvec[0]
    y_prev = xvec[1]
    theta_prev = xvec[2]
    V = u[0]
    om = u[1]

    if abs(om) < EPSILON_OMEGA:
        x_t = x_prev + V * dt * np.cos(theta_prev)
        y_t = y_prev + V * dt * np.sin(theta_prev)
        theta_t = theta_prev + om * dt
    else:
        theta_t = theta_prev + om * dt
        x_t = x_prev + V/om * (np.sin(theta_t) - np.sin(theta_prev))
        y_t = y_prev - V/om * (np.cos(theta_t) - np.cos(theta_prev))

    g = np.array([x_t, y_t, theta_t])
    Gx = compute_Gx(xvec, u, dt)
    Gu = compute_Gu(xvec, u, dt)

    ########## Code ends here ##########

    if not compute_jacobians:
        return g

    return g, Gx, Gu

def transform_line_to_scanner_frame(line, x, tf_base_to_camera, compute_jacobian=True):
    """
    Given a single map line in the world frame, outputs the line parameters
    in the scanner frame so it can be associated with the lines extracted
    from the scanner measurements.

    Input:
                     line: np.array[2,] - map line (alpha, r) in world frame.
                        x: np.array[3,] - pose of base (x, y, theta) in world frame.
        tf_base_to_camera: np.array[3,] - pose of camera (x, y, theta) in base frame.
         compute_jacobian: bool         - compute Jacobian Hx if true.
    Outputs:
         h: np.array[2,]  - line parameters in the scanner (camera) frame.
        Hx: np.array[2,3] - Jacobian of h with respect to x.
    """
    alpha, r = line

    ########## Code starts here ##########
    # TODO: Compute h, Hx
    # HINT: Calculate the pose of the camera in the world frame (x_cam, y_cam, th_cam), a rotation matrix may be useful.
    # HINT: To compute line parameters in the camera frame h = (alpha_in_cam, r_in_cam), 
    #       draw a diagram with a line parameterized by (alpha,r) in the world frame and 
    #       a camera frame with origin at x_cam, y_cam rotated by th_cam wrt to the world frame
    # HINT: What is the projection of the camera location (x_cam, y_cam) on the line r? 
    # HINT: To find Hx, write h in terms of the pose of the base in world frame (x_base, y_base, th_base)

    x_base = x[0]
    y_base = x[1]
    th_base = x[2]

    x_cam = tf_base_to_camera[0]
    y_cam = tf_base_to_camera[1]
    th_cam = tf_base_to_camera[2]

    x_cam_in_w = x_base + x_cam*np.cos(th_base) - y_cam*np.sin(th_base)
    y_cam_in_w = y_base + x_cam*np.sin(th_base) + y_cam*np.cos(th_base)
    th_cam_in_w = th_base + th_cam

    px = r * np.cos(alpha)
    py = r * np.sin(alpha)
    angle_line = np.pi/2 + alpha

    r_new = np.cos(angle_line)*(py - y_cam_in_w) - np.sin(angle_line)*(px - x_cam_in_w)
    r_new_abs = np.absolute(np.cos(angle_line)*(py - y_cam_in_w) - np.sin(angle_line)*(px - x_cam_in_w))

    cam_pos = np.array([x_cam_in_w, y_cam_in_w])
    p1 = np.array([r*np.cos(alpha), r*np.sin(alpha)])
    p2 = np.array([r*np.cos(alpha) - np.tan(alpha), r*np.sin(alpha)+1])
    p_proj_w = p1 + (np.dot(cam_pos-p1, p2-p1) * (p2-p1))/(np.dot(p2-p1,p2-p1))

    alpha_new = np.arctan2(p_proj_w[1]-y_cam_in_w, p_proj_w[0]-x_cam_in_w) - (th_base + th_cam)

    #alpha_new = alpha - (th_base + th_cam)

    h = np.array([alpha_new, r_new_abs]).reshape((2,))

    # Compute Hx
    drdx = (r_new/r_new_abs) * np.sin(angle_line)
    drdy = (r_new/r_new_abs) * -np.cos(angle_line)
    drdth = (r_new/r_new_abs) * (np.cos(angle_line) * (-x_cam*np.cos(th_base) + y_cam*np.sin(th_base)) - np.sin(angle_line) * (x_cam*np.sin(th_base) + y_cam*np.cos(th_base)))

    dalphadx = 0
    dalphady = 0
    dalphadth = -1


    Hx = np.array([[dalphadx, dalphady, dalphadth],[drdx, drdy, drdth]])

    ########## Code ends here ##########

    if not compute_jacobian:
        return h

    return h, Hx


def normalize_line_parameters(h, Hx=None):
    """
    Ensures that r is positive and alpha is in the range [-pi, pi].

    Inputs:
         h: np.array[2,]  - line parameters (alpha, r).
        Hx: np.array[2,n] - Jacobian of line parameters with respect to x.
    Outputs:
         h: np.array[2,]  - normalized parameters.
        Hx: np.array[2,n] - Jacobian of normalized line parameters. Edited in place.
    """
    alpha, r = h
    if r < 0:
        alpha += np.pi
        r *= -1
        if Hx is not None:
            Hx[1,:] *= -1
    alpha = (alpha + np.pi) % (2*np.pi) - np.pi
    h = np.array([alpha, r])

    if Hx is not None:
        return h, Hx
    return h
