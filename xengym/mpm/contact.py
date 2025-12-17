"""
Contact and Friction Module
Implements SDF penalty-based contact with regularized elastoplastic friction
"""
import taichi as ti


@ti.func
def sdf_sphere(x: ti.template(), center: ti.template(), radius: ti.f32) -> ti.f32:
    """
    Signed distance function for sphere

    Args:
        x: Query point
        center: Sphere center
        radius: Sphere radius

    Returns:
        Signed distance (negative inside, positive outside)
    """
    return (x - center).norm() - radius


@ti.func
def sdf_plane(x: ti.template(), point: ti.template(), normal: ti.template()) -> ti.f32:
    """
    Signed distance function for plane

    Args:
        x: Query point
        point: Point on plane
        normal: Plane normal (unit vector)

    Returns:
        Signed distance (negative below, positive above)
    """
    return (x - point).dot(normal)


@ti.func
def sdf_box(x: ti.template(), center: ti.template(), half_extents: ti.template()) -> ti.f32:
    """
    Signed distance function for box

    Args:
        x: Query point
        center: Box center
        half_extents: Half extents in each dimension

    Returns:
        Signed distance
    """
    q = ti.abs(x - center) - half_extents
    return ti.max(q, 0.0).norm() + ti.min(ti.max(q[0], ti.max(q[1], q[2])), 0.0)


@ti.func
def evaluate_sdf(
    x: ti.template(),
    sdf_type: ti.i32,
    center: ti.template(),
    normal: ti.template(),
    half_extents: ti.template()
) -> ti.f32:
    """
    Evaluate SDF for configurable obstacle type

    Args:
        x: Query point
        sdf_type: 0=plane, 1=sphere, 2=box
        center: Center/point on plane
        normal: Normal for plane (ignored for sphere/box)
        half_extents: Half extents for box, or (radius, 0, 0) for sphere

    Returns:
        Signed distance (negative = penetration)
    """
    phi = 0.0
    if sdf_type == 0:  # Plane
        phi = sdf_plane(x, center, normal)
    elif sdf_type == 1:  # Sphere
        phi = sdf_sphere(x, center, half_extents[0])
    else:  # Box (sdf_type == 2)
        phi = sdf_box(x, center, half_extents)
    return phi


@ti.func
def compute_sdf_normal(
    x: ti.template(),
    sdf_type: ti.i32,
    center: ti.template(),
    normal_plane: ti.template(),
    half_extents: ti.template()
) -> ti.template():
    """
    Compute outward normal for SDF obstacle

    Args:
        x: Query point
        sdf_type: 0=plane, 1=sphere, 2=box
        center: Center/point on plane
        normal_plane: Normal for plane
        half_extents: Half extents for box, or (radius, 0, 0) for sphere

    Returns:
        Outward normal vector (unit)
    """
    n = ti.Vector([0.0, 0.0, 1.0])

    if sdf_type == 0:  # Plane
        n = normal_plane
    elif sdf_type == 1:  # Sphere
        diff = x - center
        dist = diff.norm()
        if dist > 1e-8:
            n = diff / dist
        else:
            n = ti.Vector([0.0, 0.0, 1.0])
    else:  # Box (sdf_type == 2)
        # For box, compute normal from closest face
        q = x - center
        abs_q = ti.abs(q) - half_extents

        # Find which face is closest
        max_comp = ti.max(abs_q[0], ti.max(abs_q[1], abs_q[2]))
        n = ti.Vector([0.0, 0.0, 0.0])

        if abs_q[0] >= abs_q[1] and abs_q[0] >= abs_q[2]:
            n[0] = ti.select(q[0] >= 0, 1.0, -1.0)
        elif abs_q[1] >= abs_q[0] and abs_q[1] >= abs_q[2]:
            n[1] = ti.select(q[1] >= 0, 1.0, -1.0)
        else:
            n[2] = ti.select(q[2] >= 0, 1.0, -1.0)

    return n


@ti.func
def compute_contact_force(
    phi: ti.f32,
    v_rel: ti.template(),
    normal: ti.template(),
    u_t: ti.template(),
    dt: ti.f32,
    k_normal: ti.f32,
    k_tangent: ti.f32,
    mu_s: ti.f32,
    mu_k: ti.f32,
    v_transition: ti.f32
) -> ti.template():
    """
    Compute contact force with regularized elastoplastic friction

    Args:
        phi: Signed distance (negative = penetration)
        v_rel: Relative velocity
        normal: Contact normal (pointing outward from obstacle)
        u_t: Tangential displacement (elastic spring)
        dt: Time step
        k_normal: Normal contact stiffness
        k_tangent: Tangential spring stiffness
        mu_s: Static friction coefficient
        mu_k: Kinetic friction coefficient
        v_transition: Velocity for tanh transition

    Returns:
        f_contact: Contact force
        u_t_new: Updated tangential displacement
        is_contact: Contact flag (1 if in contact, 0 otherwise)
    """
    f_contact = ti.Vector([0.0, 0.0, 0.0])
    u_t_new = u_t
    is_contact = 0

    if phi < 0.0:  # Penetration
        is_contact = 1

        # Normal force (penalty method)
        f_n = -k_normal * phi * normal

        # Tangential velocity
        v_n = v_rel.dot(normal) * normal
        v_t = v_rel - v_n

        # Update tangential displacement (elastic spring)
        u_t_trial = u_t + v_t * dt

        # Compute tangential force (using tangential stiffness)
        f_t_trial = -k_tangent * u_t_trial

        # Friction limit
        f_n_mag = f_n.norm()
        f_t_mag = f_t_trial.norm()

        # Regularized friction coefficient (tanh transition from static to kinetic)
        v_t_mag = v_t.norm()
        mu_eff = mu_k + (mu_s - mu_k) * ti.tanh(v_transition / (v_t_mag + 1e-8))

        f_t_max = mu_eff * f_n_mag

        # Initialize f_t
        f_t = ti.Vector([0.0, 0.0, 0.0])

        # Elastoplastic friction: if |f_t| > f_t_max, slide
        if f_t_mag > f_t_max:
            # Sliding: limit tangential force and update u_t
            if f_t_mag > 1e-10:
                f_t = f_t_trial * (f_t_max / f_t_mag)
                u_t_new = -f_t / k_tangent
            else:
                f_t = ti.Vector([0.0, 0.0, 0.0])
                u_t_new = ti.Vector([0.0, 0.0, 0.0])
        else:
            # Sticking: use trial force
            f_t = f_t_trial
            u_t_new = u_t_trial

        f_contact = f_n + f_t

    return f_contact, u_t_new, is_contact


@ti.func
def update_contact_age(
    is_contact: ti.i32,
    age: ti.i32,
    K_clear: ti.i32
) -> ti.template():
    """
    Update no-contact age counter for hysteresis-based cleanup

    Args:
        is_contact: Contact flag (1 if in contact, 0 otherwise)
        age: Current no-contact age
        K_clear: Threshold for clearing tangential displacement

    Returns:
        age_new: Updated age
        should_clear: Flag indicating whether to clear u_t
    """
    age_new = age
    should_clear = 0

    if is_contact == 1:
        # In contact: reset age
        age_new = 0
    else:
        # Not in contact: increment age
        age_new = age + 1

        # Clear u_t if age exceeds threshold
        if age_new >= K_clear:
            should_clear = 1
            age_new = 0  # Reset age after clearing

    return age_new, should_clear
