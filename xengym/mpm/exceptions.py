"""
MPM Solver Custom Exceptions

Provides a structured exception hierarchy for better error handling and debugging.
All MPM-specific exceptions inherit from MPMError for easy catching.
"""


class MPMError(RuntimeError):
    """Base exception for all MPM solver errors.

    Catch this to handle any MPM-related error generically.
    """
    pass


class ConfigurationError(MPMError):
    """Configuration or user input validation error.

    Raised when:
    - Invalid configuration values are provided
    - Required configuration is missing
    - Configuration constraints are violated
    """
    pass


class MaterialError(ConfigurationError):
    """Material parameter validation error.

    Raised when:
    - Ogden parameters violate Drucker stability
    - Maxwell branch parameters are invalid
    - Material model constraints are violated
    """
    pass


class StabilityError(ConfigurationError):
    """Numerical stability constraint violation.

    Raised when:
    - Time step exceeds CFL condition
    - Deformation gradient becomes singular
    - Energy becomes unbounded
    """
    pass


class AutodiffError(MPMError):
    """Automatic differentiation limitation or error.

    Raised when:
    - Taichi AD limitations are encountered
    - Gradient computation is not supported for a configuration
    """
    pass


class GradientError(AutodiffError):
    """Gradient computation error.

    Raised when:
    - Manual adjoint computation fails
    - Numerical gradient verification fails
    - Gradient mode configuration is invalid
    """
    pass


class ScaleGuardError(GradientError):
    """Scale guard violation in experimental gradient mode.

    Raised when:
    - Particle count exceeds experimental mode limit
    - Step count exceeds experimental mode limit
    """
    pass


class TargetNotSetError(GradientError):
    """Target data not set for loss computation.

    Raised when:
    - Target positions not set for position loss
    - Target velocities not set for velocity loss
    - Target energy not set for energy loss
    """
    pass
