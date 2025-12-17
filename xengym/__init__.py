from pathlib import Path
PROJ_DIR = Path(__file__).resolve().parent
ASSET_DIR = PROJ_DIR / "assets"

# Lazy imports to avoid cv2 dependency when only using mpm submodule
def __getattr__(name):
    """Lazy import for optional dependencies (cv2, etc.)"""
    if name == "Xensim":
        from .render.robotScene import RobotScene
        return RobotScene
    elif name == "main":
        from .main import main
        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# For backwards compatibility, try importing but don't fail
try:
    from .render.robotScene import RobotScene as Xensim
    from .main import main
except ImportError:
    # cv2 or other dependencies not available
    # Xensim and main will be available via __getattr__ if needed
    pass