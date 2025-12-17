"""
MPM Test Package

Contains unit tests and integration tests for the MPM solver.

Recommended test commands (run from xengym/mpm directory):
    # Run all tests
    pytest tests/ -v

    # Run tests excluding slow ones
    pytest tests/ -v -m "not slow"

    # Run only gradient verification tests
    pytest tests/ -v -m gradient

    # Run tests that don't require Taichi
    pytest tests/test_gradient_mode.py -v

Note: Most tests require Taichi to be installed. Tests will be
automatically skipped if Taichi is not available.
"""
