import importlib.util
import unittest


class OpenDPValidationTests(unittest.TestCase):
    @unittest.skipIf(importlib.util.find_spec("opendp") is None, "OpenDP is optional")
    def test_opendp_validation_script_passes_when_dependency_is_installed(self) -> None:
        from scripts.validate_against_opendp import validate_gaussian, validate_laplace

        self.assertTrue(validate_laplace(samples=1500, seed=2026).passed)
        self.assertTrue(validate_gaussian(samples=1500, seed=2026).passed)


if __name__ == "__main__":
    unittest.main()
