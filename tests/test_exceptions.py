from __future__ import annotations

import unittest

from epilink import ConfigurationError, EpiLinkError, ScenarioError, SimulationError


class TestExceptions(unittest.TestCase):
    def test_exception_hierarchy(self) -> None:
        self.assertTrue(issubclass(ScenarioError, EpiLinkError))
        self.assertTrue(issubclass(ScenarioError, ValueError))

        self.assertTrue(issubclass(ConfigurationError, EpiLinkError))
        self.assertTrue(issubclass(ConfigurationError, ValueError))

        self.assertTrue(issubclass(SimulationError, EpiLinkError))
        self.assertTrue(issubclass(SimulationError, RuntimeError))

        self.assertTrue(issubclass(EpiLinkError, Exception))

    def test_exception_instantiation(self) -> None:
        with self.assertRaises(ScenarioError) as cm:
            raise ScenarioError("Invalid scenario")
        self.assertEqual(str(cm.exception), "Invalid scenario")

        with self.assertRaises(ConfigurationError) as cm:
            raise ConfigurationError("Misconfigured")
        self.assertEqual(str(cm.exception), "Misconfigured")

        with self.assertRaises(SimulationError) as cm:
            raise SimulationError("Simulation failed")
        self.assertEqual(str(cm.exception), "Simulation failed")

if __name__ == "__main__":
    unittest.main()
