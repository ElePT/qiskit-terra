# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests PassManagerConfig"""

from qiskit import QuantumRegister
from qiskit.providers.backend import Backend
from qiskit.test import QiskitTestCase
from qiskit.providers.fake_provider import Fake20QV1, Fake27QV1Pulse, FakeGeneric
from qiskit.providers.fake_provider.fake_backend_v2 import FakeBackendSimple
from qiskit.providers.basicaer import QasmSimulatorPy
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.passmanager_config import PassManagerConfig


class TestPassManagerConfig(QiskitTestCase):
    """Test PassManagerConfig.from_backend()."""

    def test_config_from_backend(self):
        """Test from_backend() with a valid backend.

        `Fake27QV1Pulse` is used in this testcase. This backend has `defaults` attribute
        that contains an instruction schedule map.
        """
        backend = Fake27QV1Pulse()
        config = PassManagerConfig.from_backend(backend)
        self.assertEqual(config.basis_gates, backend.configuration().basis_gates)
        self.assertEqual(config.inst_map, backend.defaults().instruction_schedule_map)
        self.assertEqual(
            str(config.coupling_map), str(CouplingMap(backend.configuration().coupling_map))
        )

    def test_config_from_backend_v2(self):
        """Test from_backend() with a BackendV2 instance."""
        backend = FakeGeneric(basis_gates=["cx", "id", "rz", "sx", "x"], num_qubits=27)
        config = PassManagerConfig.from_backend(backend)
        self.assertEqual(config.basis_gates, backend.operation_names)
        self.assertEqual(config.inst_map, backend.instruction_schedule_map)
        self.assertEqual(config.coupling_map.get_edges(), backend.coupling_map.get_edges())

    def test_invalid_backend(self):
        """Test from_backend() with an invalid backend."""
        with self.assertRaises(AttributeError):
            PassManagerConfig.from_backend(Backend())

    def test_from_backend_and_user(self):
        """Test from_backend() with a backend and user options.

        `FakeMelbourne` is used in this testcase. This backend does not have
        `defaults` attribute and thus not provide an instruction schedule map.
        """
        qr = QuantumRegister(4, "qr")
        initial_layout = [None, qr[0], qr[1], qr[2], None, qr[3]]

        backend = Fake20QV1()
        config = PassManagerConfig.from_backend(
            backend, basis_gates=["user_gate"], initial_layout=initial_layout
        )
        self.assertEqual(config.basis_gates, ["user_gate"])
        self.assertNotEqual(config.basis_gates, backend.configuration().basis_gates)
        self.assertIsNone(config.inst_map)
        self.assertEqual(
            str(config.coupling_map), str(CouplingMap(backend.configuration().coupling_map))
        )
        self.assertEqual(config.initial_layout, initial_layout)

    def test_from_backendv1_inst_map_is_none(self):
        """Test that from_backend() works with backend that has defaults defined as None."""
        backend = Fake27QV1Pulse()
        backend.defaults = lambda: None
        config = PassManagerConfig.from_backend(backend)
        self.assertIsInstance(config, PassManagerConfig)
        self.assertIsNone(config.inst_map)

    def test_simulator_backend_v1(self):
        """Test that from_backend() works with backendv1 simulator."""
        backend = QasmSimulatorPy()
        config = PassManagerConfig.from_backend(backend)
        self.assertIsInstance(config, PassManagerConfig)
        self.assertIsNone(config.inst_map)
        self.assertIsNone(config.coupling_map)

    def test_invalid_user_option(self):
        """Test from_backend() with an invalid user option."""
        with self.assertRaises(TypeError):
            PassManagerConfig.from_backend(Fake20QV1(), invalid_option=None)

    def test_str(self):
        """Test string output."""
        pm_config = PassManagerConfig.from_backend(FakeBackendSimple())
        # For testing remove instruction schedule map, its str output is non-deterministic
        # based on hash seed
        pm_config.inst_map = None
        str_out = str(pm_config)
        expected = """Pass Manager Config:
	initial_layout: None
	basis_gates: ['sx', 'x', 'rz', 'cx', 'measure']
	inst_map: None
	coupling_map: None
	layout_method: None
	routing_method: None
	translation_method: None
	scheduling_method: None
	instruction_durations: 
	backend_properties: None
	approximation_degree: None
	seed_transpiler: None
	timing_constraints: None
	unitary_synthesis_method: default
	unitary_synthesis_plugin_config: None
	target: Target
	Number of qubits: 20
	Instructions:
		sx
		x
		rz
		cx
		measure
"""
        self.assertEqual(str_out[:-2], expected)
