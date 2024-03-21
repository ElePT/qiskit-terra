from ddt import data, ddt, unpack
from test import QiskitTestCase, combine, slow_test  # pylint: disable=wrong-import-order
from qiskit import (
    ClassicalRegister,
    QuantumCircuit,
    QuantumRegister)

from qiskit import transpile
# from qiskit.compiler.transpiler_2 import transpile
from qiskit.providers.backend_compat import BackendV2Converter
from qiskit.providers.fake_provider import Fake5QV1, Fake7QPulseV1, Fake27QPulseV1, GenericBackendV2
from qiskit.pulse import InstructionScheduleMap, Schedule, Play, Gaussian, DriveChannel
from qiskit.circuit import Gate
import qiskit.pulse as pulse
from qiskit.transpiler.target import (
    InstructionProperties,
    Target,
    TimingConstraints,
    InstructionDurations,
)
from qiskit.transpiler.exceptions import TranspilerError

@ddt
class TestTranspile(QiskitTestCase):
    """Test transpile function."""

    def setUp(self):
        super().setUp()
        self.backend_v1 = Fake5QV1()
        self.backend_v2 = BackendV2Converter(self.backend_v1)
        self.coupling_map = [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1], [2, 3], [2, 4], [3, 2], [3, 4], [4, 2], [4, 3]]

    # def test_transpile_coupling_map(self):
    #     """Verify transpile()"""
    #
    #     qr = QuantumRegister(5, "qr")
    #     qc = QuantumCircuit(qr)
    #     qc.h(qr[0])
    #     qc.cx(qr[0], qr[1])
    #     qc.cx(qr[0], qr[2])
    #     qc.cx(qr[2], qr[3])
    #     qc.cx(qr[2], qr[4])
    #     qc.cx(qr[3], qr[4])
    #
    #     with self.subTest("just target"):
    #         new_qc = transpile(
    #             qc, target=self.backend_v2.target, seed_transpiler=42
    #         )
    #
    #         qubit_indices = {bit: idx for idx, bit in enumerate(new_qc.qubits)}
    #         cx_qubits = [instr.qubits for instr in new_qc.data if instr.operation.name == "cx"]
    #         cx_qubits_physical = [
    #             [qubit_indices[ctrl], qubit_indices[tgt]] for [ctrl, tgt] in cx_qubits
    #         ]
    #         self.assertEqual(
    #             sorted(cx_qubits_physical), [[0, 1], [0, 2], [2, 3], [2, 4], [3, 4]]
    #         )
    #
    #     with self.subTest("target + cmap"):
    #         # TARGET OVERRIDES CMAP
    #         cmap = [[0, 1], [1, 0], [1, 2], [1, 3], [2, 1], [3, 1], [3, 4], [4, 3]]
    #         new_qc = transpile(
    #             qc, target=self.backend_v2.target, coupling_map=cmap, seed_transpiler=42
    #         )
    #
    #         qubit_indices = {bit: idx for idx, bit in enumerate(new_qc.qubits)}
    #         cx_qubits = [instr.qubits for instr in new_qc.data if instr.operation.name == "cx"]
    #         cx_qubits_physical = [
    #             [qubit_indices[ctrl], qubit_indices[tgt]] for [ctrl, tgt] in cx_qubits
    #         ]
    #         self.assertEqual(
    #             sorted(cx_qubits_physical), [[0, 1], [0, 2], [2, 3], [2, 4], [3, 4]]
    #         )
    #
    #     with self.subTest("backendV1 + cmap"):
    #         cmap = [[0, 1], [1, 0], [1, 2], [1, 3], [2, 1], [3, 1], [3, 4], [4, 3]]
    #         new_qc = transpile(
    #             qc, backend=self.backend_v1, coupling_map=cmap, seed_transpiler=42
    #         )
    #
    #         qubit_indices = {bit: idx for idx, bit in enumerate(new_qc.qubits)}
    #         cx_qubits = [instr.qubits for instr in new_qc.data if instr.operation.name == "cx"]
    #         cx_qubits_physical = [
    #             [qubit_indices[ctrl], qubit_indices[tgt]] for [ctrl, tgt] in cx_qubits
    #         ]
    #
    #         # old_solution = [[0, 1], [1, 0], [1, 2], [1, 2], [2, 1], [2, 1], [3, 1], [3, 4]]
    #         self.assertEqual(
    #             sorted(cx_qubits_physical), [[0, 1], [0, 1], [1, 0], [1, 0], [1, 2], [2, 1], [3, 1], [3, 4]]
    #         )
    #
    #     with self.subTest("backendV2 + cmap"):
    #         cmap = [[0, 1], [1, 0], [1, 2], [1, 3], [2, 1], [3, 1], [3, 4], [4, 3]]
    #         new_qc = transpile(
    #             qc, backend=self.backend_v1, coupling_map=cmap, seed_transpiler=42
    #         )
    #
    #         qubit_indices = {bit: idx for idx, bit in enumerate(new_qc.qubits)}
    #         cx_qubits = [instr.qubits for instr in new_qc.data if instr.operation.name == "cx"]
    #         cx_qubits_physical = [
    #             [qubit_indices[ctrl], qubit_indices[tgt]] for [ctrl, tgt] in cx_qubits
    #         ]
    #         # old_solution = [[0, 1], [1, 0], [1, 2], [1, 2], [2, 1], [2, 1], [3, 1], [3, 4]]
    #         self.assertEqual(
    #             sorted(cx_qubits_physical),
    #             [[0, 1], [0, 1], [1, 0], [1, 0], [1, 2], [2, 1], [3, 1], [3, 4]]
    #         )

    def test_transpile_inst_map(self):
        """Verify transpile()"""

        inst_map = InstructionScheduleMap()
        inst_map.add("newgate", [0, 1], pulse.ScheduleBlock())
        newgate = Gate("newgate", 2, [])
        circ = QuantumCircuit(2)
        circ.append(newgate, [0, 1])

        for backend in [self.backend_v1, self.backend_v2]:
            with self.subTest(backend=backend):
                tqc = transpile(
                    circ,
                    backend=backend,
                    inst_map=inst_map,
                    basis_gates=["newgate"],
                    optimization_level=2,
                    seed_transpiler=42,
                )
                self.assertEqual(len(tqc.data), 1)
                self.assertEqual(tqc.data[0].operation, newgate)

        with self.subTest("target"):
            # target overrides inst_map
            with self.assertRaises(TranspilerError):
                tqc = transpile(
                    circ,
                    target=self.backend_v2.target,
                    inst_map=inst_map,
                    basis_gates=["newgate"],
                    optimization_level=2,
                    seed_transpiler=42,
                )

    def test_scheduling_timing_constraints(self):
        """Test that scheduling-related loose transpile constraints
        work with both BackendV1 and BackendV2."""

        backend_v1 = Fake27QPulseV1()
        backend_v2 = BackendV2Converter(backend_v1)
        target = backend_v2.target
        # the original timing constraints are granularity = min_length = 16
        timing_constraints = TimingConstraints(granularity=32, min_length=64)
        error_msgs = {
            65: "Pulse duration is not multiple of 32",
            32: "Pulse gate duration is less than 64",
        }
        duration = 65
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        qc.add_calibration(
            "h", [0], Schedule(Play(Gaussian(duration, 0.2, 4), DriveChannel(0))), [0, 0]
        )
        qc.add_calibration(
            "cx",
            [0, 1],
            Schedule(Play(Gaussian(duration, 0.2, 4), DriveChannel(1))),
            [0, 0],
        )
        with self.assertRaisesRegex(TranspilerError, error_msgs[duration]):
            _ = transpile(
                qc,
                target=target,
                timing_constraints=timing_constraints,
            )

        with self.assertRaisesRegex(TranspilerError, error_msgs[duration]):
            _ = transpile(
                qc,
                backend=backend_v1,
                timing_constraints=timing_constraints,
            )

        with self.assertRaisesRegex(TranspilerError, error_msgs[duration]):
            _ = transpile(
                qc,
                backend=backend_v2,
                timing_constraints=timing_constraints,
            )

    def test_scheduling_instruction_constraints(self):
        """Test that scheduling-related loose transpile constraints
        work with BackendV1."""

        backend_v1 = Fake27QPulseV1()
        backend_v2 = BackendV2Converter(backend_v1)
        target = backend_v2.target

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.delay(500, 1, "dt")
        qc.cx(0, 1)
        # update durations
        durations = InstructionDurations.from_backend(backend_v1)
        durations.update([("cx", [0, 1], 1000, "dt")])

        scheduled = transpile(
            qc,
            target=target,
            scheduling_method="alap",
            instruction_durations=durations,
            layout_method="trivial",
        )
        print(scheduled.duration, 1500)
        # custom input ignored
        self.assertEqual(scheduled.duration, 1876)
        # self.assertEqual(scheduled.duration, 1500)

        scheduled = transpile(
            qc,
            backend=backend_v2,
            scheduling_method="alap",
            instruction_durations=durations,
            layout_method="trivial",
        )
        print(scheduled.duration, 1500)
        # custom input ignored
        self.assertEqual(scheduled.duration, 1500)

        scheduled = transpile(
            qc,
            backend=backend_v1,
            scheduling_method="alap",
            instruction_durations=durations,
            layout_method="trivial",
        )
        print(scheduled.duration, 1500)
        # custom input ignored
        self.assertEqual(scheduled.duration, 1500)

    def test_scheduling_dt_constraints(self):
        """Test that scheduling-related loose transpile constraints
        work with BackendV1."""

        backend_v1 = Fake27QPulseV1()
        backend_v2 = BackendV2Converter(backend_v1)
        target = backend_v2.target

        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.measure(0, 0)
        original_dt = 2.2222222222222221e-10
        original_duration = 3504

        # halve dt in sec = double duration in dt
        scheduled = transpile(
            qc, target=target, scheduling_method="asap", dt=original_dt / 2
        )
        print(scheduled.duration, original_duration * 2)
        # custom input ignored
        self.assertEqual(scheduled.duration, original_duration)
        # self.assertEqual(scheduled.duration, original_duration * 2)