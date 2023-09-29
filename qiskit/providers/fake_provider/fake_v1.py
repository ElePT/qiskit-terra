# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Fake Singapore device (20 qubit).
"""

import os
import json

from qiskit.providers.models import GateConfig, QasmBackendConfiguration, BackendProperties
from qiskit.providers.fake_provider import fake_pulse_backend, fake_qasm_backend
from qiskit.providers.fake_provider.fake_backend import FakeBackend


class Fake5QV1(fake_qasm_backend.FakeQasmBackend):
    """A fake 5 qubit backend (Yorktown)

    .. code-block:: text

            1
          / |
        0 - 2 - 3
            | /
            4
    """

    dirname = os.path.dirname(__file__)
    conf_filename = "data/conf_yorktown.json"
    props_filename = "data/props_yorktown.json"
    backend_name = "fake_5q_v1"


class Fake20QV1Dense(FakeBackend):
    """A fake 20 qubit backend."""

    def __init__(self):
        """

        .. code-block:: text

            00 ↔ 01 ↔ 02 ↔ 03 ↔ 04
             ↕    ↕    ↕    ↕ ⤫  ↕
            05 ↔ 06 ↔ 07 ↔ 08 ↔ 09
             ↕ ⤫ ↕    ↕ ⤫ ↕
            10 ↔ 11 ↔ 12 ↔ 13 ↔ 14
             ↕    ↕ ⤫      ↕ ⤫  ↕
            15 ↔ 16 ↔ 17   18   19
        """
        cmap = [
            [0, 1],
            [0, 5],
            [1, 0],
            [1, 2],
            [1, 6],
            [1, 7],
            [2, 1],
            [2, 6],
            [3, 8],
            [4, 8],
            [4, 9],
            [5, 0],
            [5, 6],
            [5, 10],
            [5, 11],
            [6, 1],
            [6, 2],
            [6, 5],
            [6, 7],
            [6, 10],
            [6, 11],
            [7, 1],
            [7, 6],
            [7, 8],
            [7, 12],
            [8, 3],
            [8, 4],
            [8, 7],
            [8, 9],
            [8, 12],
            [8, 13],
            [9, 4],
            [9, 8],
            [10, 5],
            [10, 6],
            [10, 11],
            [10, 15],
            [11, 5],
            [11, 6],
            [11, 10],
            [11, 12],
            [11, 16],
            [11, 17],
            [12, 7],
            [12, 8],
            [12, 11],
            [12, 13],
            [12, 16],
            [13, 8],
            [13, 12],
            [13, 14],
            [13, 18],
            [13, 19],
            [14, 13],
            [14, 18],
            [14, 19],
            [15, 10],
            [15, 16],
            [16, 11],
            [16, 12],
            [16, 15],
            [16, 17],
            [17, 11],
            [17, 16],
            [17, 18],
            [18, 13],
            [18, 14],
            [18, 17],
            [19, 13],
            [19, 14],
        ]

        configuration = QasmBackendConfiguration(
            backend_name="fake_tokyo",
            backend_version="0.0.0",
            n_qubits=20,
            basis_gates=["u1", "u2", "u3", "cx", "id"],
            simulator=False,
            local=True,
            conditional=False,
            open_pulse=False,
            memory=False,
            max_shots=65536,
            max_experiments=900,
            gates=[GateConfig(name="TODO", parameters=[], qasm_def="TODO")],
            coupling_map=cmap,
        )

        super().__init__(configuration)

    def properties(self):
        """Returns a snapshot of device properties as recorded on 8/30/19."""
        dirname = os.path.dirname(__file__)
        filename = "props_tokyo.json"
        with open(os.path.join(dirname, filename)) as f_prop:
            props = json.load(f_prop)
        return BackendProperties.from_dict(props)


class Fake20QV1(fake_qasm_backend.FakeQasmBackend):
    """A fake Singapore backend.

    .. code-block:: text

        00 ↔ 01 ↔ 02 ↔ 03 ↔ 04
              ↕         ↕
        05 ↔ 06 ↔ 07 ↔ 08 ↔ 09
         ↕         ↕         ↕
        10 ↔ 11 ↔ 12 ↔ 13 ↔ 14
              ↕         ↕
        15 ↔ 16 ↔ 17 ↔ 18 ↔ 19
    """

    dirname = os.path.dirname(__file__)
    conf_filename = "data/conf_singapore.json"
    props_filename = "data/props_singapore.json"
    backend_name = "fake_20q_v1"


class Fake14QV1(fake_qasm_backend.FakeQasmBackend):
    """A fake Melbourne backend."""

    def __init__(self):
        """

        .. code-block:: text

            0 ← 1 →  2 →  3 ←  4 ← 5 → 6
                ↑    ↑    ↑    ↓   ↓   ↓
               13 → 12 ← 11 → 10 ← 9 → 8 ← 7
        """
        cmap = [
            [1, 0],
            [1, 2],
            [2, 3],
            [4, 3],
            [4, 10],
            [5, 4],
            [5, 6],
            [5, 9],
            [6, 8],
            [7, 8],
            [9, 8],
            [9, 10],
            [11, 3],
            [11, 10],
            [11, 12],
            [12, 2],
            [13, 1],
            [13, 12],
        ]

        configuration = QasmBackendConfiguration(
            backend_name="fake_melbourne",
            backend_version="0.0.0",
            n_qubits=14,
            basis_gates=["u1", "u2", "u3", "cx", "id"],
            simulator=False,
            local=True,
            conditional=False,
            open_pulse=False,
            memory=False,
            max_shots=65536,
            max_experiments=900,
            gates=[GateConfig(name="TODO", parameters=[], qasm_def="TODO")],
            coupling_map=cmap,
        )

        super().__init__(configuration)

    def properties(self):
        """Returns a snapshot of device properties"""
        dirname = os.path.dirname(__file__)
        filename = "props_melbourne.json"
        with open(os.path.join(dirname, filename)) as f_prop:
            props = json.load(f_prop)
        return BackendProperties.from_dict(props)


class Fake7QV1Pulse(fake_pulse_backend.FakePulseBackend):
    """A fake 7 qubit backend."""

    dirname = os.path.dirname(__file__)
    conf_filename = "data/conf_nairobi.json"
    props_filename = "data/props_nairobi.json"
    defs_filename = "data/defs_nairobi.json"
    backend_name = "fake_nairobi"


class Fake65QV1Pulse(fake_pulse_backend.FakePulseBackend):
    """A fake 65 qubit backend."""

    dirname = os.path.dirname(__file__)
    conf_filename = "data/conf_manhattan.json"
    props_filename = "data/props_manhattan.json"
    defs_filename = "data/defs_manhattan.json"
    backend_name = "fake_manhattan"


class Fake27QV1Pulse(fake_pulse_backend.FakePulseBackend):
    """A fake Paris backend.

    .. code-block:: text

                       06                  17
                       ↕                    ↕
        00 ↔ 01 ↔ 04 ↔ 07 ↔ 10 ↔ 12 ↔ 15 ↔ 18 ↔ 20 ↔ 23
             ↕                   ↕                    ↕
             02                  13                  24
             ↕                   ↕                    ↕
             03 ↔ 05 ↔ 08 ↔ 11 ↔ 14 ↔ 16 ↔ 19 ↔ 22 ↔ 25 ↔ 26
                       ↕                    ↕
                       09                  20
    """

    dirname = os.path.dirname(__file__)
    conf_filename = "data/conf_hanoi.json"
    props_filename = "data/props_hanoi.json"
    defs_filename = "data/defs_hanoi.json"
    backend_name = "fake_paris"


class Fake16QConnectedV1(FakeBackend):
    """A fake 16 qubit backend."""

    def __init__(self):
        """

        .. code-block:: text

            1 →  2 →  3 →  4 ←  5 ←  6 →  7 ← 8
            ↓    ↑    ↓    ↓    ↑    ↓    ↓   ↑
            0 ← 15 → 14 ← 13 ← 12 → 11 → 10 ← 9
        """
        cmap = [
            [1, 0],
            [1, 2],
            [2, 3],
            [3, 4],
            [3, 14],
            [5, 4],
            [6, 5],
            [6, 7],
            [6, 11],
            [7, 10],
            [8, 7],
            [9, 8],
            [9, 10],
            [11, 10],
            [12, 5],
            [12, 11],
            [12, 13],
            [13, 4],
            [13, 14],
            [15, 0],
            [15, 2],
            [15, 14],
        ]

        configuration = QasmBackendConfiguration(
            backend_name="fake_rueschlikon",
            backend_version="0.0.0",
            n_qubits=16,
            basis_gates=["u1", "u2", "u3", "cx", "id"],
            simulator=False,
            local=True,
            conditional=False,
            open_pulse=False,
            memory=False,
            max_shots=65536,
            max_experiments=900,
            gates=[GateConfig(name="TODO", parameters=[], qasm_def="TODO")],
            coupling_map=cmap,
        )

        super().__init__(configuration)
