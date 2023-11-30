# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Fake ``BackendV1`` Backends.
"""

import os
from qiskit.providers.fake_provider import fake_pulse_backend, fake_qasm_backend


class Fake5QV1(fake_qasm_backend.FakeQasmBackend):
    """A fake backend with the following characteristics:

    * num_qubits: 5
    * coupling_map:

        .. code-block:: text

                1
              / |
            0 - 2 - 3
                | /
                4

    * basis_gates: ``["id", "rz", "sx", "x", "cx", "reset"]``
    """

    dirname = os.path.dirname(__file__)
    conf_filename = "5q/conf_yorktown.json"
    props_filename = "5q/props_yorktown.json"
    backend_name = "fake_5q_v1"


class Fake20QV1(fake_qasm_backend.FakeQasmBackend):
    """A fake backend with the following characteristics:

    * num_qubits: 20
    * coupling_map:

        .. code-block:: text

            00 ↔ 01 ↔ 02 ↔ 03 ↔ 04
                  ↕         ↕
            05 ↔ 06 ↔ 07 ↔ 08 ↔ 09
             ↕         ↕         ↕
            10 ↔ 11 ↔ 12 ↔ 13 ↔ 14
                  ↕         ↕
            15 ↔ 16 ↔ 17 ↔ 18 ↔ 19

    * basis_gates: ``["id", "u1", "u2", "u3", "cx"]``
    """

    dirname = os.path.dirname(__file__)
    conf_filename = "20q/conf_singapore.json"
    props_filename = "20q/props_singapore.json"
    backend_name = "fake_20q_v1"


class Fake7QV1Pulse(fake_pulse_backend.FakePulseBackend):
    """A fake **pulse** backend with the following characteristics:

    * num_qubits: 7
    * coupling_map:

        .. code-block:: text

            0 ↔ 1 ↔ 3 ↔ 5 ↔ 6
                ↕       ↕
                2       4

    * basis_gates: ``["id", "rz", "sx", "x", "cx", "reset"]``
    * scheduled instructions:
        # ``{'u3', 'id', 'measure', 'u2', 'x', 'u1', 'sx', 'rz'}`` for all individual qubits
        # ``{'cx'}`` for all edges
        # ``{'measure'}`` for (0, 1, 2, 3, 4, 5, 6)
    """

    dirname = os.path.dirname(__file__)
    conf_filename = "7q_pulse/conf_nairobi.json"
    props_filename = "7q_pulse/props_nairobi.json"
    defs_filename = "7q_pulse/defs_nairobi.json"
    backend_name = "fake_7q_v1_pulse"


class Fake27QV1Pulse(fake_pulse_backend.FakePulseBackend):
    """A fake **pulse** backend with the following characteristics:

    * num_qubits: 27
    * coupling_map:

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

    * basis_gates: ``["id", "rz", "sx", "x", "cx", "reset"]``
    * scheduled instructions:
        # ``{'id', 'rz', 'u2', 'x', 'u3', 'sx', 'measure', 'u1'}`` for all individual qubits
        # ``{'cx'}`` for all edges
        # ``{'measure'}`` for (0, ..., 26)
    """

    dirname = os.path.dirname(__file__)
    conf_filename = "27q_pulse/conf_hanoi.json"
    props_filename = "27q_pulse/props_hanoi.json"
    defs_filename = "27q_pulse/defs_hanoi.json"
    backend_name = "fake_27q_v1_pulse"


class Fake65QV1Pulse(fake_pulse_backend.FakePulseBackend):
    """A fake **pulse** backend with the following characteristics:

    * num_qubits: 65
    * coupling_map: heavy-hex based
    * basis_gates: ``["id", "rz", "sx", "x", "cx", "reset"]``
    * scheduled instructions:
        # ``{'id', 'measure', 'u2', 'rz', 'x', 'u3', 'sx', 'u1'}`` for all individual qubits
        # ``{'cx'}`` for all edges
        # ``{'measure'}`` for (0, ..., 65)
    """

    dirname = os.path.dirname(__file__)
    conf_filename = "65q_pulse/conf_manhattan.json"
    props_filename = "65q_pulse/props_manhattan.json"
    defs_filename = "65q_pulse/defs_manhattan.json"
    backend_name = "fake_65q_v1_pulse"
