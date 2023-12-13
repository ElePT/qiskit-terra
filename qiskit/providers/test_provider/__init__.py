# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
====================================================================
TestProvider: Python-based Simulators (:mod:`qiskit.providers.test_provider`)
====================================================================

.. currentmodule:: qiskit.providers.test_provider

A module of Python-based quantum simulators.  Simulators are accessed
via the `TestProvider` provider, e.g.:

.. code-block::

   from qiskit import TestProvider

   backend = TestProvider.get_backend('qasm_simulator')


Simulators
==========

.. autosummary::
   :toctree: ../stubs/

   TestSimulator

Provider
========

.. autosummary::
   :toctree: ../stubs/

   TestProvider

Job Class
=========

.. autosummary::
   :toctree: ../stubs/

   TestProviderJob

Exceptions
==========

.. autosummary::
   :toctree: ../stubs/

   TestProviderError
"""

from .test_provider import TestProvider
from .test_provider_job import TestProviderJob
from .test_simulator import TestSimulator
from .exceptions import TestProviderError

# Global instance to be used as the entry point for convenience.

TestProvider = TestProvider()
