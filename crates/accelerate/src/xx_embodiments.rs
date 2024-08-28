// // This code is part of Qiskit.
// //
// // (C) Copyright IBM 2024
// //
// // This code is licensed under the Apache License, Version 2.0. You may
// // obtain a copy of this license in the LICENSE.txt file in the root directory
// // of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
// //
// // Any modifications or derivative works of this code must retain this
// // copyright notice, and modified files need to carry a notice indicating
// // that they have been altered from the originals.

// // A library of known embodiments of RXXGate in terms of other gates,
// // for some generic or specific angles.

// use qiskit_circuit::{circuit_data::CircuitData, operations::{Operation, Param, StandardGate}};
// use std::str::FromStr;

// enum XXEmbodiments{
//     RXXGate,
//     RYYGate,
//     RZZGate,
//     RZXGate,
//     CRXGate,
//     CRYGate,
//     CRZGate,
//     CPhaseGate,
//     CXGate,
//     CYGate,
//     CZGate,
//     CHGate,
//     ECRGate,
// }

// impl FromStr for XXEmbodiments {
//     type Err = ();

//     fn from_str(s: &str) -> Result<Self, Self::Err> {
//         match s {
//             "rxx" => Ok(XXEmbodiments::RXXGate),
//             "ryy" => Ok(XXEmbodiments::RYYGate),
//             "rzz" => Ok(XXEmbodiments::RZZGate),
//             "rzx" => Ok(XXEmbodiments::RZXGate),
//             "crx" => Ok(XXEmbodiments::CRXGate),
//             "cry" => Ok(XXEmbodiments::CRYGate),
//             "crz" => Ok(XXEmbodiments::CRZGate),
//             "cp" => Ok(XXEmbodiments::CPhaseGate),
//             "cx" => Ok(XXEmbodiments::CXGate),
//             "cy" => Ok(XXEmbodiments::CYGate),
//             "cz" => Ok(XXEmbodiments::CZGate),
//             "ch" => Ok(XXEmbodiments::CHGate),
//             "ecr" => Ok(XXEmbodiments::ECRGate),
//             _ => Err(()),

//         }
//     }
// }

// #[inline]
// pub fn xx_embodiments_from_name(py, name:&str) -> CircuitData {
//     match XXEmbodiments::from_str(name) {
//         XXEmbodiments::RXXGate => CircuitData::from_standard_gates(py, 2, (StandardGate::RXXGate, smallvec![PI / 2.], &[Qubit(0), Qubit(1)])),
//         XXEmbodiments::RYYGate => ,
//         XXEmbodiments::RZZGate => ,
//         XXEmbodiments::RZXGate => ,
//         XXEmbodiments::CRXGate => ,
//         XXEmbodiments::CRYGate => ,
//         XXEmbodiments::CRZGate => ,
//         XXEmbodiments::CPhaseGate => ,
//         XXEmbodiments::CXGate => ,
//         XXEmbodiments::CYGate => ,
//         XXEmbodiments::CZGate => ,
//         XXEmbodiments::CHGate => ,
//         XXEmbodiments::ECRGate => ,
//     }
// }
