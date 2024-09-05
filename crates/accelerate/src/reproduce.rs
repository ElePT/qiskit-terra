use crate::unitary_synthesis::TwoQubitUnitarySequence;

// #[cfg(feature = "cache_pygates")]
use std::cell::OnceCell;


use hashbrown::{HashSet};
use qiskit_circuit::bit_data::BitData;
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::{Clbit, Qubit};
use smallvec::{smallvec, SmallVec};
// use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::Python;
use num_complex::{Complex, Complex64};

use qiskit_circuit::dag_circuit::{add_global_phase, DAGCircuit, NodeType};
use qiskit_circuit::imports::{CIRCUIT_TO_DAG, DAG_TO_CIRCUIT};
use qiskit_circuit::operations::{Operation, OperationRef};
use qiskit_circuit::operations::{Param, StandardGate};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperationType};
use crate::two_qubit_decompose::{TwoQubitGateSequence};


// fn dag_from_2q_gate_sequence(
//     py: Python<'_>,
//     sequence:  Vec<(Option<StandardGate>, SmallVec<[f64; 3]>, SmallVec<[u8; 2]>)>,
// ) -> PyResult<DAGCircuit> {

//     let mut target_dag = DAGCircuit::with_capacity(py, 2, 0, None, None, None)?;
//     let _ = target_dag.set_global_phase(Param::Float(0));

//     let mut counter = 0;
//     let mut instructions = Vec::new();

//     for (gate, params, qubit_ids) in sequence {
//         counter += 1;
//         println!(
//             "(Print in Rust) Iteration: {:?}. Gate: {:?}, qubit_ids: {:?} \n",
//             counter, gate, qubit_ids
//         );

//         let gate_node = match gate {
//             None => StandardGate::HGate,
//             Some(gate) => gate,
//         };

//         let qubits = match gate_node.num_qubits() {
//             1 => vec![Qubit(qubit_ids[0] as u32)],
//             2 => vec![Qubit(qubit_ids[0] as u32), Qubit(qubit_ids[1] as u32)],
//             _ => unreachable!(),
//         };

//         let new_params: SmallVec<[Param; 3]> = params.iter().map(|p| Param::Float(*p)).collect();

//         let pi = PackedInstruction {
//             op: PackedOperation::from_standard(gate_node),
//             qubits: target_dag.qargs_interner.insert(&qubits),
//             clbits: target_dag.cargs_interner.get_default(),
//             params: Some(Box::new(new_params)),
//             extra_attrs: None,
//             // #[cfg(feature = "cache_pygates")]
//             py_op: OnceCell::new(),
//         };
//         instructions.push(pi);
//     }

//     let _ = target_dag.add_from_iter(py, instructions.into_iter());
//     println!("Qubit io map: {:?}", target_dag.qubit_io_map);
//     Ok(target_dag)
// }

fn dag_from_2q_gate_sequence(
    py: Python<'_>,
    sequence: TwoQubitUnitarySequence,
) -> PyResult<DAGCircuit> {
    let gate_vec = &sequence.gate_sequence.gates;
    let mut target_dag = DAGCircuit::with_capacity(py, 2, 0, None, None, None)?;
    let _ = target_dag.set_global_phase(Param::Float(sequence.gate_sequence.global_phase));

    let mut counter = 0;
    let mut instructions = Vec::new();

    for (gate, params, qubit_ids) in gate_vec {
        counter += 1;
        println!(
            "(Print in Rust) Iteration: {:?}. Gate: {:?}, qubit_ids: {:?} \n",
            counter, gate, qubit_ids
        );

        let gate_node = match gate {
            None => sequence.get_decomp_gate().clone().unwrap(),
            Some(gate) => *gate,
        };

        let qubits = match gate_node.num_qubits() {
            1 => vec![Qubit(qubit_ids[0] as u32)],
            2 => vec![Qubit(qubit_ids[0] as u32), Qubit(qubit_ids[1] as u32)],
            _ => unreachable!(),
        };

        let new_params: SmallVec<[Param; 3]> = params.iter().map(|p| Param::Float(*p)).collect();

        let pi = PackedInstruction {
            op: PackedOperation::from_standard(gate_node),
            qubits: target_dag.qargs_interner.insert(&qubits),
            clbits: target_dag.cargs_interner.get_default(),
            params: Some(Box::new(new_params)),
            extra_attrs: None,
            // #[cfg(feature = "cache_pygates")]
            py_op: OnceCell::new(),
        };
        instructions.push(pi);
    }

    let _ = target_dag.add_from_iter(py, instructions.into_iter());
    println!("Qubit io map: {:?}", target_dag.qubit_io_map);
    println!("Qubits: {:?}", target_dag.qubits);

    Ok(target_dag)
}

#[pyfunction]
#[pyo3(name = "create_dag_from_scratch_1_op")]
fn py_create_dag_from_scratch_1_op(py: Python<'_>) -> PyResult<DAGCircuit> {
    let gate_sequence = TwoQubitGateSequence {
        gates: vec![(Some(StandardGate::CXGate), SmallVec::new(), smallvec![0, 1])],
        global_phase: 0.0,
    };

    let mut sequence = TwoQubitUnitarySequence::new();
    sequence.set_state((gate_sequence, Some(StandardGate::CYGate.name().to_string())));

    Ok(dag_from_2q_gate_sequence(py, sequence)?)
}

#[pyfunction]
#[pyo3(name = "create_dag_from_scratch_3_ops")]
fn py_create_dag_from_scratch_3_ops(py: Python<'_>) -> PyResult<DAGCircuit> {
    let gate_sequence = TwoQubitGateSequence {
        gates: vec![
            (Some(StandardGate::CXGate), SmallVec::new(), smallvec![0, 1]),
            (Some(StandardGate::XGate), SmallVec::new(), smallvec![0]),
            (Some(StandardGate::YGate), SmallVec::new(), smallvec![1]),
        ],
        global_phase: 0.0,
    };

    let mut sequence = TwoQubitUnitarySequence::new();
    sequence.set_state((gate_sequence, Some(StandardGate::CYGate.name().to_string())));

    Ok(dag_from_2q_gate_sequence(py, sequence)?)
}

#[pymodule]
pub fn reproduce_elenas_issues(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_create_dag_from_scratch_1_op))?;
    m.add_wrapped(wrap_pyfunction!(py_create_dag_from_scratch_3_ops))?;
    Ok(())
}
