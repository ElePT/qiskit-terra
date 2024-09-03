// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use approx::relative_eq;
use core::panic;
use hashbrown::{HashMap, HashSet};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::{Clbit, Qubit};
use smallvec::{smallvec, SmallVec};
use std::f64::consts::PI;
use std::hash::Hash;
use std::mem;

// use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::{IntoPyDict, PyDict, PyList, PyString, PyTuple};
use pyo3::wrap_pyfunction;
use pyo3::Python;

use ndarray::prelude::*;
use numpy::PyArrayMethods;
use numpy::{IntoPyArray, PyArray, PyArray2, PyReadonlyArray2, ToPyArray};
// use ndarray::array;
use num_complex::{Complex, Complex64, ComplexFloat};

use rustworkx_core::petgraph::stable_graph::{EdgeReference, NodeIndex};

use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::dag_circuit::{add_global_phase, DAGCircuit, NodeType};
use qiskit_circuit::imports::{CIRCUIT_TO_DAG, DAG_TO_CIRCUIT};
use qiskit_circuit::operations::{Operation, OperationRef, PyInstruction};
use qiskit_circuit::operations::{Param, StandardGate};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperationType};

use crate::euler_one_qubit_decomposer::{
    euler_one_qubit_decomposer, unitary_to_gate_sequence, EulerBasis, OneQubitGateErrorMap,
    OneQubitGateSequence,
};
use crate::nlayout::PhysicalQubit;
use crate::target_transpiler::target;
use crate::target_transpiler::{NormalOperation, Target};
use crate::two_qubit_decompose::{
    TwoQubitBasisDecomposer, TwoQubitGateSequence, TwoQubitWeylDecomposition,
};
use crate::QiskitError;

const PI2: f64 = PI / 2.;
const PI4: f64 = PI / 4.;

#[derive(Clone, Debug)]
enum DecomposerType {
    TwoQubitBasisDecomposerType(TwoQubitBasisDecomposer),
    XXDecomposerType(PyObject),
}

#[derive(Clone, Debug)]
enum UnitarySynthesisReturnType {
    DAGType(DAGCircuit),
    OneQSequenceType(OneQubitGateSequence),
    TwoQSequenceType(TwoQubitGateSequence),
}

// pub struct TwoQubitGateSequence {
//     gates: TwoQubitSequenceVec,
//     #[pyo3(get)]
//     global_phase: f64,
// }
// type TwoQubitSequenceVec = Vec<(Option<StandardGate>, SmallVec<[f64; 3]>, SmallVec<[u8; 2]>)>;

// This function converts the 2q synthesis output from the TwoQubitBasisDecomposer (sequence of gates)
// into a DAGCircuit for easier manipulation. Used in synth_su4 and reversed_synth_su4.
fn dag_from_2q_gate_sequence(
    py: Python<'_>,
    sequence: TwoQubitGateSequence,
) -> PyResult<DAGCircuit> {
    let gate_vec = sequence.gates;
    // is num_ops correct here?
    let mut target_dag = DAGCircuit::with_capacity(py, 2, 0, Some(gate_vec.len()), None, None)?;
    target_dag.set_global_phase(Param::Float(sequence.global_phase));

    // we need to collect "instructions" to avoid borrowing mutably in 2 places at the same time
    let instructions: Vec<PackedInstruction> = gate_vec
        .iter()
        .map(|(gate, params, qubit_ids)| {
            let qubits = vec![Qubit(qubit_ids[0] as u32), Qubit(qubit_ids[1] as u32)];
            PackedInstruction {
                op: PackedOperation::from_standard(gate.unwrap()),
                qubits: target_dag.qargs_interner.insert_owned(qubits),
                clbits: target_dag.cargs_interner.get_default(),
                params: Some(Box::new(smallvec![
                    Param::Float(params[0]),
                    Param::Float(params[1]),
                    Param::Float(params[2])
                ])),
                extra_attrs: None,
            }
        })
        .collect();

    // so we create an iterator again to call target_dag.add_from_iter
    target_dag.add_from_iter(py, instructions.into_iter());
    Ok(target_dag)
}

// main loop for default method. This loop calls the main run function defined below for all nodes in the dag.
#[pyfunction]
#[pyo3(name = "run_default_main_loop")]
fn py_run_default_main_loop(
    py: Python,
    dag: &mut DAGCircuit,
    // qubit_indices is a dict {bit:i for i,bit in enumerate dag.qubits}
    qubit_indices: &Bound<'_, PyDict>,
    min_qubits: usize,
    approximation_degree: Option<f64>,
    basis_gates: Option<Vec<PyBackedStr>>,
    coupling_map: Option<PyObject>,
    natural_direction: Option<bool>,
    pulse_optimize: Option<&str>,
    gate_lengths: Option<Bound<'_, PyDict>>,
    gate_errors: Option<Bound<'_, PyDict>>,
    target: Option<Target>,
) -> PyResult<DAGCircuit> {
    // Collect node ids into vec not to mix mutable and unmutable refs
    let node_ids: Vec<NodeIndex> = dag.op_nodes(false).collect();
    // Iterate recursively over control flow blocks and "unwrap"
    for node in node_ids {
        if let NodeType::Operation(inst) = &dag.dag[node] {
            if inst.op.control_flow() {
                if let OperationRef::Instruction(py_inst) = inst.op.view() {
                    let raw_blocks = py_inst.instruction.getattr(py, "blocks")?;
                    let circuit_to_dag = CIRCUIT_TO_DAG.get_bound(py);
                    let dag_to_circuit = DAG_TO_CIRCUIT.get_bound(py);

                    let mut new_blocks = Vec::new();

                    for raw_block in raw_blocks.bind(py).iter().unwrap() {
                        let block_obj = raw_block?;
                        let block = block_obj
                            .getattr(intern!(py, "_data"))?
                            .extract::<CircuitData>()?;
                        let mut new_dag: DAGCircuit =
                            circuit_to_dag.call1((block_obj.clone(),))?.extract()?;
                        let block_qargs = (0..block.num_qubits()).into_iter().map(|x| x as usize);
                        let node_qargs = (0..inst.op.num_qubits()).into_iter().map(|x| x as usize);

                        // remap qubit indices (it's a dict of len 2)
                        let new_qubit_indices = PyDict::new_bound(py);
                        for (inner, outer) in block_qargs.zip(node_qargs) {
                            new_qubit_indices.set_item(inner, qubit_indices.get_item(outer)?)?;
                        }

                        let res = py_run_default_main_loop(
                            py,
                            &mut new_dag,
                            &new_qubit_indices,
                            min_qubits,
                            approximation_degree,
                            basis_gates.clone(),
                            coupling_map.clone(),
                            natural_direction,
                            pulse_optimize,
                            gate_lengths.clone(),
                            gate_errors.clone(),
                            target.clone(),
                        )?;
                        let res_circuit = dag_to_circuit.call1((res,))?;
                        new_blocks.push(res_circuit);
                    }

                    let bound_py_inst = py_inst.instruction.bind(py);
                    let new_op = bound_py_inst.call_method1("replace_blocks", (new_blocks,))?;

                    let other_node = dag.get_node(py, node).unwrap().clone();

                    let _ = dag.substitute_node(other_node.bind(py), &new_op, true, false);
                }
            }
        }
    }

    // Create new empty dag
    let mut out_dag = dag.copy_empty_like(py, "alike")?;

    // Iterate over nodes, find decomposers and run synthesis
    for node in dag.topological_op_nodes()? {
        if let NodeType::Operation(packed_instr) = &dag.dag[node] {
            let n_qubits: usize = packed_instr.op.num_qubits().try_into().unwrap();
            if packed_instr.op.name() == "unitary" && n_qubits >= min_qubits {
                let unitary: Array<Complex<f64>, Dim<[usize; 2]>> =
                    packed_instr.op.matrix(&[]).unwrap();

                // SmallVec of physical qubits of len 2. We will map the instruction qubits to the provided qubit_indices
                // in the original code, this info is provided through the "coupling_map" option in an obscure way
                let qubits: SmallVec<[PhysicalQubit; 2]> =
                    if let NodeType::Operation(inst) = &dag.dag[node] {
                        let mut p_qubits = SmallVec::new();
                        for q in dag.get_qargs(inst.qubits) {
                            let mapped_q = qubit_indices
                                .get_item(q.0)
                                .and_then(|item| item.unwrap().extract::<u32>())?;
                            p_qubits.push(PhysicalQubit::new(mapped_q));
                        }
                        p_qubits
                    } else {
                        unreachable!("nodes in runs will always be op nodes")
                    };

                let raw_synth_output: Option<UnitarySynthesisReturnType> =
                    run_default_unitary_synthesis(
                        py,
                        unitary,
                        qubits,
                        approximation_degree,
                        basis_gates.clone(),
                        coupling_map.clone(),
                        natural_direction,
                        pulse_optimize,
                        gate_lengths.clone(),
                        gate_errors.clone(),
                        target.clone(),
                    )?;

                // the output can be None, a DAGCircuit or a TwoQubitGateSequence
                match raw_synth_output {
                    None => {
                        out_dag.push_back(py, packed_instr.clone());
                    }

                    Some(synth_output) => {
                        match synth_output {
                            UnitarySynthesisReturnType::DAGType(synth_dag) => {
                                for synth_node in synth_dag.topological_op_nodes()? {
                                    out_dag.set_global_phase(add_global_phase(
                                        py,
                                        &out_dag.get_global_phase(),
                                        &synth_dag.get_global_phase(),
                                    )?);

                                    if let NodeType::Operation(synth_packed_instr) =
                                        &synth_dag.dag[synth_node]
                                    {
                                        let qargs = dag.qargs_interner.get(packed_instr.qubits);
                                        synth_packed_instr.to_owned().qubits =
                                            out_dag.qargs_interner.insert_owned(
                                                out_dag
                                                    .qubits
                                                    .map_bits(
                                                        synth_dag
                                                            .qubits
                                                            .clone()
                                                            .map_indices(qargs)
                                                            .map(|b| b.bind(py).clone()),
                                                    )?
                                                    .collect(),
                                            );

                                        out_dag.push_back(py, synth_packed_instr.clone());
                                    }
                                }
                            }
                            UnitarySynthesisReturnType::TwoQSequenceType(synth_sequence) => {
                                // let (node_list, global_phase, gate) = raw_synth_dag.extract::<((&str, Param, Vec<usize>), Param, &str)>(py)?;
                                todo!()
                            }
                            UnitarySynthesisReturnType::OneQSequenceType(synth_sequence) => {
                                // let (node_list, global_phase, gate) = raw_synth_dag.extract::<((&str, Param, Vec<usize>), Param, &str)>(py)?;
                                todo!()
                            }
                        }
                    }
                }
            } else {
                out_dag.push_back(py, packed_instr.clone());
            }
        }
    }
    Ok(out_dag)
}

// this function is used in the 1q synthesis path
fn build_error_map(target: &Option<Target>) -> Option<OneQubitGateErrorMap> {
    match target {
        Some(target) => {
            let mut e_map: OneQubitGateErrorMap = OneQubitGateErrorMap::new(target.num_qubits);
            for qubit in 0..target.num_qubits.unwrap() {
                let mut gate_error = HashMap::new();
                for gate in target.operation_names() {
                    // the unwrap is a bit suspicious. we might need to actually handle the case.
                    if let Some(e) = target[gate]
                        [Some(&smallvec![PhysicalQubit::new(qubit as u32)])]
                    .clone()
                    .unwrap()
                    .error
                    {
                        gate_error.insert(gate.to_string(), e);
                    }
                    e_map.add_qubit(gate_error)
                }
            }
            Some(e_map)
        }
        None => None,
    }
}

// the synthesis workflow is decided in this function. We decide whether to run 1q, 2q or shannon depending on the unitary.
// * 1q synthesis, we run `unitary_to_gate_sequence` (Rust) and return the sequence (to be converted to DAG)
// * for 2q synthesis, we first need to collect all potential decomposers (instantiated) and then run them to select the best circuit
// * for shannon, we call python because the decomposer isn't implemented in rust.
fn run_default_unitary_synthesis(
    py: Python,
    unitary: Array2<Complex64>,
    qubits: SmallVec<[PhysicalQubit; 2]>,
    approximation_degree: Option<f64>,
    basis_gates: Option<Vec<PyBackedStr>>,
    coupling_map: Option<PyObject>,
    natural_direction: Option<bool>,
    pulse_optimize: Option<&str>,
    gate_lengths: Option<Bound<'_, PyDict>>,
    gate_errors: Option<Bound<'_, PyDict>>,
    target: Option<Target>,
) -> PyResult<Option<UnitarySynthesisReturnType>> {
    match unitary.shape() {
        [2, 2] => {
            // run 1q decomposition -> fully in rust
            let error_map = build_error_map(&target);
            match unitary_to_gate_sequence(
                unitary.into_pyarray_bound(py).readonly(),
                basis_gates.unwrap(),
                qubits[0].0 as usize,
                error_map.as_ref(),
                false,
                None,
            )? {
                None => return Ok(None),
                Some(synth_sequence) => {
                    return Ok(Some(UnitarySynthesisReturnType::OneQSequenceType(
                        synth_sequence,
                    )))
                }
            }
        }
        [4, 4] => {
            // run 2q decomposition -> in rust except for XXDecomposer
            // step1: select decomposers
            let decomposers = match target {
                Some(target) => {
                    let decomposers_2q =
                        get_2q_decomposers_from_target(py, &target, &qubits, approximation_degree)?;
                    match decomposers_2q {
                        Some(decomp) => decomp,
                        None => Vec::new(),
                    }
                }
                None => todo!(),
            };

            // If we have a single TwoQubitBasisDecomposer skip dag creation as we don't need to
            // store and can instead manually create the synthesized gates directly in the output dag
            if decomposers.len() == 1 {
                let decomposer_item = decomposers.iter().next().unwrap();
                match &decomposer_item {
                    &DecomposerType::TwoQubitBasisDecomposerType(decomposer) => {
                        let preferred_dir = preferred_direction(
                            py,
                            decomposer_item,
                            &qubits,
                            natural_direction,
                            coupling_map,
                            gate_lengths,
                            gate_errors,
                        )?;
                        let synth = synth_su4_no_dag(
                            py,
                            &unitary,
                            decomposer,
                            preferred_dir,
                            approximation_degree,
                        )?;
                        return Ok(Some(synth));
                    }
                    _ => (), //skip action
                }
            }

            let mut synth_circuits = Vec::new();

            for decomposer in &decomposers {
                let preferred_dir = preferred_direction(
                    py,
                    decomposer,
                    &qubits,
                    natural_direction,
                    coupling_map.clone(),
                    gate_lengths.clone(),
                    gate_errors.clone(),
                )?;
                let synth_circuit = synth_su4(
                    py,
                    &unitary,
                    decomposer,
                    preferred_dir,
                    approximation_degree,
                );
                synth_circuits.push(synth_circuit)
            }
            // get the minimum of synth_circuits and return
            // TODO

            let synth_circuit = synth_circuits[0]?.clone();
            Ok(Some(synth_circuit))
        }
        _ => {
            todo!()
            // run qsd -> in python
        }
    }
    // The output will either be a dag circuit or a list of synthesized gates or None (need to use Option)
}

// f64 is not hashable so we divide it into a mantissa-exponent-sign triplet
fn integer_decode_f64(val: f64) -> (u64, i16, i8) {
    let bits: u64 = unsafe { mem::transmute(val) };
    let sign: i8 = if bits >> 63 == 0 { 1 } else { -1 };
    let mut exponent: i16 = ((bits >> 52) & 0x7ff) as i16;
    let mantissa = if exponent == 0 {
        (bits & 0xfffffffffffff) << 1
    } else {
        (bits & 0xfffffffffffff) | 0x10000000000000
    };

    exponent -= 1023 + 52;
    (mantissa, exponent, sign)
}

fn integer_encode_f64(mantissa: u64, exponent: i16, sign: i8) -> f64 {
    // Adjust exponent back by adding the bias and number of mantissa bits
    let exponent = (exponent + 1023 + 52) as u64;

    // Pack sign, exponent, and mantissa into a u64
    let sign_bit = if sign == -1 { 1u64 << 63 } else { 0 };
    let exponent_bits = (exponent & 0x7ff) << 52;
    let mantissa_bits = mantissa & 0xfffffffffffff;

    // Combine all parts into a single u64
    let bits = sign_bit | exponent_bits | mantissa_bits;

    // Transmute the u64 bits back into an f64
    unsafe { mem::transmute(bits) }
}

#[derive(Hash, Eq, PartialEq)]
struct InteractionStrength((u64, i16, i8));

impl InteractionStrength {
    fn new(val: f64) -> InteractionStrength {
        InteractionStrength(integer_decode_f64(val))
    }
    fn to_f64(&self) -> f64 {
        let (mantissa, exponent, sign) = self.0;
        integer_encode_f64(mantissa, exponent, sign)
    }
}

// helper function for decomposer_2q_from_target
fn replace_parametrized_gate(mut op: NormalOperation) -> NormalOperation {
    if let Some(std_gate) = op.operation.try_standard_gate() {
        match (std_gate.name(), &op.params[0]) {
            ("rxx", Param::ParameterExpression(_)) => op.params[0] = Param::Float(PI2),
            ("rzx", Param::ParameterExpression(_)) => op.params[0] = Param::Float(PI4),
            ("rzz", Param::ParameterExpression(_)) => op.params[0] = Param::Float(PI2),
            _ => todo!(), // how can we pass?
        }
    }
    op
}

// taken from optimize_1q_decomposition
fn possible_decomposers(basis_set: Option<HashSet<&str>>) -> HashSet<&str> {
    let one_q_euler_basis_gates: HashMap<&str, Vec<&str>> = {
        let mut m = HashMap::new();
        m.insert("U3", vec!["u3"]);
        m.insert("U321", vec!["u3", "u2", "u1"]);
        m.insert("U", vec!["u"]);
        m.insert("PSX", vec!["p", "sx"]);
        m.insert("U1X", vec!["u1", "rx"]);
        m.insert("RR", vec!["r"]);
        m.insert("ZYZ", vec!["rz", "ry"]);
        m.insert("ZXZ", vec!["rz", "rx"]);
        m.insert("XZX", vec!["rz", "rx"]);
        m.insert("XYX", vec!["rx", "ry"]);
        m.insert("ZSXX", vec!["rz", "sx", "x"]);
        m.insert("ZSX", vec!["rz", "sx"]);
        m
    };

    let mut decomposers = HashSet::new();
    match basis_set {
        None => decomposers = HashSet::from_iter(one_q_euler_basis_gates.keys().map(|key| *key)),
        Some(basis_set) => {
            for (euler_basis_name, gates) in one_q_euler_basis_gates.into_iter() {
                if HashSet::from_iter(gates.iter().cloned()).is_subset(&basis_set) {
                    decomposers.insert(euler_basis_name);
                }
            }
            if decomposers.contains("U3") && decomposers.contains("U321") {
                decomposers.remove("U3");
            }
            if decomposers.contains("ZSX") && decomposers.contains("ZSXX") {
                decomposers.remove("ZSX");
            }
        }
    }
    decomposers
}

fn get_2q_decomposers_from_target(
    py: Python,
    target: &Target,
    qubits: &SmallVec<[PhysicalQubit; 2]>,
    approximation_degree: Option<f64>,
) -> PyResult<Option<Vec<DecomposerType>>> {
    let mut qubits: SmallVec<[PhysicalQubit; 2]> = qubits.clone();
    qubits.sort();
    let reverse_qubits: SmallVec<[PhysicalQubit; 2]> = qubits.iter().rev().map(|x| *x).collect();
    // TODO: here return cache --> implementation?

    let mut available_2q_basis: HashMap<&str, NormalOperation> = HashMap::new();
    let mut available_2q_props: HashMap<&str, (Option<f64>, Option<f64>)> = HashMap::new();

    // try both directions for the qubits tuple
    let mut keys = HashSet::new();
    let mut tuples = HashSet::new();
    match target.operation_names_for_qargs(Some(&qubits)) {
        Ok(direct_keys) => {
            keys = direct_keys;
            tuples.insert(qubits.clone());
            if let Ok(reverse_keys) = target.operation_names_for_qargs(Some(&reverse_qubits)) {
                keys.extend(reverse_keys);
                tuples.insert(reverse_qubits);
            }
        }
        Err(_) => {
            if let Ok(reverse_keys) = target.operation_names_for_qargs(Some(&reverse_qubits)) {
                keys = reverse_keys;
                tuples.insert(reverse_qubits);
            } else {
                return Err(QiskitError::new_err(
                    "Target has no gates available on qubits to synthesize over.",
                ));
            }
        }
    }

    for (key, q_pair) in keys.iter().zip(tuples.iter()) {
        let op = target.operation_from_name(key).unwrap().clone();
        if let PackedOperationType::Gate = op.operation.discriminant() {
            available_2q_basis.insert(key, replace_parametrized_gate(op));
            available_2q_props.insert(
                key,
                (
                    target[key][Some(q_pair)].clone().unwrap().duration,
                    target[key][Some(q_pair)].clone().unwrap().error,
                ),
            );
        }
    }

    if available_2q_basis.is_empty() {
        return Err(QiskitError::new_err(
            "Target has no gates available on qubits to synthesize over.",
        ));
    }

    // available decomposition basis on each of the qubits of the pair
    // NOTE: assumes both qubits have the same single-qubit gates
    let available_1q_basis: HashSet<&str> = possible_decomposers(Some(
        target
            .operation_names_for_qargs(Some(&smallvec![qubits[0]]))
            .expect("Sth should be returned"),
    ));

    // find all decomposers
    let mut decomposers: Vec<DecomposerType> = Vec::new();

    fn is_supercontrolled(op: &NormalOperation) -> bool {
        match op.operation.matrix(&[]) {
            None => false,
            Some(unitary_matrix) => {
                let kak = TwoQubitWeylDecomposition::new_inner(unitary_matrix.view(), None, None)
                    .unwrap();
                relative_eq!(kak.a, PI4) && relative_eq!(kak.c, 0.0)
            }
        }
    }

    fn is_controlled(op: &NormalOperation) -> bool {
        match op.operation.matrix(&[]) {
            None => false,
            Some(unitary_matrix) => {
                let kak = TwoQubitWeylDecomposition::new_inner(unitary_matrix.view(), None, None)
                    .unwrap();
                relative_eq!(kak.b, 0.0) && relative_eq!(kak.c, 0.0)
            }
        }
    }

    // Iterate over 1q and 2q supercontrolled basis, append TwoQubitBasisDecomposers
    let supercontrolled_basis: HashMap<&str, NormalOperation> = available_2q_basis
        .clone()
        .into_iter()
        .filter(|(_, v)| is_supercontrolled(v))
        .collect();
    for (basis_1q, basis_2q) in available_1q_basis
        .iter()
        .zip(supercontrolled_basis.keys().into_iter())
    {
        let mut basis_2q_fidelity: f64 = match available_2q_props.get(basis_2q) {
            Some(&(_, error)) => {
                if error.is_none() {
                    1.0
                } else {
                    1.0 - error.unwrap()
                }
            }
            None => 1.0,
        };

        if approximation_degree.is_some() {
            basis_2q_fidelity = basis_2q_fidelity * approximation_degree.unwrap();
        }

        let gate = &supercontrolled_basis[basis_2q];
        let decomposer = TwoQubitBasisDecomposer::new_inner(
            gate.operation.name().to_owned(),
            gate.operation.matrix(&[]).unwrap().view(),
            basis_2q_fidelity,
            basis_1q,
            None,
        )?;

        decomposers.push(DecomposerType::TwoQubitBasisDecomposerType(decomposer));

        // If our 2q basis gates are a subset of cx, ecr, or cz then we know TwoQubitBasisDecomposer
        // is an ideal decomposition and there is no need to bother calculating the XX embodiments
        // or try the XX decomposer
        let mut goodbye_set = HashSet::new();
        goodbye_set.insert("cx");
        goodbye_set.insert("cz");
        goodbye_set.insert("ecr");

        let available_basis_set =
            HashSet::from_iter(available_2q_basis.clone().keys().into_iter().map(|&k| k));
        if goodbye_set.is_superset(&available_basis_set) {
            // TODO: decomposer cache thingy
            return Ok(Some(decomposers));
        }

        // Let's now look for possible controlled decomposers (i.e. XXDecomposer)
        let controlled_basis: HashMap<&str, NormalOperation> = available_2q_basis
            .clone()
            .into_iter()
            .filter(|(_, v)| is_controlled(v))
            .collect();
        let mut basis_2q_fidelity: HashMap<InteractionStrength, f64> = HashMap::new();

        // the embodiments will be a list of circuit representations
        let mut embodiments: HashMap<InteractionStrength, CircuitData> = HashMap::new();
        let mut pi2_basis: Option<&str> = None;

        for (k, v) in controlled_basis.iter() {
            let strength = 2.0
                * TwoQubitWeylDecomposition::new_inner(
                    v.operation.matrix(&[]).unwrap().view(),
                    None,
                    None,
                )
                .unwrap()
                .a;
            // each strength has its own fidelity
            let fidelity_value = match available_2q_props.get(basis_2q) {
                Some(&(_, error)) => {
                    if error.is_none() {
                        1.0
                    } else {
                        1.0 - error.unwrap()
                    }
                }
                None => 1.0,
            };
            basis_2q_fidelity.insert(InteractionStrength::new(strength), fidelity_value);

            // rewrite XX of the same strength in terms of it
            let xx_embodiments =
                PyModule::import_bound(py, "qiskit.synthesis.two_qubit.xx_decompose")?
                    .getattr("XXEmbodiments")?;

            // The embodiment should be a py object representing a quantum circuit
            let embodiment =
                xx_embodiments.get_item(v.clone().into_py(py).getattr(py, "base_class")?)?; //XXEmbodiments[v.base_class];

            // This is 100% gonna fail
            if embodiment.getattr("parameters")?.len()? == 1 {
                embodiments.insert(
                    InteractionStrength::new(strength),
                    embodiment
                        .call_method1("assign_parameters", (strength,))?
                        .extract()?,
                );
            } else {
                embodiments.insert(InteractionStrength::new(strength), embodiment.extract()?);
            }

            // basis equivalent to CX are well optimized so use for the pi/2 angle if available
            if relative_eq!(strength, PI2) && supercontrolled_basis.contains_key(k) {
                pi2_basis = Some(v.operation.name());
            }
        }

        // if we are using the approximation_degree knob, use it to scale already-given fidelities
        if approximation_degree.is_some() {
            for fidelity in basis_2q_fidelity.values_mut() {
                *fidelity *= approximation_degree.unwrap();
            }
        }

        // Iterate over 2q fidelities ans select decomposers

        if !basis_2q_fidelity.is_empty() {
            let xx_decomposer: Bound<'_, PyAny> =
                PyModule::import_bound(py, "qiskit.synthesis.two_qubit.xx_decompose")?
                    .getattr("XXDecomposer")?;

            for basis_1q in &available_1q_basis {
                let mut fidelity: f64 = 1.0;
                let mut pi2_decomposer: Option<TwoQubitBasisDecomposer> = None;
                // check that these are the expected strings (caps??)
                if pi2_basis.unwrap() == "cx" && *basis_1q == "ZSX" {
                    if approximation_degree.is_none() {
                        let error = target["cx"][Some(&qubits)].clone().unwrap().error;
                        fidelity = match error {
                            Some(error) => 1.0 - error,
                            None => 1.0,
                        }
                    } else {
                        fidelity = approximation_degree.unwrap()
                    }
                    pi2_decomposer = Some(TwoQubitBasisDecomposer::new_inner(
                        pi2_basis.unwrap().to_string(),
                        StandardGate::CXGate.matrix(&[]).unwrap().view(),
                        fidelity,
                        basis_1q,
                        Some(true),
                    )?);
                }

                let basis_2q_fidelity_dict = PyDict::new_bound(py);
                for (k, v) in basis_2q_fidelity.iter() {
                    basis_2q_fidelity_dict.set_item(k.to_f64().clone(), v)?;
                }

                let embodiments_dict = PyDict::new_bound(py);

                // Use iterator to populate PyDict
                embodiments.iter().for_each(|(key, value)| {
                    embodiments_dict
                        .set_item(key.to_f64().clone(), value.clone().into_py(py))
                        .unwrap();
                });

                let decomposer = xx_decomposer.call1((
                    basis_2q_fidelity_dict,
                    PyString::new_bound(py, basis_1q),
                    embodiments_dict,
                    pi2_decomposer,
                ))?; //instantiate properly
                decomposers.push(DecomposerType::XXDecomposerType(decomposer.into()));
            }
        }
    }
    Ok(Some(decomposers))
}

// output of preferred_direction (bool):
// true = [0,1]
// false = [1,0]
// gate_lengths is a dict of {qubits_tuple: (gate_name, duration)}
// gate_errors is a dict of {qubits_tuple: (gate_name, error)}
fn preferred_direction(
    py: Python,
    decomposer: &DecomposerType,
    qubits: &SmallVec<[PhysicalQubit; 2]>,
    natural_direction: Option<bool>,
    coupling_map: Option<PyObject>,
    gate_lengths: Option<Bound<'_, PyDict>>,
    gate_errors: Option<Bound<'_, PyDict>>,
) -> PyResult<Option<bool>> {
    let reverse_qubits: SmallVec<[PhysicalQubit; 2]> = qubits.iter().rev().map(|x| *x).collect();
    let mut preferred_direction: Option<bool> = None;
    if natural_direction.is_none() || natural_direction.unwrap() == true {
        // find native gate directions from a (non-bidirectional) coupling map
        match coupling_map {
            Some(cmap) => {
                let neighbors0 = cmap
                    .call_method1(py, "neighbors", PyTuple::new_bound(py, [qubits[0].0]))?
                    .extract::<Vec<u32>>(py)?;
                let zero_one = neighbors0.contains(&qubits[0].0);
                let neighbors1 = cmap
                    .call_method1(py, "neighbors", PyTuple::new_bound(py, [qubits[1].0]))?
                    .extract::<Vec<u32>>(py)?;
                let one_zero = neighbors1.contains(&qubits[1].0);
                match (zero_one, one_zero) {
                    (true, false) => preferred_direction = Some(true),
                    (false, true) => preferred_direction = Some(false),
                    _ => (),
                }
            }
            None => (),
        }
    }
    let decomposer2q_gate = match decomposer {
        DecomposerType::TwoQubitBasisDecomposerType(decomp) => &decomp.gate,
        DecomposerType::XXDecomposerType(decomp) => {
            &decomp.getattr(py, "gate")?.extract::<String>(py)?
        }
    };
    // go back and see how gate_dict looks like
    let compute_cost = |gate_dict: &Bound<'_, PyDict>,
                        q_tuple: &SmallVec<[PhysicalQubit; 2]>,
                        cost: Option<f64>|
     -> () {
        let ids: (u32, u32) = (q_tuple[0].0, q_tuple[1].0);

        // let gate_dict_vec = gate_dict.get_item(key).unwrap_or()
        // if let gate_dict_vec = gate_dict
        //     .get_item(&ids)
        //     .iter()
        //     .map(|item| item.unwrap().extract().expect("idk what's wrong with ?"))
        // {
        //     if let Some(value) = gate_dict_vec
        //         .find(|(gate, _)| gate == decomposer2q_gate)
        //         .map(|(_, value)| *value)
        //     {
        //         cost = value;
        //     }
        // }
    };

    if preferred_direction.is_none() && (gate_lengths.is_some() || gate_errors.is_some()) {
        let mut cost_0_1: Option<f64> = None;
        let mut cost_1_0: Option<f64> = None;

        // Try to find the cost in gate_lengths
        compute_cost(&gate_lengths.unwrap(), qubits, cost_0_1);
        compute_cost(&gate_lengths.unwrap(), &reverse_qubits, cost_1_0);

        // If no valid cost was found in gate_lengths, check gate_errors
        if cost_0_1.is_none() && cost_1_0.is_none() {
            compute_cost(&gate_errors.unwrap(), qubits, cost_0_1);
            compute_cost(&gate_errors.unwrap(), &reverse_qubits, cost_1_0);
        }

        if cost_0_1.is_some() && cost_1_0.is_some() {
            if cost_0_1 < cost_1_0 {
                preferred_direction = Some(true)
            } else if cost_1_0 < cost_0_1 {
                preferred_direction = Some(false)
            }
        }
    }
    Ok(preferred_direction)
}

// synth function for 2q gates (4x4)
fn synth_su4(
    py: Python,
    su4_mat: &Array2<Complex64>,
    decomposer_2q: &DecomposerType,
    preferred_direction: Option<bool>,
    approximation_degree: Option<f64>,
) -> PyResult<UnitarySynthesisReturnType> {
    // double check approximation_degree None
    let is_approximate = approximation_degree.is_none() || approximation_degree.unwrap() != 1.0;
    let synth_dag = match decomposer_2q {
        DecomposerType::XXDecomposerType(decomposer) => {
            let mut kwargs = HashMap::<&str, bool>::new();
            kwargs.insert("approximate", is_approximate);
            kwargs.insert("use_dag", true);
            // can we avoid cloning the matrix to pass it to python?
            decomposer
                .call_method_bound(
                    py,
                    "__call__",
                    (su4_mat.clone().into_pyarray_bound(py),),
                    Some(&kwargs.into_py_dict_bound(py)),
                )?
                .extract::<DAGCircuit>(py)?
        }
        DecomposerType::TwoQubitBasisDecomposerType(decomposer) => {
            // we don't have access to basis_fidelity, right???
            let synth = decomposer.call_inner(su4_mat.view(), None, is_approximate, None)?;
            dag_from_2q_gate_sequence(py, synth)?
        }
    };

    match preferred_direction {
        None => return Ok(UnitarySynthesisReturnType::DAGType(synth_dag)),
        Some(preferred_dir) => {
            let mut synth_direction: Option<Vec<u32>> = None;
            for node in synth_dag.topological_op_nodes()? {
                if let NodeType::Operation(inst) = &synth_dag.dag[node] {
                    if inst.op.num_qubits() == 2 {
                        // not sure if these are the right qargs
                        let qargs = synth_dag.get_qargs(inst.qubits);
                        synth_direction = Some(vec![qargs[0].0, qargs[1].0]);
                    }
                }
            }
            match synth_direction {
                None => return Ok(UnitarySynthesisReturnType::DAGType(synth_dag)),
                Some(synth_direction) => {
                    let synth_dir = match synth_direction.as_slice() {
                        [0, 1] => true,
                        [1, 0] => false,
                        _ => panic!(),
                    };
                    if synth_dir != preferred_dir {
                        return reversed_synth_su4(
                            py,
                            su4_mat,
                            decomposer_2q,
                            approximation_degree,
                        );
                    } else {
                        return Ok(UnitarySynthesisReturnType::DAGType(synth_dag));
                    }
                }
            }
        }
    }
}

fn synth_su4_no_dag(
    py: Python<'_>,
    su4_mat: &Array2<Complex64>,
    decomposer_2q: &TwoQubitBasisDecomposer,
    preferred_direction: Option<bool>,
    approximation_degree: Option<f64>,
) -> PyResult<UnitarySynthesisReturnType> {
    let is_approximate = approximation_degree.is_none() || approximation_degree.unwrap() != 1.0;
    let synth = decomposer_2q.call_inner(su4_mat.view(), None, is_approximate, None)?;
    // return (synth_circ, synth_circ.global_phase, decomposer2q.gate)

    match preferred_direction {
        None => return Ok(UnitarySynthesisReturnType::TwoQSequenceType(synth)),
        Some(preferred_dir) => {
            let mut synth_direction: Option<SmallVec<[u8; 2]>> = None;
            for (gate, _, qubits) in synth.gates.clone() {
                if gate.is_none() || gate.unwrap().name() == "cx" {
                    synth_direction = Some(qubits);
                }
            }

            match synth_direction {
                None => return Ok(UnitarySynthesisReturnType::TwoQSequenceType(synth)),
                Some(synth_direction) => {
                    let synth_dir = match synth_direction.as_slice() {
                        [0, 1] => true,
                        [1, 0] => false,
                        _ => panic!(),
                    };
                    if synth_dir != preferred_dir {
                        return reversed_synth_su4(
                            py,
                            su4_mat,
                            &DecomposerType::TwoQubitBasisDecomposerType(*decomposer_2q),
                            approximation_degree,
                        );
                    } else {
                        return Ok(UnitarySynthesisReturnType::TwoQSequenceType(synth));
                    }
                }
            }
        }
    }
}

fn reversed_synth_su4(
    py: Python<'_>,
    su4_mat: &Array2<Complex64>,
    decomposer_2q: &DecomposerType,
    approximation_degree: Option<f64>,
) -> PyResult<UnitarySynthesisReturnType> {
    let is_approximate = approximation_degree.is_none() || approximation_degree.unwrap() != 1.0;
    let mut su4_mat_mm = su4_mat.clone();
    su4_mat_mm[[1, 2]] = su4_mat_mm[[2, 1]];
    // In Python, this is: su4_mat_mm[:, [1, 2]] = su4_mat_mm[:, [2, 1]];
    su4_mat_mm
        .slice_mut(s![1, ..])
        .assign(&su4_mat_mm.slice_mut(s![.., 2]).to_owned());
    su4_mat_mm
        .slice_mut(s![2, ..])
        .assign(&su4_mat_mm.slice_mut(s![.., 1]).to_owned());

    let synth_dag = match decomposer_2q {
        DecomposerType::XXDecomposerType(decomposer) => {
            let kwargs = HashMap::<&str, bool>::new();
            kwargs.insert("approximate", is_approximate);
            kwargs.insert("use_dag", true);
            let synth = decomposer
                .call_method_bound(
                    py,
                    "__call__",
                    (su4_mat_mm.clone().into_pyarray_bound(py),),
                    Some(&kwargs.into_py_dict_bound(py)),
                )?
                .extract::<DAGCircuit>(py)?;

            let mut target_dag = DAGCircuit::with_capacity(py, 2, 0, None, None, None)?;
            target_dag.set_global_phase(synth.get_global_phase());
            // out_dag.add_qubits(list(reversed(synth_circ.qubits)))
            // let flip_bits = target_dag.qubits;
            // for node in synth.topological_op_nodes():
            //     qubits = tuple(flip_bits[synth.find_bit(x).index] for x in node.qargs)
            //     node = DAGOpNode.from_instruction(
            //         node._to_circuit_instruction().replace(qubits=qubits, params=node.params)
            //     )
            //     out_dag._apply_op_node_back(node)
        }
        DecomposerType::TwoQubitBasisDecomposerType(decomposer) => {
            // we don't have access to basis_fidelity, right???
            let synth = decomposer.call_inner(su4_mat_mm.view(), None, is_approximate, None)?;
        }
    };
}

#[pymodule]
pub fn unitary_synthesis(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_run_default_main_loop))?;
    Ok(())
}
