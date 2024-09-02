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
use hashbrown::{HashMap, HashSet};
use numpy::PyReadonlyArray2;
use pyo3::types::PyString;
use std::collections::BTreeMap;
use std::hash::Hash;

use ndarray::prelude::*;
use num_complex::{Complex, Complex64, ComplexFloat};
use numpy::{IntoPyArray, ToPyArray};
use std::f64::consts::{FRAC_1_SQRT_2, PI};

use pulp::x86;
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::PyTuple;
use pyo3::types::{IntoPyDict, PyDict, PyList};
use pyo3::wrap_pyfunction;

use crate::nlayout::PhysicalQubit;
use smallvec::SmallVec;

use crate::target_transpiler::{NormalOperation, Target};

use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::dag_circuit::{add_global_phase, DAGCircuit, NodeType};
use qiskit_circuit::imports::{CIRCUIT_TO_DAG, DAG_TO_CIRCUIT};
use qiskit_circuit::operations::{Operation, OperationRef, PyInstruction};
use qiskit_circuit::operations::{Param, StandardGate};
use qiskit_circuit::packed_instruction::PackedInstruction;
use rustworkx_core::petgraph::stable_graph::{EdgeReference, NodeIndex};

use crate::euler_one_qubit_decomposer::{
    euler_one_qubit_decomposer, unitary_to_gate_sequence, EulerBasis, OneQubitGateErrorMap,
    OneQubitGateSequence,
};
use crate::two_qubit_decompose::{
    TwoQubitBasisDecomposer, TwoQubitGateSequence, TwoQubitWeylDecomposition,
};
// use crate::optimize_1q_gates::
use crate::QiskitError;

const PI2: f64 = PI / 2.;
const PI4: f64 = PI / 4.;

use std::mem;

// main loop for default method. This loop calls the main run function defined below for all nodes in the dag.
#[pyfunction]
#[pyo3(name = "run_default_main_loop")]
fn py_run_default_main_loop(
    py: Python,
    dag: &mut DAGCircuit,
    qubit_indices: HashMap<usize, usize>,
    min_qubits: usize,
    approximation_degree: Option<f64>,
    basis_gates: Option<Vec<PyBackedStr>>,
    coupling_map: Option<PyObject>,
    natural_direction: Option<&str>,
    pulse_optimize: Option<&str>,
    gate_lengths: Option<&str>,
    gate_errors: Option<&str>,
    qubits: Option<Vec<usize>>,
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

                        let qubit_ids: HashMap<_, _> = block_qargs
                            .zip(node_qargs)
                            .map(|(inner, outer)| (inner, qubit_indices[&outer]))
                            .collect();
                        let res = py_run_default_main_loop(
                            py,
                            &mut new_dag,
                            qubit_ids,
                            min_qubits,
                            approximation_degree,
                            basis_gates,
                            coupling_map,
                            natural_direction,
                            pulse_optimize,
                            gate_lengths,
                            gate_errors,
                            qubits,
                            target,
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
                let unitary: ArrayBase<OwnedRepr<Complex<f64>>, Dim<[usize; 2]>> =
                    packed_instr.op.matrix(&[]).unwrap();

                // cmap is always supported in the default plugin (no need to check)
                let new_q_ids: Vec<usize> = (0..n_qubits)
                    .into_iter()
                    .map(|x| qubit_indices[&x])
                    .collect();
                // let coupling_map: (Py<PyAny>, Vec<usize>) =(cmap.clone(), new_q_ids);

                let raw_synth_output: Option<UnitarySynthesisType> = run_default_unitary_synthesis(
                    py,
                    unitary,
                    approximation_degree,
                    basis_gates,
                    coupling_map,
                    natural_direction,
                    pulse_optimize,
                    gate_lengths,
                    gate_errors,
                    qubits,
                    target,
                )?;

                // the output can be None, a DAGCircuit or a TwoQubitGateSequence
                match raw_synth_output {
                    None => {
                        out_dag.push_back(py, packed_instr.clone());
                    }

                    Some(synth_output) => {
                        match synth_output {
                            UnitarySynthesisType::DAGType(synth_dag) => {
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
                            UnitarySynthesisType::TwoQSequenceType(synth_sequence) => {
                                // let (node_list, global_phase, gate) = raw_synth_dag.extract::<((&str, Param, Vec<usize>), Param, &str)>(py)?;
                                todo!()
                            }
                            UnitarySynthesisType::OneQSequenceType(synth_sequence) => {
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
fn build_error_map(target: &Option<Target>) -> Option<&OneQubitGateErrorMap> {
    match target {
        Some(target) => {
            let mut e_map: OneQubitGateErrorMap = OneQubitGateErrorMap::new(target.num_qubits);
            for qubit in 0..target.num_qubits.unwrap() {
                let mut single_qubit: SmallVec<[PhysicalQubit; 2]> = SmallVec::new();
                single_qubit.push(PhysicalQubit::new(qubit as u32));

                let mut gate_error = HashMap::new();

                for gate in target.operation_names() {
                    // the unwrap is a bit suspicious. we might need to actually handle the case.
                    if let Some(e) = target[gate][Some(&single_qubit)].clone().unwrap().error {
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
    unitary: PyReadonlyArray2<Complex64>,
    approximation_degree: Option<f64>,
    basis_gates: Option<Vec<PyBackedStr>>,
    coupling_map: Option<PyObject>,
    natural_direction: Option<&str>,
    pulse_optimize: Option<&str>,
    gate_lengths: Option<&str>,
    gate_errors: Option<&str>,
    qubits: Option<Vec<usize>>,
    target: Option<Target>,
) -> PyResult<Option<UnitarySynthesisType>> {
    match unitary.as_array().shape() {
        [2, 2] => {
            // run 1q decomposition -> fully in rust
            let error_map = build_error_map(&target);
            match unitary_to_gate_sequence(unitary, basis_gates.unwrap(), qubits.unwrap()[0], error_map, false, None)?
            {
                None => return Ok(None),
                Some(synth_sequence) => {
                    return Ok(Some(UnitarySynthesisType::OneQSequenceType(synth_sequence)))
                }
            }
        }
        [4, 4] => {
            // run 2q decomposition -> in rust except for XXDecomposer
            // step1: select decomposers
            let decomposers = match target{
                Some(target){
                    let decomposers_2q = get_2q_decomposers_from_target(py, &target, qubits, approximation_degree)?;
                    match decomposers_2q{
                        Some(decomp) => {
                            decomp
                        }
                        None => Vec::new()
                    }
                }
                None => todo!()
            };

            for decomposer in decomposers{
                todo!()
            }
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

#[derive(Clone, Debug)]
enum DecomposerType {
    TwoQubitBasisDecomposerType(TwoQubitBasisDecomposer),
    TwoQubitWeylDecompositionType(TwoQubitWeylDecomposition),
    XXDecomposerType(PyObject),
}

#[derive(Clone, Debug)]
enum UnitarySynthesisType {
    DAGType(DAGCircuit),
    OneQSequenceType(OneQubitGateSequence),
    TwoQSequenceType(TwoQubitGateSequence),
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
    qubits: Vec<u32>,
    approximation_degree: Option<f64>,
) -> PyResult<Option<Vec<DecomposerType>>> {
    let mut qubits = qubits.clone();
    qubits.sort();
    let qubits_tuple = qubits.iter().map(|&q| PhysicalQubit::new(q)).collect();
    let reverse_tuple = qubits
        .iter()
        .rev()
        .map(|&q| PhysicalQubit::new(q))
        .collect();
    // TODO: here return cache --> implementation?

    let mut available_2q_basis: HashMap<&str, NormalOperation> = HashMap::new();
    let mut available_2q_props: HashMap<&str, Option<f64>> = HashMap::new();

    // try both directions for the qubits tuple
    let mut keys = HashSet::new();
    let mut tuples = HashSet::new();
    match target.operation_names_for_qargs(Some(&qubits_tuple)) {
        Ok(direct_keys) => {
            keys = direct_keys;
            tuples.insert(qubits_tuple.clone());
            if let Ok(reverse_keys) = target.operation_names_for_qargs(Some(&reverse_tuple)) {
                keys.extend(reverse_keys);
                tuples.insert(reverse_tuple);
            }
        }
        Err(_) => {
            if let Ok(reverse_keys) = target.operation_names_for_qargs(Some(&reverse_tuple)) {
                keys = reverse_keys;
                tuples.insert(reverse_tuple);
            } else {
                return Err(QiskitError::new_err(
                    "Target has no gates available on qubits to synthesize over.",
                ));
            }
        }
    }

    for (key, q_pair) in keys.iter().zip(tuples.iter()) {
        let op = target.operation_from_name(key).unwrap().clone();
        // TODO:
        // if not isinstance(op, Gate):
        // continue
        available_2q_basis.insert(key, replace_parametrized_gate(op));
        available_2q_props.insert(key, target[key][Some(q_pair)].clone().unwrap().error);
    }

    if available_2q_basis.is_empty() {
        return Err(QiskitError::new_err(
            "Target has no gates available on qubits to synthesize over.",
        ));
    }

    // available decomposition basis on each of the qubits of the pair
    // NOTE: assumes both qubits have the same single-qubit gates
    let mut single_qubit: SmallVec<[PhysicalQubit; 2]> = SmallVec::new();
    single_qubit.push(qubits_tuple[0].clone());
    let available_1q_basis: HashSet<&str> = possible_decomposers(Some(
        target
            .operation_names_for_qargs(Some(&single_qubit))
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
            Some(&error) => {
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
            return Ok(decomposers);
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
                Some(error) => {
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
                        let error = target["cx"][Some(&qubits_tuple)].clone().unwrap().error;
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
    Ok(decomposers)
}

// // synth function for 2q gates (4x4)
// fn synth_su4(su4_mat: Array2<Complex64>, decomposer_2q: &Bound<'_, PyAny>, preferred_direction: Option<Vec<usize>>, approximation_degree: f64) -> DAGCircuit{
//     let py = decomposer_2q.py();
//     let kwargs = HashMap::<&str, bool>::new();
//     kwargs.insert("approximate", approximation_degree != 1.0);
//     kwargs.insert("use_dag", true);
//     let synth_circ = decomposer_2q.call_method(
//         "__call__",
//         su4_mat.to_pyarray_bound(py).into(),
//         Some(&kwargs.into_py_dict_bound(py)),
//     )?.extract::<DAGCircuit>()?;

//     match preferred_direction{
//         None => return synth_circ,
//         Some(preferred_dir) =>{
//             let synth_direction: Option<Vec<usize>> = None;
//             // if the gates in synthesis are in the opposite direction of the preferred direction,
//             // resynthesize a new operator which is the original conjugated by swaps.
//             // this new operator is doubly mirrored from the original and is locally equivalent.
//             for node in synth_circ.topological_op_nodes(){
//                 if let NodeType::Operation(inst) = &synth_circ.dag[node] {
//                     if inst.op.num_qubits() == 2 {
//                         synth_direction = (0..inst.op.num_qubits()).into_iter().map(|q| synth_circ.find_bit(py,q).index)
//                     }
//                 }
//             }
//             match synth_direction {
//                 None => return synth_circ,
//                 Some(synth_dir) => if synth_dir != preferred_dir {return reversed_synth_su4(su4_mat, decomposer_2q, approximation_degree)}
//             }
//         }
//     }
//     synth_circ
// }

#[pymodule]
pub fn unitary_synthesis(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_run_default_main_loop))?;
    Ok(())
}
