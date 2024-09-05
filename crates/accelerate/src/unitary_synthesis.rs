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

use core::num;
// #[cfg(feature = "cache_pygates")]
use std::cell::OnceCell;

use approx::relative_eq;
use qiskit_circuit::bit_data::BitData;
use core::panic;
use hashbrown::{HashMap, HashSet};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::{Clbit, Qubit};
use qiskit_circuit::TupleLikeArg;
use smallvec::{smallvec, SmallVec};
use std::f64::consts::PI;
use std::hash::Hash;
use std::mem;

// use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::{IntoPyDict, PyDict, PyList, PySet, PyString, PyTuple};
use pyo3::wrap_pyfunction;
use pyo3::Python;

use ndarray::prelude::*;
use num_complex::{Complex, Complex64};
use numpy::IntoPyArray;
use numpy::PyArrayMethods;

use rustworkx_core::petgraph::stable_graph::NodeIndex;

use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::dag_circuit::{add_global_phase, DAGCircuit, NodeType};
use qiskit_circuit::imports::{CIRCUIT_TO_DAG, DAG_TO_CIRCUIT};
use qiskit_circuit::operations::{Operation, OperationRef};
use qiskit_circuit::operations::{Param, StandardGate};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperationType};
use qiskit_circuit::slice::{PySequenceIndex, SequenceIndex};

use crate::euler_one_qubit_decomposer::{
    unitary_to_gate_sequence, EulerBasis, OneQubitGateErrorMap, OneQubitGateSequence,
};
use crate::nlayout::PhysicalQubit;
use crate::target_transpiler::{NormalOperation, Target};
use crate::two_qubit_decompose::{
    TwoQubitBasisDecomposer, TwoQubitGateSequence, TwoQubitWeylDecomposition,
};
use qiskit_circuit::imports;
use crate::QiskitError;

const PI2: f64 = PI / 2.;
const PI4: f64 = PI / 4.;

// These are the possible decomposer types for 2q synthesis.
// TODO: change name to TwoQubitDecomposerType
// [E] Are these names too long???
#[derive(Clone, Debug)]
enum DecomposerType {
    TwoQubitBasisDecomposerType(TwoQubitBasisDecomposer),
    XXDecomposerType(PyObject),
}

// These are the possible return types for 1q, 2q and 3+q synthesis.
// The data structures have already been defined in other crates.
// [E] TBD how these can be handled in Python
#[derive(Clone, Debug)]
enum UnitarySynthesisReturnType {
    DAGType(DAGCircuit),
    OneQSequenceType(OneQubitGateSequence),
    TwoQSequenceType(TwoQubitUnitarySequence),
}

#[derive(Clone, Debug)]
pub struct TwoQubitUnitarySequence {
    pub gate_sequence: TwoQubitGateSequence,
    pub decomp_gate: Option<String>,
}

impl TwoQubitUnitarySequence {
    fn new() -> Self {
        TwoQubitUnitarySequence {
            gate_sequence: TwoQubitGateSequence::new(),
            decomp_gate: None,
        }
    }

    fn __getstate__(&self) -> (TwoQubitGateSequence, Option<String>) {
        (self.gate_sequence.clone(), self.decomp_gate.clone())
    }

    fn __setstate__(&mut self, state: (TwoQubitGateSequence, Option<String>)) {
        self.gate_sequence = state.0;
        self.decomp_gate = state.1;
    }

    fn set_state(&mut self, state: (TwoQubitGateSequence, Option<String>)) {
        self.gate_sequence = state.0;
        self.decomp_gate = state.1;
    }

    fn get_decomp_gate(&self) -> Option<StandardGate> {
        println!(
            "decomp gate name {:?}",
            self.decomp_gate.as_deref() == Some("cx")
        );
        match self.decomp_gate.as_deref() {
            Some("ch") => Some(StandardGate::CHGate),       // 21
            Some("cx") => Some(StandardGate::CXGate),       // 22
            Some("cy") => Some(StandardGate::CYGate),       // 23
            Some("cz") => Some(StandardGate::CZGate),       // 24
            Some("dcx") => Some(StandardGate::DCXGate),     // 25
            Some("ecr") => Some(StandardGate::ECRGate),     // 26
            Some("swap") => Some(StandardGate::SwapGate),   // 27
            Some("iswap") => Some(StandardGate::ISwapGate), // 28
            Some("cp") => Some(StandardGate::CPhaseGate),   // 29
            Some("crx") => Some(StandardGate::CRXGate),     // 30
            Some("cry") => Some(StandardGate::CRYGate),     // 31
            Some("crz") => Some(StandardGate::CRZGate),     // 32
            Some("cs") => Some(StandardGate::CSGate),       // 33
            Some("csdg") => Some(StandardGate::CSdgGate),   // 34
            Some("csx") => Some(StandardGate::CSXGate),     // 35
            Some("cu") => Some(StandardGate::CUGate),       // 36
            Some("cu1") => Some(StandardGate::CU1Gate),     // 37
            Some("cu3") => Some(StandardGate::CU3Gate),     // 38
            Some("rxx") => Some(StandardGate::RXXGate),     // 39
            Some("ryy") => Some(StandardGate::RYYGate),     // 40
            Some("rzz") => Some(StandardGate::RZZGate),     // 41
            Some("rzx") => Some(StandardGate::RZXGate),     // 42
            Some("xx_minus_yy") => Some(StandardGate::XXMinusYYGate), // 43
            Some("xx_plus_yy") => Some(StandardGate::XXPlusYYGate), // 44
            _ => None,
        }
    }
}

fn create_qreg<'py>(py: Python<'py>, size: u32) -> PyResult<Bound<'py, PyAny>> {
    imports::QUANTUM_REGISTER.get_bound(py).call1((size,))
}

fn qreg_bit<'py>(py: Python, qreg: &Bound<'py, PyAny>, index: u32) -> PyResult<Bound<'py, PyAny>> {
    qreg.call_method1(intern!(py, "__getitem__"), (index,))
}

fn std_gate(py: Python, gate: StandardGate, params: SmallVec<[Param; 3]>) -> PyResult<Py<PyAny>> {
    gate.create_py_op(py, Some(&params), None)
}

fn parameterized_std_gate(py: Python, gate: StandardGate, param: Param) -> PyResult<Py<PyAny>> {
    gate.create_py_op(py, Some(&[param]), None)
}

fn apply_op_back(py: Python, dag: &mut DAGCircuit, op: &Py<PyAny>, qargs: &[&Bound<PyAny>]) -> PyResult<()> {
    dag.py_apply_operation_back(py,
        op.bind(py).clone(),
        Some( TupleLikeArg::extract_bound( &PyTuple::new_bound(py, qargs))? ),
        None,
        false)?;

    Ok(())
}


// This function converts the 2q synthesis output from the TwoQubitBasisDecomposer (sequence of gates)
// into a DAGCircuit for easier manipulation. Should we try to get rid of it for performance reasons? TBD
// Used in `synth_su4` and `reversed_synth_su4`.
fn dag_from_2q_gate_sequence(
    py: Python<'_>,
    sequence: TwoQubitUnitarySequence,
) -> PyResult<DAGCircuit> {
    // For reference:
    // pub struct TwoQubitGateSequence {
    //     gates: TwoQubitSequenceVec,
    //     #[pyo3(get)]
    //     global_phase: f64,
    // }
    // type TwoQubitSequenceVec = Vec<(Option<StandardGate>, SmallVec<[f64; 3]>, SmallVec<[u8; 2]>)>;
    let gate_vec = &sequence.gate_sequence.gates;

    // let target_dag = &mut DAGCircuit::with_capacity(py, 2, 1, None, None, None)?;
    let target_dag = &mut DAGCircuit::new(py)?;
    let _ = target_dag.set_global_phase(Param::Float(sequence.gate_sequence.global_phase));
    
    let qreg = create_qreg(py, 2)?;
    target_dag.add_qreg(py, &qreg)?;

    let (q0, q1) = (qreg_bit(py, &qreg, 0)?, qreg_bit(py, &qreg, 0)?);

    for (gate, params, qubit_ids) in gate_vec{
        let gate_node = match gate {
            None => sequence.get_decomp_gate().clone().unwrap(), // this is initialized to None but should always have a value in this case
            Some(gate) => *gate,
        };

        let new_params: SmallVec<[Param; 3]> = params.iter().map(|p| Param::Float(*p)).collect();

        let mut qubits = Vec::new();
        for q in qubit_ids{
            match q{
                0 => qubits.push(&q0),
                1 => qubits.push(&q1),
                _ => (),
            }
        }
        apply_op_back(py, target_dag, &std_gate(py, gate_node, new_params)?, &qubits);
    }

    Ok(target_dag.clone())
}

// This is the cost function for choosing the best 2q synthesis output.
// Used in `run_default_unitary_synthesis`.
fn compute_2q_error(
    py: Python<'_>,
    synth_circuit: &PyResult<UnitarySynthesisReturnType>,
    target: &Option<Target>,
    qubits: &SmallVec<[PhysicalQubit; 2]>,
) -> f64 {
    match target {
        None => {
            match synth_circuit {
                Ok(UnitarySynthesisReturnType::DAGType(synth_dag)) => {
                    return synth_dag.op_nodes(false).count() as f64
                }
                Ok(UnitarySynthesisReturnType::TwoQSequenceType(synth_sequence)) => {
                    return synth_sequence.gate_sequence.gates.len() as f64
                }
                _ => unreachable!(), // we only compute the error for 2q synthesis
            }
        }
        Some(target) => {
            let mut gate_fidelities = Vec::new();

            let mut score_instruction = |instruction: &PackedInstruction,
                                         inst_qubits: SmallVec<[PhysicalQubit; 2]>|
             -> PyResult<()> {
                match target.operation_names_for_qargs(Some(&inst_qubits)) {
                    Ok(names) => {
                        for name in names {
                            let target_op = target.operation_from_name(name).unwrap();
                            let op = instruction;
                            // is this a good replacement for is_instance?
                            // is this a good replacement for is parameterized
                            // missing one condition: all(isclose(float(p1), float(p2)) for p1, p2 in zip(target_op.params, op.params))
                            if target_op.operation.name() == op.op.name()
                                && (!target_op.params.is_empty())
                            {
                                // this is inst_props
                                match target[name].get(Some(&inst_qubits)) {
                                    None => gate_fidelities.push(1.0),
                                    Some(props) => gate_fidelities
                                        .push(1.0 - props.clone().unwrap().error.unwrap()),
                                }
                                break;
                            }
                        }
                        return Ok(());
                    }
                    Err(_) => {
                        // how does this formatting look like? These things don't implement Display.
                        return Err(QiskitError::new_err(
                            format!("Encountered a bad synthesis. Target has no {instruction:?} on qubits {inst_qubits:?}.")
                        ));
                    }
                }
            };

            match synth_circuit {
                Ok(UnitarySynthesisReturnType::DAGType(synth_dag)) => {
                    for node in synth_dag
                        .topological_op_nodes()
                        .expect("cannot return error here (don't know how to handle it later)")
                    {
                        if let NodeType::Operation(inst) = &synth_dag.dag[node] {
                            let inst_qubits: SmallVec<[PhysicalQubit; 2]> = synth_dag
                                .qargs_interner
                                .get(inst.qubits)
                                .iter()
                                .map(|&q| qubits[q.0 as usize])
                                .collect();
                            let _ = score_instruction(inst, inst_qubits);
                        }
                    }
                }
                Ok(UnitarySynthesisReturnType::TwoQSequenceType(synth_sequence)) => {
                    for (gate, params, qubit_ids) in &synth_sequence.gate_sequence.gates {
                        let inst_qubits: SmallVec<[PhysicalQubit; 2]> =
                            qubit_ids.iter().map(|&q| qubits[q as usize]).collect();
                        let packed_qubits = inst_qubits.iter().map(|q| Qubit(q.0)).collect();
                        let mut dummy_dag = DAGCircuit::new(py).expect("dag");
                        let inst = PackedInstruction {
                            op: PackedOperation::from_standard(gate.clone().unwrap()),
                            qubits: dummy_dag.qargs_interner.insert_owned(packed_qubits),
                            clbits: dummy_dag.cargs_interner.get_default(),
                            params: Some(Box::new(smallvec![
                                Param::Float(params.clone()[0]),
                                Param::Float(params.clone()[1]),
                                Param::Float(params.clone()[2])
                            ])),
                            extra_attrs: None,
                            // #[cfg(feature = "cache_pygates")]
                            py_op: OnceCell::new(),
                        };
                        let _ = score_instruction(&inst, inst_qubits);
                    }
                }
                _ => unreachable!(), // we only compute the error for 2q synthesis
            }
            return 1.0 - gate_fidelities.into_iter().sum::<f64>();
        }
    }
}

// This is the outer-most run function. It is meant to be called from Python inside `UnitarySynthesis.run()`
// This loop iterates over the dag and calls `run_default_unitary_synthesis` (defined below).
#[pyfunction]
#[pyo3(name = "run_default_main_loop")]
fn py_run_default_main_loop(
    py: Python,
    dag: &mut DAGCircuit,
    // originally, qubit indices = {bit: i for i, bit in enumerate(dag.qubits)}
    // for example: {Qubit(QuantumRegister(2, 'q1'), 0): 0, Qubit(QuantumRegister(2, 'q1'), 1): 1}
    qubit_indices: &Bound<'_, PyList>,
    min_qubits: usize,
    approximation_degree: Option<f64>,
    basis_gates: Option<HashSet<PyBackedStr>>,
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
                    let raw_blocks = raw_blocks.bind(py).iter()?;
                    for raw_block in raw_blocks {
                        let block_obj = raw_block?;
                        // let block = block_obj
                        //     .getattr(intern!(py, "_data"))?
                        //     .extract::<CircuitData>()?;
                        let mut new_dag: DAGCircuit =
                            circuit_to_dag.call1((block_obj.clone(),))?.extract()?;

                        // let block_qargs = (0..block.num_qubits()).into_iter().map(|x| x as usize);
                        // let node_qargs = (0..inst.op.num_qubits()).into_iter().map(|x| x as usize);

                        // // remap qubit indices
                        // let new_ids = node_qargs.map(|outer| qubit_indices.get_item(outer))

                        // let new_qubit_indices = PyList::new_bound(py,);
                        // for (inner, outer) in block_qargs.zip(node_qargs) {
                        //     new_qubit_indices.set_item(inner, qubit_indices.get_item(outer)?)?;
                        // }

                        let res = py_run_default_main_loop(
                            py,
                            &mut new_dag,
                            qubit_indices,
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
                    let other_node = dag.get_node(py, node)?.clone();
                    let _ = dag.substitute_node(other_node.bind(py), &new_op, true, false);
                }
            }
        }
    }

    // Create new empty dag
    let mut out_dag = dag.copy_empty_like(py, "alike")?;

    // Iterate over nodes, find decomposers and run synthesis
    // [E] implement as an iterator once it works (if it ever works)
    for node in dag.topological_op_nodes()? {
        if let NodeType::Operation(packed_instr) = &dag.dag[node] {
            let n_qubits = packed_instr.op.num_qubits().try_into()?;
            match n_qubits {
                None => {
                    return Err(QiskitError::new_err("The instruction has no qubits"));
                }
                Some(n_qubits) => {
                    if packed_instr.op.name() == "unitary" && n_qubits >= min_qubits as u32 {
                        let unitary: Option<Array<Complex<f64>, Dim<[usize; 2]>>> =
                            packed_instr.op.matrix(&[]);

                        match unitary {
                            None => {
                                return Err(QiskitError::new_err("Unitary not found"));
                            }
                            Some(unitary) => {
                                // SmallVec of physical qubits of len 2. We will map the instruction qubits to the provided qubit_indices
                                // in the original code, this info is provided through the "coupling_map" option in an obscure way
                                let qubits: SmallVec<[PhysicalQubit; 2]> =
                                    if let NodeType::Operation(inst) = &dag.dag[node] {
                                        let mut p_qubits = SmallVec::new();
                                        for q in dag.get_qargs(inst.qubits) {
                                            let mapped_id = qubit_indices
                                                .get_item(q.0 as usize)?
                                                .extract::<u32>()?;
                                            p_qubits.push(PhysicalQubit::new(mapped_id));
                                        }
                                        p_qubits
                                    } else {
                                        unreachable!("Is this unreachable?")
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
                                        let _ = out_dag.push_back(py, packed_instr.clone());
                                    }

                                    Some(synth_output) => {
                                        match synth_output {
                                            UnitarySynthesisReturnType::DAGType(synth_dag) => {
                                                for synth_node in
                                                    synth_dag.topological_op_nodes()?
                                                {
                                                    let _ =
                                                        out_dag.set_global_phase(add_global_phase(
                                                            py,
                                                            &out_dag.get_global_phase(),
                                                            &synth_dag.get_global_phase(),
                                                        )?);

                                                    if let NodeType::Operation(synth_packed_instr) =
                                                        &synth_dag.dag[synth_node]
                                                    {
                                                        let qargs = dag
                                                            .qargs_interner
                                                            .get(packed_instr.qubits);
                                                        synth_packed_instr.to_owned().qubits =
                                                            out_dag.qargs_interner.insert_owned(
                                                                out_dag
                                                                    .qubits
                                                                    .map_bits(
                                                                        synth_dag
                                                                            .qubits
                                                                            .clone()
                                                                            .map_indices(qargs)
                                                                            .map(|b| {
                                                                                b.bind(py).clone()
                                                                            }),
                                                                    )?
                                                                    .collect(),
                                                            );

                                                        let _ = out_dag.push_back(
                                                            py,
                                                            synth_packed_instr.clone(),
                                                        );
                                                    }
                                                }
                                            }
                                            UnitarySynthesisReturnType::TwoQSequenceType(
                                                synth_sequence,
                                            ) => {
                                                // let (node_list, global_phase, gate) = raw_synth_dag.extract::<((&str, Param, Vec<usize>), Param, &str)>(py)?;
                                                // HERE
                                                println!("synth sequence {:?}", synth_sequence);
                                                out_dag =
                                                    dag_from_2q_gate_sequence(py, synth_sequence)?;
                                            }
                                            UnitarySynthesisReturnType::OneQSequenceType(
                                                synth_sequence,
                                            ) => {
                                                // let (node_list, global_phase, gate) = raw_synth_dag.extract::<((&str, Param, Vec<usize>), Param, &str)>(py)?;
                                                // HERE
                                                println!("The return type is 1q sequence and I haven't implemented it yet");
                                                todo!()
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        let _ = out_dag.push_back(py, packed_instr.clone());
                    }
                }
            }
        }
    }
    Ok(out_dag)
}

// Tis function is used in `run_default_unitary_synthesis` for the 1q synthesis path.
// The OneQubitGateErrorMap is necessary for calling `euler_one_qubit_decomposer::unitary_to_gate_sequence`.
// [E] Should we move this to euler_one_qubit_decomposer?
fn build_1q_error_map(target: &Option<Target>) -> Option<OneQubitGateErrorMap> {
    match target {
        Some(target) => {
            let mut e_map: OneQubitGateErrorMap = OneQubitGateErrorMap::new(target.num_qubits);
            match target.num_qubits {
                None => None,
                Some(n_qubits) => {
                    for qubit in 0..n_qubits {
                        let mut gate_error = HashMap::new();
                        for gate in target.operation_names() {
                            if let Some(inst_props) = target[gate]
                                [Some(&smallvec![PhysicalQubit::new(qubit as u32)])]
                            .clone()
                            {
                                match inst_props.error {
                                    Some(e) => {
                                        gate_error.insert(gate.to_string(), e);
                                    }
                                    None => continue,
                                }
                            }
                            // Reason for clone: move occurs because `gate_error` has type
                            // `hashbrown::HashMap<String, f64>`, which does not implement the `Copy` trait
                            e_map.add_qubit(gate_error.clone())
                        }
                    }
                    Some(e_map)
                }
            }
        }
        None => None,
    }
}

// This function is meant to be used instead of `DefaultUnitarySynthesisPlugin.run()`. If handles the main synthesis logic:
// * 1q synthesis, we run `euler_one_qubit_decomposer::unitary_to_gate_sequence` -> return OneQSequenceType
// * for 2q synthesis, we first need to collect all potential decomposer instances (see DecomposerType) and then run
//      them to select the best circuit -> return DAGType or TwoQSequenceType
// * for qsd, we call python because the decomposer isn't implemented in rust -> return DAGType
// [E] Double-check that the Option return type is correct.
fn run_default_unitary_synthesis(
    py: Python,
    unitary: Array2<Complex64>,
    qubits: SmallVec<[PhysicalQubit; 2]>,
    approximation_degree: Option<f64>,
    basis_gates: Option<HashSet<PyBackedStr>>,
    coupling_map: Option<PyObject>,
    natural_direction: Option<bool>,
    pulse_optimize: Option<&str>,
    gate_lengths: Option<Bound<'_, PyDict>>,
    gate_errors: Option<Bound<'_, PyDict>>,
    target: Option<Target>,
) -> PyResult<Option<UnitarySynthesisReturnType>> {
    match unitary.shape() {
        [2, 2] => {
            // run 1q decomposition -> return OneQubitGateSequence
            let error_map = build_1q_error_map(&target);
            match unitary_to_gate_sequence(
                unitary.into_pyarray_bound(py).readonly(),
                basis_gates.unwrap().into_iter().collect(),
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
            // run 2q decomposition (in Rust except for XXDecomposer) -> Return types will vary.
            // step1: select decomposers
            let decomposers = match &target {
                Some(target_ref) => {
                    let decomposers_2q = get_2q_decomposers_from_target(
                        py,
                        target_ref,
                        &qubits,
                        approximation_degree,
                    )?;
                    match decomposers_2q {
                        Some(decomp) => decomp,
                        None => Vec::new(),
                    }
                }
                None => todo!(), // HERE decomposer_2q_from_basis_gates -> this one uses pulse_optimize
            };
            // println!("These are the found decomposers: {:?}", decomposers);

            // If we have a single TwoQubitBasisDecomposer, skip dag creation as we don't need to
            // store and can instead manually create the synthesized gates directly in the output dag
            if decomposers.len() == 1 {
                println!("Special case");

                let decomposer_item = decomposers.iter().next().unwrap();
                match &decomposer_item {
                    &DecomposerType::TwoQubitBasisDecomposerType(decomposer) => {
                        println!("Before preferred dir");
                        println!("Gate lengths {:?}", gate_lengths);
                        println!("Gate errors {:?}", gate_lengths);
                        let preferred_dir = preferred_direction(
                            py,
                            decomposer_item,
                            &qubits,
                            natural_direction,
                            coupling_map,
                            &gate_lengths,
                            &gate_errors,
                        )?;
                        println!("After preferred dir");

                        let synth = synth_su4_no_dag(
                            py,
                            &unitary,
                            decomposer,
                            preferred_dir,
                            approximation_degree,
                        )?;
                        return Ok(Some(synth));
                    }
                    _ => (),
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
                    &gate_lengths,
                    &gate_errors,
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
            let synth_circuit = synth_circuits
                .into_iter()
                .min_by(|circ1, circ2| {
                    compute_2q_error(py, circ1, &target, &qubits)
                        .partial_cmp(&compute_2q_error(py, circ2, &target, &qubits))
                        .unwrap()
                })
                .unwrap()?;
            Ok(Some(synth_circuit))
        }
        _ => {
            todo!()
            // HERE
            // run qsd -> in python
        }
    }
    // The output will either be a dag circuit or a list of synthesized gates or None (need to use Option)
}

// [E] This is a tempoerary workaround to deal with some parts of `get_2q_decomposers_from_target`
// that rely on hashing a float :'). It probably makes the most sense to just use the gate names
// as keys (what are the chances to get 2 values for the same float key?), but for the moment I
// just naively ported the logic.
#[derive(Hash, Eq, PartialEq)]
struct InteractionStrength((u64, i16, i8));

// Utilities for `InteractionStrength`
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
// Utilities for `InteractionStrength`
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

impl InteractionStrength {
    fn new(val: f64) -> InteractionStrength {
        InteractionStrength(integer_decode_f64(val))
    }
    fn to_f64(&self) -> f64 {
        let (mantissa, exponent, sign) = self.0;
        integer_encode_f64(mantissa, exponent, sign)
    }
}

// Utility for `decomposer_2q_from_target`
// [E] we could move this definition to `decomposer_2q_from_target`, but it's
// already pretty long.
fn replace_parametrized_gate(mut op: NormalOperation) -> NormalOperation {
    if let Some(std_gate) = op.operation.try_standard_gate() {
        match std_gate.name() {
            "rxx" => {
                if let Param::ParameterExpression(_) = op.params[0] {
                    op.params[0] = Param::Float(PI2)
                }
            }
            "rzx" => {
                if let Param::ParameterExpression(_) = op.params[0] {
                    op.params[0] = Param::Float(PI4)
                }
            }
            "rzz" => {
                if let Param::ParameterExpression(_) = op.params[0] {
                    op.params[0] = Param::Float(PI2)
                }
            }
            _ => (),
        }
        print!("After replacing params {:?}", op.params)
        // match (std_gate.name(), &op.params[0]) {
        //     ("rxx", Param::ParameterExpression(_)) => op.params[0] = Param::Float(PI2),
        //     ("rzx", Param::ParameterExpression(_)) => op.params[0] = Param::Float(PI4),
        //     ("rzz", Param::ParameterExpression(_)) => op.params[0] = Param::Float(PI2),
        //     _ => (),
        // }
    }
    op
}

// [E] to be replaced with EulerBasis
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

// This function collects a bunch of decomposer instances that will be used in `run_default_unitary_synthesis`
fn get_2q_decomposers_from_target(
    py: Python,
    target: &Target,
    qubits: &SmallVec<[PhysicalQubit; 2]>,
    approximation_degree: Option<f64>,
) -> PyResult<Option<Vec<DecomposerType>>> {
    let mut qubits: SmallVec<[PhysicalQubit; 2]> = qubits.clone();
    qubits.sort();
    let reverse_qubits: SmallVec<[PhysicalQubit; 2]> = qubits.iter().rev().map(|x| *x).collect();
    // HERE: caching
    // TODO: here return cache --> implementation?

    let mut available_2q_basis: HashMap<&str, NormalOperation> = HashMap::new();
    let mut available_2q_props: HashMap<&str, (Option<f64>, Option<f64>)> = HashMap::new();

    // try both directions for the qubits tuple
    let mut keys = HashSet::new();
    let mut tuples = HashSet::new();
    println!("These are the qubits {:?}", qubits);
    println!("These are the reverse qubits {:?}", reverse_qubits);
    println!(
        "This is the output from operation_names_for_qargs {:?}",
        target.operation_names_for_qargs(Some(&qubits))
    );
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
    println!(
        "These are the key and tuples at the end {:?}, {:?}",
        keys, tuples
    );

    for (key, q_pair) in keys.iter().zip(tuples.iter()) {
        let op = target.operation_from_name(key).unwrap().clone();
        println!(
            "This is the discriminant {:?}, {:?}",
            op,
            op.operation.discriminant()
        );

        // if it's not a gate, move on to next iteration
        match op.operation.discriminant() {
            PackedOperationType::Gate => (),
            PackedOperationType::StandardGate => (),
            _ => continue,
        }

        available_2q_basis.insert(key, replace_parametrized_gate(op));
        available_2q_props.insert(
            key,
            (
                target[key][Some(q_pair)].clone().unwrap().duration.clone(),
                target[key][Some(q_pair)].clone().unwrap().error.clone(),
            ),
        );
    }
    println!(
        "This is the available 2q basis: {:?}, {:?}",
        available_2q_basis, available_2q_props
    );

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

    println!("This is the available 1q basis: {:?}", available_1q_basis);

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

    println!(
        "This is the supercontrolled basis: {:?}",
        supercontrolled_basis
    );

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

        println!("1");

        if approximation_degree.is_some() {
            basis_2q_fidelity = basis_2q_fidelity * approximation_degree.unwrap();
        }
        println!("2");

        let gate = &supercontrolled_basis[basis_2q];
        println!("3");

        let decomposer = TwoQubitBasisDecomposer::new_inner(
            gate.operation.name().to_owned(),
            gate.operation.matrix(&[]).unwrap().view(),
            basis_2q_fidelity,
            basis_1q,
            None,
        )?;
        println!("4");

        decomposers.push(DecomposerType::TwoQubitBasisDecomposerType(decomposer));

        // If our 2q basis gates are a subset of cx, ecr, or cz then we know TwoQubitBasisDecomposer
        // is an ideal decomposition and there is no need to bother calculating the XX embodiments
        // or try the XX decomposer
        let mut goodbye_set = HashSet::new();
        goodbye_set.insert("cx");
        goodbye_set.insert("cz");
        goodbye_set.insert("ecr");
        println!("5");

        let available_basis_set =
            HashSet::from_iter(available_2q_basis.clone().keys().into_iter().map(|&k| k));

        println!("6");
        println!("{:?}", goodbye_set.is_superset(&available_basis_set));

        if goodbye_set.is_superset(&available_basis_set) {
            // TODO: decomposer cache thingy
            return Ok(Some(decomposers));
        }

        println!("7");

        // Let's now look for possible controlled decomposers (i.e. XXDecomposer)
        let controlled_basis: HashMap<&str, NormalOperation> = available_2q_basis
            .clone()
            .into_iter()
            .filter(|(_, v)| is_controlled(v))
            .collect();

        println!("8");

        let mut basis_2q_fidelity: HashMap<InteractionStrength, f64> = HashMap::new();

        // the embodiments will be a list of circuit representations
        let mut embodiments: HashMap<InteractionStrength, CircuitData> = HashMap::new();
        let mut pi2_basis: Option<&str> = None;
        println!("9");

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

// This is used in synth_su4 & company. I changed the logic with respect to the python reference code to return a bool
// instead of a tuple of qubits. The equivalence is the following:
// true = [0,1]
// false = [1,0]
// gate_lengths comes from Python and is a dict of {qubits_tuple: (gate_name, duration)}
// gate_errors comes from Python and is a dict of {qubits_tuple: (gate_name, error)}
fn preferred_direction(
    py: Python,
    decomposer: &DecomposerType,
    qubits: &SmallVec<[PhysicalQubit; 2]>,
    natural_direction: Option<bool>,
    coupling_map: Option<PyObject>,
    gate_lengths: &Option<Bound<'_, PyDict>>,
    gate_errors: &Option<Bound<'_, PyDict>>,
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
        DecomposerType::TwoQubitBasisDecomposerType(decomp) => decomp.gate.clone(),
        // Reason for clone: creates a temporary value which is freed while still in use
        DecomposerType::XXDecomposerType(decomp) => {
            decomp.getattr(py, "gate")?.extract::<String>(py)?
        }
    };

    // In python, gate_dict has the form: {(qubits,): {gate_name: duration}} (wrongly documented in the docstring. Fix!)
    let compute_cost = |gate_dict: &Option<Bound<'_, PyDict>>,
                        q_tuple: &SmallVec<[PhysicalQubit; 2]>,
                        in_cost: Option<f64>|
     -> PyResult<Option<f64>> {
        let ids = vec![q_tuple[0].0, q_tuple[1].0];

        match gate_dict {
            Some(gate_dict) => {
                let gate_value_dict = gate_dict.get_item(&decomposer2q_gate)?;
                println!("HERE with gate value dict {:?}", gate_value_dict);

                let cost = match gate_value_dict {
                    Some(val_dict) => val_dict
                        .extract::<&PyDict>()?
                        .iter()
                        .map(|(qargs, value)| {
                            (
                                qargs
                                    .extract::<Vec<u32>>()
                                    .expect("Cannot use ? inside this closure. Find solution"),
                                value
                                    .extract::<f64>()
                                    .expect("Cannot use ? inside this closure. Find solution"),
                            )
                        })
                        .find(|(qargs, _)| *qargs == ids)
                        .map(|(_, value)| value)
                        .clone(),
                    None => in_cost,
                };
                println!("HERE AFTER. {:?}", cost);

                return Ok(cost);
            }
            None => return Ok(in_cost),
        }
    };

    if preferred_direction.is_none() && (gate_lengths.is_some() || gate_errors.is_some()) {
        let mut cost_0_1: Option<f64> = None;
        let mut cost_1_0: Option<f64> = None;

        // Try to find the cost in gate_lengths
        cost_0_1 = compute_cost(gate_lengths, qubits, cost_0_1)?;
        cost_1_0 = compute_cost(gate_lengths, &reverse_qubits, cost_1_0)?;

        // If no valid cost was found in gate_lengths, check gate_errors
        if cost_0_1.is_none() && cost_1_0.is_none() {
            cost_0_1 = compute_cost(gate_errors, qubits, cost_0_1)?;
            cost_1_0 = compute_cost(gate_errors, &reverse_qubits, cost_1_0)?;
        }

        if cost_0_1.is_some() && cost_1_0.is_some() {
            if cost_0_1 < cost_1_0 {
                preferred_direction = Some(true)
            } else if cost_1_0 < cost_0_1 {
                preferred_direction = Some(false)
            }
        }
        println!(
            "Preferred dir {:?},{:?},{:?},",
            preferred_direction, cost_0_1, cost_1_0
        );
    }
    Ok(preferred_direction)
}

// generic synth function for 2q gates (4x4)
// used in `run_default_unitary_synthesis`
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
            let decomp_gate = decomposer.gate.clone();
            let mut sequence = TwoQubitUnitarySequence::new();
            sequence.set_state((synth.clone(), Some(decomp_gate)));
            dag_from_2q_gate_sequence(py, sequence)?
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

// special-case synth function for the TwoQubitBasisDecomposer
// used in `run_default_unitary_synthesis`
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
    let decomp_gate = decomposer_2q.gate.clone();
    let mut sequence = TwoQubitUnitarySequence::new();
    sequence.set_state((synth.clone(), Some(decomp_gate)));

    match preferred_direction {
        None => return Ok(UnitarySynthesisReturnType::TwoQSequenceType(sequence)),
        Some(preferred_dir) => {
            let mut synth_direction: Option<SmallVec<[u8; 2]>> = None;
            for (gate, _, qubits) in synth.gates.clone() {
                if gate.is_none() || gate.unwrap().name() == "cx" {
                    synth_direction = Some(qubits);
                }
            }

            match synth_direction {
                None => return Ok(UnitarySynthesisReturnType::TwoQSequenceType(sequence)),
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
                            &DecomposerType::TwoQubitBasisDecomposerType(decomposer_2q.clone()),
                            approximation_degree,
                        );
                    } else {
                        return Ok(UnitarySynthesisReturnType::TwoQSequenceType(sequence));
                    }
                }
            }
        }
    }
}

// generic synth function for 2q gates (4x4) called from synth_su4
fn reversed_synth_su4(
    py: Python<'_>,
    su4_mat: &Array2<Complex64>,
    decomposer_2q: &DecomposerType,
    approximation_degree: Option<f64>,
) -> PyResult<UnitarySynthesisReturnType> {
    let is_approximate = approximation_degree.is_none() || approximation_degree.unwrap() != 1.0;
    let mut su4_mat_mm = su4_mat.clone();
    // we need the temporary variables to avoid borrowing `su4_mat_mm` as mutable more than once at a time
    let temp_1 = su4_mat_mm.slice_mut(s![.., 1]).to_owned();
    let temp_2 = su4_mat_mm.slice_mut(s![.., 2]).to_owned();
    su4_mat_mm.slice_mut(s![1, ..]).assign(&temp_1);
    su4_mat_mm.slice_mut(s![2, ..]).assign(&temp_2);
    su4_mat_mm[[1, 2]] = su4_mat_mm[[2, 1]];

    let synth_dag = match decomposer_2q {
        DecomposerType::XXDecomposerType(decomposer) => {
            let mut kwargs = HashMap::<&str, bool>::new();
            kwargs.insert("approximate", is_approximate);
            kwargs.insert("use_dag", true);
            decomposer
                .call_method_bound(
                    py,
                    "__call__",
                    (su4_mat_mm.clone().into_pyarray_bound(py),),
                    Some(&kwargs.into_py_dict_bound(py)),
                )?
                .extract::<DAGCircuit>(py)?
        }
        DecomposerType::TwoQubitBasisDecomposerType(decomposer) => {
            // we don't have access to basis_fidelity, right???
            let synth = decomposer.call_inner(su4_mat_mm.view(), None, is_approximate, None)?;
            let decomp_gate = decomposer.gate.clone();
            let mut sequence = TwoQubitUnitarySequence::new();
            sequence.set_state((synth.clone(), Some(decomp_gate)));
            dag_from_2q_gate_sequence(py, sequence)?
        }
    };

    let mut target_dag = DAGCircuit::with_capacity(py, 2, 0, None, None, None)?;
    let _ = target_dag.set_global_phase(synth_dag.get_global_phase());

    for node in synth_dag.topological_op_nodes()? {
        if let NodeType::Operation(inst) = &synth_dag.dag[node] {
            let qubits = target_dag
                .qargs_interner
                .get(inst.qubits)
                .iter()
                .rev()
                .map(|&x| x)
                .collect();
            inst.to_owned().qubits = target_dag.qargs_interner.insert_owned(qubits);
            let _ = target_dag.push_back(py, inst.clone());
        }
    }
    Ok(UnitarySynthesisReturnType::DAGType(target_dag))
}

#[pymodule]
pub fn unitary_synthesis(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_run_default_main_loop))?;
    Ok(())
}
