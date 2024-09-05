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
            "Iteration: {:?}. Gate: {:?}, qubit_ids: {:?}",
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
    
    let _ = target_dag.add_from_iter(py, instructions.into_iter(), true);
    Ok(target_dag)
}