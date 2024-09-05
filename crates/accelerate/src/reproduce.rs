

let gate_sequence = vec![(StandardGate::CXGate, &[], [0,1]), (StandardGate::CXGate, &[], [1,0])];

// let's build a dag from the gate sequence

let mut target_dag = DAGCircuit::with_capacity(py, 2, 0, None, None, None)?;

