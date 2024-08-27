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

use pyo3::prelude::*;
use qiskit_circuit::{
    dag_circuit::{DAGCircuit, NodeType},
    operations::Operation,
};

/// Check if the two-qubit gates follow the right direction with respect to instructions supported in the given target.
///
/// Args:
///     dag: the DAGCircuit to analyze
///
///     target: the Target against which gate directionality compliance is checked
#[pyfunction]
#[pyo3(name = "collect_final_ops")]
fn py_collect_final_ops(dag: &DAGCircuit) -> PyResult<Vec<usize>> {

    // Collect DAG nodes which are followed only by barriers or other measures.
    let final_op_types = vec!["measure".to_string(), "barrier".to_string()];
    let mut final_ops = Vec::new();
    let mut is_final_op = true;

    for (node_id, _weight) in dag.rust_named_nodes(final_op_types.clone()) {
        for (_, child_successors) in dag.bfs_successors(node_id){
        
            for suc in child_successors{
                match dag.dag.node_weight(suc) {
                    Some(NodeType::Operation(packed)) => {
                        if !final_op_types.contains(&packed.op.name().to_string()){
                            is_final_op = false;
                            break;
                        }
                    }
                    _ => continue
                }
            }
        }

        if is_final_op{
            final_ops.push(node_id.index())
        }
    }
    Ok(final_ops)
}

#[pymodule]
pub fn barrier_before_final_measurements(m: &Bound<PyModule>) -> Result<(), PyErr>{
    m.add_wrapped(wrap_pyfunction!(py_collect_final_ops))?;
    Ok(())
}