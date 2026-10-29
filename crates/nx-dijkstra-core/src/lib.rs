//! nx-dijkstra-core: Rust Dijkstra kernel for networkx-rust.
//!
//! Design (whole-graph crossing, P1): the caller's `G._adj` dict-of-dict
//! crosses once per call. Rust walks it with borrowed C-API references
//! (no per-item incref), builds one CSR, runs one binary-heap loop over
//! flat `f64`/`usize` arrays, and returns fresh `{label: dist}` (+ optional
//! `{label: [preds]}`) dicts. One crossing in, one crossing out.
//!
//! Semantic contract: the same observable behavior as
//! `networkx.algorithms.shortest_paths.weighted._dijkstra_multisource`
//! on the fast path (non-multigraph, non-callable weight key):
//! same settlement order (push counter breaks ties, exactly like
//! upstream's `itertools.count`), same first-predecessor choice, same
//! equal-cost multi-predecessor rule, same negative-weight `ValueError`,
//! same cutoff/target handling. Anything outside the fast path
//! (non-numeric weight value, unknown node) raises and the Python layer
//! falls back to stock networkx, which then produces the error.

use pyo3::ffi as ffi;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::BinaryHeap;

#[derive(Clone, Copy, PartialEq)]
struct Item {
    dist: f64,
    count: u64,
    node: usize,
}
impl Eq for Item {}
impl Ord for Item {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // reversed: BinaryHeap is a max-heap, we want min-dist-first;
        // ties break by push order, exactly like upstream's (dist, count).
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| other.count.cmp(&self.count))
    }
}
impl PartialOrd for Item {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

unsafe fn index_of(
    idxmap: *mut ffi::PyObject,
    p: *mut ffi::PyObject,
) -> Option<usize> {
    let io = ffi::PyDict_GetItemWithError(idxmap, p);
    if io.is_null() || !ffi::PyErr_Occurred().is_null() {
        return None;
    }
    Some(ffi::PyLong_AsLong(io) as usize)
}

/// Dijkstra over `adj` (`G._adj`).
///
/// * `sources`: labels, in caller iteration order (push order preserved).
/// * `weight_key`: edge-attr key object, or None for unweighted (all 1.0).
/// * `want_pred`: 0 = distances only; 1 = first predecessors (paths);
///   2 = all equal-cost predecessors (pred= passed by caller).
/// * `cutoff`/`target`: mirrored semantics.
/// Returns `(dist, pred_or_None)` with labels as keys; `dist` iterates in
/// settlement (increasing-distance) order like upstream.
#[pyfunction]
#[pyo3(signature = (adj, sources, weight_key=None, cutoff=None, target=None, want_pred=0))]
unsafe fn dijkstra(
    py: Python,
    adj: Bound<PyDict>,
    sources: Vec<Bound<PyAny>>,
    weight_key: Option<Bound<PyAny>>,
    cutoff: Option<f64>,
    target: Option<Bound<PyAny>>,
    want_pred: u8,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let adj_ptr = adj.as_ptr();
    let n = ffi::PyDict_Size(adj_ptr) as usize;
    let keys = ffi::PyDict_Keys(adj_ptr); // new ref
    if keys.is_null() {
        return Err(PyErr::fetch(py));
    }
    // label -> row index
    let idxmap = ffi::PyDict_New();
    if idxmap.is_null() {
        ffi::Py_DecRef(keys);
        return Err(PyErr::fetch(py));
    }
    for i in 0..n {
        let k = ffi::PyList_GetItem(keys, i as isize); // borrowed
        if k.is_null() {
            ffi::Py_DecRef(keys);
            ffi::Py_DecRef(idxmap);
            return Err(PyErr::fetch(py));
        }
        let io = ffi::PyLong_FromLong(i as i64);
        if io.is_null() || ffi::PyDict_SetItem(idxmap, k, io) != 0 {
            ffi::Py_DecRef(io);
            ffi::Py_DecRef(keys);
            ffi::Py_DecRef(idxmap);
            return Err(PyErr::fetch(py));
        }
        ffi::Py_DecRef(io);
    }
    let wkey_ptr = weight_key.as_ref().map(|k| k.as_ptr());
    // ---- CSR build: single walk, borrowed refs ----
    let mut offsets: Vec<usize> = Vec::with_capacity(n + 1);
    let mut nbrs: Vec<usize> = Vec::new();
    let mut wgts: Vec<f64> = Vec::new();
    offsets.push(0);
    let mut proto_err: Option<&str> = None;
    for i in 0..n {
        let k = ffi::PyList_GetItem(keys, i as isize);
        let inner = ffi::PyDict_GetItemWithError(adj_ptr, k);
        if inner.is_null() {
            proto_err = Some("adj lookup failed");
            break;
        }
        let mut pos: isize = 0;
        let mut uk: *mut ffi::PyObject = std::ptr::null_mut();
        let mut ed: *mut ffi::PyObject = std::ptr::null_mut();
        while ffi::PyDict_Next(inner, &mut pos, &mut uk, &mut ed) != 0 {
            let Some(j) = index_of(idxmap, uk) else {
                proto_err = Some("neighbor not a node");
                break;
            };
            let w = match wkey_ptr {
                None => 1.0,
                Some(wk) => {
                    let wobj = ffi::PyDict_GetItemWithError(ed, wk);
                    if wobj.is_null() {
                        if ffi::PyErr_Occurred().is_null() {
                            1.0 // missing attr -> 1
                        } else {
                            proto_err = Some("weight lookup failed");
                            break;
                        }
                    } else if wobj == ffi::Py_None() {
                        continue; // hidden edge
                    } else if ffi::PyFloat_Check(wobj) != 0 {
                        ffi::PyFloat_AsDouble(wobj)
                    } else if ffi::PyLong_Check(wobj) != 0 {
                        // bool is a PyLong subclass: True -> 1.0, as upstream
                        ffi::PyLong_AsDouble(wobj)
                    } else {
                        proto_err = Some("non-numeric weight (fallback)");
                        break;
                    }
                }
            };
            if !ffi::PyErr_Occurred().is_null() {
                proto_err = Some("weight conversion failed");
                break;
            }
            nbrs.push(j);
            wgts.push(w);
        }
        if proto_err.is_some() {
            break;
        }
        offsets.push(nbrs.len());
    }
    if let Some(msg) = proto_err {
        ffi::Py_DecRef(keys);
        ffi::Py_DecRef(idxmap);
        if ffi::PyErr_Occurred().is_null() {
            return Err(pyo3::exceptions::PyTypeError::new_err(msg));
        }
        return Err(PyErr::fetch(py));
    }
    // ---- Dijkstra (mirrors _dijkstra_multisource arm for arm) ----
    const INF: f64 = f64::INFINITY;
    let mut dist = vec![INF; n];
    let mut seen = vec![false; n];
    let mut settled = vec![false; n];
    let mut pred: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut pop_order: Vec<usize> = Vec::new();
    let mut heap: BinaryHeap<Item> = BinaryHeap::new();
    let mut count: u64 = 0;
    for s in &sources {
        let Some(si) = index_of(idxmap, s.as_ptr()) else {
            ffi::Py_DecRef(keys);
            ffi::Py_DecRef(idxmap);
            return Err(pyo3::exceptions::PyKeyError::new_err(
                "source not in graph",
            ));
        };
        seen[si] = true;
        dist[si] = 0.0;
        heap.push(Item { dist: 0.0, count, node: si });
        count += 1;
    }
    let target_idx: Option<usize> = match &target {
        // Unresolvable target (missing or unhashable) means unreachable:
        // expand fully and let the wrapper raise NetworkXNoPath via
        // `dist[target]`, exactly like upstream.
        Some(t) => {
            let ti = index_of(idxmap, t.as_ptr());
            ffi::PyErr_Clear();
            ti
        }
        None => None,
    };
    let cut = cutoff.unwrap_or(INF);
    let mut contradiction = false;
    while let Some(Item { dist: dv, count: _, node: v }) = heap.pop() {
        if settled[v] || dv > dist[v] {
            continue;
        }
        settled[v] = true;
        pop_order.push(v);
        if target_idx == Some(v) {
            break;
        }
        for e in offsets[v]..offsets[v + 1] {
            let u = nbrs[e];
            let vu = dv + wgts[e];
            if vu > cut {
                continue;
            }
            if settled[u] {
                // settled arm: negative-weight tripwire + equal-cost
                // multi-pred when the caller tracks all preds.
                if vu < dist[u] {
                    contradiction = true;
                    break;
                } else if want_pred == 2 && vu == dist[u] {
                    pred[u].push(v);
                }
            } else if !seen[u] || vu < dist[u] {
                seen[u] = true;
                dist[u] = vu;
                heap.push(Item { dist: vu, count, node: u });
                count += 1;
                if want_pred >= 1 {
                    pred[u].clear();
                    pred[u].push(v);
                }
            } else if want_pred == 2 && vu == dist[u] {
                // in-fringe equal cost, caller tracks all preds
                pred[u].push(v);
            }
        }
        if contradiction {
            break;
        }
    }
    if contradiction {
        ffi::Py_DecRef(keys);
        ffi::Py_DecRef(idxmap);
        return Err(pyo3::exceptions::PyValueError::new_err((
            "Contradictory paths found:",
            "negative weights?",
        )));
    }
    // ---- result dicts in settlement (pop) order ----
    let out_dist = ffi::PyDict_New();
    if out_dist.is_null() {
        ffi::Py_DecRef(keys);
        ffi::Py_DecRef(idxmap);
        return Err(PyErr::fetch(py));
    }
    let out_pred = if want_pred >= 1 {
        let p = ffi::PyDict_New();
        if p.is_null() {
            ffi::Py_DecRef(out_dist);
            ffi::Py_DecRef(keys);
            ffi::Py_DecRef(idxmap);
            return Err(PyErr::fetch(py));
        }
        Some(p)
    } else {
        None
    };
    for &i in &pop_order {
        let k = ffi::PyList_GetItem(keys, i as isize);
        let val = ffi::PyFloat_FromDouble(dist[i]);
        if val.is_null() || ffi::PyDict_SetItem(out_dist, k, val) != 0 {
            ffi::Py_DecRef(val);
            ffi::Py_DecRef(out_dist);
            if let Some(p) = out_pred {
                ffi::Py_DecRef(p);
            }
            ffi::Py_DecRef(keys);
            ffi::Py_DecRef(idxmap);
            return Err(PyErr::fetch(py));
        }
        ffi::Py_DecRef(val);
        if let Some(p) = out_pred {
            let lst = ffi::PyList_New(pred[i].len() as isize);
            if lst.is_null() {
                ffi::Py_DecRef(out_dist);
                ffi::Py_DecRef(p);
                ffi::Py_DecRef(keys);
                ffi::Py_DecRef(idxmap);
                return Err(PyErr::fetch(py));
            }
            for (j, &pv) in pred[i].iter().enumerate() {
                let pk = ffi::PyList_GetItem(keys, pv as isize); // borrowed
                ffi::Py_IncRef(pk);
                ffi::PyList_SetItem(lst, j as isize, pk); // steals ref
            }
            if ffi::PyDict_SetItem(p, k, lst) != 0 {
                ffi::Py_DecRef(lst);
                ffi::Py_DecRef(out_dist);
                ffi::Py_DecRef(p);
                ffi::Py_DecRef(keys);
                ffi::Py_DecRef(idxmap);
                return Err(PyErr::fetch(py));
            }
            ffi::Py_DecRef(lst);
        }
    }
    ffi::Py_DecRef(keys);
    ffi::Py_DecRef(idxmap);
    let none = py.None();
    Ok((
        Py::from_owned_ptr(py, out_dist),
        match out_pred {
            Some(p) => Py::from_owned_ptr(py, p),
            None => none,
        },
    ))
}

#[pymodule]
fn _rust(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dijkstra, m)?)?;
    Ok(())
}
