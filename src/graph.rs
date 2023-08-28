use std::ops::Deref;
use std::os::raw::c_void;

use halo2curves::{bn256::Fr, ff::Field};
use serde::{Deserialize, Serialize};

use crate::value_source::{Calculation, CalculationInfo, ValueSource};

use crate::stub::*;

#[derive(Debug)]
pub struct WrappedGraph {
    inner: *const c_void,
}

unsafe impl Sync for WrappedGraph {}

impl Default for WrappedGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl WrappedGraph {
    pub fn new() -> Self {
        unsafe {
            Self {
                inner: create_graph(),
            }
        }
    }

    pub fn reset(&mut self, num: usize) {
        unsafe {
            reset_graph(self.inner, num);
        }
    }

    pub fn eval<P: Deref<Target = [Fr]>>(
        &self,
        values: &mut [Fr],
        rotations: &[i32],
        constants: &[Fr],
        fixed: Option<&[P]>,
        advice: Option<&[P]>,
        instance: Option<&[P]>,
        challenges: &[Fr],
        beta: &Fr,
        gamma: &Fr,
        theta: &Fr,
        y: &Fr,
        rot_scale: i32,
        isize: i32,
    ) {
        unsafe {
            let f = |v: Option<&[P]>| -> (Vec<*const Fr>, usize, usize) {
                if let Some(fixed) = v {
                    let ptr = fixed.iter().map(|t| t.as_ptr()).collect::<Vec<_>>();
                    let col = fixed.len();
                    let row = fixed.get(0).and_then(|p| Some(p.len())).unwrap_or_default();
                    (ptr, col, row)
                } else {
                    (vec![], 0, 0)
                }
            };

            let (fixed, fixed_col, fixed_row) = f(fixed);
            let (advice, advice_col, advice_row) = f(advice);
            let (instance, instance_col, instance_row) = f(instance);
            evaluate_batch(
                values.as_mut_ptr() as *mut _,
                values.len(),
                self.inner,
                rotations.as_ptr(),
                rotations.len(),
                constants.as_ptr() as *const _,
                constants.len(),
                fixed.as_ptr() as *const _,
                fixed_col,
                fixed_row,
                advice.as_ptr() as *const _,
                advice_col,
                advice_row,
                instance.as_ptr() as *const _,
                instance_col,
                instance_row,
                challenges.as_ptr() as *const _,
                challenges.len(),
                beta as *const Fr as *const _,
                gamma as *const Fr as *const _,
                theta as *const Fr as *const _,
                y as *const Fr as *const _,
                rot_scale,
                isize,
            );
        }
    }

    pub fn push(&mut self, cal: &CalculationInfo) {
        let target = cal.target;
        match cal.calculation.clone() {
            Calculation::Add(l, r) => {
                let vs: Vec<ValueSource> = vec![l, r];
                unsafe {
                    push_node(
                        self.inner,
                        target,
                        CalculationTag::Add,
                        vs.as_ptr() as *const _,
                        vs.len(),
                    );
                }
            }
            Calculation::Sub(l, r) => {
                let vs: Vec<ValueSource> = vec![l, r];
                unsafe {
                    push_node(
                        self.inner,
                        target,
                        CalculationTag::Sub,
                        vs.as_ptr() as *const _,
                        vs.len(),
                    );
                }
            }
            Calculation::Mul(l, r) => {
                let vs: Vec<ValueSource> = vec![l, r];
                unsafe {
                    push_node(
                        self.inner,
                        target,
                        CalculationTag::Mul,
                        vs.as_ptr() as *const _,
                        vs.len(),
                    );
                }
            }
            Calculation::Double(l) => {
                let vs: Vec<ValueSource> = vec![l];
                unsafe {
                    push_node(
                        self.inner,
                        target,
                        CalculationTag::Double,
                        vs.as_ptr() as *const _,
                        vs.len(),
                    );
                }
            }
            Calculation::Square(l) => {
                let vs: Vec<ValueSource> = vec![l];
                unsafe {
                    push_node(
                        self.inner,
                        target,
                        CalculationTag::Square,
                        vs.as_ptr() as *const _,
                        vs.len(),
                    );
                }
            }
            Calculation::Negate(l) => {
                let vs: Vec<ValueSource> = vec![l];
                unsafe {
                    push_node(
                        self.inner,
                        target,
                        CalculationTag::Negate,
                        vs.as_ptr() as *const _,
                        vs.len(),
                    );
                }
            }
            Calculation::Store(l) => unsafe {
                let vs: Vec<ValueSource> = vec![l];
                push_node(
                    self.inner,
                    target,
                    CalculationTag::Store,
                    vs.as_ptr() as *const _,
                    vs.len(),
                );
            },
            Calculation::Horner(l, v, r) => {
                let mut vs: Vec<ValueSource> = vec![l];
                for &i in v.iter() {
                    vs.push(i)
                }
                vs.push(r);
                unsafe {
                    push_node(
                        self.inner,
                        target,
                        CalculationTag::Horner,
                        vs.as_ptr() as *const _,
                        vs.len(),
                    );
                }
            }
        }
    }
}

impl Drop for WrappedGraph {
    fn drop(&mut self) {
        unsafe {
            delete_graph(self.inner);
        }
    }
}

#[derive(Default, Debug, Deserialize, Serialize)]
#[repr(C)]
pub struct Graph {
    pub calculations: Vec<CalculationInfo>,
    pub num_intermediates: usize,
    #[serde(skip)]
    wrapped: WrappedGraph,
}

impl Graph {
    fn wrap(&mut self) -> &WrappedGraph {
        self.wrapped.reset(self.num_intermediates);
        for i in self.calculations.iter() {
            self.wrapped.push(i);
        }
        &self.wrapped
    }

    fn evaluate_batch_inner<F: Field, P: Deref<Target = [F]> + Sync + Send>(
        &self,
        values: &mut [F],
        rotations: &[i32],
        constants: &[F],
        fixed: &[P],
        advice: &[P],
        instance: &[P],
        challenges: &[F],
        beta: &F,
        gamma: &F,
        theta: &F,
        y: &F,
        rot_scale: i32,
        isize: i32,
    ) {
        fn parallelize<T: Send, F: Fn(&mut [T], usize) + Send + Sync + Clone>(v: &mut [T], f: F) {
            let n = v.len();
            let num_threads = rayon::current_num_threads();
            let mut chunk = (n as usize) / num_threads;
            if chunk < num_threads {
                chunk = 1;
            }

            rayon::scope(|scope| {
                for (chunk_num, v) in v.chunks_mut(chunk).enumerate() {
                    let f = f.clone();
                    scope.spawn(move |_| {
                        let start = chunk_num * chunk;
                        f(v, start);
                    });
                }
            });
        }

        fn get_rotation_idx(idx: usize, rot: i32, rot_scale: i32, isize: i32) -> usize {
            (((idx as i32) + (rot * rot_scale)).rem_euclid(isize)) as usize
        }

        parallelize(values, |values, start| {
            for (i, value) in values.iter_mut().enumerate() {
                let idx = start + i;
                let rotations = rotations
                    .iter()
                    .map(|rot| get_rotation_idx(idx, *rot, rot_scale, isize))
                    .collect::<Vec<_>>();

                let mut intermediates = vec![F::ZERO; self.num_intermediates];
                for calc in self.calculations.iter() {
                    intermediates[calc.target] = calc.calculation.eval(
                        &rotations,
                        &constants,
                        &intermediates,
                        fixed,
                        advice,
                        instance,
                        challenges,
                        beta,
                        gamma,
                        theta,
                        y,
                        value,
                    );
                }

                if let Some(calc) = self.calculations.last() {
                    *value = intermediates[calc.target];
                } else {
                    *value = F::ZERO;
                }
            }
        })
    }
}

pub trait GraphEval<F: Field> {
    fn update_and_eval<P: Deref<Target = [F]> + Sync + Send>(
        &mut self,
        info: Option<(&[CalculationInfo], usize)>,
        values: &mut [F],
        rotations: &[i32],
        constants: &[F],
        fixed: Option<&[P]>,
        advice: Option<&[P]>,
        instance: Option<&[P]>,
        challenges: &[F],
        beta: &F,
        gamma: &F,
        theta: &F,
        y: &F,
        rot_scale: i32,
        isize: i32,
    );
}

impl<F: Field> GraphEval<F> for Graph {
    default fn update_and_eval<P: Deref<Target = [F]> + Sync + Send>(
        &mut self,
        info: Option<(&[CalculationInfo], usize)>,
        values: &mut [F],
        rotations: &[i32],
        constants: &[F],
        fixed: Option<&[P]>,
        advice: Option<&[P]>,
        instance: Option<&[P]>,
        challenges: &[F],
        beta: &F,
        gamma: &F,
        theta: &F,
        y: &F,
        rot_scale: i32,
        isize: i32,
    ) {
        if let Some((calculations, num_intermediates)) = info {
            self.calculations = calculations.to_vec();
            self.num_intermediates = num_intermediates;
        }
        self.evaluate_batch_inner::<F, P>(
            values,
            rotations,
            constants,
            fixed.unwrap(),
            advice.unwrap(),
            instance.unwrap(),
            challenges,
            beta,
            gamma,
            theta,
            y,
            rot_scale,
            isize,
        );
    }
}

impl GraphEval<Fr> for Graph {
    fn update_and_eval<P: Deref<Target = [Fr]> + Sync + Send>(
        &mut self,
        info: Option<(&[CalculationInfo], usize)>,
        values: &mut [Fr],
        rotations: &[i32],
        constants: &[Fr],
        fixed: Option<&[P]>,
        advice: Option<&[P]>,
        instance: Option<&[P]>,
        challenges: &[Fr],
        beta: &Fr,
        gamma: &Fr,
        theta: &Fr,
        y: &Fr,
        rot_scale: i32,
        isize: i32,
    ) {
        if let Some((calculations, num_intermediates)) = info {
            self.calculations = calculations.to_vec();
            self.num_intermediates = num_intermediates;
        }

        let graph = self.wrap();
        graph.eval(
            values, rotations, constants, fixed, advice, instance, challenges, beta, gamma, theta,
            y, rot_scale, isize,
        );
    }
}
