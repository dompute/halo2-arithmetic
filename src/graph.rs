use std::{ffi::c_void, ops::Deref};

use ff::Field;

#[cfg(feature = "cuda")]
use halo2curves::bn256::Fr;

use halo2curves::{bn256::G1Affine, CurveAffine};

use crate::value_source::{Calculation, CalculationInfo, Rotation, ValueSource};

/// GraphEvaluator
#[derive(Clone, Debug)]
pub struct GraphEvaluator<C: CurveAffine> {
    /// Constants
    pub constants: Vec<C::ScalarExt>,
    /// Rotations
    pub rotations: Vec<i32>,
    /// Calculations
    pub calculations: Vec<CalculationInfo>,
    /// Number of intermediates
    pub num_intermediates: usize,
    #[cfg(feature = "cuda")]
    inner: *const c_void,
}

// #[cfg(feature = "cuda")]
// impl<C: PrimeField> Drop for GraphEvaluator<C> {
//     fn drop(&mut self) {
//         unsafe {
//             crate::stub::delete_graph(self.inner);
//         }
//     }
// }

impl<C: CurveAffine> Default for GraphEvaluator<C> {
    fn default() -> Self {
        Self {
            // Fixed positions to allow easy access
            constants: vec![
                C::ScalarExt::zero(),
                C::ScalarExt::one(),
                C::ScalarExt::from(2u64),
            ],
            rotations: Vec::new(),
            calculations: Vec::new(),
            num_intermediates: 0,
            #[cfg(feature = "cuda")]
            inner: unsafe { crate::stub::create_graph() },
        }
    }
}

impl<C: CurveAffine> GraphEvaluator<C> {
    /// Adds a rotation
    pub fn add_rotation(&mut self, rotation: &Rotation) -> usize {
        let position = self.rotations.iter().position(|&c| c == rotation.0);
        match position {
            Some(pos) => pos,
            None => {
                self.rotations.push(rotation.0);
                self.rotations.len() - 1
            }
        }
    }

    /// Adds a constant
    pub fn add_constant(&mut self, constant: &C::ScalarExt) -> ValueSource {
        let position = self.constants.iter().position(|&c| c == *constant);
        ValueSource::Constant(match position {
            Some(pos) => pos,
            None => {
                self.constants.push(*constant);
                self.constants.len() - 1
            }
        })
    }

    /// Adds a calculation.
    /// Currently does the simplest thing possible: just stores the
    /// resulting value so the result can be reused  when that calculation
    /// is done multiple times.
    pub fn add_calculation(&mut self, calculation: Calculation) -> ValueSource {
        let existing_calculation = self
            .calculations
            .iter()
            .find(|c| c.calculation == calculation);
        match existing_calculation {
            Some(existing_calculation) => ValueSource::Intermediate(existing_calculation.target),
            None => {
                let target = self.num_intermediates;
                let info = CalculationInfo {
                    calculation,
                    target,
                };
                #[cfg(feature = "cuda")]
                self.push(&info);
                self.calculations.push(info);
                self.num_intermediates += 1;
                ValueSource::Intermediate(target)
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn push(&mut self, cal: &CalculationInfo) {
        use crate::stub::{push_node, CalculationTag};

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

    fn evaluate_inner<P: Deref<Target = [C::ScalarExt]> + Sync + Send>(
        &self,
        values: &mut [C::ScalarExt],
        fixed: &[P],
        advice: &[P],
        instance: &[P],
        challenges: &[C::ScalarExt],
        beta: &C::ScalarExt,
        gamma: &C::ScalarExt,
        theta: &C::ScalarExt,
        y: &C::ScalarExt,
        rot_scale: i32,
        isize: i32,
    ) {
        let rotations = &self.rotations;
        let constants = &self.constants;

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

        let num_intermediates = self.num_intermediates;
        let calculations = &self.calculations;

        parallelize(values, |values, start| {
            for (i, value) in values.iter_mut().enumerate() {
                let idx = start + i;
                let rotations = rotations
                    .iter()
                    .map(|rot| get_rotation_idx(idx, *rot, rot_scale, isize))
                    .collect::<Vec<_>>();

                let mut intermediates = vec![C::ScalarExt::zero(); num_intermediates];
                for calc in calculations.iter() {
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

                if let Some(calc) = calculations.last() {
                    *value = intermediates[calc.target];
                } else {
                    *value = C::ScalarExt::zero();
                }
            }
        })
    }

    pub fn evaluate<P: Deref<Target = [C::ScalarExt]> + Sync + Send>(
        &self,
        values: &mut [C::ScalarExt],
        fixed: &[P],
        advice: &[P],
        instance: &[P],
        challenges: &[C::ScalarExt],
        beta: &C::ScalarExt,
        gamma: &C::ScalarExt,
        theta: &C::ScalarExt,
        y: &C::ScalarExt,
        rot_scale: i32,
        isize: i32,
        round: usize,
    ) {
        trait Functor<F: CurveAffine> {
            fn invoke<P: Deref<Target = [F::ScalarExt]> + Sync + Send>(
                graph: &GraphEvaluator<F>,
                values: &mut [F::ScalarExt],
                fixed: &[P],
                advice: &[P],
                instance: &[P],
                challenges: &[F::ScalarExt],
                beta: &F::ScalarExt,
                gamma: &F::ScalarExt,
                theta: &F::ScalarExt,
                y: &F::ScalarExt,
                rot_scale: i32,
                isize: i32,
                round: usize,
            );
        }

        impl<F: CurveAffine> Functor<F> for () {
            default fn invoke<P: Deref<Target = [F::ScalarExt]> + Sync + Send>(
                graph: &GraphEvaluator<F>,
                values: &mut [F::ScalarExt],
                fixed: &[P],
                advice: &[P],
                instance: &[P],
                challenges: &[F::ScalarExt],
                beta: &F::ScalarExt,
                gamma: &F::ScalarExt,
                theta: &F::ScalarExt,
                y: &F::ScalarExt,
                rot_scale: i32,
                isize: i32,
                _round: usize,
            ) {
                let now = std::time::Instant::now();
                graph.evaluate_inner(
                    values, fixed, advice, instance, challenges, beta, gamma, theta, y, rot_scale,
                    isize,
                );
                println!("Eval(Host) elapsed: {:?}", now.elapsed());
            }
        }

        #[cfg(feature = "cuda")]
        impl Functor<G1Affine> for () {
            fn invoke<P: Deref<Target = [Fr]> + Sync + Send>(
                graph: &GraphEvaluator<G1Affine>,
                values: &mut [Fr],
                fixed: &[P],
                advice: &[P],
                instance: &[P],
                challenges: &[Fr],
                beta: &Fr,
                gamma: &Fr,
                theta: &Fr,
                y: &Fr,
                rot_scale: i32,
                isize: i32,
                round: usize,
            ) {
                unsafe {
                    let f = |v: &[P]| -> (Vec<*const c_void>, usize, usize) {
                        let ptr = v
                            .iter()
                            .map(|v| v.as_ptr() as *const c_void)
                            .collect::<Vec<_>>();
                        let col = v.len();
                        let row = v.get(0).and_then(|v| Some(v.len())).unwrap_or(0);
                        (ptr, col, row)
                    };

                    let (fixed, fiexd_col, fixed_row) = f(fixed);
                    let (advice, advice_col, advice_row) = f(advice);
                    let (instance, instance_col, instance_row) = f(instance);
                    #[cfg(feature = "profile")]
                    let now = std::time::Instant::now();
                    crate::stub::evaluate_batch(
                        values.as_mut_ptr() as *mut _,
                        values.len(),
                        graph.inner,
                        graph.rotations.as_ptr(),
                        graph.rotations.len(),
                        graph.constants.as_ptr() as *const c_void,
                        graph.constants.len(),
                        fixed.as_ptr(),
                        fiexd_col,
                        fixed_row,
                        advice.as_ptr(),
                        advice_col,
                        advice_row,
                        instance.as_ptr(),
                        instance_col,
                        instance_row,
                        challenges.as_ptr() as *const c_void,
                        challenges.len(),
                        beta as *const _ as *const c_void,
                        gamma as *const _ as *const c_void,
                        theta as *const _ as *const c_void,
                        y as *const _ as *const c_void,
                        rot_scale,
                        isize,
                        round,
                    );
                    #[cfg(feature = "profile")]
                    println!("Eval elapsed: {:?}", now.elapsed());
                }
            }
        }

        <() as Functor<C>>::invoke(
            self, values, fixed, advice, instance, challenges, beta, gamma, theta, y, rot_scale,
            isize, round,
        );
    }
}
