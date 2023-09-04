use std::ops::Deref;

use ff::PrimeField;

use crate::value_source::{Calculation, CalculationInfo, Rotation, ValueSource};

/// GraphEvaluator
#[derive(Clone, Debug)]
pub struct GraphEvaluator<C: PrimeField> {
    /// Constants
    pub constants: Vec<C>,
    /// Rotations
    pub rotations: Vec<i32>,
    /// Calculations
    pub calculations: Vec<CalculationInfo>,
    /// Number of intermediates
    pub num_intermediates: usize,
}

impl<C: PrimeField> Default for GraphEvaluator<C> {
    fn default() -> Self {
        Self {
            // Fixed positions to allow easy access
            constants: vec![C::ZERO, C::ONE, C::from(2u64)],
            rotations: Vec::new(),
            calculations: Vec::new(),
            num_intermediates: 0,
        }
    }
}

impl<C: PrimeField> GraphEvaluator<C> {
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
    pub fn add_constant(&mut self, constant: &C) -> ValueSource {
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
                self.calculations.push(CalculationInfo {
                    calculation,
                    target,
                });
                self.num_intermediates += 1;
                ValueSource::Intermediate(target)
            }
        }
    }

    fn evaluate_inner<P: Deref<Target = [C]> + Sync + Send>(
        &self,
        values: &mut [C],
        fixed: &[P],
        advice: &[P],
        instance: &[P],
        challenges: &[C],
        beta: &C,
        gamma: &C,
        theta: &C,
        y: &C,
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

        parallelize(values, |values, start| {
            for (i, value) in values.iter_mut().enumerate() {
                let idx = start + i;
                let rotations = rotations
                    .iter()
                    .map(|rot| get_rotation_idx(idx, *rot, rot_scale, isize))
                    .collect::<Vec<_>>();

                let mut intermediates = vec![C::ZERO; self.num_intermediates];
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
                    *value = C::ZERO;
                }
            }
        })
    }

    pub fn evaluate<P: Deref<Target = [C]> + Sync + Send>(
        &self,
        values: &mut [C],
        fixed: &[P],
        advice: &[P],
        instance: &[P],
        challenges: &[C],
        beta: &C,
        gamma: &C,
        theta: &C,
        y: &C,
        rot_scale: i32,
        isize: i32,
    ) {
        trait Functor<C: PrimeField> {
            fn invoke<P: Deref<Target = [C]> + Sync + Send>(
                graph: &GraphEvaluator<C>,
                values: &mut [C],
                fixed: &[P],
                advice: &[P],
                instance: &[P],
                challenges: &[C],
                beta: &C,
                gamma: &C,
                theta: &C,
                y: &C,
                rot_scale: i32,
                isize: i32,
            );
        }

        impl<C: PrimeField> Functor<C> for () {
            default fn invoke<P: Deref<Target = [C]> + Sync + Send>(
                graph: &GraphEvaluator<C>,
                values: &mut [C],
                fixed: &[P],
                advice: &[P],
                instance: &[P],
                challenges: &[C],
                beta: &C,
                gamma: &C,
                theta: &C,
                y: &C,
                rot_scale: i32,
                isize: i32,
            ) {
                graph.evaluate_inner(
                    values, fixed, advice, instance, challenges, beta, gamma, theta, y, rot_scale,
                    isize,
                );
            }
        }

        <() as Functor<C>>::invoke(
            self, values, fixed, advice, instance, challenges, beta, gamma, theta, y, rot_scale,
            isize,
        );
    }
}

// pub trait Evaluable<C: PrimeField> {
//     fn evaluate<P: Deref<Target = [C]> + Sync + Send>(
//         &self,
//         values: &mut [C],
//         fixed: &[P],
//         advice: &[P],
//         instance: &[P],
//         challenges: &[C],
//         beta: &C,
//         gamma: &C,
//         theta: &C,
//         y: &C,
//         rot_scale: i32,
//         isize: i32,
//     );
// }

// impl<C: PrimeField> Evaluable<C> for GraphEvaluator<C> {
//     default fn evaluate<P: Deref<Target = [C]> + Sync + Send>(
//         &self,
//         values: &mut [C],
//         fixed: &[P],
//         advice: &[P],
//         instance: &[P],
//         challenges: &[C],
//         beta: &C,
//         gamma: &C,
//         theta: &C,
//         y: &C,
//         rot_scale: i32,
//         isize: i32,
//     ) {
//         self.evaluate_inner(
//             values, fixed, advice, instance, challenges, beta, gamma, theta, y, rot_scale, isize,
//         );
//     };
// }
