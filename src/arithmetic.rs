// use ff::{Field, PrimeField};
// use group::Group as GroupTrait;

use halo2curves::{
    group::ff::{Field, PrimeField},
    group::Group as GroupTrait,
    CurveAffine, Group,
};

#[cfg(feature = "cuda")]
use halo2curves::bn256::{Fr, G1Affine, G1};

fn multiexp_serial<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C], acc: &mut C::Curve) {
    let coeffs: Vec<_> = coeffs.iter().map(|a| a.to_repr()).collect();

    let c = if bases.len() < 4 {
        1
    } else if bases.len() < 32 {
        3
    } else {
        (f64::from(bases.len() as u32)).ln().ceil() as usize
    };

    fn get_at<F: PrimeField>(segment: usize, c: usize, bytes: &F::Repr) -> usize {
        let skip_bits = segment * c;
        let skip_bytes = skip_bits / 8;

        if skip_bytes >= 32 {
            return 0;
        }

        let mut v = [0; 8];
        for (v, o) in v.iter_mut().zip(bytes.as_ref()[skip_bytes..].iter()) {
            *v = *o;
        }

        let mut tmp = u64::from_le_bytes(v);
        tmp >>= skip_bits - (skip_bytes * 8);
        tmp = tmp % (1 << c);

        tmp as usize
    }

    let segments = (256 / c) + 1;

    for current_segment in (0..segments).rev() {
        for _ in 0..c {
            *acc = acc.double();
        }

        #[derive(Clone, Copy)]
        enum Bucket<C: CurveAffine> {
            None,
            Affine(C),
            Projective(C::Curve),
        }

        impl<C: CurveAffine> Bucket<C> {
            fn add_assign(&mut self, other: &C) {
                *self = match *self {
                    Bucket::None => Bucket::Affine(*other),
                    Bucket::Affine(a) => Bucket::Projective(a + *other),
                    Bucket::Projective(mut a) => {
                        a += *other;
                        Bucket::Projective(a)
                    }
                }
            }

            fn add(self, mut other: C::Curve) -> C::Curve {
                match self {
                    Bucket::None => other,
                    Bucket::Affine(a) => {
                        other += a;
                        other
                    }
                    Bucket::Projective(a) => other + &a,
                }
            }
        }

        let mut buckets: Vec<Bucket<C>> = vec![Bucket::None; (1 << c) - 1];

        for (coeff, base) in coeffs.iter().zip(bases.iter()) {
            let coeff = get_at::<C::Scalar>(current_segment, c, coeff);
            if coeff != 0 {
                buckets[coeff - 1].add_assign(base);
            }
        }

        // Summation by parts
        // e.g. 3a + 2b + 1c = a +
        //                    (a) + b +
        //                    ((a) + b) + c
        let mut running_sum = C::Curve::identity();
        for exp in buckets.into_iter().rev() {
            running_sum = exp.add(running_sum);
            *acc = *acc + &running_sum;
        }
    }
}

fn best_multiexp_inner<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
    assert_eq!(coeffs.len(), bases.len());

    let num_threads = rayon::current_num_threads();
    if coeffs.len() > num_threads {
        let chunk = coeffs.len() / num_threads;
        let num_chunks = coeffs.chunks(chunk).len();
        let mut results = vec![C::Curve::identity(); num_chunks];
        rayon::scope(|scope| {
            let chunk = coeffs.len() / num_threads;

            for ((coeffs, bases), acc) in coeffs
                .chunks(chunk)
                .zip(bases.chunks(chunk))
                .zip(results.iter_mut())
            {
                scope.spawn(move |_| {
                    multiexp_serial(coeffs, bases, acc);
                });
            }
        });
        results.iter().fold(C::Curve::identity(), |a, b| a + b)
    } else {
        let mut acc = C::Curve::identity();
        multiexp_serial(coeffs, bases, &mut acc);
        acc
    }
}

/// Performs a multi-exponentiation operation.
///
/// This function will panic if coeffs and bases have a different length.
///
/// This will use multithreading if beneficial.
pub fn best_multiexp<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
    trait Functor<C: CurveAffine> {
        fn invoke(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve;
    }

    impl<C: CurveAffine> Functor<C> for () {
        default fn invoke(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
            best_multiexp_inner(coeffs, bases)
        }
    }

    #[cfg(feature = "cuda")]
    impl Functor<G1Affine> for () {
        fn invoke(coeffs: &[Fr], bases: &[G1Affine]) -> G1 {
            cuda::msm(coeffs, bases)
        }
    }

    <() as Functor<C>>::invoke(coeffs, bases)
}

/// This perform recursive butterfly arithmetic
fn recursive_butterfly_arithmetic<G: Group>(
    a: &mut [G],
    n: usize,
    twiddle_chunk: usize,
    twiddles: &[G::Scalar],
) {
    if n == 2 {
        let t = a[1];
        a[1] = a[0];
        a[0].group_add(&t);
        a[1].group_sub(&t);
    } else {
        let (left, right) = a.split_at_mut(n / 2);
        rayon::join(
            || recursive_butterfly_arithmetic(left, n / 2, twiddle_chunk * 2, twiddles),
            || recursive_butterfly_arithmetic(right, n / 2, twiddle_chunk * 2, twiddles),
        );

        // case when twiddle factor is one
        let (a, left) = left.split_at_mut(1);
        let (b, right) = right.split_at_mut(1);
        let t = b[0];
        b[0] = a[0];
        a[0].group_add(&t);
        b[0].group_sub(&t);

        left.iter_mut()
            .zip(right.iter_mut())
            .enumerate()
            .for_each(|(i, (a, b))| {
                let mut t = *b;
                t.group_scale(&twiddles[(i + 1) * twiddle_chunk]);
                *b = *a;
                a.group_add(&t);
                b.group_sub(&t);
            });
    }
}

fn best_fft_inner<G: Group>(a: &mut [G], omega: G::Scalar, log_n: u32) {
    fn bitreverse(mut n: usize, l: usize) -> usize {
        let mut r = 0;
        for _ in 0..l {
            r = (r << 1) | (n & 1);
            n >>= 1;
        }
        r
    }

    fn log2_floor(num: usize) -> u32 {
        assert!(num > 0);

        let mut pow = 0;

        while (1 << (pow + 1)) <= num {
            pow += 1;
        }

        pow
    }

    let threads = rayon::current_num_threads();
    let log_threads = log2_floor(threads);
    let n = a.len() as usize;
    assert_eq!(n, 1 << log_n);

    for k in 0..n {
        let rk = bitreverse(k, log_n as usize);
        if k < rk {
            a.swap(rk, k);
        }
    }

    // precompute twiddle factors
    let twiddles: Vec<_> = (0..(n / 2) as usize)
        .scan(G::Scalar::one(), |w, _| {
            let tw = *w;
            *w *= &omega;
            Some(tw)
        })
        .collect();

    if log_n <= log_threads {
        let mut chunk = 2_usize;
        let mut twiddle_chunk = (n / 2) as usize;
        for _ in 0..log_n {
            a.chunks_mut(chunk).for_each(|coeffs| {
                let (left, right) = coeffs.split_at_mut(chunk / 2);

                // case when twiddle factor is one
                let (a, left) = left.split_at_mut(1);
                let (b, right) = right.split_at_mut(1);
                let t = b[0];
                b[0] = a[0];
                a[0].group_add(&t);
                b[0].group_sub(&t);

                left.iter_mut()
                    .zip(right.iter_mut())
                    .enumerate()
                    .for_each(|(i, (a, b))| {
                        let mut t = *b;
                        t.group_scale(&twiddles[(i + 1) * twiddle_chunk]);
                        *b = *a;
                        a.group_add(&t);
                        b.group_sub(&t);
                    });
            });
            chunk *= 2;
            twiddle_chunk /= 2;
        }
    } else {
        recursive_butterfly_arithmetic(a, n, 1, &twiddles)
    }
}

/// Performs a radix-$2$ Fast-Fourier Transformation (FFT) on a vector of size
/// $n = 2^k$, when provided `log_n` = $k$ and an element of multiplicative
/// order $n$ called `omega` ($\omega$). The result is that the vector `a`, when
/// interpreted as the coefficients of a polynomial of degree $n - 1$, is
/// transformed into the evaluations of this polynomial at each of the $n$
/// distinct powers of $\omega$. This transformation is invertible by providing
/// $\omega^{-1}$ in place of $\omega$ and dividing each resulting field element
/// by $n$.
///
/// This will use multithreading if beneficial.
pub fn best_fft<G: Group>(a: &mut [G], omega: G::Scalar, log_n: u32) {
    trait Functor<G: Group> {
        fn invoke(a: &mut [G], omega: G::Scalar, log_n: u32);
    }

    impl<G: Group> Functor<G> for () {
        default fn invoke(a: &mut [G], omega: G::Scalar, log_n: u32) {
            best_fft_inner(a, omega, log_n)
        }
    }

    #[cfg(feature = "cuda")]
    impl Functor<Fr> for () {
        fn invoke(a: &mut [Fr], omega: Fr, log_n: u32) {
            cuda::fft(a, omega, log_n);
        }
    }

    <() as Functor<G>>::invoke(a, omega, log_n)
}

#[cfg(feature = "cuda")]
mod cuda {
    use std::ffi::c_void;

    use halo2curves::{
        bn256::{Fr, G1Affine, G1},
        group::Group,
    };

    pub fn msm(scalars: &[Fr], bases: &[G1Affine]) -> G1 {
        let mut out = G1::identity();
        let buf_len = bases.len();
        unsafe {
            crate::stub::msm_fr_g1(
                scalars.as_ptr() as *const c_void,
                bases.as_ptr() as *const c_void,
                buf_len as u32,
                &mut out as *mut _ as *mut c_void,
            );
        };
        out
    }

    pub fn fft(a: &mut [Fr], omega: Fr, log_n: u32) {
        assert!(a.len() == (1 << log_n));
        let twiddles_len = a.len() / 2;
        let twiddles: Vec<_> = (0..twiddles_len)
            .scan(Fr::one(), |w, _| {
                let tw = *w;
                *w *= &omega;
                Some(tw)
            })
            .collect();
        unsafe {
            crate::stub::fft_fr(
                a.as_mut_ptr() as *mut c_void,
                twiddles.as_ptr() as *const c_void,
                log_n,
            );
        };
    }

    #[cfg(test)]
    mod tests {
        use std::ops::Mul;

        use halo2curves::{
            bn256::{Fq, Fr, G1Affine, G1},
            group::{ff::Field, Curve},
        };
        use rand::rngs::OsRng;

        use crate::arithmetic::{best_fft_inner, best_multiexp_inner};

        fn fr_from_str(r: &str) -> Fr {
            let a = u64::from_str_radix(&r[0..16], 16).unwrap();
            let b = u64::from_str_radix(&r[16..32], 16).unwrap();
            let c = u64::from_str_radix(&r[32..48], 16).unwrap();
            let d = u64::from_str_radix(&r[48..64], 16).unwrap();
            Fr::from_raw([d, c, b, a])
        }

        #[allow(dead_code)]
        fn fq_from_str(r: &str) -> Fq {
            let a = u64::from_str_radix(&r[0..16], 16).unwrap();
            let b = u64::from_str_radix(&r[16..32], 16).unwrap();
            let c = u64::from_str_radix(&r[32..48], 16).unwrap();
            let d = u64::from_str_radix(&r[48..64], 16).unwrap();
            Fq::from_raw([d, c, b, a])
        }

        #[allow(dead_code)]
        fn g1_from_str(r: &str) -> G1Affine {
            let x = fq_from_str(&r[3..67]);
            let y = fq_from_str(&r[71..135]);
            G1Affine { x, y }
        }

        #[test]
        fn test_msm() {
            const N: usize = 1 << 18;
            // let scalars = (0..N).map(|_| Fr::random(OsRng)).collect::<Vec<_>>();
            // let bases = (0..N).map(|_| G1Affine::random(OsRng)).collect::<Vec<_>>();
            let scalars = (0..N).map(|_| Fr::one()).collect::<Vec<_>>();
            let bases = (0..N).map(|_| G1Affine::generator()).collect::<Vec<_>>();

            for i in 0..100 {
                let now = std::time::Instant::now();
                let gpu = super::msm(&scalars, &bases);
                println!("GPU: {:?}", now.elapsed());

                let now = std::time::Instant::now();
                let cpu = best_multiexp_inner(&scalars, &bases);
                println!("CPU: {:?}", now.elapsed());
                assert_eq!(gpu.to_affine(), cpu.to_affine());
            }
        }

        #[test]
        fn test_fft() {
            const K: u32 = 18u32;
            let ori = (0..(1 << K)).map(|i| Fr::random(OsRng)).collect::<Vec<_>>();
            let omega = Fr::random(OsRng);

            for i in 0..100 {
                let mut g_a = ori.clone();
                let mut g_b = ori.clone();

                let now = std::time::Instant::now();
                best_fft_inner(&mut g_a, omega, K);
                println!("CPU: {:?}", now.elapsed());

                let now = std::time::Instant::now();
                super::fft(&mut g_b, omega, K);
                println!("GPU: {:?}", now.elapsed());

                for (i, (cpu, gpu)) in g_a.iter().zip(g_b.iter()).enumerate() {
                    assert_eq!(cpu, gpu);
                }
            }
        }
    }
}
