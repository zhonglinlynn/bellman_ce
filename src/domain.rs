//! This module contains an [`EvaluationDomain`] abstraction for performing
//! various kinds of polynomial arithmetic on top of the scalar field.
//!
//! In pairing-based SNARKs, we need to calculate a quotient
//! polynomial over a target polynomial with roots at distinct points associated
//! with each constraint of the constraint system. In order to be efficient, we
//! choose these roots to be the powers of a 2<sup>n</sup> root of unity in the
//! field. This allows us to perform polynomial operations in O(n) by performing
//! an O(n log n) FFT over such a domain.
//!
//! [`EvaluationDomain`]: crate::domain::EvaluationDomain

pub use super::group::*;
use super::worker::Worker;
use super::SynthesisError;
use pairing::{
    ff::{Field, PrimeField, ScalarEngine},
    CurveProjective, Engine,
};


use crate::locks::LockedMultiFFTKernel;
use crate::locks::LockedMultiexpKernel;
use ec_gpu_gen::EcError;
use ec_gpu_gen::EcResult;
use ec_gpu_gen::fft::FftKernel;
use ec_gpu_gen::multiexp::MultiexpKernel;


use crate::plonk::domains::Domain;
use log::{info, warn};

pub struct EvaluationDomain<E: Engine, G: Group<E>> {
    coeffs: Vec<G>,
    exp: u32,
    omega: E::Fr,
    omegainv: E::Fr,
    geninv: E::Fr,
    minv: E::Fr,
}

impl<E: Engine, G: Group<E>> AsRef<[G]> for EvaluationDomain<E, G> {
    fn as_ref(&self) -> &[G] {
        &self.coeffs
    }
}

impl<E: Engine, G: Group<E>> AsMut<[G]> for EvaluationDomain<E, G> {
    fn as_mut(&mut self) -> &mut [G] {
        &mut self.coeffs
    }
}

impl<E: Engine, G: Group<E>> EvaluationDomain<E, G> {
    pub fn into_coeffs(self) -> Vec<G> {
        self.coeffs
    }

    pub fn from_coeffs(mut coeffs: Vec<G>) -> Result<EvaluationDomain<E, G>, SynthesisError> {
        // Compute the size of our evaluation domain
        let mut m = 1;
        let mut exp = 0;
        while m < coeffs.len() {
            m *= 2;
            exp += 1;

            // The pairing-friendly curve may not be able to support
            // large enough (radix2) evaluation domains.
            if exp >= E::Fr::S {
                return Err(SynthesisError::PolynomialDegreeTooLarge);
            }
        }
        // Compute omega, the 2^exp primitive root of unity
        let mut omega = E::Fr::root_of_unity();
        for _ in exp..E::Fr::S {
            omega.square();
        }

        // Extend the coeffs vector with zeroes if necessary
        coeffs.resize(m, G::group_zero());

        Ok(EvaluationDomain {
            coeffs,
            exp,
            omega,
            omegainv: omega.inverse().unwrap(),
            geninv: E::Fr::multiplicative_generator().inverse().unwrap(),
            minv: E::Fr::from_str(&format!("{}", m))
                .unwrap()
                .inverse()
                .unwrap(),
        })
    }

    pub fn fft(
        &mut self,
        worker: &Worker,
        kern: &mut Option<LockedMultiFFTKernel<E>>,
    ) -> EcResult<()> {
        best_fft(kern, &mut self.coeffs, worker, &self.omega, self.exp)?;
        Ok(())
    }

    pub fn ifft(
        &mut self,
        worker: &Worker,
        kern: &mut Option<LockedMultiFFTKernel<E>>,
    ) -> EcResult<()> {
        best_fft(kern, &mut self.coeffs, worker, &self.omegainv, self.exp)?;

        worker.scope(self.coeffs.len(), |scope, chunk| {
            let minv = self.minv;

            for v in self.coeffs.chunks_mut(chunk) {
                scope.spawn(move |_| {
                    for v in v {
                        v.group_mul_assign(&minv);
                    }
                });
            }
        });

        Ok(())
    }

    pub fn distribute_powers(&mut self, worker: &Worker, g: E::Fr) {
        worker.scope(self.coeffs.len(), |scope, chunk| {
            for (i, v) in self.coeffs.chunks_mut(chunk).enumerate() {
                scope.spawn(move |_| {
                    let mut u = g.pow(&[(i * chunk) as u64]);
                    for v in v.iter_mut() {
                        v.group_mul_assign(&u);
                        u.mul_assign(&g);
                    }
                });
            }
        });
    }

    pub fn coset_fft(
        &mut self,
        worker: &Worker,
        kern: &mut Option<LockedMultiFFTKernel<E>>,
    ) -> EcResult<()> {
        self.distribute_powers(worker, E::Fr::multiplicative_generator());
        self.fft(worker, kern)?;
        Ok(())
    }

    pub fn icoset_fft(
        &mut self,
        worker: &Worker,
        kern: &mut Option<LockedMultiFFTKernel<E>>,
    ) -> EcResult<()> {
        let geninv = self.geninv;
        self.ifft(worker, kern)?;
        self.distribute_powers(worker, geninv);
        Ok(())
    }

    //TODO: consider fft kernel
    pub fn transform_powers_of_tau_into_lagrange_basis(&mut self, worker: &Worker) {
        self.ifft(&worker, &mut None).unwrap();
    }

    //TODO: consider fft kernel
    pub fn transform_powers_of_tau_into_lagrange_basis_on_coset(&mut self, worker: &Worker) {
        let geninv = self.geninv;
        self.distribute_powers(worker, geninv);

        self.ifft(worker, &mut None).unwrap();
    }

    /// This evaluates t(tau) for this domain, which is
    /// tau^m - 1 for these radix-2 domains.
    pub fn z(&self, tau: &E::Fr) -> E::Fr {
        let mut tmp = tau.pow(&[self.coeffs.len() as u64]);
        tmp.sub_assign(&E::Fr::one());

        tmp
    }

    /// The target polynomial is the zero polynomial in our
    /// evaluation domain, so we must perform division over
    /// a coset.
    pub fn divide_by_z_on_coset(&mut self, worker: &Worker) {
        let i = self
            .z(&E::Fr::multiplicative_generator())
            .inverse()
            .unwrap();

        worker.scope(self.coeffs.len(), |scope, chunk| {
            for v in self.coeffs.chunks_mut(chunk) {
                scope.spawn(move |_| {
                    for v in v {
                        v.group_mul_assign(&i);
                    }
                });
            }
        });
    }

    /// Perform O(n) multiplication of two polynomials in the domain.
    pub fn mul_assign(&mut self, worker: &Worker, other: &EvaluationDomain<E, Scalar<E>>) {
        assert_eq!(self.coeffs.len(), other.coeffs.len());

        worker.scope(self.coeffs.len(), |scope, chunk| {
            for (a, b) in self
                .coeffs
                .chunks_mut(chunk)
                .zip(other.coeffs.chunks(chunk))
            {
                scope.spawn(move |_| {
                    for (a, b) in a.iter_mut().zip(b.iter()) {
                        a.group_mul_assign(&b.0);
                    }
                });
            }
        });
    }

    /// Perform O(n) subtraction of one polynomial from another in the domain.
    pub fn sub_assign(&mut self, worker: &Worker, other: &EvaluationDomain<E, G>) {
        assert_eq!(self.coeffs.len(), other.coeffs.len());

        worker.scope(self.coeffs.len(), |scope, chunk| {
            for (a, b) in self
                .coeffs
                .chunks_mut(chunk)
                .zip(other.coeffs.chunks(chunk))
            {
                scope.spawn(move |_| {
                    for (a, b) in a.iter_mut().zip(b.iter()) {
                        a.group_sub_assign(&b);
                    }
                });
            }
        });
    }
}

fn best_fft<E: Engine, T: Group<E>>(
    kern: &mut Option<LockedMultiFFTKernel<E>>,
    a: &mut [T],
    worker: &Worker,
    omega: &E::Fr,
    log_n: u32,
) -> EcResult<()> {
    if let Some(ref mut kern) = kern {
        if kern
            .with(|k: &mut FftKernel<E>| gpu_fft(k, a, omega, log_n))
            .is_ok()
        {
            return Ok(());
        }
    }

    let log_cpus = worker.log_num_cpus();
    if log_n <= log_cpus {
        serial_fft(a, omega, log_n);
    } else {
        parallel_fft(a, worker, omega, log_n, log_cpus);
    }

    Ok(())
}

pub fn gpu_fft<E: Engine, T: Group<E>>(
    kern: &mut FftKernel<E>,
    a: &mut [T],
    omega: &E::Fr,
    log_n: u32,
) -> EcResult<()> {
    // EvaluationDomain module is supposed to work only with E::Fr elements, and not CurveProjective
    // points. The Bellman authors have implemented an unnecessarry abstraction called Group<E>
    // which is implemented for both PrimeField and CurveProjective elements. As nowhere in the code
    // is the CurveProjective version used, T and E::Fr are guaranteed to be equal and thus have same
    // size.
    // For compatibility/performance reasons we decided to transmute the array to the desired type
    // as it seems safe and needs less modifications in the current structure of Bellman library.
    // So we use E::Fr instead of Group<E> directly.
    let a = unsafe { std::mem::transmute::<&mut [T], &mut [E::Fr]>(a) };
    kern.fft_multiple(&mut [a], omega, log_n)?;

    Ok(())
}

pub fn serial_fft<E: Engine, T: Group<E>>(a: &mut [T], omega: &E::Fr, log_n: u32) {
    fn bitreverse(mut n: u32, l: u32) -> u32 {
        let mut r = 0;
        for _ in 0..l {
            r = (r << 1) | (n & 1);
            n >>= 1;
        }
        r
    }

    let n = a.len() as u32;
    assert_eq!(n, 1 << log_n);

    for k in 0..n {
        let rk = bitreverse(k, log_n);
        if k < rk {
            a.swap(rk as usize, k as usize);
        }
    }

    let mut m = 1;
    for _ in 0..log_n {
        let w_m = omega.pow(&[u64::from(n / (2 * m))]);

        let mut k = 0;
        while k < n {
            let mut w = E::Fr::one();
            for j in 0..m {
                let mut t = a[(k + j + m) as usize];
                t.group_mul_assign(&w);
                let mut tmp = a[(k + j) as usize];
                tmp.group_sub_assign(&t);
                a[(k + j + m) as usize] = tmp;
                a[(k + j) as usize].group_add_assign(&t);
                w.mul_assign(&w_m);
            }

            k += 2 * m;
        }

        m *= 2;
    }
}

fn parallel_fft<E: Engine, T: Group<E>>(
    a: &mut [T],
    worker: &Worker,
    omega: &E::Fr,
    log_n: u32,
    log_cpus: u32,
) {
    assert!(log_n >= log_cpus);

    let num_cpus = 1 << log_cpus;
    let log_new_n = log_n - log_cpus;
    let mut tmp = vec![vec![T::group_zero(); 1 << log_new_n]; num_cpus];
    let new_omega = omega.pow(&[num_cpus as u64]);

    worker.scope(0, |scope, _| {
        let a = &*a;

        for (j, tmp) in tmp.iter_mut().enumerate() {
            scope.spawn(move |_scope| {
                // Shuffle into a sub-FFT
                let omega_j = omega.pow(&[j as u64]);
                let omega_step = omega.pow(&[(j as u64) << log_new_n]);

                let mut elt = E::Fr::one();
                for (i, tmp) in tmp.iter_mut().enumerate() {
                    for s in 0..num_cpus {
                        let idx = (i + (s << log_new_n)) % (1 << log_n);
                        let mut t = a[idx];
                        t.group_mul_assign(&elt);
                        tmp.group_add_assign(&t);
                        elt.mul_assign(&omega_step);
                    }
                    elt.mul_assign(&omega_j);
                }

                // Perform sub-FFT
                serial_fft(tmp, &new_omega, log_new_n);
            });
        }
    });

    worker.scope(a.len(), |scope, chunk| {
        let tmp = &tmp;

        for (idx, a) in a.chunks_mut(chunk).enumerate() {
            scope.spawn(move |_scope| {
                let mut idx = idx * chunk;
                let mask = (1 << log_cpus) - 1;
                for a in a {
                    *a = tmp[idx & mask][idx >> log_cpus];
                    idx += 1;
                }
            });
        }
    });
}

pub fn distribute_powers<E: Engine>(coeffs: &mut [E::Fr], worker: &Worker, g: E::Fr) {
    worker.scope(coeffs.len(), |scope, chunk| {
        for (i, v) in coeffs.chunks_mut(chunk).enumerate() {
            scope.spawn(move |_| {
                let mut u = g.pow(&[(i * chunk) as u64]);
                for v in v.iter_mut() {
                    v.mul_assign(&u);
                    u.mul_assign(&g);
                }
            });
        }
    });
}

use rust_gpu_tools::Device;
pub fn create_fft_kernel<E>(_log_d: usize, priority: &[&Device]) -> Option<FftKernel<E>>
where
    E: Engine,
{
    match FftKernel::create(&priority) {
        Ok(k) => {
            info!("GPU FFT kernel instantiated!");
            Some(k)
        }
        Err(e) => {
            warn!("Cannot instantiate GPU FFT kernel! Error: {}", e);
            None
        }
    }
}

pub fn best_fft_multiple_gpu<E: Engine>(
    kern: &mut Option<LockedMultiFFTKernel<E>>,
    polys: &mut [&mut [E::Fr]],
    worker: &Worker,
    omega: &E::Fr,
    log_n: u32,
) -> EcResult<()> {
    if let Some(ref mut kern) = kern {
        if kern
            .with(|k: &mut FftKernel<E>| gpu_fft_multiple(k, polys, omega, log_n))
            .is_ok()
        {
            return Ok(());
        }
    }

    let log_cpus = worker.log_num_cpus();
    if log_n <= log_cpus {
        for poly in polys.iter_mut() {
            serial_fft_fr::<E>(poly, omega, log_n);
        }
    } else {
        for poly in polys.iter_mut() {
            parallel_fft_fr::<E>(poly, worker, omega, log_n, log_cpus);
        }
    }

    Ok(())
}

pub fn gpu_fft_multiple<E: Engine>(
    kern: &mut FftKernel<E>,
    polys: &mut [&mut [E::Fr]],
    omega: &E::Fr,
    log_n: u32,
) -> EcResult<()> {
    kern.fft_multiple(polys, omega, log_n)?;

    Ok(())
}

pub fn serial_fft_fr<E: Engine>(a: &mut [E::Fr], omega: &E::Fr, log_n: u32) {
    fn bitreverse(mut n: u32, l: u32) -> u32 {
        let mut r = 0;
        for _ in 0..l {
            r = (r << 1) | (n & 1);
            n >>= 1;
        }
        r
    }

    let n = a.len() as u32;
    assert_eq!(n, 1 << log_n);

    for k in 0..n {
        let rk = bitreverse(k, log_n);
        if k < rk {
            a.swap(rk as usize, k as usize);
        }
    }

    let mut m = 1;
    for _ in 0..log_n {
        let w_m = omega.pow(&[u64::from(n / (2 * m))]);

        let mut k = 0;
        while k < n {
            let mut w = E::Fr::one();
            for j in 0..m {
                let mut t = a[(k + j + m) as usize];
                t.mul_assign(&w);
                let mut tmp = a[(k + j) as usize];
                tmp.sub_assign(&t);
                a[(k + j + m) as usize] = tmp;
                a[(k + j) as usize].add_assign(&t);
                w.mul_assign(&w_m);
            }

            k += 2 * m;
        }

        m *= 2;
    }
}

fn parallel_fft_fr<E: Engine>(
    a: &mut [E::Fr],
    worker: &Worker,
    omega: &E::Fr,
    log_n: u32,
    log_cpus: u32,
) {
    assert!(log_n >= log_cpus);

    let num_cpus = 1 << log_cpus;
    let log_new_n = log_n - log_cpus;
    let mut tmp = vec![vec![E::Fr::zero(); 1 << log_new_n]; num_cpus];
    let new_omega = omega.pow(&[num_cpus as u64]);

    worker.scope(0, |scope, _| {
        let a = &*a;

        for (j, tmp) in tmp.iter_mut().enumerate() {
            scope.spawn(move |_scope| {
                // Shuffle into a sub-FFT
                let omega_j = omega.pow(&[j as u64]);
                let omega_step = omega.pow(&[(j as u64) << log_new_n]);

                let mut elt = E::Fr::one();
                for (i, tmp) in tmp.iter_mut().enumerate() {
                    for s in 0..num_cpus {
                        let idx = (i + (s << log_new_n)) % (1 << log_n);
                        let mut t = a[idx];
                        t.mul_assign(&elt);
                        tmp.add_assign(&t);
                        elt.mul_assign(&omega_step);
                    }
                    elt.mul_assign(&omega_j);
                }

                // Perform sub-FFT
                serial_fft_fr::<E>(tmp, &new_omega, log_new_n);
            });
        }
    });

    worker.scope(a.len(), |scope, chunk| {
        let tmp = &tmp;

        for (idx, a) in a.chunks_mut(chunk).enumerate() {
            scope.spawn(move |_scope| {
                let mut idx = idx * chunk;
                let mask = (1 << log_cpus) - 1;
                for a in a {
                    *a = tmp[idx & mask][idx >> log_cpus];
                    idx += 1;
                }
            });
        }
    });
}

pub fn best_fft_recursive_gpu<E: Engine>(
    kern: &mut Option<LockedMultiFFTKernel<E>>,
    a: &mut [E::Fr],
    worker: &Worker,
    omega: &E::Fr,
    log_n: u32,
) -> EcResult<()> {
    if let Some(ref mut kern) = kern {
        let size = a.len();
        let mut copy_a: Vec<E::Fr> = Vec::with_capacity(size);
        unsafe { copy_a.set_len(size) };

        // let r = &mut copy_a[..] as *mut [E::Fr];
        // copy a
        let half_size = size / 2;
        let (left, right) = copy_a.split_at_mut(half_size);
        let a_ref = &a;

        use std::time::Instant;
        let now = Instant::now();
        worker.scope(half_size, |scope, chunk| {
            for (thread_idx, (left_chunk, right_chunk)) in left
                .chunks_mut(chunk)
                .zip(right.chunks_mut(chunk))
                .enumerate()
            {
                scope.spawn(move |_| {
                    for (idx, (left_element, right_element)) in left_chunk
                        .iter_mut()
                        .zip(right_chunk.iter_mut())
                        .enumerate()
                    {
                        let i = thread_idx * chunk + idx;
                        *left_element = a_ref[i * 2];
                        *right_element = a_ref[i * 2 + 1];
                    }
                });
            }
        });
        println!("extract vector taken {:?}", now.elapsed());

        if kern
            .with(|k: &mut FftKernel<E>| {
                let domain = Domain::<E::Fr>::new_for_size(half_size as u64).unwrap();
                let omega = domain.generator.inverse().unwrap();
                gpu_fft_multiple(k, &mut [left, right], &omega, domain.power_of_two as u32)
            })
            .is_ok()
        {
            println!("use ONE GPU");
            // distribute powers of in the second half of copy_a

            let now = Instant::now();
            distribute_powers::<E>(right, &worker, omega.clone());
            println!("distribute_powers taken {:?}", now.elapsed());

            let now = Instant::now();
            worker.scope(half_size, |scope, chunk| {
                for (thread_idx, (left_chunk, right_chunk)) in left
                    .chunks_mut(chunk)
                    .zip(right.chunks_mut(chunk))
                    .enumerate()
                {
                    scope.spawn(move |_| {
                        for (idx, (left_element, right_element)) in left_chunk
                            .iter_mut()
                            .zip(right_chunk.iter_mut())
                            .enumerate()
                        {
                            let tmp = left_element.clone();
                            left_element.add_assign(&right_element);
                            right_element.negate();
                            right_element.add_assign(&tmp);
                        }
                    });
                }
            });
            println!("butterfly taken {:?}", now.elapsed());

            // copy to a
            let r = &mut a[..] as *mut [E::Fr];
            worker.in_place_scope(size, |scope, chunk| {
                for (i, v) in copy_a.chunks(chunk).enumerate() {
                    let r = unsafe { &mut *r };
                    scope.spawn(move |_| {
                        let start = i * chunk;
                        let end = if start + chunk <= size {
                            start + chunk
                        } else {
                            size
                        };
                        let copy_start_pointer: *mut E::Fr = r[start..end].as_mut_ptr();

                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                v.as_ptr(),
                                copy_start_pointer,
                                end - start,
                            )
                        };
                    });
                }
            });

            return Ok(());
        }
    }
    println!("use CPU");
    let log_cpus = worker.log_num_cpus();
    if log_n <= log_cpus {
        serial_fft_fr::<E>(a, omega, log_n);
    } else {
        parallel_fft_fr::<E>(a, worker, omega, log_n, log_cpus);
    }

    Ok(())
}

// direction is true, do FFT, otherwise iFFT.
pub fn fft_parallel<E: Engine>(
    kern: &mut Option<LockedMultiFFTKernel<E>>,
    polys: &mut [&mut [E::Fr]],
    worker: &Worker,
    omega: &E::Fr,
    minv: &E::Fr,
    log_n: u32,
    direction: bool,
) -> EcResult<()> {
    let size = 1 << log_n;

    use std::time::Instant;
    let now = Instant::now();
    best_fft_multiple_gpu(kern, polys, worker, omega, log_n)?;
    println!("FFTs taken {:?}", now.elapsed());

    if !direction {
        let now = Instant::now();

        for poly in polys.iter_mut() {
            worker.scope(size, |scope, chunk| {
                for v in poly.chunks_mut(chunk) {
                    scope.spawn(move |_| {
                        for v in v {
                            v.mul_assign(&minv);
                        }
                    });
                }
            });
        }

        println!("MINVs taken {:?}", now.elapsed());
    }

    Ok(())
}

#[test]
fn test_best_fft_recursive_gpu_consistency() {
    use crate::pairing::ff::{Field, PrimeField};
    use crate::pairing::bn256::Fr;
    use crate::plonk::polynomials::{Polynomial, Values};
    use pairing::bn256::Bn256;

    let log_n = 8;
    let worker = Worker::new();

    println!("size = 2^{:?}", log_n);
    let max_size = 1 << log_n;
    let scalars1 =
        crate::kate_commitment::test::make_random_field_elements::<Fr>(&worker, max_size);
    let mut as_coeffs1: Polynomial<Bn256, Values> = Polynomial::from_values(scalars1).unwrap();
    let omegainv = as_coeffs1.omegainv.clone();
    let mut coeff1 = as_coeffs1.into_coeffs();

    let scalars2 =
        crate::kate_commitment::test::make_random_field_elements::<Fr>(&worker, max_size);
    let mut as_coeffs2: Polynomial<Bn256, Values> = Polynomial::from_values(scalars2).unwrap();
    let mut coeff2 = as_coeffs2.into_coeffs();

    let mut fft_kern = Some(LockedMultiFFTKernel::<Bn256>::new(log_n, false));

    best_fft_multiple_gpu(
        &mut fft_kern,
        &mut [&mut coeff1],
        &worker,
        &omegainv,
        log_n as u32,
    )
    .unwrap();
    best_fft_recursive_gpu(&mut fft_kern, &mut coeff2, &worker, &omegainv, log_n as u32).unwrap();
}
