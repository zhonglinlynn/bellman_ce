use crate::domain;
use crate::domain::distribute_powers as distribute_powers_new;
use crate::domain::{best_fft_multiple_gpu, best_fft_recursive_gpu};
use crate::pairing::ff::{Field, PrimeField};
use crate::pairing::Engine;
use crate::plonk::domains::*;
use crate::plonk::fft::with_precomputation;
use crate::plonk::fft::with_precomputation::FftPrecomputations;
use crate::plonk::fft::*;
use crate::worker::Worker;
use crate::{gpu, SynthesisError};


use crate::locks::LockedMultiFFTKernel;
use crate::locks::LockedMultiexpKernel;
// use ec_gpu_gen::EcError;
// use ec_gpu_gen::EcResult;

use crate::plonk::fft::cooley_tukey_ntt;
use crate::plonk::fft::cooley_tukey_ntt::partial_reduction;
use crate::plonk::fft::cooley_tukey_ntt::CTPrecomputations;

use crate::plonk::commitments::transparent::utils::log2_floor;
use crate::plonk::transparent_engine::PartialTwoBitReductionField;
use crate::plonk::utils::{fast_clone, fast_initialize_to_element};
use std::time::Instant;

pub trait PolynomialForm: Sized + Copy + Clone + Send {}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Coefficients {}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Values {}

impl PolynomialForm for Coefficients {}
impl PolynomialForm for Values {}

// TODO: Enforce bitreversed values as a separate form

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Polynomial<E: Engine, P: PolynomialForm> {
    coeffs: Vec<E::Fr>,
    pub exp: u32,
    pub omega: E::Fr,
    pub omegainv: E::Fr,
    pub geninv: E::Fr,
    pub minv: E::Fr,
    _marker: std::marker::PhantomData<P>,
}

impl<E: Engine, P: PolynomialForm> Polynomial<E, P> {
    pub fn size(&self) -> usize {
        self.coeffs.len()
    }

    pub fn as_ref(&self) -> &[E::Fr] {
        &self.coeffs
    }

    pub fn as_mut(&mut self) -> &mut [E::Fr] {
        &mut self.coeffs
    }

    pub fn into_coeffs(self) -> Vec<E::Fr> {
        self.coeffs
    }

    pub fn fast_clone(&self, worker: &Worker) -> Polynomial<E, P> {
        let size = self.size();
        let mut coeffs: Vec<E::Fr> = Vec::with_capacity(size);
        unsafe {
            coeffs.set_len(size);
        }

        fast_clone(self.as_ref(), &mut coeffs, worker);

        Polynomial {
            coeffs,
            exp: self.exp,
            omega: self.omega,
            omegainv: self.omegainv,
            geninv: self.geninv,
            minv: self.minv,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn distribute_powers(&mut self, worker: &Worker, g: E::Fr) {
        domain::distribute_powers::<E>(&mut self.coeffs, &worker, g);
    }

    pub fn reuse_allocation<PP: PolynomialForm>(&mut self, other: &Polynomial<E, PP>) {
        assert_eq!(self.coeffs.len(), other.coeffs.len());
        self.coeffs.copy_from_slice(&other.coeffs);
    }

    pub fn reuse_allocation_parallel<PP: PolynomialForm>(
        &mut self,
        worker: &Worker,
        other: &Polynomial<E, PP>,
    ) {
        let size = self.coeffs.len();
        assert_eq!(size, other.coeffs.len());

        let r = &mut self.coeffs[..] as *mut [E::Fr];
        worker.in_place_scope(size, |scope, chunk| {
            for (i, v) in other.coeffs.chunks(chunk).enumerate() {
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
                        std::ptr::copy_nonoverlapping(v.as_ptr(), copy_start_pointer, end - start)
                    };
                });
            }
        });
    }

    pub fn bitreverse_enumeration(&mut self, worker: &Worker) {
        let total_len = self.coeffs.len();
        let log_n = self.exp as usize;
        if total_len <= worker.get_num_cpus() {
            for j in 0..total_len {
                let rj = cooley_tukey_ntt::bitreverse(j, log_n);
                if j < rj {
                    self.coeffs.swap(j, rj);
                }
            }

            return;
        }

        let r = &mut self.coeffs[..] as *mut [E::Fr];

        let to_spawn = worker.get_num_cpus();

        let chunk = Worker::chunk_size_for_num_spawned_threads(total_len, to_spawn);

        // while it's unsafe we don't care cause swapping is always one to one

        worker.in_place_scope(0, |scope, _| {
            for thread_id in 0..to_spawn {
                let r = unsafe { &mut *r };
                scope.spawn(move |_| {
                    let start = thread_id * chunk;
                    let end = if start + chunk <= total_len {
                        start + chunk
                    } else {
                        total_len
                    };
                    for j in start..end {
                        let rj = cooley_tukey_ntt::bitreverse(j, log_n);
                        if j < rj {
                            r.swap(j, rj);
                        }
                    }
                });
            }
        });
    }

    pub fn bitreverse_enumeration_partial(
        &mut self,
        worker: &Worker,
        from: usize,
        total_len: usize,
    ) {
        assert!(from + total_len <= self.coeffs.len(), "to should <= len()");

        assert!(total_len.is_power_of_two(), "slice should be power of two");

        let log_n = log2_floor(total_len) as usize;
        if total_len <= worker.get_num_cpus() {
            for j in 0..total_len {
                let rj = cooley_tukey_ntt::bitreverse(j, log_n);
                if j < rj {
                    self.coeffs.swap(from + j, from + rj);
                }
            }

            return;
        }

        let r = &mut self.coeffs[from..from + total_len] as *mut [E::Fr];

        let to_spawn = worker.get_num_cpus();

        let chunk = Worker::chunk_size_for_num_spawned_threads(total_len, to_spawn);

        // while it's unsafe we don't care cause swapping is always one to one

        worker.in_place_scope(0, |scope, _| {
            for thread_id in 0..to_spawn {
                let r = unsafe { &mut *r };
                scope.spawn(move |_| {
                    let start = thread_id * chunk;
                    let end = if start + chunk <= total_len {
                        start + chunk
                    } else {
                        total_len
                    };
                    for j in start..end {
                        let rj = cooley_tukey_ntt::bitreverse(j, log_n);
                        if j < rj {
                            r.swap(j, rj);
                        }
                    }
                });
            }
        });
    }

    pub fn scale(&mut self, worker: &Worker, g: E::Fr) {
        if g == E::Fr::one() {
            return;
        }

        worker.scope(self.coeffs.len(), |scope, chunk| {
            for v in self.coeffs.chunks_mut(chunk) {
                scope.spawn(move |_| {
                    for v in v.iter_mut() {
                        v.mul_assign(&g);
                    }
                });
            }
        });
    }

    pub fn negate(&mut self, worker: &Worker) {
        worker.scope(self.coeffs.len(), |scope, chunk| {
            for v in self.coeffs.chunks_mut(chunk) {
                scope.spawn(move |_| {
                    for v in v.iter_mut() {
                        v.negate();
                    }
                });
            }
        });
    }

    pub fn map<M: Fn(&mut E::Fr) -> () + Send + Sync + Copy>(&mut self, worker: &Worker, func: M) {
        worker.scope(self.coeffs.len(), |scope, chunk| {
            for v in self.coeffs.chunks_mut(chunk) {
                scope.spawn(move |_| {
                    for v in v.iter_mut() {
                        func(v);
                    }
                });
            }
        });
    }

    pub fn map_indexed<M: Fn(usize, &mut E::Fr) -> () + Send + Sync + Copy>(
        &mut self,
        worker: &Worker,
        func: M,
    ) {
        worker.scope(self.coeffs.len(), |scope, chunk| {
            for (chunk_idx, v) in self.coeffs.chunks_mut(chunk).enumerate() {
                scope.spawn(move |_| {
                    let base = chunk_idx * chunk;
                    for (in_chunk_idx, v) in v.iter_mut().enumerate() {
                        func(base + in_chunk_idx, v);
                    }
                });
            }
        });
    }

    pub fn pad_by_factor(&mut self, factor: usize) -> Result<(), SynthesisError> {
        debug_assert!(self.coeffs.len().is_power_of_two());

        if factor == 1 {
            return Ok(());
        }
        let next_power_of_two = factor.next_power_of_two();
        if factor != next_power_of_two {
            return Err(SynthesisError::Unsatisfiable);
        }

        let new_size = self.coeffs.len() * factor;
        self.coeffs.resize(new_size, E::Fr::zero());

        let domain = Domain::new_for_size(new_size as u64)?;
        self.exp = domain.power_of_two as u32;
        let m = domain.size as usize;
        self.omega = domain.generator;
        self.omegainv = self.omega.inverse().unwrap();
        self.minv = E::Fr::from_str(&format!("{}", m))
            .unwrap()
            .inverse()
            .unwrap();

        Ok(())
    }

    pub fn pad_to_size(&mut self, new_size: usize) -> Result<(), SynthesisError> {
        if new_size < self.coeffs.len() {
            return Err(SynthesisError::PolynomialDegreeTooLarge);
        }
        let next_power_of_two = new_size.next_power_of_two();
        if new_size != next_power_of_two {
            return Err(SynthesisError::Unsatisfiable);
        }
        self.coeffs.resize(new_size, E::Fr::zero());

        let domain = Domain::new_for_size(new_size as u64)?;
        self.exp = domain.power_of_two as u32;
        let m = domain.size as usize;
        self.omega = domain.generator;
        self.omegainv = self.omega.inverse().unwrap();
        self.minv = E::Fr::from_str(&format!("{}", m))
            .unwrap()
            .inverse()
            .unwrap();

        Ok(())
    }

    pub fn pad_to_domain(&mut self) -> Result<(), SynthesisError> {
        let domain = Domain::<E::Fr>::new_for_size(self.coeffs.len() as u64)?;
        self.coeffs.resize(domain.size as usize, E::Fr::zero());

        Ok(())
    }

    pub fn clone_padded_to_domain(&self, worker: &Worker) -> Result<Self, SynthesisError> {
        let mut padded = self.fast_clone(worker);
        let domain = Domain::<E::Fr>::new_for_size(self.coeffs.len() as u64)?;
        padded.coeffs.resize(domain.size as usize, E::Fr::zero());

        Ok(padded)
    }

    pub fn trim_to_degree(&mut self, degree: usize) -> Result<(), SynthesisError> {
        let size = self.coeffs.len();
        if size <= degree + 1 {
            return Ok(());
        }
        self.coeffs.truncate(degree + 1);
        self.coeffs.resize(size, E::Fr::zero());

        Ok(())
    }
}

impl<E: Engine> Polynomial<E, Coefficients> {
    pub fn new_for_size(
        size: usize,
        worker: &Worker,
    ) -> Result<Polynomial<E, Coefficients>, SynthesisError> {
        let new_size = size.next_power_of_two();
        let mut coeffs = fast_initialize_to_element(new_size, E::Fr::zero(), worker);

        Self::from_coeffs(coeffs)
    }

    pub fn from_coeffs(
        mut coeffs: Vec<E::Fr>,
    ) -> Result<Polynomial<E, Coefficients>, SynthesisError> {
        let coeffs_len = coeffs.len();

        let domain = Domain::new_for_size(coeffs_len as u64)?;
        let exp = domain.power_of_two as u32;
        let m = domain.size as usize;
        let omega = domain.generator;

        coeffs.resize(m, E::Fr::zero());

        Ok(Polynomial::<E, Coefficients> {
            coeffs: coeffs,
            exp: exp,
            omega: omega,
            omegainv: omega.inverse().unwrap(),
            geninv: E::Fr::multiplicative_generator().inverse().unwrap(),
            minv: E::Fr::from_str(&format!("{}", m))
                .unwrap()
                .inverse()
                .unwrap(),
            _marker: std::marker::PhantomData,
        })
    }

    pub fn from_roots(
        roots: Vec<E::Fr>,
        worker: &Worker,
    ) -> Result<Polynomial<E, Coefficients>, SynthesisError> {
        let coeffs_len = roots.len() + 1;

        let domain = Domain::<E::Fr>::new_for_size(coeffs_len as u64)?;
        let num_threads = worker.get_num_cpus();

        // vector of vectors of polynomial coefficients for subproducts
        let mut subterms = vec![vec![]; num_threads];

        worker.scope(roots.len(), |scope, chunk| {
            for (r, s) in roots.chunks(chunk).zip(subterms.chunks_mut(1)) {
                scope.spawn(move |_| {
                    for r in r.iter() {
                        if s[0].len() == 0 {
                            let mut tmp = *r;
                            tmp.negate();
                            s[0] = vec![tmp, E::Fr::one()];
                        } else {
                            let mut tmp = Vec::with_capacity(s[0].len() + 1);
                            tmp.push(E::Fr::zero());
                            tmp.extend(s[0].clone());
                            for (c, n) in s[0].iter().zip(tmp.iter_mut()) {
                                let mut t = *c;
                                t.mul_assign(&r);
                                n.sub_assign(&t);
                            }
                            s[0] = tmp;
                        }
                    }
                });
            }
        });

        // now we have subproducts in a coefficient form

        let mut result: Option<Polynomial<E, Values>> = None;
        let result_len = domain.size as usize;

        for s in subterms.into_iter() {
            if s.len() == 0 {
                continue;
            }
            let t = Polynomial::<E, Coefficients>::from_coeffs(s)?;
            let factor = result_len / t.size();
            let t = t.lde(&worker, factor)?;
            if let Some(res) = result.as_mut() {
                res.mul_assign(&worker, &t);
            } else {
                result = Some(t);
            }
        }

        let result = result.expect("is some");
        let result = result.ifft(&worker, &mut None);

        Ok(result)
    }

    pub fn evaluate_at_domain_for_degree_one(
        &self,
        worker: &Worker,
        domain_size: u64,
    ) -> Result<Polynomial<E, Values>, SynthesisError> {
        assert_eq!(self.coeffs.len(), 2);
        let alpha = self.coeffs[1];
        let c = self.coeffs[0];

        let domain = Domain::<E::Fr>::new_for_size(domain_size)?;

        let mut result = vec![E::Fr::zero(); domain.size as usize];
        let g = domain.generator;
        worker.scope(result.len(), |scope, chunk| {
            for (i, v) in result.chunks_mut(chunk).enumerate() {
                scope.spawn(move |_| {
                    let mut u = g.pow(&[(i * chunk) as u64]);
                    for v in v.iter_mut() {
                        let mut tmp = alpha;
                        tmp.mul_assign(&u);
                        tmp.add_assign(&c);
                        *v = tmp;
                        u.mul_assign(&g);
                    }
                });
            }
        });

        Polynomial::from_values(result)
    }

    pub fn coset_evaluate_at_domain_for_degree_one(
        &self,
        worker: &Worker,
        domain_size: u64,
    ) -> Result<Polynomial<E, Values>, SynthesisError> {
        assert_eq!(self.coeffs.len(), 2);
        let alpha = self.coeffs[1];
        let c = self.coeffs[0];

        let domain = Domain::<E::Fr>::new_for_size(domain_size)?;

        let mut result = vec![E::Fr::zero(); domain.size as usize];
        let g = domain.generator;
        worker.scope(result.len(), |scope, chunk| {
            for (i, v) in result.chunks_mut(chunk).enumerate() {
                scope.spawn(move |_| {
                    let mut u = g.pow(&[(i * chunk) as u64]);
                    u.mul_assign(&E::Fr::multiplicative_generator());
                    for v in v.iter_mut() {
                        let mut tmp = alpha;
                        tmp.mul_assign(&u);
                        tmp.add_assign(&c);
                        *v = tmp;
                        u.mul_assign(&g);
                    }
                });
            }
        });

        Polynomial::from_values(result)
    }

    #[inline(always)]
    pub fn break_into_multiples(
        self,
        size: usize,
    ) -> Result<Vec<Polynomial<E, Coefficients>>, SynthesisError> {
        let mut coeffs = self.coeffs;

        let (mut num_parts, last_size) = if coeffs.len() % size == 0 {
            let num_parts = coeffs.len() / size;

            (num_parts, 0)
        } else {
            let num_parts = coeffs.len() / size;
            let last_size = coeffs.len() - num_parts * size;

            (num_parts, last_size)
        };

        let mut rev_results = Vec::with_capacity(num_parts);

        if last_size != 0 {
            let drain = coeffs.split_off(coeffs.len() - last_size);
            rev_results.push(drain);
            num_parts -= 1;
        }

        for _ in 0..num_parts {
            let drain = coeffs.split_off(coeffs.len() - size);
            rev_results.push(drain);
        }

        let mut results = Vec::with_capacity(num_parts);

        for c in rev_results.into_iter().rev() {
            let poly = Polynomial::<E, Coefficients>::from_coeffs(c)?;
            results.push(poly);
        }

        Ok(results)
    }

    #[inline(always)]
    pub fn lde(
        self,
        worker: &Worker,
        factor: usize,
    ) -> Result<Polynomial<E, Values>, SynthesisError> {
        self.lde_using_multiple_cosets(worker, factor)
        // self.filtering_lde(worker, factor)
    }

    #[inline(always)]
    pub fn coset_lde(
        self,
        worker: &Worker,
        factor: usize,
    ) -> Result<Polynomial<E, Values>, SynthesisError> {
        self.coset_lde_using_multiple_cosets(worker, factor)
        // self.filtering_coset_lde(worker, factor)
    }

    pub fn filtering_lde(
        self,
        worker: &Worker,
        factor: usize,
    ) -> Result<Polynomial<E, Values>, SynthesisError> {
        debug_assert!(self.coeffs.len().is_power_of_two());

        if factor == 1 {
            return Ok(self.fft(&worker, &mut None));
        }
        assert!(factor.is_power_of_two());
        let new_size = self.coeffs.len() * factor;
        let domain = Domain::new_for_size(new_size as u64)?;

        let mut lde = self.coeffs;
        lde.resize(new_size as usize, E::Fr::zero());
        best_lde(
            &mut lde,
            worker,
            &domain.generator,
            domain.power_of_two as u32,
            factor,
        );

        Polynomial::from_values(lde)
    }

    pub fn lde_using_multiple_cosets_naive(
        self,
        worker: &Worker,
        factor: usize,
    ) -> Result<Polynomial<E, Values>, SynthesisError> {
        debug_assert!(self.coeffs.len().is_power_of_two());

        if factor == 1 {
            return Ok(self.fft(&worker, &mut None));
        }
        assert!(factor.is_power_of_two());
        let new_size = self.coeffs.len() * factor;
        let domain = Domain::new_for_size(new_size as u64)?;

        let mut results = vec![];

        let mut coset_generator = E::Fr::one();

        let one = E::Fr::one();

        for _ in 0..factor {
            let coeffs = self.clone();
            let lde = if coset_generator == one {
                coeffs.fft(&worker, &mut None)
            } else {
                coeffs.coset_fft_for_generator(&worker, coset_generator)
            };

            results.push(lde.into_coeffs());
            coset_generator.mul_assign(&domain.generator);
        }

        let mut final_values = vec![E::Fr::zero(); new_size];

        let results_ref = &results;

        worker.scope(final_values.len(), |scope, chunk| {
            for (i, v) in final_values.chunks_mut(chunk).enumerate() {
                scope.spawn(move |_| {
                    let mut idx = i * chunk;
                    for v in v.iter_mut() {
                        let coset_idx = idx % factor;
                        let element_idx = idx / factor;
                        *v = results_ref[coset_idx][element_idx];

                        idx += 1;
                    }
                });
            }
        });

        Polynomial::from_values(final_values)
    }

    pub fn lde_using_multiple_cosets(
        self,
        worker: &Worker,
        factor: usize,
    ) -> Result<Polynomial<E, Values>, SynthesisError> {
        debug_assert!(self.coeffs.len().is_power_of_two());

        if factor == 1 {
            return Ok(self.fft(&worker, &mut None));
        }

        let num_cpus = worker.get_num_cpus();
        let num_cpus_hint = if num_cpus <= factor {
            Some(1)
        } else {
            let threads_per_coset = factor / num_cpus;
            // TODO: it's not immediately clean if having more threads than (virtual) cores benefits
            // over underutilization of some (virtual) cores
            // let mut threads_per_coset = factor / num_cpus;
            // if factor % num_cpus != 0 {
            //     if (threads_per_coset + 1).is_power_of_two() {
            //         threads_per_coset += 1;
            //     }
            // }
            Some(threads_per_coset)
        };

        assert!(factor.is_power_of_two());
        let new_size = self.coeffs.len() * factor;
        let domain = Domain::<E::Fr>::new_for_size(new_size as u64)?;

        let mut results = vec![self.coeffs; factor];

        let coset_omega = domain.generator;
        let this_domain_omega = self.omega;

        let log_n = self.exp;

        worker.scope(results.len(), |scope, chunk| {
            for (i, r) in results.chunks_mut(chunk).enumerate() {
                scope.spawn(move |_| {
                    let mut coset_generator = coset_omega.pow(&[i as u64]);
                    for r in r.iter_mut() {
                        if coset_generator != E::Fr::one() {
                            distribute_powers_serial(&mut r[..], coset_generator);
                            // distribute_powers(&mut r[..], &worker, coset_generator);
                        }
                        best_fft(
                            &mut r[..],
                            &worker,
                            &this_domain_omega,
                            log_n,
                            num_cpus_hint,
                        );
                        coset_generator.mul_assign(&coset_omega);
                    }
                });
            }
        });

        let mut final_values = vec![E::Fr::zero(); new_size];

        let results_ref = &results;

        worker.scope(final_values.len(), |scope, chunk| {
            for (i, v) in final_values.chunks_mut(chunk).enumerate() {
                scope.spawn(move |_| {
                    let mut idx = i * chunk;
                    for v in v.iter_mut() {
                        let coset_idx = idx % factor;
                        let element_idx = idx / factor;
                        *v = results_ref[coset_idx][element_idx];

                        idx += 1;
                    }
                });
            }
        });

        Polynomial::from_values(final_values)
    }

    pub fn lde_using_multiple_cosets_with_precomputation<P: FftPrecomputations<E::Fr>>(
        self,
        worker: &Worker,
        factor: usize,
        precomputed_omegas: &P,
    ) -> Result<Polynomial<E, Values>, SynthesisError> {
        debug_assert!(self.coeffs.len().is_power_of_two());
        debug_assert_eq!(self.size(), precomputed_omegas.domain_size());

        if factor == 1 {
            return Ok(self.fft(&worker, &mut None));
        }

        let num_cpus = worker.get_num_cpus();
        let num_cpus_hint = if num_cpus <= factor {
            Some(1)
        } else {
            let threads_per_coset = factor / num_cpus;
            // let mut threads_per_coset = factor / num_cpus;
            // if factor % num_cpus != 0 {
            //     if (threads_per_coset + 1).is_power_of_two() {
            //         threads_per_coset += 1;
            //     }
            // }
            Some(threads_per_coset)
        };

        assert!(factor.is_power_of_two());
        let new_size = self.coeffs.len() * factor;
        let domain = Domain::<E::Fr>::new_for_size(new_size as u64)?;

        let mut results = vec![self.coeffs; factor];

        let coset_omega = domain.generator;
        let this_domain_omega = self.omega;

        let log_n = self.exp;

        worker.scope(results.len(), |scope, chunk| {
            for (i, r) in results.chunks_mut(chunk).enumerate() {
                scope.spawn(move |_| {
                    let mut coset_generator = coset_omega.pow(&[i as u64]);
                    for r in r.iter_mut() {
                        distribute_powers(&mut r[..], &worker, coset_generator);
                        with_precomputation::fft::best_fft(
                            &mut r[..],
                            &worker,
                            &this_domain_omega,
                            log_n,
                            num_cpus_hint,
                            precomputed_omegas,
                        );
                        coset_generator.mul_assign(&coset_omega);
                    }
                });
            }
        });

        let mut final_values = vec![E::Fr::zero(); new_size];

        let results_ref = &results;

        worker.scope(final_values.len(), |scope, chunk| {
            for (i, v) in final_values.chunks_mut(chunk).enumerate() {
                scope.spawn(move |_| {
                    let mut idx = i * chunk;
                    for v in v.iter_mut() {
                        let coset_idx = idx % factor;
                        let element_idx = idx / factor;
                        *v = results_ref[coset_idx][element_idx];

                        idx += 1;
                    }
                });
            }
        });

        Polynomial::from_values(final_values)
    }

    pub fn lde_using_bitreversed_ntt<P: CTPrecomputations<E::Fr>>(
        self,
        worker: &Worker,
        factor: usize,
        precomputed_omegas: &P,
    ) -> Result<Polynomial<E, Values>, SynthesisError> {
        debug_assert!(self.coeffs.len().is_power_of_two());
        debug_assert_eq!(self.size(), precomputed_omegas.domain_size());

        if factor == 1 {
            return Ok(self.fft(&worker, &mut None));
        }

        let num_cpus = worker.get_num_cpus();
        let num_cpus_hint = if num_cpus <= factor {
            Some(1)
        } else {
            let threads_per_coset = factor / num_cpus;
            // let mut threads_per_coset = factor / num_cpus;
            // if factor % num_cpus != 0 {
            //     if (threads_per_coset + 1).is_power_of_two() {
            //         threads_per_coset += 1;
            //     }
            // }
            Some(threads_per_coset)
        };

        assert!(factor.is_power_of_two());
        let current_size = self.coeffs.len();
        let new_size = current_size * factor;
        let domain = Domain::<E::Fr>::new_for_size(new_size as u64)?;

        let mut results = vec![self.coeffs; factor];

        let coset_omega = domain.generator;

        let log_n_u32 = self.exp;
        let log_n = log_n_u32 as usize;

        // for r in results.iter_mut().skip(1) {
        //     let mut coset_generator = coset_omega;
        //     distribute_powers(&mut r[..], &worker, coset_generator);
        //     coset_generator.mul_assign(&coset_omega);
        // }

        worker.scope(results.len(), |scope, chunk| {
            for (i, r) in results.chunks_mut(chunk).enumerate() {
                scope.spawn(move |_| {
                    let mut coset_generator = coset_omega.pow(&[i as u64]);
                    let mut gen_power = i;
                    for r in r.iter_mut() {
                        if gen_power != 0 {
                            distribute_powers_serial(&mut r[..], coset_generator);
                        }
                        // distribute_powers(&mut r[..], &worker, coset_generator);
                        cooley_tukey_ntt::best_ct_ntt(
                            &mut r[..],
                            &worker,
                            log_n_u32,
                            num_cpus_hint,
                            precomputed_omegas,
                        );

                        coset_generator.mul_assign(&coset_omega);
                        gen_power += 1;
                    }
                });
            }
        });

        // let mut final_values = vec![E::Fr::zero(); new_size];

        let mut final_values = Vec::with_capacity(new_size);
        unsafe { final_values.set_len(new_size) };

        // copy here is more complicated: to have the value in a natural order
        // one has to use coset_idx to address the result element
        // and use bit-reversed lookup for an element index

        let results_ref = &results;

        worker.scope(final_values.len(), |scope, chunk| {
            for (i, v) in final_values.chunks_mut(chunk).enumerate() {
                scope.spawn(move |_| {
                    let mut idx = i * chunk;
                    for v in v.iter_mut() {
                        let coset_idx = idx % factor;
                        let element_idx = idx / factor;
                        let element_idx = cooley_tukey_ntt::bitreverse(element_idx, log_n);
                        *v = results_ref[coset_idx][element_idx];

                        idx += 1;
                    }
                });
            }
        });


        Polynomial::from_values(final_values)
    }

    pub fn coset_filtering_lde(
        mut self,
        worker: &Worker,
        factor: usize,
    ) -> Result<Polynomial<E, Values>, SynthesisError> {
        debug_assert!(self.coeffs.len().is_power_of_two());

        if factor == 1 {
            return Ok(self.coset_fft(&worker, &mut None));
        }
        assert!(factor.is_power_of_two());
        self.distribute_powers(worker, E::Fr::multiplicative_generator());

        let new_size = self.coeffs.len() * factor;
        let domain = Domain::new_for_size(new_size as u64)?;

        let mut lde = self.coeffs;
        lde.resize(new_size as usize, E::Fr::zero());
        best_lde(
            &mut lde,
            worker,
            &domain.generator,
            domain.power_of_two as u32,
            factor,
        );

        Polynomial::from_values(lde)
    }

    pub fn coset_lde_using_multiple_cosets_naive(
        self,
        worker: &Worker,
        factor: usize,
    ) -> Result<Polynomial<E, Values>, SynthesisError> {
        debug_assert!(self.coeffs.len().is_power_of_two());

        if factor == 1 {
            return Ok(self.coset_fft(&worker, &mut None));
        }
        assert!(factor.is_power_of_two());
        let new_size = self.coeffs.len() * factor;
        let domain = Domain::new_for_size(new_size as u64)?;

        let mut results = vec![];

        let mut coset_generator = E::Fr::multiplicative_generator();

        for _ in 0..factor {
            let coeffs = self.clone();
            let lde = coeffs.coset_fft_for_generator(&worker, coset_generator);

            results.push(lde.into_coeffs());
            coset_generator.mul_assign(&domain.generator);
        }

        let mut final_values = vec![E::Fr::zero(); new_size];

        let results_ref = &results;

        worker.scope(final_values.len(), |scope, chunk| {
            for (i, v) in final_values.chunks_mut(chunk).enumerate() {
                scope.spawn(move |_| {
                    let mut idx = i * chunk;
                    for v in v.iter_mut() {
                        let coset_idx = idx % factor;
                        let element_idx = idx / factor;
                        *v = results_ref[coset_idx][element_idx];

                        idx += 1;
                    }
                });
            }
        });

        Polynomial::from_values(final_values)
    }

    pub fn coset_lde_using_multiple_cosets(
        self,
        worker: &Worker,
        factor: usize,
    ) -> Result<Polynomial<E, Values>, SynthesisError> {
        debug_assert!(self.coeffs.len().is_power_of_two());

        if factor == 1 {
            return Ok(self.coset_fft(&worker, &mut None));
        }

        let num_cpus = worker.get_num_cpus();
        let num_cpus_hint = if num_cpus <= factor {
            Some(1)
        } else {
            let threads_per_coset = factor / num_cpus;
            // let mut threads_per_coset = factor / num_cpus;
            // if factor % num_cpus != 0 {
            //     if (threads_per_coset + 1).is_power_of_two() {
            //         threads_per_coset += 1;
            //     }
            // }
            Some(threads_per_coset)
        };

        assert!(factor.is_power_of_two());
        let new_size = self.coeffs.len() * factor;
        let domain = Domain::<E::Fr>::new_for_size(new_size as u64)?;

        let mut results = vec![self.coeffs; factor];

        let coset_omega = domain.generator;
        let this_domain_omega = self.omega;

        let log_n = self.exp;

        worker.scope(results.len(), |scope, chunk| {
            for (i, r) in results.chunks_mut(chunk).enumerate() {
                scope.spawn(move |_| {
                    let mut coset_generator = coset_omega.pow(&[i as u64]);
                    coset_generator.mul_assign(&E::Fr::multiplicative_generator());
                    for r in r.iter_mut() {
                        distribute_powers_serial(&mut r[..], coset_generator);
                        // distribute_powers(&mut r[..], &worker, coset_generator);
                        best_fft(
                            &mut r[..],
                            &worker,
                            &this_domain_omega,
                            log_n,
                            num_cpus_hint,
                        );
                        coset_generator.mul_assign(&coset_omega);
                    }
                });
            }
        });

        let mut final_values = vec![E::Fr::zero(); new_size];

        let results_ref = &results;

        worker.scope(final_values.len(), |scope, chunk| {
            for (i, v) in final_values.chunks_mut(chunk).enumerate() {
                scope.spawn(move |_| {
                    let mut idx = i * chunk;
                    for v in v.iter_mut() {
                        let coset_idx = idx % factor;
                        let element_idx = idx / factor;
                        *v = results_ref[coset_idx][element_idx];

                        idx += 1;
                    }
                });
            }
        });

        Polynomial::from_values(final_values)
    }

    pub fn fft(
        mut self,
        worker: &Worker,
        kern: &mut Option<LockedMultiFFTKernel<E>>,
    ) -> Polynomial<E, Values> {
        debug_assert!(self.coeffs.len().is_power_of_two());

        best_fft_multiple_gpu(kern, &mut [&mut self.coeffs], worker, &self.omega, self.exp)
            .unwrap();

        Polynomial::<E, Values> {
            coeffs: self.coeffs,
            exp: self.exp,
            omega: self.omega,
            omegainv: self.omegainv,
            geninv: self.geninv,
            minv: self.minv,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn fft_2(
        mut self,
        mut other: Polynomial<E, Coefficients>,
        worker: &Worker,
        kern: &mut Option<LockedMultiFFTKernel<E>>,
    ) -> (Polynomial<E, Values>, Polynomial<E, Values>) {
        debug_assert!(self.coeffs.len().is_power_of_two());

        let now = Instant::now();
        best_fft_multiple_gpu(
            kern,
            &mut [&mut self.coeffs, &mut other.coeffs],
            worker,
            &self.omega,
            self.exp,
        )
        .unwrap();
        println!("fft_2 taken {:?}", now.elapsed());

        (
            Polynomial::<E, Values> {
                coeffs: self.coeffs,
                exp: self.exp,
                omega: self.omega,
                omegainv: self.omegainv,
                geninv: self.geninv,
                minv: self.minv,
                _marker: std::marker::PhantomData,
            },
            Polynomial::<E, Values> {
                exp: other.exp,
                omega: other.omega,
                omegainv: other.omegainv,
                geninv: other.geninv,
                minv: other.minv,
                coeffs: other.into_coeffs(),
                _marker: std::marker::PhantomData,
            },
        )
    }

    pub fn coset_fft(
        mut self,
        worker: &Worker,
        kern: &mut Option<LockedMultiFFTKernel<E>>,
    ) -> Polynomial<E, Values> {
        debug_assert!(self.coeffs.len().is_power_of_two());
        self.distribute_powers(worker, E::Fr::multiplicative_generator());
        self.fft(worker, kern)
    }

    pub fn coset_fft_for_generator(mut self, worker: &Worker, gen: E::Fr) -> Polynomial<E, Values> {
        debug_assert!(self.coeffs.len().is_power_of_two());
        self.distribute_powers(worker, gen);
        self.fft(worker, &mut None)
    }

    pub fn add_assign(&mut self, worker: &Worker, other: &Polynomial<E, Coefficients>) {
        assert!(self.coeffs.len() >= other.coeffs.len());

        worker.scope(other.coeffs.len(), |scope, chunk| {
            for (a, b) in self
                .coeffs
                .chunks_mut(chunk)
                .zip(other.coeffs.chunks(chunk))
            {
                scope.spawn(move |_| {
                    for (a, b) in a.iter_mut().zip(b.iter()) {
                        a.add_assign(&b);
                    }
                });
            }
        });
    }

    pub fn add_assign_scaled(
        &mut self,
        worker: &Worker,
        other: &Polynomial<E, Coefficients>,
        scaling: &E::Fr,
    ) {
        assert!(self.coeffs.len() >= other.coeffs.len());

        worker.scope(other.coeffs.len(), |scope, chunk| {
            for (a, b) in self
                .coeffs
                .chunks_mut(chunk)
                .zip(other.coeffs.chunks(chunk))
            {
                scope.spawn(move |_| {
                    for (a, b) in a.iter_mut().zip(b.iter()) {
                        let mut tmp = *b;
                        tmp.mul_assign(&scaling);
                        a.add_assign(&tmp);
                    }
                });
            }
        });
    }

    /// Perform O(n) subtraction of one polynomial from another in the domain.
    pub fn sub_assign(&mut self, worker: &Worker, other: &Polynomial<E, Coefficients>) {
        assert!(self.coeffs.len() >= other.coeffs.len());

        worker.scope(other.coeffs.len(), |scope, chunk| {
            for (a, b) in self
                .coeffs
                .chunks_mut(chunk)
                .zip(other.coeffs.chunks(chunk))
            {
                scope.spawn(move |_| {
                    for (a, b) in a.iter_mut().zip(b.iter()) {
                        a.sub_assign(&b);
                    }
                });
            }
        });
    }

    pub fn sub_assign_scaled(
        &mut self,
        worker: &Worker,
        other: &Polynomial<E, Coefficients>,
        scaling: &E::Fr,
    ) {
        assert!(self.coeffs.len() >= other.coeffs.len());

        worker.scope(other.coeffs.len(), |scope, chunk| {
            for (a, b) in self
                .coeffs
                .chunks_mut(chunk)
                .zip(other.coeffs.chunks(chunk))
            {
                scope.spawn(move |_| {
                    for (a, b) in a.iter_mut().zip(b.iter()) {
                        let mut tmp = *b;
                        tmp.mul_assign(&scaling);
                        a.sub_assign(&tmp);
                    }
                });
            }
        });
    }

    pub fn evaluate_at(&self, worker: &Worker, g: E::Fr) -> E::Fr {
        let num_threads = worker.get_num_spawned_threads(self.coeffs.len());
        let mut subvalues = vec![E::Fr::zero(); num_threads];

        worker.scope(self.coeffs.len(), |scope, chunk| {
            for (i, (a, s)) in self
                .coeffs
                .chunks(chunk)
                .zip(subvalues.chunks_mut(1))
                .enumerate()
            {
                scope.spawn(move |_| {
                    let mut x = g.pow([(i * chunk) as u64]);
                    for a in a.iter() {
                        let mut value = x;
                        value.mul_assign(&a);
                        s[0].add_assign(&value);
                        x.mul_assign(&g);
                    }
                });
            }
        });

        let mut result = E::Fr::zero();
        for v in subvalues.iter() {
            result.add_assign(&v);
        }

        result
    }

    pub fn from_coeffs_unpadded_and_domain(
        coeffs: Vec<E::Fr>,
        exp: u32,
        omega: E::Fr,
        omegainv: E::Fr,
        geninv: E::Fr,
        minv: E::Fr,
    ) -> Result<Polynomial<E, Coefficients>, SynthesisError> {
        Ok(Polynomial::<E, Coefficients> {
            coeffs,
            exp,
            omega,
            omegainv,
            geninv,
            minv,
            _marker: std::marker::PhantomData,
        })
    }
}

impl<E: Engine> Polynomial<E, Values> {
    pub fn default() -> Polynomial<E, Values> {
        Polynomial::<E, Values> {
            coeffs: vec![],
            exp: 0,
            omega: E::Fr::zero(),
            omegainv: E::Fr::zero(),
            geninv: E::Fr::zero(),
            minv: E::Fr::zero(),
            _marker: std::marker::PhantomData,
        }
    }
    pub fn new_for_size(
        size: usize,
        worker: &Worker,
    ) -> Result<Polynomial<E, Values>, SynthesisError> {
        let new_size = size.next_power_of_two();
        let mut coeffs = fast_initialize_to_element(new_size, E::Fr::zero(), worker);
        Self::from_values(coeffs)
    }

    pub fn from_values(mut values: Vec<E::Fr>) -> Result<Polynomial<E, Values>, SynthesisError> {
        let coeffs_len = values.len();

        let domain = Domain::new_for_size(coeffs_len as u64)?;
        let exp = domain.power_of_two as u32;
        let m = domain.size as usize;
        let omega = domain.generator;

        values.resize(m, E::Fr::zero());

        Ok(Polynomial::<E, Values> {
            coeffs: values,
            exp: exp,
            omega: omega,
            omegainv: omega.inverse().unwrap(),
            geninv: E::Fr::multiplicative_generator().inverse().unwrap(),
            minv: E::Fr::from_str(&format!("{}", m))
                .unwrap()
                .inverse()
                .unwrap(),
            _marker: std::marker::PhantomData,
        })
    }

    pub fn from_coeffs_and_domain(
        mut coeffs: Vec<E::Fr>,
        exp: u32,
        omega: &E::Fr,
        omegainv: &E::Fr,
        geninv: &E::Fr,
        minv: &E::Fr,
    ) -> Polynomial<E, Coefficients> {
        let coeffs_len = coeffs.len();
        let m = 1 << exp;
        coeffs.resize(m, E::Fr::zero());

        Polynomial::<E, Coefficients> {
            coeffs: coeffs,
            exp: exp.clone(),
            omega: omega.clone(),
            omegainv: omegainv.clone(),
            geninv: geninv.clone(),
            minv: minv.clone(),
            _marker: std::marker::PhantomData,
        }
    }

    pub fn from_values_unpadded(
        values: Vec<E::Fr>,
    ) -> Result<Polynomial<E, Values>, SynthesisError> {
        let coeffs_len = values.len();

        let domain = Domain::new_for_size(coeffs_len as u64)?;
        let exp = domain.power_of_two as u32;
        let m = domain.size as usize;
        let omega = domain.generator;

        Ok(Polynomial::<E, Values> {
            coeffs: values,
            exp: exp,
            omega: omega,
            omegainv: omega.inverse().unwrap(),
            geninv: E::Fr::multiplicative_generator().inverse().unwrap(),
            minv: E::Fr::from_str(&format!("{}", m))
                .unwrap()
                .inverse()
                .unwrap(),
            _marker: std::marker::PhantomData,
        })
    }

    pub fn from_values_unpadded_and_domain(
        values: Vec<E::Fr>,
        exp: u32,
        omega: E::Fr,
        omegainv: E::Fr,
        geninv: E::Fr,
        minv: E::Fr,
    ) -> Result<Polynomial<E, Values>, SynthesisError> {
        Ok(Polynomial::<E, Values> {
            coeffs: values,
            exp,
            omega,
            omegainv,
            geninv,
            minv,
            _marker: std::marker::PhantomData,
        })
    }

    // this function should only be used on the values that are bitreverse enumerated
    pub fn clone_subset_assuming_bitreversed(
        &self,
        subset_factor: usize,
    ) -> Result<Polynomial<E, Values>, SynthesisError> {
        if subset_factor == 1 {
            return Ok(self.clone());
        }

        assert!(subset_factor.is_power_of_two());

        let current_size = self.coeffs.len();
        let new_size = current_size / subset_factor;

        let mut result = Vec::with_capacity(new_size);
        unsafe { result.set_len(new_size) };

        // copy elements. If factor is 2 then non-reversed we would output only elements that are == 0 mod 2
        // If factor is 2 and we are bit-reversed - we need to only output first half of the coefficients
        // If factor is 4 then we need to output only the first 4th part
        // if factor is 8 - only the first 8th part

        let start = 0;
        let end = new_size;

        result.copy_from_slice(&self.coeffs[start..end]);

        // unsafe { result.set_len(new_size)};
        // let copy_to_start_pointer: *mut E::Fr = result[..].as_mut_ptr();
        // let copy_from_start_pointer: *const E::Fr = self.coeffs[start..end].as_ptr();

        // unsafe { std::ptr::copy_nonoverlapping(copy_from_start_pointer, copy_to_start_pointer, new_size) };

        Polynomial::from_values(result)
    }

    pub fn pow(&mut self, worker: &Worker, exp: u64) {
        if exp == 2 {
            return self.square(&worker);
        }
        worker.scope(self.coeffs.len(), |scope, chunk| {
            for v in self.coeffs.chunks_mut(chunk) {
                scope.spawn(move |_| {
                    for v in v.iter_mut() {
                        *v = v.pow([exp]);
                    }
                });
            }
        });
    }

    pub fn square(&mut self, worker: &Worker) {
        worker.scope(self.coeffs.len(), |scope, chunk| {
            for v in self.coeffs.chunks_mut(chunk) {
                scope.spawn(move |_| {
                    for v in v.iter_mut() {
                        v.square();
                    }
                });
            }
        });
    }

    pub fn ifft(
        mut self,
        worker: &Worker,
        kern: &mut Option<LockedMultiFFTKernel<E>>,
    ) -> Polynomial<E, Coefficients> {
        debug_assert!(self.coeffs.len().is_power_of_two());

        if self.exp != 28 {
            best_fft_multiple_gpu(
                kern,
                &mut [&mut self.coeffs],
                worker,
                &self.omegainv,
                self.exp,
            )
            .unwrap();
        } else {
            best_fft_recursive_gpu(kern, &mut self.coeffs, worker, &self.omegainv, self.exp)
                .unwrap();
        }

        worker.scope(self.coeffs.len(), |scope, chunk| {
            let minv = self.minv;

            for v in self.coeffs.chunks_mut(chunk) {
                scope.spawn(move |_| {
                    for v in v {
                        v.mul_assign(&minv);
                    }
                });
            }
        });

        Polynomial::<E, Coefficients> {
            coeffs: self.coeffs,
            exp: self.exp,
            omega: self.omega,
            omegainv: self.omegainv,
            geninv: self.geninv,
            minv: self.minv,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn ifft_2(
        mut self,
        mut other: Polynomial<E, Values>,
        worker: &Worker,
        kern: &mut Option<LockedMultiFFTKernel<E>>,
    ) -> (Polynomial<E, Coefficients>, Polynomial<E, Coefficients>) {
        debug_assert!(self.coeffs.len().is_power_of_two());

        let now = Instant::now();
        best_fft_multiple_gpu(
            kern,
            &mut [&mut self.coeffs, &mut other.coeffs],
            worker,
            &self.omegainv,
            self.exp,
        )
        .unwrap();
        println!("ifft_2 taken {:?}", now.elapsed());

        let now = Instant::now();
        worker.scope(self.coeffs.len(), |scope, chunk| {
            let minv = self.minv;

            for v in self.coeffs.chunks_mut(chunk) {
                scope.spawn(move |_| {
                    for v in v {
                        v.mul_assign(&minv);
                    }
                });
            }
        });

        worker.scope(other.coeffs.len(), |scope, chunk| {
            let minv = other.minv;

            for v in other.coeffs.chunks_mut(chunk) {
                scope.spawn(move |_| {
                    for v in v {
                        v.mul_assign(&minv);
                    }
                });
            }
        });
        println!("minv_2 taken {:?}", now.elapsed());

        (
            Polynomial::<E, Coefficients> {
                coeffs: self.coeffs,
                exp: self.exp,
                omega: self.omega,
                omegainv: self.omegainv,
                geninv: self.geninv,
                minv: self.minv,
                _marker: std::marker::PhantomData,
            },
            Polynomial::<E, Coefficients> {
                exp: other.exp,
                omega: other.omega,
                omegainv: other.omegainv,
                geninv: other.geninv,
                minv: other.minv,
                coeffs: other.into_coeffs(),
                _marker: std::marker::PhantomData,
            },
        )
    }

    pub fn icoset_fft(
        self,
        worker: &Worker,
        kern: &mut Option<LockedMultiFFTKernel<E>>,
    ) -> Polynomial<E, Coefficients> {
        debug_assert!(self.coeffs.len().is_power_of_two());
        let geninv = self.geninv;
        let mut res = self.ifft(worker, kern);
        res.distribute_powers(worker, geninv);

        res
    }

    pub fn icoset_fft_for_generator(
        self,
        worker: &Worker,
        coset_generator: &E::Fr,
        kern: &mut Option<LockedMultiFFTKernel<E>>,
    ) -> Polynomial<E, Coefficients> {
        debug_assert!(self.coeffs.len().is_power_of_two());
        let geninv = coset_generator.inverse().expect("must exist");
        let mut res = self.ifft(worker, kern);
        res.distribute_powers(worker, geninv);

        res
    }

    pub fn add_assign(&mut self, worker: &Worker, other: &Polynomial<E, Values>) {
        assert_eq!(self.coeffs.len(), other.coeffs.len());

        worker.scope(self.coeffs.len(), |scope, chunk| {
            for (a, b) in self
                .coeffs
                .chunks_mut(chunk)
                .zip(other.coeffs.chunks(chunk))
            {
                scope.spawn(move |_| {
                    for (a, b) in a.iter_mut().zip(b.iter()) {
                        a.add_assign(&b);
                    }
                });
            }
        });
    }

    pub fn add_constant(&mut self, worker: &Worker, constant: &E::Fr) {
        worker.scope(self.coeffs.len(), |scope, chunk| {
            for a in self.coeffs.chunks_mut(chunk) {
                scope.spawn(move |_| {
                    for a in a.iter_mut() {
                        a.add_assign(&constant);
                    }
                });
            }
        });
    }

    pub fn sub_constant(&mut self, worker: &Worker, constant: &E::Fr) {
        worker.scope(self.coeffs.len(), |scope, chunk| {
            for a in self.coeffs.chunks_mut(chunk) {
                scope.spawn(move |_| {
                    for a in a.iter_mut() {
                        a.sub_assign(&constant);
                    }
                });
            }
        });
    }

    pub fn rotate(mut self, by: usize) -> Result<Polynomial<E, Values>, SynthesisError> {
        let mut values: Vec<_> = self.coeffs.drain(by..).collect();

        for c in self.coeffs.into_iter() {
            values.push(c);
        }

        Polynomial::from_values(values)
    }

    pub fn barycentric_evaluate_at(
        &self,
        worker: &Worker,
        g: E::Fr,
    ) -> Result<E::Fr, SynthesisError> {
        // use a barycentric formula

        // L_i(X) = (omega^i / N) / (X - omega^i) * (X^N - 1)
        // we'll have to pay more for batch inversion at some point, but
        // it's still useful
        let domain_size = self.size() as u64;
        assert!(domain_size.is_power_of_two());

        let mut vanishing_at_g = g.pow(&[domain_size]);
        vanishing_at_g.sub_assign(&E::Fr::one());

        // now generate (X - omega^i)
        let mut tmp = vec![g; domain_size as usize];

        let generator = self.omega;

        // constant factor = 1 / ( (1 / N) * (X^N - 1) ) = N / (X^N - 1)

        worker.scope(tmp.len(), |scope, chunk| {
            for (i, vals) in tmp.chunks_mut(chunk).enumerate() {
                scope.spawn(move |_| {
                    // let mut one_over_omega_pow = generator_inv.pow([(i*chunk) as u64]);
                    // one_over_omega_pow.mul_assign(&constant_factor);
                    let mut omega_power = generator.pow([(i * chunk) as u64]);
                    for val in vals.iter_mut() {
                        val.sub_assign(&omega_power); // (X - omega^i)
                                                      // val.mul_assign(&one_over_omega_pow); // (X - omega^i) * N / (X^N - 1) * omega^(-i), so when we inverse it's valid evaluation
                        omega_power.mul_assign(&generator);
                        // one_over_omega_pow.mul_assign(&generator_inv);
                    }
                });
            }
        });

        let mut values: Polynomial<E, Values> = Polynomial::from_values(tmp)?;
        values.batch_inversion(&worker)?;

        // now multiply by omega^i / N * (X^N - 1) and value for L_i(X)

        let mut constant_factor = vanishing_at_g;
        constant_factor.mul_assign(&self.minv);

        worker.scope(values.size(), |scope, chunk| {
            for (i, (vals, coeffs)) in values
                .as_mut()
                .chunks_mut(chunk)
                .zip(self.coeffs.chunks(chunk))
                .enumerate()
            {
                scope.spawn(move |_| {
                    let mut omega_power = generator.pow([(i * chunk) as u64]);
                    omega_power.mul_assign(&constant_factor);
                    for (val, coeff) in vals.iter_mut().zip(coeffs.iter()) {
                        val.mul_assign(&omega_power);
                        val.mul_assign(coeff);
                        omega_power.mul_assign(&generator);
                    }
                });
            }
        });

        values.calculate_sum(&worker)
    }

    pub fn barycentric_over_coset_evaluate_at(
        &self,
        worker: &Worker,
        x: E::Fr,
        coset_factor: &E::Fr,
    ) -> Result<E::Fr, SynthesisError> {
        // use a barycentric formula
        // L_i(x) = \prod_{i != j} (X - x_j) / \prod_{i != j} (x_i - x_j)
        // that for a case when x_j = g*omega^j is simplified

        // \prod_{i != j} (X - x_j) = (X^N - g^N) / (X - g * omega^i)

        // \prod_{i != j} (x_i - x_j) = g * (omega)^i / N

        // L_i(X) = (g*(omega)^i / N) / (X - g*(omega)^i) * (X^N - g^N)
        // we'll have to pay more for batch inversion at some point, but
        // it's still useful
        let domain_size = self.size() as u64;
        assert!(domain_size.is_power_of_two());

        // let normalization_factor = ..pow(&[domain_size]);

        let offset = coset_factor.pow(&[domain_size]);

        let normalization_factor = offset.inverse().ok_or(SynthesisError::DivisionByZero)?;

        let mut vanishing_at_x = x.pow(&[domain_size]);
        vanishing_at_x.sub_assign(&offset);

        // now generate (X - g*omega^i)
        let mut tmp = vec![x; domain_size as usize];

        let generator = self.omega;

        // constant factor = 1 / ( (1 / N) * (X^N - g^N) ) = N / (X^N - g^N)

        worker.scope(tmp.len(), |scope, chunk| {
            for (i, vals) in tmp.chunks_mut(chunk).enumerate() {
                scope.spawn(move |_| {
                    let mut omega_power = generator.pow([(i * chunk) as u64]);
                    omega_power.mul_assign(&coset_factor);
                    for val in vals.iter_mut() {
                        val.sub_assign(&omega_power); // (X - omega^i)
                        omega_power.mul_assign(&generator);
                    }
                });
            }
        });

        let mut values: Polynomial<E, Values> = Polynomial::from_values(tmp)?;
        values.batch_inversion(&worker)?;

        // now multiply by g*omega^i / N * (X^N - g^N) and value for L_i(X)

        let mut constant_factor = vanishing_at_x;
        constant_factor.mul_assign(&self.minv);
        constant_factor.mul_assign(&coset_factor);
        constant_factor.mul_assign(&normalization_factor);

        worker.scope(values.size(), |scope, chunk| {
            for (i, (vals, coeffs)) in values
                .as_mut()
                .chunks_mut(chunk)
                .zip(self.coeffs.chunks(chunk))
                .enumerate()
            {
                scope.spawn(move |_| {
                    let mut omega_power = generator.pow([(i * chunk) as u64]);
                    omega_power.mul_assign(&constant_factor);
                    for (val, coeff) in vals.iter_mut().zip(coeffs.iter()) {
                        val.mul_assign(&omega_power);
                        val.mul_assign(coeff);
                        omega_power.mul_assign(&generator);
                    }
                });
            }
        });

        values.calculate_sum(&worker)
    }

    pub fn split_into_even_and_odd_assuming_natural_ordering(
        self,
        worker: &Worker,
        coset_offset: &E::Fr,
    ) -> Result<(Polynomial<E, Values>, Polynomial<E, Values>), SynthesisError> {
        // Classical trick: E::Fr(x) = f_even(X^2) + x * f_odd(X^2)

        // E::Fr(g) = c_0 + c_1 * g + c_2 * g + c_3 * g
        // E::Fr(-g) = c_0 - c_1 * g + c_2 * g - c_3 * g
        // f_even(g) = c_0 + c_2 * g + ...
        // f_odd(g) = c_1 * g + c_3 * g + ...

        // E::Fr(g*Omega) = c_0 + c_1 * g * Omega + c_2 * g * Omega^2 + c_3 * g * Omega^3
        // E::Fr(-g*Omega) = c_0 - c_1 * g * Omega + c_2 * g * Omega^2 - c_3 * g * Omega^3

        // what should be
        // f_even(g*Omega^2) = c_0 + c_2 * g*Omega^2 + ...
        // f_odd(g*Omega^2/g) * g*Omega = c_1 * g * Omega + c_3 * g * Omega^3 + ...

        // (E::Fr(g*Omega) + E::Fr(-g*Omega))/2 = c_0 + c_2 * g*Omega^2 + ... - those are values of the even coefficient polynomial at X^2/g
        // (E::Fr(g*Omega) - E::Fr(-g*Omega))/2 / (g * Omega) = c_1 + c_3 * Omega^2 + ... - those are values of the even coefficient polynomial at X^2/g^2

        // to make it homogenius (cause we don't care about particular coefficients)
        // we make it as
        // (E::Fr(g*Omega) + E::Fr(-g*Omega))/2 / g = c_0/g + c_2 * Omega^2 - values for some polynomial over (X^2 / g^2)
        // (E::Fr(g*Omega) - E::Fr(-g*Omega))/2 / (g * Omega) = c_1 + c_3 * Omega^2 - values for some polynomial over (X^2 / g^2)
        assert!(self.coeffs.len().is_power_of_two());
        assert!(self.coeffs.len() > 1);

        let result_len = self.coeffs.len() / 2;

        let mut coeffs = self.coeffs;

        let mut second: Vec<_> = coeffs.drain(result_len..(2 * result_len)).collect();
        let mut first = coeffs;

        let generator_inv = self.omegainv;

        let mut two = E::Fr::one();
        two.double();

        let coset_offset_inv = coset_offset
            .inverse()
            .ok_or(SynthesisError::DivisionByZero)?;

        let two_inv = two.inverse().ok_or(SynthesisError::DivisionByZero)?;

        let mut constant_factor = two_inv;
        constant_factor.mul_assign(&coset_offset_inv);

        let divisor_even = two_inv;
        // let divisor_even = constant_factor;

        // f_even(X^2) = (E::Fr(x) + E::Fr(-x))/ 2
        // f_odd(X^2) = (E::Fr(x) - E::Fr(-x))/ 2x

        worker.scope(first.len(), |scope, chunk| {
            for (i, (first, second)) in first
                .chunks_mut(chunk)
                .zip(second.chunks_mut(chunk))
                .enumerate()
            {
                scope.spawn(move |_| {
                    let mut divisor_odd = generator_inv.pow([(i * chunk) as u64]);
                    divisor_odd.mul_assign(&constant_factor);
                    for (f, s) in first.iter_mut().zip(second.iter_mut()) {
                        let f_at_x = *f;
                        let f_at_minus_x = *s;

                        let mut even = f_at_x;
                        even.add_assign(&f_at_minus_x);
                        even.mul_assign(&divisor_even);

                        let mut odd = f_at_x;
                        odd.sub_assign(&f_at_minus_x);
                        odd.mul_assign(&divisor_odd);

                        *f = even;
                        *s = odd;

                        divisor_odd.mul_assign(&generator_inv);
                    }
                });
            }
        });

        let even = Polynomial::from_values(first)?;
        let odd = Polynomial::from_values(second)?;

        Ok((even, odd))
    }

    pub fn calculate_shifted_grand_product(
        &self,
        worker: &Worker,
    ) -> Result<Polynomial<E, Values>, SynthesisError> {
        // let mut result = vec![E::Fr::zero(); self.coeffs.len() + 1];
        let mut result: Vec<E::Fr> = Vec::with_capacity(self.size() + 1);
        unsafe { result.set_len(self.size() + 1) };

        result[0] = E::Fr::one();

        // let not_shifted_product = self.calculate_grand_product(&worker)?;
        // result[1..].copy_from_slice(&not_shifted_product.into_coeffs()[..]);

        // Polynomial::from_values_unpadded(result)

        let work_chunk = &mut result[1..];
        assert!(work_chunk.len() == self.coeffs.len());

        let num_threads = worker.get_num_spawned_threads(self.coeffs.len());
        let mut subproducts = vec![E::Fr::one(); num_threads as usize];

        worker.scope(self.coeffs.len(), |scope, chunk| {
            for ((g, c), s) in work_chunk
                .chunks_mut(chunk)
                .zip(self.coeffs.chunks(chunk))
                .zip(subproducts.chunks_mut(1))
            {
                scope.spawn(move |_| {
                    for (g, c) in g.iter_mut().zip(c.iter()) {
                        s[0].mul_assign(&c);
                        *g = s[0];
                    }
                });
            }
        });

        // subproducts are [abc, def, xyz]

        // we do not need the last one
        subproducts.pop().expect("has at least one value");

        let mut tmp = E::Fr::one();
        for s in subproducts.iter_mut() {
            tmp.mul_assign(&s);
            *s = tmp;
        }

        let chunk_len = worker.get_chunk_size(self.coeffs.len());

        worker.scope(0, |scope, _| {
            for (g, s) in work_chunk[chunk_len..]
                .chunks_mut(chunk_len)
                .zip(subproducts.chunks(1))
            {
                scope.spawn(move |_| {
                    for g in g.iter_mut() {
                        g.mul_assign(&s[0]);
                    }
                });
            }
        });

        Polynomial::from_values(result)
    }

    pub fn calculate_grand_product(
        &self,
        worker: &Worker,
    ) -> Result<Polynomial<E, Values>, SynthesisError> {
        let mut result = vec![E::Fr::zero(); self.coeffs.len()];

        let num_threads = worker.get_num_spawned_threads(self.coeffs.len());
        let mut subproducts = vec![E::Fr::one(); num_threads as usize];

        worker.scope(self.coeffs.len(), |scope, chunk| {
            for ((g, c), s) in result
                .chunks_mut(chunk)
                .zip(self.coeffs.chunks(chunk))
                .zip(subproducts.chunks_mut(1))
            {
                scope.spawn(move |_| {
                    for (g, c) in g.iter_mut().zip(c.iter()) {
                        s[0].mul_assign(&c);
                        *g = s[0];
                    }
                });
            }
        });

        // subproducts are [abc, def, xyz]

        // we do not need the last one
        subproducts.pop().expect("has at least one value");

        let mut tmp = E::Fr::one();
        for s in subproducts.iter_mut() {
            tmp.mul_assign(&s);
            *s = tmp;
        }

        let chunk_len = worker.get_chunk_size(self.coeffs.len());

        worker.scope(0, |scope, _| {
            for (g, s) in result[chunk_len..]
                .chunks_mut(chunk_len)
                .zip(subproducts.chunks(1))
            {
                scope.spawn(move |_| {
                    let c = s[0];
                    for g in g.iter_mut() {
                        g.mul_assign(&c);
                    }
                });
            }
        });

        Polynomial::from_values_unpadded(result)
    }

    pub fn calculate_grand_product_serial(&self) -> Result<Polynomial<E, Values>, SynthesisError> {
        let mut result = Vec::with_capacity(self.coeffs.len());

        let mut tmp = E::Fr::one();
        for c in self.coeffs.iter() {
            tmp.mul_assign(&c);
            result.push(tmp);
        }

        Polynomial::from_values_unpadded(result)
    }

    pub fn calculate_sum(&self, worker: &Worker) -> Result<E::Fr, SynthesisError> {
        let num_threads = worker.get_num_spawned_threads(self.coeffs.len());
        let mut subresults = vec![E::Fr::zero(); num_threads as usize];

        worker.scope(self.coeffs.len(), |scope, chunk| {
            for (c, s) in self.coeffs.chunks(chunk).zip(subresults.chunks_mut(1)) {
                scope.spawn(move |_| {
                    for c in c.iter() {
                        s[0].add_assign(&c);
                    }
                });
            }
        });

        let mut sum = E::Fr::zero();

        for el in subresults.iter() {
            sum.add_assign(&el);
        }

        Ok(sum)
    }

    pub fn calculate_grand_sum(
        &self,
        worker: &Worker,
    ) -> Result<(E::Fr, Polynomial<E, Values>), SynthesisError> {
        // first value is zero, then first element, then first + second, ...
        let mut result = vec![E::Fr::zero(); self.coeffs.len() + 1];

        let num_threads = worker.get_num_spawned_threads(self.coeffs.len());
        let mut subsums = vec![E::Fr::zero(); num_threads as usize];

        worker.scope(self.coeffs.len(), |scope, chunk| {
            for ((g, c), s) in result[1..]
                .chunks_mut(chunk)
                .zip(self.coeffs.chunks(chunk))
                .zip(subsums.chunks_mut(1))
            {
                scope.spawn(move |_| {
                    for (g, c) in g.iter_mut().zip(c.iter()) {
                        s[0].add_assign(&c);
                        *g = s[0];
                    }
                });
            }
        });

        // subsums are [a+b+c, d+e+E::Fr, x+y+z]

        let mut tmp = E::Fr::zero();
        for s in subsums.iter_mut() {
            tmp.add_assign(&s);
            *s = tmp;
        }

        // sum over the full domain is the last element
        let domain_sum = subsums.pop().expect("has at least one value");

        let chunk_len = worker.get_chunk_size(self.coeffs.len());

        worker.scope(0, |scope, _| {
            for (g, s) in result[(chunk_len + 1)..]
                .chunks_mut(chunk_len)
                .zip(subsums.chunks(1))
            {
                scope.spawn(move |_| {
                    let c = s[0];
                    for g in g.iter_mut() {
                        g.add_assign(&c);
                    }
                });
            }
        });

        // let result = result.drain(1..).collect();

        let alt_total_sum = result.pop().expect("must pop the last element");

        assert_eq!(alt_total_sum, domain_sum);

        Ok((domain_sum, Polynomial::from_values_unpadded(result)?))
    }

    pub fn add_assign_scaled(
        &mut self,
        worker: &Worker,
        other: &Polynomial<E, Values>,
        scaling: &E::Fr,
    ) {
        assert_eq!(self.coeffs.len(), other.coeffs.len());

        worker.scope(other.coeffs.len(), |scope, chunk| {
            for (a, b) in self
                .coeffs
                .chunks_mut(chunk)
                .zip(other.coeffs.chunks(chunk))
            {
                scope.spawn(move |_| {
                    for (a, b) in a.iter_mut().zip(b.iter()) {
                        let mut tmp = *b;
                        tmp.mul_assign(&scaling);
                        a.add_assign(&tmp);
                    }
                });
            }
        });
    }

    /// Perform O(n) subtraction of one polynomial from another in the domain.
    pub fn sub_assign(&mut self, worker: &Worker, other: &Polynomial<E, Values>) {
        assert_eq!(self.coeffs.len(), other.coeffs.len());

        worker.scope(self.coeffs.len(), |scope, chunk| {
            for (a, b) in self
                .coeffs
                .chunks_mut(chunk)
                .zip(other.coeffs.chunks(chunk))
            {
                scope.spawn(move |_| {
                    for (a, b) in a.iter_mut().zip(b.iter()) {
                        a.sub_assign(&b);
                    }
                });
            }
        });
    }

    pub fn sub_assign_scaled(
        &mut self,
        worker: &Worker,
        other: &Polynomial<E, Values>,
        scaling: &E::Fr,
    ) {
        assert_eq!(self.coeffs.len(), other.coeffs.len());

        worker.scope(other.coeffs.len(), |scope, chunk| {
            for (a, b) in self
                .coeffs
                .chunks_mut(chunk)
                .zip(other.coeffs.chunks(chunk))
            {
                scope.spawn(move |_| {
                    for (a, b) in a.iter_mut().zip(b.iter()) {
                        let mut tmp = *b;
                        tmp.mul_assign(&scaling);
                        a.sub_assign(&tmp);
                    }
                });
            }
        });
    }

    /// Perform O(n) multiplication of two polynomials in the domain.
    pub fn mul_assign(&mut self, worker: &Worker, other: &Polynomial<E, Values>) {
        assert_eq!(self.coeffs.len(), other.coeffs.len());

        worker.scope(self.coeffs.len(), |scope, chunk| {
            for (a, b) in self
                .coeffs
                .chunks_mut(chunk)
                .zip(other.coeffs.chunks(chunk))
            {
                scope.spawn(move |_| {
                    for (a, b) in a.iter_mut().zip(b.iter()) {
                        a.mul_assign(&b);
                    }
                });
            }
        });
    }

    pub fn batch_inversion(&mut self, worker: &Worker) -> Result<(), SynthesisError> {
        let num_threads = worker.get_num_spawned_threads(self.coeffs.len());

        // let mut grand_products: Vec<E::Fr> = vec![E::Fr::one(); self.coeffs.len()];
        let mut grand_products: Vec<E::Fr> = Vec::with_capacity(self.size());
        unsafe { grand_products.set_len(self.size()) };
        let mut subproducts: Vec<E::Fr> = vec![E::Fr::one(); num_threads as usize];

        worker.scope(self.coeffs.len(), |scope, chunk| {
            for ((a, g), s) in self
                .coeffs
                .chunks(chunk)
                .zip(grand_products.chunks_mut(chunk))
                .zip(subproducts.chunks_mut(1))
            {
                scope.spawn(move |_| {
                    for (a, g) in a.iter().zip(g.iter_mut()) {
                        s[0].mul_assign(&a);
                        *g = s[0];
                    }
                });
            }
        });

        // now coeffs are [a, b, c, d, ..., z]
        // grand_products are [a, ab, abc, d, de, def, ...., xyz]
        // subproducts are [abc, def, xyz]
        // not guaranteed to have equal length

        let mut full_grand_product = E::Fr::one();
        for sub in subproducts.iter() {
            full_grand_product.mul_assign(sub);
        }

        let product_inverse = full_grand_product
            .inverse()
            .ok_or(SynthesisError::DivisionByZero)?;

        // now let's get [abc^-1, def^-1, ..., xyz^-1];
        let mut subinverses = vec![E::Fr::one(); num_threads];
        for (i, s) in subinverses.iter_mut().enumerate() {
            let mut tmp = product_inverse;
            for (j, p) in subproducts.iter().enumerate() {
                if i == j {
                    continue;
                }
                tmp.mul_assign(&p);
            }

            *s = tmp;
        }

        worker.scope(self.coeffs.len(), |scope, chunk| {
            for ((a, g), s) in self
                .coeffs
                .chunks_mut(chunk)
                .zip(grand_products.chunks(chunk))
                .zip(subinverses.chunks_mut(1))
            {
                scope.spawn(move |_| {
                    for (a, g) in a
                        .iter_mut()
                        .rev()
                        .zip(g.iter().rev().skip(1).chain(Some(E::Fr::one()).iter()))
                    {
                        // s[0] = abc^-1
                        // a = c
                        // g = ab
                        let tmp = *a; // c
                        *a = *g;
                        a.mul_assign(&s[0]); // a = ab*(abc^-1) = c^-1
                        s[0].mul_assign(&tmp); // s[0] = (ab)^-1
                    }
                });
            }
        });

        Ok(())
    }

    pub fn pop_last(&mut self) -> Result<E::Fr, SynthesisError> {
        let last = self.coeffs.pop().ok_or(SynthesisError::AssignmentMissing)?;

        Ok(last)
    }

    pub fn clone_shifted_assuming_natural_ordering(
        &self,
        by: usize,
    ) -> Result<Self, SynthesisError> {
        let len = self.coeffs.len();
        assert!(by < len);
        let mut new_coeffs = vec![E::Fr::zero(); self.coeffs.len()];
        new_coeffs[..(len - by)].copy_from_slice(&self.coeffs[by..]);
        new_coeffs[(len - by)..].copy_from_slice(&self.coeffs[..by]);

        Self::from_values(new_coeffs)
    }

    pub fn clone_shifted_assuming_bitreversed(
        &self,
        by: usize,
        worker: &Worker,
    ) -> Result<Self, SynthesisError> {
        let len = self.coeffs.len();
        assert!(by < len);
        let mut extended_clone = Vec::with_capacity(len + by);
        unsafe { extended_clone.set_len(len) };
        // extended_clone.extend_from_slice(&self.coeffs);
        fast_clone(self.coeffs.as_ref(), &mut extended_clone, worker);
        let mut tmp = Self::from_values(extended_clone)?;
        tmp.bitreverse_enumeration(&worker);

        let mut coeffs = tmp.into_coeffs();
        let tmp: Vec<_> = coeffs.drain(..by).collect();
        coeffs.extend(tmp);

        let mut tmp = Self::from_values(coeffs)?;
        tmp.bitreverse_enumeration(&worker);

        Ok(tmp)
    }
}

impl<E: Engine> Polynomial<E, Coefficients>
where
    E::Fr: PartialTwoBitReductionField,
{
    pub fn bitreversed_lde_using_bitreversed_ntt_with_partial_reduction<
        P: CTPrecomputations<E::Fr>,
    >(
        self,
        worker: &Worker,
        factor: usize,
        precomputed_omegas: &P,
        coset_factor: &E::Fr,
    ) -> Result<Polynomial<E, Values>, SynthesisError> {
        debug_assert!(self.coeffs.len().is_power_of_two());
        debug_assert_eq!(self.size(), precomputed_omegas.domain_size());

        if factor == 1 {
            return Ok(self.fft(&worker, &mut None));
        }

        let num_cpus = worker.get_num_cpus();
        let num_cpus_hint = if num_cpus <= factor {
            Some(1)
        } else {
            let mut threads_per_coset = num_cpus / factor;
            if threads_per_coset == 0 {
                threads_per_coset = 1;
            } else if num_cpus % factor != 0 {
                threads_per_coset += 1;
            }
            // let mut threads_per_coset = factor / num_cpus;
            // if factor % num_cpus != 0 {
            //     if (threads_per_coset + 1).is_power_of_two() {
            //         threads_per_coset += 1;
            //     }
            // }
            Some(threads_per_coset)
        };

        assert!(factor.is_power_of_two());
        let current_size = self.coeffs.len();
        let new_size = self.coeffs.len() * factor;
        let domain = Domain::<E::Fr>::new_for_size(new_size as u64)?;

        // let mut results = vec![self.coeffs.clone(); factor];

        let mut result = Vec::with_capacity(new_size);
        unsafe { result.set_len(new_size) };

        let r = &mut result[..] as *mut [E::Fr];

        let coset_omega = domain.generator;

        let log_n = self.exp;

        let range: Vec<usize> = (0..factor).collect();

        let self_coeffs_ref = &self.coeffs;

        // copy

        worker.in_place_scope(range.len(), |scope, chunk| {
            for coset_idx in range.chunks(chunk) {
                let r = unsafe { &mut *r };
                scope.spawn(move |_| {
                    for coset_idx in coset_idx.iter() {
                        let start = current_size * coset_idx;
                        let end = start + current_size;
                        let copy_start_pointer: *mut E::Fr = r[start..end].as_mut_ptr();

                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                self_coeffs_ref.as_ptr(),
                                copy_start_pointer,
                                current_size,
                            )
                        };
                    }
                });
            }
        });

        let to_spawn = factor;
        let coset_size = current_size;

        let factor_log = log2_floor(factor) as usize;

        // let chunk = Worker::chunk_size_for_num_spawned_threads(factor, to_spawn);

        // Each coset will produce values at specific indexes only, e.g
        // coset factor of omega^0 = 1 will produce elements that are only at places == 0 mod 16
        // coset factor of omega^1 will produce elements that are only at places == 1 mod 16
        // etc. We expect the output to be bitreversed, so
        // elements for coset factor of omega^0 = 1 will need to be placed first (00 top bits, bitreversed 00)
        // elements for coset factor of omega^1 will need to be placed after the first half (10 top bits, bitreversed 01)

        worker.in_place_scope(0, |scope, _| {
            for coset_idx in 0..to_spawn {
                let r = unsafe { &mut *r };
                scope.spawn(move |_| {
                    let from = coset_size * coset_idx;
                    let to = from + coset_size;
                    let one = E::Fr::one();
                    let bitreversed_power = cooley_tukey_ntt::bitreverse(coset_idx, factor_log);
                    let mut coset_generator = coset_omega.pow(&[bitreversed_power as u64]);
                    coset_generator.mul_assign(&coset_factor);
                    if coset_generator != one {
                        distribute_powers_with_num_cpus(
                            &mut r[from..to],
                            &worker,
                            coset_generator,
                            num_cpus_hint.expect("is some"),
                        );
                    }
                    partial_reduction::best_ct_ntt_partial_reduction(
                        &mut r[from..to],
                        &worker,
                        log_n,
                        num_cpus_hint,
                        precomputed_omegas,
                    );
                });
            }
        });

        Polynomial::from_values(result)
    }
}

impl<E: Engine> Polynomial<E, Values>
where
    E::Fr: PartialTwoBitReductionField,
{
    //TODO: use fft kernel
    pub fn ifft_using_bitreversed_ntt_with_partial_reduction<P: CTPrecomputations<E::Fr>>(
        self,
        worker: &Worker,
        precomputed_omegas: &P,
        coset_generator: &E::Fr,
    ) -> Result<Polynomial<E, Coefficients>, SynthesisError> {
        if self.coeffs.len() <= worker.get_num_cpus() * 4 {
            return Ok(self.ifft(&worker, &mut None));
        }

        let mut coeffs: Vec<_> = self.coeffs;
        let exp = self.exp;
        cooley_tukey_ntt::partial_reduction::best_ct_ntt_partial_reduction(
            &mut coeffs,
            worker,
            exp,
            Some(worker.get_num_cpus()),
            precomputed_omegas,
        );

        let mut this: Polynomial<E, Coefficients> = Polynomial::from_coeffs(coeffs)?;

        this.bitreverse_enumeration(&worker);

        let geninv = coset_generator.inverse().expect("must exist");

        worker.scope(this.coeffs.len(), |scope, chunk| {
            let minv = this.minv;

            for v in this.coeffs.chunks_mut(chunk) {
                scope.spawn(move |_| {
                    for v in v {
                        v.mul_assign(&minv);
                    }
                });
            }
        });

        if geninv != E::Fr::one() {
            this.distribute_powers(&worker, geninv);
        }

        Ok(this)
    }
}

impl<E: Engine> Polynomial<E, Values> {
    /// taken in natural enumeration
    /// outputs in natural enumeration
    pub fn ifft_using_bitreversed_ntt<P: CTPrecomputations<E::Fr>>(
        self,
        worker: &Worker,
        precomputed_omegas: &P,
        coset_generator: &E::Fr,
    ) -> Result<Polynomial<E, Coefficients>, SynthesisError> {
        let cpu_num = worker.get_num_cpus();
        if self.coeffs.len() <= cpu_num * 4 {
            return Ok(self.ifft(&worker, &mut None));
        }

        let mut coeffs: Vec<E::Fr> = self.coeffs;
        let exp = self.exp;
        cooley_tukey_ntt::best_ct_ntt(&mut coeffs, worker, exp, Some(cpu_num), precomputed_omegas);
        let mut this: Polynomial<E, Coefficients> = Polynomial::from_coeffs(coeffs)?;

        this.bitreverse_enumeration(&worker);

        let geninv = coset_generator.inverse().expect("must exist");

        worker.scope(this.coeffs.len(), |scope, chunk| {
            let minv = this.minv;

            for v in this.coeffs.chunks_mut(chunk) {
                scope.spawn(move |_| {
                    for v in v {
                        v.mul_assign(&minv);
                    }
                });
            }
        });

        if geninv != E::Fr::one() {
            this.distribute_powers(&worker, geninv);
        }

        Ok(this)
    }
}

impl<E: Engine> Polynomial<E, Coefficients> {
    pub fn bitreversed_lde_using_bitreversed_ntt(
        self,
        worker: &Worker,
        factor: usize,
        coset_factor: &E::Fr,
        fft_kern: &mut Option<LockedMultiFFTKernel<E>>,
    ) -> Result<Polynomial<E, Values>, SynthesisError> {
        debug_assert!(self.coeffs.len().is_power_of_two());

        if factor == 1 {
            return Ok(self.fft(&worker, fft_kern));
        }
        // assert!(factor.is_power_of_two());
        assert_eq!(factor, 4, "factor == 4");

        let current_size = self.coeffs.len();
        let new_size = self.coeffs.len() * factor;
        let new_domain = Domain::<E::Fr>::new_for_size(new_size as u64)?;

        let mut result = Vec::with_capacity(new_size);
        unsafe { result.set_len(new_size) };

        let r = &mut result[..] as *mut [E::Fr];

        let coset_omega = new_domain.generator;

        worker.in_place_scope(current_size, |scope, chunk| {
            for (i, v) in self.coeffs.chunks(chunk).enumerate() {
                let r = unsafe { &mut *r };
                scope.spawn(move |_| {
                    for j in 0..factor {
                        let start = i * chunk + j * current_size;
                        let end = if start + chunk <= (j + 1) * current_size {
                            start + chunk
                        } else {
                            (j + 1) * current_size
                        };
                        let copy_start_pointer: *mut E::Fr = r[start..end].as_mut_ptr();

                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                v.as_ptr(),
                                copy_start_pointer,
                                end - start,
                            )
                        };
                    }
                });
            }
        });

        let factor_log = log2_floor(factor) as usize;

        let coset_size = current_size;
        let r = unsafe { &mut *r };
        let one = E::Fr::one();
        for coset_idx in 0..factor {
            let from = coset_size * coset_idx;
            let to = from + coset_size;

            let bitreversed_power = cooley_tukey_ntt::bitreverse(coset_idx, factor_log);
            let mut coset_generator = coset_omega.pow(&[bitreversed_power as u64]);
            coset_generator.mul_assign(&coset_factor);

            if coset_generator != one {
                distribute_powers_new::<E>(&mut r[from..to], &worker, coset_generator);
            }
        }

        let (left, right) = r.split_at_mut(coset_size * 2);
        let (r1, r2) = left.split_at_mut(coset_size);
        let (r3, r4) = right.split_at_mut(coset_size);

        best_fft_multiple_gpu(
            fft_kern,
            &mut [r1, r2, r3, r4],
            &worker,
            &self.omega,
            self.exp,
        )
        .unwrap();

        Polynomial::from_values(result)
    }

    /// taken in natural enumeration
    /// outputs in natural enumeration
    pub fn fft_using_bitreversed_ntt<P: CTPrecomputations<E::Fr>>(
        self,
        worker: &Worker,
        precomputed_omegas: &P,
        coset_generator: &E::Fr,
    ) -> Result<Polynomial<E, Values>, SynthesisError> {
        if self.coeffs.len() <= worker.get_num_cpus() * 4 {
            return Ok(self.coset_fft_for_generator(&worker, *coset_generator));
        }

        let mut this = self;
        if coset_generator != &E::Fr::one() {
            this.distribute_powers(&worker, *coset_generator);
        }

        let mut coeffs: Vec<_> = this.coeffs;
        let exp = this.exp;
        cooley_tukey_ntt::best_ct_ntt(
            &mut coeffs,
            worker,
            exp,
            Some(worker.get_num_cpus()),
            precomputed_omegas,
        );
        let mut this = Polynomial::from_values(coeffs)?;

        this.bitreverse_enumeration(&worker);

        Ok(this)
    }

    /// taken in natural enumeration
    /// outputs in natural enumeration
    pub fn fft_using_bitreversed_ntt_output_bitreversed<P: CTPrecomputations<E::Fr>>(
        self,
        worker: &Worker,
        precomputed_omegas: &P,
        coset_generator: &E::Fr,
    ) -> Result<Polynomial<E, Values>, SynthesisError> {
        if self.coeffs.len() <= worker.get_num_cpus() * 4 {
            return Ok(self.coset_fft_for_generator(&worker, *coset_generator));
        }

        let mut this = self;
        if coset_generator != &E::Fr::one() {
            this.distribute_powers(&worker, *coset_generator);
        }

        let mut coeffs: Vec<_> = this.coeffs;
        let exp = this.exp;
        cooley_tukey_ntt::best_ct_ntt(
            &mut coeffs,
            worker,
            exp,
            Some(worker.get_num_cpus()),
            precomputed_omegas,
        );
        let this = Polynomial::from_values(coeffs)?;

        Ok(this)
    }
}

pub fn fft_multiple<E: Engine>(
    mut polys: Vec<Polynomial<E, Coefficients>>,
    worker: &Worker,
    kern: &mut Option<LockedMultiFFTKernel<E>>,
) -> Vec<Polynomial<E, Values>> {
    if polys.is_empty() {
        return vec![];
    }
    let exp = polys[0].exp;
    let omega = polys[0].omega;
    let omegainv = polys[0].omegainv;
    let geninv = polys[0].geninv;
    let minv = polys[0].minv;

    let mut vecs = vec![];
    for poly in polys.iter_mut() {
        vecs.push(poly.as_mut());
    }
    let now = Instant::now();
    best_fft_multiple_gpu(kern, &mut vecs, worker, &omega, exp).unwrap();
    println!("FFT multiple taken {:?}", now.elapsed());

    let mut res = vec![];
    for p in polys.into_iter() {
        res.push(Polynomial::<E, Values> {
            exp,
            omega,
            omegainv,
            geninv,
            minv,
            coeffs: p.into_coeffs(),
            _marker: std::marker::PhantomData,
        });
    }

    res
}

pub fn ifft_multiple<E: Engine>(
    mut polys: Vec<Polynomial<E, Values>>,
    worker: &Worker,
    kern: &mut Option<LockedMultiFFTKernel<E>>,
) -> Vec<Polynomial<E, Coefficients>> {
    if polys.is_empty() {
        return vec![];
    }
    let omegainv = polys[0].omegainv;
    let exp = polys[0].exp;

    let mut vecs = vec![];
    for poly in polys.iter_mut() {
        vecs.push(poly.as_mut());
    }

    best_fft_multiple_gpu(kern, &mut vecs, worker, &omegainv, exp).unwrap();

    let minv = polys[0].minv;
    let size = polys[0].size();

    for p in polys.iter_mut() {
        worker.scope(size, |scope, chunk| {
            for v in p.as_mut().chunks_mut(chunk) {
                scope.spawn(move |_| {
                    for v in v {
                        v.mul_assign(&minv);
                    }
                });
            }
        });
    }

    let mut res = vec![];
    for p in polys.into_iter() {
        res.push(Polynomial::<E, Coefficients> {
            exp,
            omega: p.omega,
            omegainv: p.omegainv,
            geninv: p.geninv,
            minv,
            coeffs: p.into_coeffs(),
            _marker: std::marker::PhantomData,
        });
    }

    res
}

pub fn bit_reverse<E: Engine>(a: &mut [E::Fr], log_n: u32) {
    fn bitrev(mut n: u32, l: u32) -> u32 {
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
        let rk = bitrev(k, log_n);
        if k < rk {
            a.swap(rk as usize, k as usize);
        }
    }
}

#[test]
fn test_fft_2_parallel() {
    use crate::ff::{Field, PrimeField};
    //use crate::LockedMultiFFTKernel;
    use crate::pairing::bn256::Fr;
    use pairing::bn256::Bn256;

    let logs = vec![24, 25, 26, 27];
    let worker = Worker::new();

    for log in logs {
        println!("size = 2^{:?}", log);
        let max_size = 1 << log;
        let scalars1 =
            crate::kate_commitment::test::make_random_field_elements::<Fr>(&worker, max_size);
        let mut as_coeffs1: Polynomial<Bn256, Values> = Polynomial::from_values(scalars1).unwrap();

        let scalars2 =
            crate::kate_commitment::test::make_random_field_elements::<Fr>(&worker, max_size);
        let mut as_coeffs2: Polynomial<Bn256, Values> = Polynomial::from_values(scalars2).unwrap();

        let mut fft_kern = Some(LockedMultiFFTKernel::<Bn256>::new(
            as_coeffs1.exp as usize,
            false,
        ));

        let (res1, res2) = as_coeffs1.ifft_2(as_coeffs2, &worker, &mut fft_kern);

        let scalars3 =
            crate::kate_commitment::test::make_random_field_elements::<Fr>(&worker, max_size);
        let mut as_coeffs3: Polynomial<Bn256, Values> = Polynomial::from_values(scalars3).unwrap();

        let scalars4 =
            crate::kate_commitment::test::make_random_field_elements::<Fr>(&worker, max_size);
        let mut as_coeffs4: Polynomial<Bn256, Values> = Polynomial::from_values(scalars4).unwrap();

        let (res3, res4) = as_coeffs3.ifft_2(as_coeffs4, &worker, &mut fft_kern);

        println!("============================");
    }
}

#[cfg(test)]
mod test {
    use pairing::bn256::Bn256;

    #[test]
    fn test_shifted_grand_product() {
        use super::*;
        use crate::ff::{Field, PrimeField};
        use crate::pairing::bn256::Fr;

        use crate::worker::Worker;
        use rand::{Rand, Rng, SeedableRng, XorShiftRng};

        let samples: usize = 1 << 20;
        let rng = &mut XorShiftRng::from_seed([0x3dbe6259, 0x8d313d76, 0x3237db17, 0xe5bc0654]);

        let v = (0..samples).map(|_| Fr::rand(rng)).collect::<Vec<_>>();

        let mut manual = vec![];
        manual.push(Fr::one());

        let mut tmp = Fr::one();

        for v in v.iter() {
            tmp.mul_assign(&v);
            manual.push(tmp);
        }

        let as_poly: Polynomial<Bn256, Values> = Polynomial::from_values(v).unwrap();
        let worker = Worker::new();
        let as_poly = as_poly.calculate_shifted_grand_product(&worker).unwrap();
        let as_poly = as_poly.into_coeffs();
        for idx in 0..manual.len() {
            assert_eq!(manual[idx], as_poly[idx], "failed at idx = {}", idx);
        }
    }

    #[test]
    fn test_grand_product() {
        use super::*;
        use crate::ff::{Field, PrimeField};
        use crate::pairing::bn256::Fr;

        use crate::worker::Worker;
        use rand::{Rand, Rng, SeedableRng, XorShiftRng};

        let samples: usize = 1 << 20;
        let rng = &mut XorShiftRng::from_seed([0x3dbe6259, 0x8d313d76, 0x3237db17, 0xe5bc0654]);

        let v = (0..samples).map(|_| Fr::rand(rng)).collect::<Vec<_>>();

        let mut manual = vec![];

        let mut tmp = Fr::one();

        for v in v.iter() {
            tmp.mul_assign(&v);
            manual.push(tmp);
        }

        let as_poly: Polynomial<Bn256, Values> = Polynomial::from_values(v).unwrap();
        let worker = Worker::new();
        let as_poly = as_poly.calculate_grand_product(&worker).unwrap();
        let as_poly = as_poly.into_coeffs();
        for idx in 0..manual.len() {
            assert_eq!(manual[idx], as_poly[idx], "failed at idx = {}", idx);
        }
    }
}
