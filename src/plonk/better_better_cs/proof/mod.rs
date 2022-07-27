use super::cs::*;
use super::data_structures::{self, *};
use crate::pairing::ff::*;
use crate::pairing::*;
use crate::plonk::polynomials::*;
use std::collections::HashMap;

use std::time::Instant;

use crate::plonk::domains::*;
use crate::worker::Worker;
use crate::SynthesisError;

use crate::kate_commitment::*;

use super::super::better_cs::utils::*;
use super::setup::*;
use super::utils::*;

use crate::plonk::fft::cooley_tukey_ntt::*;

use crate::byteorder::BigEndian;
use crate::byteorder::ReadBytesExt;
use crate::byteorder::WriteBytesExt;
use std::io::{Read, Write};

use crate::plonk::better_cs::keys::*;

pub fn write_tuple_with_one_index<F: PrimeField, W: Write>(
    tuple: &(usize, F),
    mut writer: W,
) -> std::io::Result<()> {
    writer.write_u64::<BigEndian>(tuple.0 as u64)?;
    write_fr(&tuple.1, &mut writer)?;

    Ok(())
}

pub fn write_tuple_with_one_index_vec<F: PrimeField, W: Write>(
    p: &[(usize, F)],
    mut writer: W,
) -> std::io::Result<()> {
    writer.write_u64::<BigEndian>(p.len() as u64)?;
    for p in p.iter() {
        write_tuple_with_one_index(p, &mut writer)?;
    }
    Ok(())
}

pub fn read_tuple_with_one_index<F: PrimeField, R: Read>(
    mut reader: R,
) -> std::io::Result<(usize, F)> {
    let index = reader.read_u64::<BigEndian>()?;
    let el = read_fr(&mut reader)?;

    Ok((index as usize, el))
}

pub fn read_tuple_with_one_index_vec<F: PrimeField, R: Read>(
    mut reader: R,
) -> std::io::Result<Vec<(usize, F)>> {
    let num_elements = reader.read_u64::<BigEndian>()?;
    let mut elements = vec![];
    for _ in 0..num_elements {
        let el = read_tuple_with_one_index(&mut reader)?;
        elements.push(el);
    }

    Ok(elements)
}

pub fn write_tuple_with_two_indexes<F: PrimeField, W: Write>(
    tuple: &(usize, usize, F),
    mut writer: W,
) -> std::io::Result<()> {
    writer.write_u64::<BigEndian>(tuple.0 as u64)?;
    writer.write_u64::<BigEndian>(tuple.1 as u64)?;
    write_fr(&tuple.2, &mut writer)?;

    Ok(())
}

pub fn write_tuple_with_two_indexes_vec<F: PrimeField, W: Write>(
    p: &[(usize, usize, F)],
    mut writer: W,
) -> std::io::Result<()> {
    writer.write_u64::<BigEndian>(p.len() as u64)?;
    for p in p.iter() {
        write_tuple_with_two_indexes(p, &mut writer)?;
    }
    Ok(())
}

pub fn read_tuple_with_two_indexes<F: PrimeField, R: Read>(
    mut reader: R,
) -> std::io::Result<(usize, usize, F)> {
    let index_0 = reader.read_u64::<BigEndian>()?;
    let index_1 = reader.read_u64::<BigEndian>()?;
    let el = read_fr(&mut reader)?;

    Ok((index_0 as usize, index_1 as usize, el))
}

pub fn read_tuple_with_two_indexes_vec<F: PrimeField, R: Read>(
    mut reader: R,
) -> std::io::Result<Vec<(usize, usize, F)>> {
    let num_elements = reader.read_u64::<BigEndian>()?;
    let mut elements = vec![];
    for _ in 0..num_elements {
        let el = read_tuple_with_two_indexes(&mut reader)?;
        elements.push(el);
    }

    Ok(elements)
}

#[derive(Clone, PartialEq, Eq)]
pub struct Proof<E: Engine, C: Circuit<E>> {
    pub n: usize,
    pub inputs: Vec<E::Fr>,
    pub state_polys_commitments: Vec<E::G1Affine>,
    pub witness_polys_commitments: Vec<E::G1Affine>,
    pub copy_permutation_grand_product_commitment: E::G1Affine,

    pub lookup_s_poly_commitment: Option<E::G1Affine>,
    pub lookup_grand_product_commitment: Option<E::G1Affine>,

    pub quotient_poly_parts_commitments: Vec<E::G1Affine>,

    pub state_polys_openings_at_z: Vec<E::Fr>,
    pub state_polys_openings_at_dilations: Vec<(usize, usize, E::Fr)>,
    pub witness_polys_openings_at_z: Vec<E::Fr>,
    pub witness_polys_openings_at_dilations: Vec<(usize, usize, E::Fr)>,

    pub gate_setup_openings_at_z: Vec<(usize, usize, E::Fr)>,
    pub gate_selectors_openings_at_z: Vec<(usize, E::Fr)>,

    pub copy_permutation_polys_openings_at_z: Vec<E::Fr>,
    pub copy_permutation_grand_product_opening_at_z_omega: E::Fr,

    pub lookup_s_poly_opening_at_z_omega: Option<E::Fr>,
    pub lookup_grand_product_opening_at_z_omega: Option<E::Fr>,

    pub lookup_t_poly_opening_at_z: Option<E::Fr>,
    pub lookup_t_poly_opening_at_z_omega: Option<E::Fr>,

    pub lookup_selector_poly_opening_at_z: Option<E::Fr>,
    pub lookup_table_type_poly_opening_at_z: Option<E::Fr>,

    pub quotient_poly_opening_at_z: E::Fr,

    pub linearization_poly_opening_at_z: E::Fr,

    pub opening_proof_at_z: E::G1Affine,
    pub opening_proof_at_z_omega: E::G1Affine,

    _marker: std::marker::PhantomData<C>,
}

impl<E: Engine, C: Circuit<E>> Proof<E, C> {
    pub fn empty() -> Self {
        Self {
            n: 0,
            inputs: vec![],
            state_polys_commitments: vec![],
            witness_polys_commitments: vec![],
            copy_permutation_grand_product_commitment: E::G1Affine::zero(),

            lookup_s_poly_commitment: None,
            lookup_grand_product_commitment: None,

            quotient_poly_parts_commitments: vec![],

            state_polys_openings_at_z: vec![],
            state_polys_openings_at_dilations: vec![],
            witness_polys_openings_at_z: vec![],
            witness_polys_openings_at_dilations: vec![],

            gate_setup_openings_at_z: vec![],
            gate_selectors_openings_at_z: vec![],

            copy_permutation_polys_openings_at_z: vec![],
            copy_permutation_grand_product_opening_at_z_omega: E::Fr::zero(),

            lookup_s_poly_opening_at_z_omega: None,
            lookup_grand_product_opening_at_z_omega: None,

            lookup_t_poly_opening_at_z: None,
            lookup_t_poly_opening_at_z_omega: None,

            lookup_selector_poly_opening_at_z: None,
            lookup_table_type_poly_opening_at_z: None,

            quotient_poly_opening_at_z: E::Fr::zero(),

            linearization_poly_opening_at_z: E::Fr::zero(),

            opening_proof_at_z: E::G1Affine::zero(),
            opening_proof_at_z_omega: E::G1Affine::zero(),

            _marker: std::marker::PhantomData,
        }
    }

    pub fn write<W: Write>(&self, mut writer: W) -> std::io::Result<()> {
        writer.write_u64::<BigEndian>(self.n as u64)?;

        write_fr_vec(&self.inputs, &mut writer)?;

        write_curve_affine_vec(&self.state_polys_commitments, &mut writer)?;
        write_curve_affine_vec(&self.witness_polys_commitments, &mut writer)?;

        write_curve_affine(&self.copy_permutation_grand_product_commitment, &mut writer)?;

        write_optional_curve_affine(&self.lookup_s_poly_commitment, &mut writer)?;
        write_optional_curve_affine(&self.lookup_grand_product_commitment, &mut writer)?;

        write_curve_affine_vec(&self.quotient_poly_parts_commitments, &mut writer)?;

        write_fr_vec(&self.state_polys_openings_at_z, &mut writer)?;
        write_tuple_with_two_indexes_vec(&self.state_polys_openings_at_dilations, &mut writer)?;

        write_fr_vec(&self.witness_polys_openings_at_z, &mut writer)?;
        write_tuple_with_two_indexes_vec(&self.witness_polys_openings_at_dilations, &mut writer)?;

        write_tuple_with_two_indexes_vec(&self.gate_setup_openings_at_z, &mut writer)?;
        write_tuple_with_one_index_vec(&self.gate_selectors_openings_at_z, &mut writer)?;

        write_fr_vec(&self.copy_permutation_polys_openings_at_z, &mut writer)?;
        write_fr(
            &self.copy_permutation_grand_product_opening_at_z_omega,
            &mut writer,
        )?;

        write_optional_fr(&self.lookup_s_poly_opening_at_z_omega, &mut writer)?;
        write_optional_fr(&self.lookup_grand_product_opening_at_z_omega, &mut writer)?;

        write_optional_fr(&self.lookup_t_poly_opening_at_z, &mut writer)?;
        write_optional_fr(&self.lookup_t_poly_opening_at_z_omega, &mut writer)?;

        write_optional_fr(&self.lookup_selector_poly_opening_at_z, &mut writer)?;
        write_optional_fr(&self.lookup_table_type_poly_opening_at_z, &mut writer)?;

        write_fr(&self.quotient_poly_opening_at_z, &mut writer)?;
        write_fr(&self.linearization_poly_opening_at_z, &mut writer)?;

        write_curve_affine(&self.opening_proof_at_z, &mut writer)?;
        write_curve_affine(&self.opening_proof_at_z_omega, &mut writer)?;

        Ok(())
    }

    pub fn read<R: Read>(mut reader: R) -> std::io::Result<Self> {
        let new = Self {
            n: reader.read_u64::<BigEndian>()? as usize,
            inputs: read_fr_vec(&mut reader)?,
            state_polys_commitments: read_curve_affine_vector(&mut reader)?,
            witness_polys_commitments: read_curve_affine_vector(&mut reader)?,
            copy_permutation_grand_product_commitment: read_curve_affine(&mut reader)?,

            lookup_s_poly_commitment: read_optional_curve_affine(&mut reader)?,
            lookup_grand_product_commitment: read_optional_curve_affine(&mut reader)?,

            quotient_poly_parts_commitments: read_curve_affine_vector(&mut reader)?,

            state_polys_openings_at_z: read_fr_vec(&mut reader)?,
            state_polys_openings_at_dilations: read_tuple_with_two_indexes_vec(&mut reader)?,
            witness_polys_openings_at_z: read_fr_vec(&mut reader)?,
            witness_polys_openings_at_dilations: read_tuple_with_two_indexes_vec(&mut reader)?,

            gate_setup_openings_at_z: read_tuple_with_two_indexes_vec(&mut reader)?,
            gate_selectors_openings_at_z: read_tuple_with_one_index_vec(&mut reader)?,

            copy_permutation_polys_openings_at_z: read_fr_vec(&mut reader)?,
            copy_permutation_grand_product_opening_at_z_omega: read_fr(&mut reader)?,

            lookup_s_poly_opening_at_z_omega: read_optional_fr(&mut reader)?,
            lookup_grand_product_opening_at_z_omega: read_optional_fr(&mut reader)?,

            lookup_t_poly_opening_at_z: read_optional_fr(&mut reader)?,
            lookup_t_poly_opening_at_z_omega: read_optional_fr(&mut reader)?,

            lookup_selector_poly_opening_at_z: read_optional_fr(&mut reader)?,
            lookup_table_type_poly_opening_at_z: read_optional_fr(&mut reader)?,

            quotient_poly_opening_at_z: read_fr(&mut reader)?,

            linearization_poly_opening_at_z: read_fr(&mut reader)?,

            opening_proof_at_z: read_curve_affine(&mut reader)?,
            opening_proof_at_z_omega: read_curve_affine(&mut reader)?,

            _marker: std::marker::PhantomData,
        };

        Ok(new)
    }
}

use super::cs::*;
use crate::gpu::{LockedMultiFFTKernel, LockedMultiexpKernel};
use crate::plonk::commitments::transcript::*;
use crate::plonk::utils::{fast_clone, fast_initialize_to_element};

impl<E: Engine, P: PlonkConstraintSystemParams<E>, MG: MainGate<E>, S: SynthesisMode>
    Assembly<E, P, MG, S>
{
    pub fn create_proof<C: Circuit<E>, T: Transcript<E::Fr>>(
        self,
        worker: &Worker,
        setup: &Setup<E, C>,
        mon_crs: &Crs<E, CrsForMonomialForm>,
        transcript_params: Option<T::InitializationParameters>,
    ) -> Result<Proof<E, C>, SynthesisError> {
        assert!(S::PRODUCE_WITNESS);
        assert!(self.is_finalized);
        let mut transcript = if let Some(params) = transcript_params {
            T::new_from_params(params)
        } else {
            T::new()
        };

        let mut proof = Proof::<E, C>::empty();

        let input_values = self.input_assingments.clone();

        proof.n = self.n();
        proof.inputs = input_values.clone();

        for inp in input_values.iter() {
            transcript.commit_field_element(inp);
        }

        let num_state_polys = <Self as ConstraintSystem<E>>::Params::STATE_WIDTH;
        assert!(num_state_polys % 2 == 0);
        let num_witness_polys = <Self as ConstraintSystem<E>>::Params::WITNESS_WIDTH;

        let mut values_storage = self.make_assembled_poly_storage(worker, true)?;

        let required_domain_size = self.n() + 1;
        assert!(required_domain_size.is_power_of_two());

        let omegas_bitreversed =
            BitReversedOmegas::<E::Fr>::new_for_domain_size(required_domain_size);
        let omegas_inv_bitreversed =
            <OmegasInvBitreversed<E::Fr> as CTPrecomputations<E::Fr>>::new_for_domain_size(
                required_domain_size,
            );

        let mut fft_kern = Some(LockedMultiFFTKernel::<E>::new(0, false));
        // if we simultaneously produce setup then grab permutation polys in values forms
        if S::PRODUCE_SETUP {
            let permutation_polys = self.make_permutations(&worker)?;
            assert_eq!(permutation_polys.len(), num_state_polys);

            for (idx, poly) in permutation_polys.into_iter().enumerate() {
                let key = PolyIdentifier::PermutationPolynomial(idx);
                let poly = PolynomialProxy::from_owned(poly);
                values_storage.setup_map.insert(key, poly);
            }
        } else {
            // compute from setup
            let mut coeffs = vec![];
            for idx in 0..num_state_polys {
                coeffs.push(setup.permutation_monomials[idx].fast_clone(&worker));
            }
            let polys = fft_multiple(coeffs, &worker, &mut fft_kern);
            for (idx, poly) in polys.into_iter().enumerate() {
                let key = PolyIdentifier::PermutationPolynomial(idx);
                let poly = PolynomialProxy::from_owned(poly);
                values_storage.setup_map.insert(key, poly);
            }
        }

        let mut ldes_storage = AssembledPolynomialStorage::<E>::new(
            true,
            self.max_constraint_degree.next_power_of_two(),
        );

        let mut monomials_storage =
            Self::create_monomial_storage(&worker, &values_storage, true, &mut fft_kern)?;
        drop(fft_kern);

        monomials_storage.extend_from_setup(setup)?;

        let mut multiexp_kern = Some(LockedMultiexpKernel::<E>::new(0, false));
        // step 1 - commit state and witness, enumerated. Also commit sorted polynomials for table arguments
        for i in 0..num_state_polys {
            let key = PolyIdentifier::VariablesPolynomial(i);
            let poly_ref = monomials_storage.get_poly(key);
            let commitment =
                commit_using_monomials(poly_ref, mon_crs, &worker, &mut multiexp_kern)?;

            commit_point_as_xy::<E, T>(&mut transcript, &commitment);

            proof.state_polys_commitments.push(commitment);
        }

        for i in 0..num_witness_polys {
            let key = PolyIdentifier::VariablesPolynomial(i);
            let poly_ref = monomials_storage.get_poly(key);
            let commitment =
                commit_using_monomials(poly_ref, mon_crs, &worker, &mut multiexp_kern)?;

            commit_point_as_xy::<E, T>(&mut transcript, &commitment);

            proof.witness_polys_commitments.push(commitment);
        }
        drop(multiexp_kern);

        // step 1.5 - if there are lookup tables then draw random "eta" to linearlize over tables
        let mut lookup_data: Option<data_structures::LookupDataHolder<E>> = if self.tables.len() > 0
        {
            unreachable!("unreachable for aggregated plonk proof");
        } else {
            None
        };

        if self.multitables.len() > 0 {
            unimplemented!("do not support multitables yet")
        }

        // step 2 - grand product arguments
        let beta_for_copy_permutation = transcript.get_challenge();
        let gamma_for_copy_permutation = transcript.get_challenge();

        // copy permutation grand product argument
        let mut grand_products_protos_with_gamma = vec![];

        for i in 0..num_state_polys {
            let id = PolyIdentifier::VariablesPolynomial(i);

            let mut p = values_storage
                .state_map
                .get(&id)
                .unwrap()
                .as_ref()
                .fast_clone(worker);
            p.add_constant(&worker, &gamma_for_copy_permutation);

            grand_products_protos_with_gamma.push(p);
        }

        let required_domain_size = required_domain_size;

        let domain = Domain::new_for_size(required_domain_size as u64)?;

        let mut domain_elements =
            materialize_domain_elements_with_natural_enumeration(&domain, &worker);

        domain_elements
            .pop()
            .expect("must pop last element for omega^i");

        let non_residues = make_non_residues::<E::Fr>(num_state_polys - 1);

        let mut domain_elements_poly_by_beta = Polynomial::from_values_unpadded(domain_elements)?;
        domain_elements_poly_by_beta.scale(&worker, beta_for_copy_permutation);

        // we take A, B, C, ... values and form (A + beta * X * non_residue + gamma), etc and calculate their grand product

        let mut z_num = {
            let mut grand_products_proto_it = grand_products_protos_with_gamma.iter().cloned();

            let mut z_1 = grand_products_proto_it.next().unwrap();
            z_1.add_assign(&worker, &domain_elements_poly_by_beta);

            for (mut p, non_res) in grand_products_proto_it.zip(non_residues.iter()) {
                p.add_assign_scaled(&worker, &domain_elements_poly_by_beta, non_res);
                z_1.mul_assign(&worker, &p);
            }

            z_1
        };

        // we take A, B, C, ... values and form (A + beta * perm_a + gamma), etc and calculate their grand product

        let mut permutation_polynomials_values_of_size_n_minus_one = vec![];

        for idx in 0..num_state_polys {
            let key = PolyIdentifier::PermutationPolynomial(idx);

            let mut coeffs = values_storage
                .get_poly(key)
                .fast_clone(worker)
                .into_coeffs();
            coeffs.pop().unwrap();

            let p = Polynomial::from_values_unpadded(coeffs)?;
            permutation_polynomials_values_of_size_n_minus_one.push(p);
        }

        let z_den = {
            assert_eq!(
                permutation_polynomials_values_of_size_n_minus_one.len(),
                grand_products_protos_with_gamma.len()
            );
            let mut grand_products_proto_it = grand_products_protos_with_gamma.into_iter();
            let mut permutation_polys_it =
                permutation_polynomials_values_of_size_n_minus_one.iter();

            let mut z_2 = grand_products_proto_it.next().unwrap();
            z_2.add_assign_scaled(
                &worker,
                permutation_polys_it.next().unwrap(),
                &beta_for_copy_permutation,
            );

            for (mut p, perm) in grand_products_proto_it.zip(permutation_polys_it) {
                // permutation polynomials
                p.add_assign_scaled(&worker, &perm, &beta_for_copy_permutation);
                z_2.mul_assign(&worker, &p);
            }

            z_2.batch_inversion(&worker)?;

            z_2
        };

        z_num.mul_assign(&worker, &z_den);
        drop(z_den);

        let z = z_num.calculate_shifted_grand_product(&worker)?;
        drop(z_num);

        assert!(z.size().is_power_of_two());

        assert!(z.as_ref()[0] == E::Fr::one());

        let mut fft_kern = Some(LockedMultiFFTKernel::<E>::new(0, false));
        let copy_permutation_z_in_monomial_form = z.ifft(&worker, &mut fft_kern);
        drop(fft_kern);

        let mut multiexp_kern = Some(LockedMultiexpKernel::<E>::new(0, false));
        let copy_permutation_z_poly_commitment = commit_using_monomials(
            &copy_permutation_z_in_monomial_form,
            mon_crs,
            &worker,
            &mut multiexp_kern,
        )?;
        drop(multiexp_kern);

        commit_point_as_xy::<E, T>(&mut transcript, &copy_permutation_z_poly_commitment);

        proof.copy_permutation_grand_product_commitment = copy_permutation_z_poly_commitment;

        let lookup_z_poly_in_monomial_form: Option<Polynomial<E, Coefficients>> =
            if let Some(data) = lookup_data.as_mut() {
                unreachable!("aggregated proof unreachable!");
            } else {
                None
            };

        // now draw alpha and add all the contributions to the quotient polynomial
        let alpha = transcript.get_challenge();

        let mut total_powers_of_alpha_for_gates = 0;
        for g in self.sorted_gates.iter() {
            total_powers_of_alpha_for_gates += g.num_quotient_terms();
        }

        println!(
            "Have {} terms from {} gates",
            total_powers_of_alpha_for_gates,
            self.sorted_gates.len()
        );

        let mut current_alpha = E::Fr::one();
        let mut powers_of_alpha_for_gates = Vec::with_capacity(total_powers_of_alpha_for_gates);
        powers_of_alpha_for_gates.push(current_alpha);
        for _ in 1..total_powers_of_alpha_for_gates {
            current_alpha.mul_assign(&alpha);
            powers_of_alpha_for_gates.push(current_alpha);
        }

        assert_eq!(
            powers_of_alpha_for_gates.len(),
            total_powers_of_alpha_for_gates
        );

        let mut all_gates = self.sorted_gates.clone();
        let num_different_gates = self.sorted_gates.len();

        let mut challenges_slice = &powers_of_alpha_for_gates[..];

        let mut lde_factor = num_state_polys;
        for g in self.sorted_gates.iter() {
            let degree = g.degree();
            if degree > lde_factor {
                lde_factor = degree;
            }
        }

        assert!(lde_factor <= 4);

        let coset_factor = E::Fr::multiplicative_generator();

        let mut fft_kern = Some(LockedMultiFFTKernel::<E>::new(0, false));
        let mut t_poly = {
            let gate = all_gates.drain(0..1).into_iter().next().unwrap();
            assert!(<Self as ConstraintSystem<E>>::MainGate::default().into_internal() == gate);
            let gate = <Self as ConstraintSystem<E>>::MainGate::default();
            let num_challenges = gate.num_quotient_terms();
            let (for_gate, rest) = challenges_slice.split_at(num_challenges);
            challenges_slice = rest;

            let input_values = self.input_assingments.clone();

            let mut t: Polynomial<E, Values> = gate.contribute_into_quotient_for_public_inputs(
                required_domain_size,
                &input_values,
                &mut ldes_storage,
                &monomials_storage,
                for_gate,
                &omegas_bitreversed,
                &omegas_inv_bitreversed,
                &worker,
                &mut fft_kern,
            )?;

            if num_different_gates > 1 {
                // we have to multiply by the masking poly (selector)
                let key = PolyIdentifier::GateSelector(gate.name());
                let monomial_selector =
                    monomials_storage.gate_selectors.get(&key).unwrap().as_ref();
                let mut selector_lde = monomial_selector
                    .clone_padded_to_domain(worker)?
                    .bitreversed_lde_using_bitreversed_ntt(
                        &worker,
                        lde_factor,
                        &coset_factor,
                        &mut fft_kern,
                    )?;
                let domain_size = monomial_selector.size().next_power_of_two();
                for i in 0..lde_factor {
                    selector_lde.bitreverse_enumeration_partial(
                        &worker,
                        i * domain_size,
                        domain_size,
                    );
                }

                t.mul_assign(&worker, &selector_lde);
                drop(selector_lde);
            }

            t
        };

        let non_main_gates = all_gates;
        println!("non_main_gates = {:?}", non_main_gates.len());

        for gate in non_main_gates.into_iter() {
            let num_challenges = gate.num_quotient_terms();
            let (for_gate, rest) = challenges_slice.split_at(num_challenges);
            challenges_slice = rest;
            //todo: step into
            let mut contribution = gate.contribute_into_quotient(
                required_domain_size,
                &mut ldes_storage,
                &monomials_storage,
                for_gate,
                &omegas_bitreversed,
                &omegas_inv_bitreversed,
                &worker,
            )?;

            {
                // we have to multiply by the masking poly (selector)
                let key = PolyIdentifier::GateSelector(gate.name());
                let monomial_selector =
                    monomials_storage.gate_selectors.get(&key).unwrap().as_ref();
                let mut selector_lde = monomial_selector
                    .clone_padded_to_domain(worker)?
                    .bitreversed_lde_using_bitreversed_ntt(
                        &worker,
                        lde_factor,
                        &coset_factor,
                        &mut fft_kern,
                    )?;

                let domain_size = monomial_selector.size().next_power_of_two();
                for i in 0..lde_factor {
                    selector_lde.bitreverse_enumeration_partial(
                        &worker,
                        i * domain_size,
                        domain_size,
                    );
                }

                contribution.mul_assign(&worker, &selector_lde);
                drop(selector_lde);
            }

            t_poly.add_assign(&worker, &contribution);
        }

        assert_eq!(challenges_slice.len(), 0);

        println!(
            "Power of alpha for a start of normal permutation argument = {}",
            total_powers_of_alpha_for_gates
        );

        // perform copy-permutation argument
        // we precompute L_{0} here cause it's necessary for both copy-permutation and lookup permutation
        // z(omega^0) - 1 == 0
        let l_0 = calculate_lagrange_poly::<E>(
            &worker,
            required_domain_size.next_power_of_two(),
            0,
            &mut fft_kern,
        )?;
        let domain_size = l_0.size();
        let mut l_0_coset_lde_bitreversed = l_0.bitreversed_lde_using_bitreversed_ntt(
            &worker,
            lde_factor,
            &coset_factor,
            &mut fft_kern,
        )?;
        for i in 0..lde_factor {
            l_0_coset_lde_bitreversed.bitreverse_enumeration_partial(
                &worker,
                i * domain_size,
                domain_size,
            );
        }

        let mut copy_grand_product_alphas = None;
        let x_poly_lde_bitreversed = {
            // now compute the permutation argument

            // bump alpha
            current_alpha.mul_assign(&alpha);
            let alpha_0 = current_alpha;

            let mut z_coset_lde_bitreversed = copy_permutation_z_in_monomial_form
                .fast_clone(worker)
                .bitreversed_lde_using_bitreversed_ntt(
                    &worker,
                    lde_factor,
                    &coset_factor,
                    &mut fft_kern,
                )?;
            let domain_size = copy_permutation_z_in_monomial_form.size();
            for i in 0..lde_factor {
                z_coset_lde_bitreversed.bitreverse_enumeration_partial(
                    &worker,
                    i * domain_size,
                    domain_size,
                );
            }

            assert!(z_coset_lde_bitreversed.size() == required_domain_size * lde_factor);

            let z_shifted_coset_lde_bitreversed =
                z_coset_lde_bitreversed.clone_shifted_assuming_bitreversed(lde_factor, &worker)?;

            assert!(z_shifted_coset_lde_bitreversed.size() == required_domain_size * lde_factor);

            // For both Z_1 and Z_2 we first check for grand products
            // z*(X)(A + beta*X + gamma)(B + beta*k_1*X + gamma)(C + beta*K_2*X + gamma) -
            // - (A + beta*perm_a(X) + gamma)(B + beta*perm_b(X) + gamma)(C + beta*perm_c(X) + gamma)*Z(X*Omega)== 0

            // we use evaluations of the polynomial X and K_i * X on a large domain's coset
            let mut contrib_z = z_coset_lde_bitreversed.fast_clone(worker);

            // precompute x poly
            let mut x_poly = Polynomial::from_values(fast_initialize_to_element(
                required_domain_size * lde_factor,
                coset_factor,
                worker,
            ))?;
            x_poly.distribute_powers(&worker, z_shifted_coset_lde_bitreversed.omega);
            x_poly.bitreverse_enumeration(&worker);

            assert_eq!(x_poly.size(), required_domain_size * lde_factor);

            // A + beta*X + gamma

            let mut tmp = ldes_storage
                .state_map
                .get(&PolyIdentifier::VariablesPolynomial(0))
                .unwrap()
                .as_ref()
                .fast_clone(worker);
            tmp.add_constant(&worker, &gamma_for_copy_permutation);
            tmp.add_assign_scaled(&worker, &x_poly, &beta_for_copy_permutation);
            contrib_z.mul_assign(&worker, &tmp);

            assert_eq!(non_residues.len() + 1, num_state_polys);

            for (poly_idx, non_res) in (1..num_state_polys).zip(non_residues.iter()) {
                let mut factor = beta_for_copy_permutation;
                factor.mul_assign(&non_res);

                let key = PolyIdentifier::VariablesPolynomial(poly_idx);
                tmp.reuse_allocation_parallel(
                    worker,
                    &ldes_storage.state_map.get(&key).unwrap().as_ref(),
                );
                tmp.add_constant(&worker, &gamma_for_copy_permutation);
                tmp.add_assign_scaled(&worker, &x_poly, &factor);
                contrib_z.mul_assign(&worker, &tmp);
            }

            t_poly.add_assign_scaled(&worker, &contrib_z, &current_alpha);

            drop(contrib_z);

            let mut contrib_z = z_shifted_coset_lde_bitreversed;

            // A + beta*perm_a + gamma

            for idx in 0..num_state_polys {
                let key = PolyIdentifier::VariablesPolynomial(idx);

                tmp.reuse_allocation_parallel(
                    worker,
                    &ldes_storage.state_map.get(&key).unwrap().as_ref(),
                );
                tmp.add_constant(&worker, &gamma_for_copy_permutation);

                let key = PolyIdentifier::PermutationPolynomial(idx);
                let mut perm = monomials_storage
                    .get_poly(key)
                    .fast_clone(worker)
                    .bitreversed_lde_using_bitreversed_ntt(
                        &worker,
                        lde_factor,
                        &coset_factor,
                        &mut fft_kern,
                    )?;
                let domain_size = monomials_storage.get_poly(key).size();
                for i in 0..lde_factor {
                    perm.bitreverse_enumeration_partial(&worker, i * domain_size, domain_size);
                }
                tmp.add_assign_scaled(&worker, &perm, &beta_for_copy_permutation);
                contrib_z.mul_assign(&worker, &tmp);
                drop(perm);
            }

            t_poly.sub_assign_scaled(&worker, &contrib_z, &current_alpha);

            drop(contrib_z);

            drop(tmp);

            // Z(x) * L_{0}(x) - 1 == 0
            current_alpha.mul_assign(&alpha);

            let alpha_1 = current_alpha;

            {
                let mut z_minus_one_by_l_0 = z_coset_lde_bitreversed;
                z_minus_one_by_l_0.sub_constant(&worker, &E::Fr::one());

                z_minus_one_by_l_0.mul_assign(&worker, &l_0_coset_lde_bitreversed);

                t_poly.add_assign_scaled(&worker, &z_minus_one_by_l_0, &current_alpha);
            }

            copy_grand_product_alphas = Some([alpha_0, alpha_1]);

            x_poly
        };

        // add contribution from grand product for loopup polys if there is one

        if let Some(z_poly_in_monomial_form) = lookup_z_poly_in_monomial_form.as_ref() {
            unreachable!("lookup_z_poly_in_monomial_form none");
        } else {
            drop(x_poly_lde_bitreversed);
            drop(l_0_coset_lde_bitreversed);
        }

        // perform the division

        let inverse_divisor_on_coset_lde_natural_ordering = {
            let mut vanishing_poly_inverse_bitreversed =
                evaluate_vanishing_polynomial_of_degree_on_domain_size::<E>(
                    required_domain_size as u64,
                    &E::Fr::multiplicative_generator(),
                    (required_domain_size * lde_factor) as u64,
                    &worker,
                )?;
            vanishing_poly_inverse_bitreversed.batch_inversion(&worker)?;
            // vanishing_poly_inverse_bitreversed.bitreverse_enumeration(&worker)?;

            vanishing_poly_inverse_bitreversed
        };

        // don't forget to bitreverse

        t_poly.bitreverse_enumeration(&worker);

        t_poly.mul_assign(&worker, &inverse_divisor_on_coset_lde_natural_ordering);

        drop(inverse_divisor_on_coset_lde_natural_ordering);

        let t_poly = t_poly.icoset_fft_for_generator(&worker, &coset_factor, &mut fft_kern);
        drop(fft_kern);

        println!("Lde factor = {}", lde_factor);
        // println!("Quotient poly = {:?}", t_poly.as_ref());

        {
            // degree is 4n-4
            let l = t_poly.size();
            if &t_poly.as_ref()[(l - 4)..] != &[E::Fr::zero(); 4][..] {
                return Err(SynthesisError::Unsatisfiable);
            }
            // assert_eq!(&t_poly.as_ref()[(l-4)..], &[E::Fr::zero(); 4][..], "quotient degree is too large");
        }

        // println!("Quotient poly degree = {}", get_degree::<E::Fr>(&t_poly));

        let mut t_poly_parts = t_poly.break_into_multiples(required_domain_size)?;

        let mut multiexp_kern = Some(LockedMultiexpKernel::<E>::new(0, false));
        for part in t_poly_parts.iter() {
            let commitment = commit_using_monomials(part, mon_crs, &worker, &mut multiexp_kern)?;

            commit_point_as_xy::<E, T>(&mut transcript, &commitment);

            proof.quotient_poly_parts_commitments.push(commitment);
        }

        // draw opening point
        let z = transcript.get_challenge();

        let omega = domain.generator;

        // evaluate quotient at z
        let quotient_at_z = {
            let mut result = E::Fr::zero();
            let mut current = E::Fr::one();
            let z_in_domain_size = z.pow(&[required_domain_size as u64]);
            for p in t_poly_parts.iter() {
                let mut subvalue_at_z = p.evaluate_at(&worker, z);

                subvalue_at_z.mul_assign(&current);
                result.add_assign(&subvalue_at_z);
                current.mul_assign(&z_in_domain_size);
            }

            result
        };

        // commit quotient value
        transcript.commit_field_element(&quotient_at_z);

        proof.quotient_poly_opening_at_z = quotient_at_z;

        // Now perform the linearization.
        // First collect and evalute all the polynomials that are necessary for linearization
        // and construction of the verification equation

        const MAX_DILATION: usize = 1;

        let queries_with_linearization =
            sort_queries_for_linearization(&self.sorted_gates, MAX_DILATION);

        let mut query_values_map = std::collections::HashMap::new();

        // go over all required queries

        for (dilation_value, ids) in queries_with_linearization.state_polys.iter().enumerate() {
            for id in ids.into_iter() {
                let (poly_ref, poly_idx) = if let PolyIdentifier::VariablesPolynomial(idx) = id {
                    (monomials_storage.state_map.get(&id).unwrap().as_ref(), idx)
                } else {
                    unreachable!();
                };

                let mut opening_point = z;
                for _ in 0..dilation_value {
                    opening_point.mul_assign(&omega);
                }

                let value = poly_ref.evaluate_at(&worker, opening_point);

                transcript.commit_field_element(&value);

                if dilation_value == 0 {
                    proof.state_polys_openings_at_z.push(value);
                } else {
                    proof.state_polys_openings_at_dilations.push((
                        dilation_value,
                        *poly_idx,
                        value,
                    ));
                }

                let key = PolynomialInConstraint::from_id_and_dilation(*id, dilation_value);

                query_values_map.insert(key, value);
            }
        }

        for (dilation_value, ids) in queries_with_linearization.witness_polys.iter().enumerate() {
            for id in ids.into_iter() {
                let (poly_ref, poly_idx) = if let PolyIdentifier::WitnessPolynomial(idx) = id {
                    (
                        monomials_storage.witness_map.get(&id).unwrap().as_ref(),
                        idx,
                    )
                } else {
                    unreachable!();
                };

                let mut opening_point = z;
                for _ in 0..dilation_value {
                    opening_point.mul_assign(&omega);
                }

                let value = poly_ref.evaluate_at(&worker, opening_point);

                transcript.commit_field_element(&value);

                if dilation_value == 0 {
                    proof.witness_polys_openings_at_z.push(value);
                } else {
                    proof.witness_polys_openings_at_dilations.push((
                        dilation_value,
                        *poly_idx,
                        value,
                    ));
                }

                let key = PolynomialInConstraint::from_id_and_dilation(*id, dilation_value);

                query_values_map.insert(key, value);
            }
        }

        for (gate_idx, queries) in queries_with_linearization
            .gate_setup_polys
            .iter()
            .enumerate()
        {
            for (dilation_value, ids) in queries.iter().enumerate() {
                for id in ids.into_iter() {
                    let (poly_ref, poly_idx) =
                        if let PolyIdentifier::GateSetupPolynomial(_, idx) = id {
                            (monomials_storage.setup_map.get(&id).unwrap().as_ref(), idx)
                        } else {
                            unreachable!();
                        };

                    let mut opening_point = z;
                    for _ in 0..dilation_value {
                        opening_point.mul_assign(&omega);
                    }

                    let value = poly_ref.evaluate_at(&worker, opening_point);

                    transcript.commit_field_element(&value);

                    if dilation_value == 0 {
                        proof
                            .gate_setup_openings_at_z
                            .push((gate_idx, *poly_idx, value));
                    } else {
                        unimplemented!("gate setup polynomials can not be time dilated");
                    }

                    let key = PolynomialInConstraint::from_id_and_dilation(*id, dilation_value);

                    query_values_map.insert(key, value);
                }
            }
        }

        // also open selectors

        let mut selector_values = vec![];
        for s in queries_with_linearization.gate_selectors.iter() {
            let gate_index = self.sorted_gates.iter().position(|r| r == s).unwrap();

            let key = PolyIdentifier::GateSelector(s.name());
            let poly_ref = monomials_storage.gate_selectors.get(&key).unwrap().as_ref();
            let value = poly_ref.evaluate_at(&worker, z);

            transcript.commit_field_element(&value);

            proof.gate_selectors_openings_at_z.push((gate_index, value));

            selector_values.push(value);
        }

        // copy-permutation polynomials queries

        let mut copy_permutation_queries = vec![];

        for idx in 0..(num_state_polys - 1) {
            let key = PolyIdentifier::PermutationPolynomial(idx);
            let value = monomials_storage.get_poly(key).evaluate_at(&worker, z);

            transcript.commit_field_element(&value);

            proof.copy_permutation_polys_openings_at_z.push(value);

            copy_permutation_queries.push(value);
        }

        // copy-permutation grand product query

        let mut z_omega = z;
        z_omega.mul_assign(&domain.generator);
        let copy_permutation_z_at_z_omega =
            copy_permutation_z_in_monomial_form.evaluate_at(&worker, z_omega);
        transcript.commit_field_element(&copy_permutation_z_at_z_omega);
        proof.copy_permutation_grand_product_opening_at_z_omega = copy_permutation_z_at_z_omega;

        // we've computed everything, so perform linearization

        let mut challenges_slice = &powers_of_alpha_for_gates[..];

        let mut all_gates = self.sorted_gates.clone();

        let mut r_poly = {
            let gate = all_gates.drain(0..1).into_iter().next().unwrap();
            assert!(
                gate.benefits_from_linearization(),
                "main gate is expected to benefit from linearization!"
            );
            assert!(<Self as ConstraintSystem<E>>::MainGate::default().into_internal() == gate);
            let gate = <Self as ConstraintSystem<E>>::MainGate::default();
            let num_challenges = gate.num_quotient_terms();
            let (for_gate, rest) = challenges_slice.split_at(num_challenges);
            challenges_slice = rest;

            let input_values = self.input_assingments.clone();

            let mut r: Polynomial<E, Coefficients> = gate
                .contribute_into_linearization_for_public_inputs(
                    required_domain_size,
                    &input_values,
                    z,
                    &query_values_map,
                    &monomials_storage,
                    for_gate,
                    &worker,
                )?;

            let mut selectors_it = selector_values.clone().into_iter();

            if num_different_gates > 1 {
                // first multiply r by the selector value at z
                r.scale(&worker, selectors_it.next().unwrap());
            }

            // now proceed per gate
            for gate in all_gates.into_iter() {
                let num_challenges = gate.num_quotient_terms();
                let (for_gate, rest) = challenges_slice.split_at(num_challenges);
                challenges_slice = rest;

                if gate.benefits_from_linearization() {
                    // gate benefits from linearization, so make temporary value
                    let tmp = gate.contribute_into_linearization(
                        required_domain_size,
                        z,
                        &query_values_map,
                        &monomials_storage,
                        for_gate,
                        &worker,
                    )?;

                    let selector_value = selectors_it.next().unwrap();

                    r.add_assign_scaled(&worker, &tmp, &selector_value);
                } else {
                    // we linearize over the selector, so take a selector and scale it
                    let gate_value_at_z = gate.contribute_into_verification_equation(
                        required_domain_size,
                        z,
                        &query_values_map,
                        for_gate,
                    )?;

                    let key = PolyIdentifier::GateSelector(gate.name());
                    let gate_selector_ref = monomials_storage
                        .gate_selectors
                        .get(&key)
                        .expect("must get monomial form of gate selector")
                        .as_ref();

                    r.add_assign_scaled(&worker, gate_selector_ref, &gate_value_at_z);
                }
            }

            assert!(selectors_it.next().is_none());
            assert_eq!(challenges_slice.len(), 0);

            r
        };

        // add contributions from copy-permutation and lookup-permutation

        // copy-permutation linearization comtribution
        {
            // + (a(z) + beta*z + gamma)*()*()*()*Z(x)

            let [alpha_0, alpha_1] = copy_grand_product_alphas
                .expect("there must be powers of alpha for copy permutation");

            let some_one = Some(E::Fr::one());
            let mut non_residues_iterator = some_one.iter().chain(&non_residues);

            let mut factor = alpha_0;

            for idx in 0..num_state_polys {
                let key = PolynomialInConstraint::from_id(PolyIdentifier::VariablesPolynomial(idx));
                let wire_value = query_values_map
                    .get(&key)
                    .ok_or(SynthesisError::AssignmentMissing)?;
                let mut t = z;
                let non_res = non_residues_iterator.next().unwrap();
                t.mul_assign(&non_res);
                t.mul_assign(&beta_for_copy_permutation);
                t.add_assign(&wire_value);
                t.add_assign(&gamma_for_copy_permutation);

                factor.mul_assign(&t);
            }

            assert!(non_residues_iterator.next().is_none());

            r_poly.add_assign_scaled(&worker, &copy_permutation_z_in_monomial_form, &factor);

            // - (a(z) + beta*perm_a + gamma)*()*()*z(z*omega) * beta * perm_d(X)

            let mut factor = alpha_0;
            factor.mul_assign(&beta_for_copy_permutation);
            factor.mul_assign(&copy_permutation_z_at_z_omega);

            for idx in 0..(num_state_polys - 1) {
                let key = PolynomialInConstraint::from_id(PolyIdentifier::VariablesPolynomial(idx));
                let wire_value = query_values_map
                    .get(&key)
                    .ok_or(SynthesisError::AssignmentMissing)?;
                let permutation_at_z = copy_permutation_queries[idx];
                let mut t = permutation_at_z;

                t.mul_assign(&beta_for_copy_permutation);
                t.add_assign(&wire_value);
                t.add_assign(&gamma_for_copy_permutation);

                factor.mul_assign(&t);
            }

            let key = PolyIdentifier::PermutationPolynomial(num_state_polys - 1);
            let last_permutation_poly_ref = monomials_storage.get_poly(key);

            r_poly.sub_assign_scaled(&worker, last_permutation_poly_ref, &factor);

            // + L_0(z) * Z(x)

            let mut factor = evaluate_l0_at_point(required_domain_size as u64, z)?;
            factor.mul_assign(&alpha_1);

            r_poly.add_assign_scaled(&worker, &copy_permutation_z_in_monomial_form, &factor);
        }

        // lookup grand product linearization

        // due to separate divisor it's not obvious if this is beneficial without some tricks
        // like multiplication by (1 - L_{n-1}) or by (x - omega^{n-1})

        // Z(x*omega)*(\gamma*(1 + \beta) + s(x) + \beta * s(x*omega))) -
        // Z(x) * (\beta + 1) * (\gamma + f(x)) * (\gamma(1 + \beta) + t(x) + \beta * t(x*omega)) == 0
        // check that (Z(x) - 1) * L_{0} == 0
        // check that (Z(x) - expected) * L_{n-1} == 0, or (Z(x*omega) - expected)* L_{n-2} == 0

        // f(x) does not need to be opened as it's made of table selector and witnesses
        // if we pursue the strategy from the linearization of a copy-permutation argument
        // then we leave something like s(x) from the Z(x*omega)*(\gamma*(1 + \beta) + s(x) + \beta * s(x*omega))) term,
        // and Z(x) from Z(x) * (\beta + 1) * (\gamma + f(x)) * (\gamma(1 + \beta) + t(x) + \beta * t(x*omega)) term,
        // with terms with lagrange polys as multipliers left intact

        let lookup_queries: Option<LookupQuery<E>> =
            if let Some(lookup_z_poly) = lookup_z_poly_in_monomial_form.as_ref() {
                unreachable!();
            } else {
                None
            };

        if let Some(queries) = lookup_queries.as_ref() {
            unreachable!();
        }

        let linearization_at_z = r_poly.evaluate_at(&worker, z);

        transcript.commit_field_element(&linearization_at_z);
        proof.linearization_poly_opening_at_z = linearization_at_z;

        // linearization is done, now perform sanity check
        // this is effectively a verification procedure

        {
            let vanishing_at_z = evaluate_vanishing_for_size(&z, required_domain_size as u64);

            // first let's aggregate gates

            let mut t_num_on_full_domain = E::Fr::zero();

            let challenges_slice = &powers_of_alpha_for_gates[..];

            let mut all_gates = self.sorted_gates.clone();

            // we've suffered and linearization polynomial captures all the gates except the public input!

            {
                let mut tmp = linearization_at_z;
                // add input values

                let gate = all_gates.drain(0..1).into_iter().next().unwrap();
                assert!(
                    gate.benefits_from_linearization(),
                    "main gate is expected to benefit from linearization!"
                );
                assert!(<Self as ConstraintSystem<E>>::MainGate::default().into_internal() == gate);
                let gate = <Self as ConstraintSystem<E>>::MainGate::default();
                let num_challenges = gate.num_quotient_terms();
                let (for_gate, _) = challenges_slice.split_at(num_challenges);

                let input_values = self.input_assingments.clone();

                let mut inputs_term = gate.add_inputs_into_quotient(
                    required_domain_size,
                    &input_values,
                    z,
                    for_gate,
                )?;

                if num_different_gates > 1 {
                    let selector_value = selector_values[0];
                    inputs_term.mul_assign(&selector_value);
                }

                tmp.add_assign(&inputs_term);

                t_num_on_full_domain.add_assign(&tmp);
            }

            // now aggregate leftovers from grand product for copy permutation
            {
                // - alpha_0 * (a + perm(z) * beta + gamma)*()*(d + gamma) * z(z*omega)
                let [alpha_0, alpha_1] = copy_grand_product_alphas
                    .expect("there must be powers of alpha for copy permutation");

                let mut factor = alpha_0;
                factor.mul_assign(&copy_permutation_z_at_z_omega);

                for idx in 0..(num_state_polys - 1) {
                    let key =
                        PolynomialInConstraint::from_id(PolyIdentifier::VariablesPolynomial(idx));
                    let wire_value = query_values_map
                        .get(&key)
                        .ok_or(SynthesisError::AssignmentMissing)?;
                    let permutation_at_z = copy_permutation_queries[idx];
                    let mut t = permutation_at_z;

                    t.mul_assign(&beta_for_copy_permutation);
                    t.add_assign(&wire_value);
                    t.add_assign(&gamma_for_copy_permutation);

                    factor.mul_assign(&t);
                }

                let key = PolynomialInConstraint::from_id(PolyIdentifier::VariablesPolynomial(
                    num_state_polys - 1,
                ));
                let mut tmp = *query_values_map
                    .get(&key)
                    .ok_or(SynthesisError::AssignmentMissing)?;
                tmp.add_assign(&gamma_for_copy_permutation);

                factor.mul_assign(&tmp);

                t_num_on_full_domain.sub_assign(&factor);

                // - L_0(z) * alpha_1

                let mut l_0_at_z = evaluate_l0_at_point(required_domain_size as u64, z)?;
                l_0_at_z.mul_assign(&alpha_1);

                t_num_on_full_domain.sub_assign(&l_0_at_z);
            }

            let mut lhs = quotient_at_z;
            lhs.mul_assign(&vanishing_at_z);

            let rhs = t_num_on_full_domain;

            if lhs != rhs {
                dbg!("Circuit is not satisfied");
                return Err(SynthesisError::Unsatisfiable);
            }
        }

        let v = transcript.get_challenge();

        // now construct two polynomials that are opened at z and z*omega

        let mut multiopening_challenge = E::Fr::one();

        let mut poly_to_divide_at_z = t_poly_parts.drain(0..1).collect::<Vec<_>>().pop().unwrap();
        let z_in_domain_size = z.pow(&[required_domain_size as u64]);
        let mut power_of_z = z_in_domain_size;
        for t_part in t_poly_parts.into_iter() {
            poly_to_divide_at_z.add_assign_scaled(&worker, &t_part, &power_of_z);
            power_of_z.mul_assign(&z_in_domain_size);
        }

        // linearization polynomial
        multiopening_challenge.mul_assign(&v);
        poly_to_divide_at_z.add_assign_scaled(&worker, &r_poly, &multiopening_challenge);

        debug_assert_eq!(multiopening_challenge, v.pow(&[1 as u64]));

        // now proceed over all queries

        const THIS_STEP_DILATION: usize = 0;
        for id in queries_with_linearization.state_polys[THIS_STEP_DILATION].iter() {
            multiopening_challenge.mul_assign(&v);
            let poly_ref = monomials_storage.get_poly(*id);
            poly_to_divide_at_z.add_assign_scaled(&worker, poly_ref, &multiopening_challenge);
        }

        for id in queries_with_linearization.witness_polys[THIS_STEP_DILATION].iter() {
            multiopening_challenge.mul_assign(&v);
            let poly_ref = monomials_storage.get_poly(*id);
            poly_to_divide_at_z.add_assign_scaled(&worker, poly_ref, &multiopening_challenge);
        }

        for queries in queries_with_linearization.gate_setup_polys.iter() {
            for id in queries[THIS_STEP_DILATION].iter() {
                multiopening_challenge.mul_assign(&v);
                let poly_ref = monomials_storage.get_poly(*id);
                poly_to_divide_at_z.add_assign_scaled(&worker, poly_ref, &multiopening_challenge);
            }
        }

        // also open selectors at z
        for s in queries_with_linearization.gate_selectors.iter() {
            multiopening_challenge.mul_assign(&v);
            let key = PolyIdentifier::GateSelector(s.name());
            let poly_ref = monomials_storage.get_poly(key);
            poly_to_divide_at_z.add_assign_scaled(&worker, poly_ref, &multiopening_challenge);
        }

        for idx in 0..(num_state_polys - 1) {
            multiopening_challenge.mul_assign(&v);
            let key = PolyIdentifier::PermutationPolynomial(idx);
            let poly_ref = monomials_storage.get_poly(key);
            poly_to_divide_at_z.add_assign_scaled(&worker, poly_ref, &multiopening_challenge);
        }

        // if lookup is present - add it
        if let Some(data) = lookup_data.as_ref() {
            // we need to add t(x), selector(x) and table type(x)
            multiopening_challenge.mul_assign(&v);
            let poly_ref = data.t_poly_monomial.as_ref().unwrap().as_ref();
            poly_to_divide_at_z.add_assign_scaled(&worker, poly_ref, &multiopening_challenge);

            multiopening_challenge.mul_assign(&v);
            let poly_ref = data.selector_poly_monomial.as_ref().unwrap().as_ref();
            poly_to_divide_at_z.add_assign_scaled(&worker, poly_ref, &multiopening_challenge);

            multiopening_challenge.mul_assign(&v);
            let poly_ref = data.table_type_poly_monomial.as_ref().unwrap().as_ref();
            poly_to_divide_at_z.add_assign_scaled(&worker, poly_ref, &multiopening_challenge);
        }

        // now proceed at z*omega
        multiopening_challenge.mul_assign(&v);
        let mut poly_to_divide_at_z_omega = copy_permutation_z_in_monomial_form;
        poly_to_divide_at_z_omega.scale(&worker, multiopening_challenge);

        const NEXT_STEP_DILATION: usize = 1;

        for id in queries_with_linearization.state_polys[NEXT_STEP_DILATION].iter() {
            multiopening_challenge.mul_assign(&v);
            let poly_ref = monomials_storage.get_poly(*id);
            poly_to_divide_at_z_omega.add_assign_scaled(&worker, poly_ref, &multiopening_challenge);
        }

        for id in queries_with_linearization.witness_polys[NEXT_STEP_DILATION].iter() {
            multiopening_challenge.mul_assign(&v);
            let poly_ref = monomials_storage.get_poly(*id);
            poly_to_divide_at_z_omega.add_assign_scaled(&worker, poly_ref, &multiopening_challenge);
        }

        for queries in queries_with_linearization.gate_setup_polys.iter() {
            for id in queries[NEXT_STEP_DILATION].iter() {
                multiopening_challenge.mul_assign(&v);
                let poly_ref = monomials_storage.get_poly(*id);
                poly_to_divide_at_z_omega.add_assign_scaled(
                    &worker,
                    poly_ref,
                    &multiopening_challenge,
                );
            }
        }

        if let Some(data) = lookup_data {
            unreachable!();
        }

        // division in monomial form is sequential, so we parallelize the divisions

        let mut z_by_omega = z;
        z_by_omega.mul_assign(&domain.generator);

        let mut polys = vec![
            (poly_to_divide_at_z, z),
            (poly_to_divide_at_z_omega, z_by_omega),
        ];

        worker.scope(polys.len(), |scope, chunk| {
            for p in polys.chunks_mut(chunk) {
                scope.spawn(move |_| {
                    let (poly, at) = &p[0];
                    let at = *at;
                    let result = divide_single::<E>(poly.as_ref(), at);
                    p[0] = (Polynomial::from_coeffs(result).unwrap(), at);
                });
            }
        });

        let open_at_z_omega = polys.pop().unwrap().0;
        let open_at_z = polys.pop().unwrap().0;

        let opening_at_z =
            commit_using_monomials(&open_at_z, &mon_crs, &worker, &mut multiexp_kern)?;

        let opening_at_z_omega =
            commit_using_monomials(&open_at_z_omega, &mon_crs, &worker, &mut multiexp_kern)?;
        drop(multiexp_kern);

        proof.opening_proof_at_z = opening_at_z;
        proof.opening_proof_at_z_omega = opening_at_z_omega;

        Ok(proof)
    }
}

#[derive(Debug)]
pub struct SortedGateQueries<E: Engine> {
    pub state_polys: Vec<Vec<PolyIdentifier>>,
    pub witness_polys: Vec<Vec<PolyIdentifier>>,
    pub gate_selectors: Vec<Box<dyn GateInternal<E>>>,
    pub gate_setup_polys: Vec<Vec<Vec<PolyIdentifier>>>,
}

/// we sort queries by:
/// - witness first
/// - gate selectors
/// - gate setups in order of gates appearing
/// - additionally we split them into buckets of different dilation
pub fn sort_queries_for_linearization<E: Engine>(
    gates: &Vec<Box<dyn GateInternal<E>>>,
    max_dilation: usize,
) -> SortedGateQueries<E> {
    let state_polys_sorted_by_dilation = vec![vec![]; max_dilation + 1];
    let witness_polys_sorted_by_dilation = vec![vec![]; max_dilation + 1];
    let gate_setup_polys_by_gate_and_dilation = vec![vec![vec![]; max_dilation + 1]; gates.len()];

    let mut queries = SortedGateQueries::<E> {
        state_polys: state_polys_sorted_by_dilation,
        witness_polys: witness_polys_sorted_by_dilation,
        gate_selectors: vec![],
        gate_setup_polys: gate_setup_polys_by_gate_and_dilation,
    };

    let mut opening_requests_before_linearization = std::collections::HashSet::new();
    let mut all_queries = std::collections::HashSet::new();
    let mut sorted_opening_requests = vec![];
    let mut sorted_selector_for_opening = vec![];
    let mut polys_in_linearization = std::collections::HashSet::new();

    let num_gate_types = gates.len();

    for (gate_idx, gate) in gates.iter().enumerate() {
        for q in gate.all_queried_polynomials().into_iter() {
            all_queries.insert(q);
        }
        let queries_to_add = if gate.benefits_from_linearization() {
            if num_gate_types > 1 {
                // there are various gates, so we need to query the selector
                sorted_selector_for_opening.push(gate.box_clone());
            }

            // it's better to linearize the gate
            for q in gate.linearizes_over().into_iter() {
                polys_in_linearization.insert(q);
            }

            gate.needs_opened_for_linearization()
        } else {
            // we will linearize over the selector, so we do not need to query it
            // and instead have to query all other polynomials

            // we blindly add all queried polys
            gate.all_queried_polynomials()
        };

        for q in queries_to_add.into_iter() {
            if !opening_requests_before_linearization.contains(&q) {
                opening_requests_before_linearization.insert(q.clone());

                // push into the corresponding bucket

                let (id, dilation_value) = q.into_id_and_raw_dilation();
                match id {
                    p @ PolyIdentifier::VariablesPolynomial(..) => {
                        queries.state_polys[dilation_value].push(p);
                    }
                    p @ PolyIdentifier::WitnessPolynomial(..) => {
                        queries.witness_polys[dilation_value].push(p);
                    }
                    p @ PolyIdentifier::GateSetupPolynomial(..) => {
                        queries.gate_setup_polys[gate_idx][dilation_value].push(p);
                    }
                    _ => {
                        unreachable!();
                    }
                };

                sorted_opening_requests.push(q);
            }
        }
    }

    // Sanity check: we open everything either in linearization or in plain text!
    {
        let must_open_without_linearization: Vec<_> =
            all_queries.difference(&polys_in_linearization).collect();

        for p in must_open_without_linearization.into_iter() {
            assert!(opening_requests_before_linearization.contains(&p));
        }
    }

    // gate selectors are always sorted by the gate order
    queries.gate_selectors = sorted_selector_for_opening;

    queries
}
