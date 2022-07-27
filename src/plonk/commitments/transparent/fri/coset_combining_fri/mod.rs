pub mod fri;
// pub mod query_producer;
// pub mod verifier;
pub mod precomputation;

use crate::ff::PrimeField;
use crate::plonk::commitments::transparent::iop_compiler::*;
use crate::worker::Worker;
use crate::SynthesisError;

use crate::plonk::commitments::transcript::Prng;
use crate::plonk::polynomials::*;
use pairing::Engine;

pub trait FriProofPrototype<F: PrimeField, I: IopInstance<F>> {
    fn get_roots(&self) -> Vec<I::Commitment>;
    fn get_final_root(&self) -> I::Commitment;
    fn get_final_coefficients(&self) -> Vec<F>;
}

pub trait FriProof<F: PrimeField, I: IopInstance<F>> {
    fn get_final_coefficients(&self) -> &[F];
    fn get_queries(&self) -> &Vec<Vec<I::Query>>;
}

pub trait FriPrecomputations<F: PrimeField> {
    fn new_for_domain_size(size: usize) -> Self;
    fn omegas_inv_bitreversed(&self) -> &[F];
    fn domain_size(&self) -> usize;
}

pub trait FriIop<E: Engine> {
    const DEGREE: usize;

    type IopType: IopInstance<E::Fr>;
    type ProofPrototype: FriProofPrototype<E::Fr, Self::IopType>;
    type Proof: FriProof<E::Fr, Self::IopType>;
    type Params: Clone + std::fmt::Debug;

    fn proof_from_lde<
        P: Prng<E::Fr, Input = <Self::IopType as IopInstance<E::Fr>>::Commitment>,
        C: FriPrecomputations<E::Fr>,
    >(
        lde_values: &Polynomial<E, Values>,
        lde_factor: usize,
        output_coeffs_at_degree_plus_one: usize,
        precomputations: &C,
        worker: &Worker,
        prng: &mut P,
        params: &Self::Params,
    ) -> Result<Self::ProofPrototype, SynthesisError>;

    fn prototype_into_proof(
        prototype: Self::ProofPrototype,
        iop_values: &Polynomial<E, Values>,
        natural_first_element_indexes: Vec<usize>,
        params: &Self::Params,
    ) -> Result<Self::Proof, SynthesisError>;

    fn get_fri_challenges<
        P: Prng<E::Fr, Input = <Self::IopType as IopInstance<E::Fr>>::Commitment>,
    >(
        proof: &Self::Proof,
        prng: &mut P,
        params: &Self::Params,
    ) -> Vec<E::Fr>;

    fn verify_proof_with_challenges(
        proof: &Self::Proof,
        natural_element_indexes: Vec<usize>,
        expected_value: &[E::Fr],
        fri_challenges: &[E::Fr],
        params: &Self::Params,
    ) -> Result<bool, SynthesisError>;
}
