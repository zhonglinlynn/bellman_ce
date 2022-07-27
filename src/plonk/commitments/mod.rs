use crate::pairing::ff::PrimeField;

use crate::plonk::commitments::transcript::*;
use crate::plonk::polynomials::*;
use pairing::Engine;

pub mod transparent;

pub mod transcript;

pub trait CommitmentScheme<E: Engine> {
    type Commitment: std::fmt::Debug;
    type OpeningProof;
    type IntermediateData;
    type Meta: Clone;
    type Prng: Prng<E::Fr>;

    const REQUIRES_PRECOMPUTATION: bool;
    const IS_HOMOMORPHIC: bool;

    fn new_for_size(max_degree_plus_one: usize, meta: Self::Meta) -> Self;
    fn precompute(&self, poly: &Polynomial<E, Coefficients>) -> Option<Self::IntermediateData>;
    fn commit_single(
        &self,
        poly: &Polynomial<E, Coefficients>,
    ) -> (Self::Commitment, Option<Self::IntermediateData>);
    fn commit_multiple(
        &self,
        polynomials: Vec<&Polynomial<E, Coefficients>>,
        degrees: Vec<usize>,
        aggregation_coefficient: E::Fr,
    ) -> (Self::Commitment, Option<Vec<Self::IntermediateData>>);
    fn open_single(
        &self,
        poly: &Polynomial<E, Coefficients>,
        at_point: E::Fr,
        opening_value: E::Fr,
        data: &Option<&Self::IntermediateData>,
        prng: &mut Self::Prng,
    ) -> Self::OpeningProof;
    fn open_multiple(
        &self,
        polynomials: Vec<&Polynomial<E, Coefficients>>,
        degrees: Vec<usize>,
        aggregation_coefficient: E::Fr,
        at_points: Vec<E::Fr>,
        opening_values: Vec<E::Fr>,
        data: &Option<Vec<&Self::IntermediateData>>,
        prng: &mut Self::Prng,
    ) -> Self::OpeningProof;
    fn verify_single(
        &self,
        commitment: &Self::Commitment,
        at_point: E::Fr,
        claimed_value: E::Fr,
        proof: &Self::OpeningProof,
        prng: &mut Self::Prng,
    ) -> bool;
    fn verify_multiple_openings(
        &self,
        commitments: Vec<&Self::Commitment>,
        at_points: Vec<E::Fr>,
        claimed_values: &Vec<E::Fr>,
        aggregation_coefficient: E::Fr,
        proof: &Self::OpeningProof,
        prng: &mut Self::Prng,
    ) -> bool;
}
