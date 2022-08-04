// For benchmarking
use std::time::{Duration, Instant};

// We'll use these interfaces to construct our circuit.
use bellman_gpu::{Circuit, ConstraintSystem, SynthesisError};
// Bring in some tools for using pairing-friendly curves
use bellman_gpu::pairing::ff::Field;
use bellman_gpu::pairing::Engine;

use bellman_gpu::plonk::{make_precomputations, prove_by_steps};

const MIMC_ROUNDS: usize = 322;

/// This is an implementation of MiMC, specifically a
/// variant named `LongsightF322p3` for BLS12-381.
/// See http://eprint.iacr.org/2016/492 for more
/// information about this construction.
///
/// ```
/// function LongsightF322p3(xL ⦂ Fp, xR ⦂ Fp) {
///     for i from 0 up to 321 {
///         xL, xR := xR + (xL + Ci)^3, xL
///     }
///     return xL
/// }
/// ```
fn mimc<E: Engine>(mut xl: E::Fr, mut xr: E::Fr, constants: &[E::Fr]) -> E::Fr {
    assert_eq!(constants.len(), MIMC_ROUNDS);

    for i in 0..MIMC_ROUNDS {
        let mut tmp1 = xl;
        tmp1.add_assign(&constants[i]);
        let mut tmp2 = tmp1;
        tmp2.square();
        tmp2.mul_assign(&tmp1);
        tmp2.add_assign(&xr);
        xr = xl;
        xl = tmp2;
    }

    xl
}

/// This is our demo circuit for proving knowledge of the
/// preimage of a MiMC hash invocation.
#[derive(Clone)]
struct MiMCDemo<'a, E: Engine> {
    xl: Option<E::Fr>,
    xr: Option<E::Fr>,
    constants: &'a [E::Fr],
}

/// Our demo circuit implements this `Circuit` trait which
/// is used during paramgen and proving in order to
/// synthesize the constraint system.
impl<'a, E: Engine> Circuit<E> for MiMCDemo<'a, E> {
    fn synthesize<CS: ConstraintSystem<E>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
        assert_eq!(self.constants.len(), MIMC_ROUNDS);

        // Allocate the first component of the preimage.
        let mut xl_value = self.xl;
        let mut xl = cs.alloc(|| "preimage xl",|| xl_value.ok_or(SynthesisError::AssignmentMissing))?;

        // Allocate the second component of the preimage.
        let mut xr_value = self.xr;
        let mut xr = cs.alloc(|| "preimage xr",|| xr_value.ok_or(SynthesisError::AssignmentMissing))?;

        for i in 0..MIMC_ROUNDS {
            // xL, xR := xR + (xL + Ci)^3, xL
            let cs = &mut cs.namespace(|| format!("round {}", i));

            // tmp = (xL + Ci)^2
            let tmp_value = xl_value.map(|mut e| {
                e.add_assign(&self.constants[i]);
                e.square();
                e
            });
            let tmp = cs.alloc(|| "tmp",|| tmp_value.ok_or(SynthesisError::AssignmentMissing))?;

            cs.enforce(
                || "tmp = (xL + Ci)^2",
                |lc| lc + xl + (self.constants[i], CS::one()),
                |lc| lc + xl + (self.constants[i], CS::one()),
                |lc| lc + tmp,
            );

            // new_xL = xR + (xL + Ci)^3
            // new_xL = xR + tmp * (xL + Ci)
            // new_xL - xR = tmp * (xL + Ci)
            let new_xl_value = xl_value.map(|mut e| {
                e.add_assign(&self.constants[i]);
                e.mul_assign(&tmp_value.unwrap());
                e.add_assign(&xr_value.unwrap());
                e
            });

            let new_xl = if i == (MIMC_ROUNDS - 1) {
                // This is the last round, xL is our image and so
                // we allocate a public input.
                cs.alloc_input(|| "image",|| new_xl_value.ok_or(SynthesisError::AssignmentMissing))?
            } else {
                cs.alloc(
                    || "new_xl",
                    || new_xl_value.ok_or(SynthesisError::AssignmentMissing),
                )?
            };

            cs.enforce(
                || "new_xL = xR + (xL + Ci)^3",
                |lc| lc + tmp,
                |lc| lc + xl + (self.constants[i], CS::one()),
                |lc| lc + new_xl - xr,
            );

            // xR = xL
            xr = xl;
            xr_value = xl_value;

            // xL = new_xL
            xl = new_xl;
            xl_value = new_xl_value;
        }

        Ok(())
    }
}

#[test]
fn transpile_and_prove_with_no_precomputations_mimc() {
    use crate::Circuit;
    use bellman_gpu::kate_commitment::*;
    use bellman_gpu::plonk::better_cs::cs::Circuit as PlonkCircuit;
    use bellman_gpu::plonk::better_cs::{
        adaptor::{AdaptorCircuit, Transpiler},
        cs::PlonkCsWidth4WithNextStepParams,
        generator::*,
        keys::*,
        verifier::*,
    };
    use bellman_gpu::plonk::commitments::transcript::keccak_transcript::*;
    use bellman_gpu::worker::Worker;
    use pairing::bn256::{Bn256, Fr};
    use rand::{thread_rng, Rng};

    let rng = &mut thread_rng();

    // Generate the MiMC round constants
    let constants = (0..MIMC_ROUNDS).map(|_| rng.gen()).collect::<Vec<_>>();
    // let constants = (0..MIMC_ROUNDS)
    //     .map(|_| Fr::from_str("3673").unwrap())
    //     .collect::<Vec<_>>();

    let c = MiMCDemo::<Bn256> {
        xl: None,
        xr: None,
        constants: &constants,
    };

    let mut transpiler = Transpiler::<Bn256, PlonkCsWidth4WithNextStepParams>::new();

    c.synthesize(&mut transpiler)
        .expect("sythesize into traspilation must succeed");

    let hints = transpiler.into_hints();

    let a = rng.gen();
    let b = rng.gen();
    // let a = Fr::from_str("48577").unwrap();
    // let b = Fr::from_str("22580").unwrap();
    let c = MiMCDemo::<Bn256> {
        xl: Some(a),
        xr: Some(b),
        constants: &constants,
    };

    let adapted_curcuit =
        AdaptorCircuit::<Bn256, PlonkCsWidth4WithNextStepParams, _>::new(c.clone(), &hints);
    let mut assembly = GeneratorAssembly4WithNextStep::<Bn256>::new();
    adapted_curcuit
        .synthesize(&mut assembly)
        .expect("sythesize of transpiled into CS must succeed");
    println!("transpile into {:?} gates", assembly.num_gates());
    assembly.finalize();
    println!("finalize into {:?} gates", assembly.num_gates());

    let worker = Worker::new();

    let setup = assembly.setup(&worker).unwrap();

    let crs_mons =
        Crs::<Bn256, CrsForMonomialForm>::crs_42(setup.permutation_polynomials[0].size(), &worker);

    let verification_key = VerificationKey::from_setup(&setup, &worker, &crs_mons).unwrap();

    type Transcr = RollingKeccakTranscript<Fr>;
    let mut total_proving = Duration::new(0, 0);
    let samples = 5;
    for test_i in 0..samples {
        println!("test {:?}\n", test_i);
        let now = Instant::now();
        let proof =
            prove_by_steps::<_, _, Transcr>(c.clone(), &hints, &setup, None, &crs_mons, None)
                .unwrap();
        let single_prove_time = now.elapsed();

        let is_valid = verify::<Bn256, PlonkCsWidth4WithNextStepParams, Transcr>(
            &proof,
            &verification_key,
            None,
        )
        .unwrap();

        assert!(is_valid, "proof verification failed");

        total_proving += single_prove_time;
    }

    let proving_avg = total_proving / samples;
    let proving_avg = proving_avg.as_millis();

    println!("Average proving time: {:?} ms", proving_avg);
}

#[test]
fn transpile_and_prove_with_precomputations_mimc() {
    use crate::Circuit;
    use bellman_gpu::kate_commitment::*;
    use bellman_gpu::plonk::better_cs::cs::Circuit as PlonkCircuit;
    use bellman_gpu::plonk::better_cs::{
        adaptor::{AdaptorCircuit, Transpiler},
        cs::PlonkCsWidth4WithNextStepParams,
        generator::*,
        keys::*,
        verifier::*,
    };
    use bellman_gpu::plonk::commitments::transcript::keccak_transcript::*;
    use bellman_gpu::worker::Worker;
    use pairing::bn256::{Bn256, Fr};
    use rand::{thread_rng, Rng};

    let rng = &mut thread_rng();

    // Generate the MiMC round constants
    let constants = (0..MIMC_ROUNDS).map(|_| rng.gen()).collect::<Vec<_>>();
    // let constants = (0..MIMC_ROUNDS)
    //     .map(|_| Fr::from_str("3673").unwrap())
    //     .collect::<Vec<_>>();

    let c = MiMCDemo::<Bn256> {
        xl: None,
        xr: None,
        constants: &constants,
    };

    let mut transpiler = Transpiler::<Bn256, PlonkCsWidth4WithNextStepParams>::new();

    c.synthesize(&mut transpiler)
        .expect("sythesize into traspilation must succeed");

    let hints = transpiler.into_hints();

    let a = rng.gen();
    let b = rng.gen();
    // let a = Fr::from_str("48577").unwrap();
    // let b = Fr::from_str("22580").unwrap();
    let c = MiMCDemo::<Bn256> {
        xl: Some(a),
        xr: Some(b),
        constants: &constants,
    };

    let adapted_curcuit =
        AdaptorCircuit::<Bn256, PlonkCsWidth4WithNextStepParams, _>::new(c.clone(), &hints);
    let mut assembly = GeneratorAssembly4WithNextStep::<Bn256>::new();
    adapted_curcuit
        .synthesize(&mut assembly)
        .expect("sythesize of transpiled into CS must succeed");
    println!("transpile into {:?} gates", assembly.num_gates());
    assembly.finalize();
    println!("finalize into {:?} gates", assembly.num_gates());

    let worker = Worker::new();

    let setup = assembly.setup(&worker).unwrap();

    let crs_mons =
        Crs::<Bn256, CrsForMonomialForm>::crs_42(setup.permutation_polynomials[0].size(), &worker);

    let verification_key = VerificationKey::from_setup(&setup, &worker, &crs_mons).unwrap();

    // precompute
    println!("precompute");
    let now = Instant::now();
    let precomputations = make_precomputations(&setup).unwrap();
    println!("precompute taken {:?}", now.elapsed());

    type Transcr = RollingKeccakTranscript<Fr>;
    let mut total_proving = Duration::new(0, 0);
    let samples = 5;
    for test_i in 0..samples {
        println!("test {:?}\n", test_i);
        let now = Instant::now();
        let proof = prove_by_steps::<_, _, Transcr>(
            c.clone(),
            &hints,
            &setup,
            Some(&precomputations),
            &crs_mons,
            None,
        )
        .unwrap();
        let single_prove_time = now.elapsed();

        let is_valid = verify::<Bn256, PlonkCsWidth4WithNextStepParams, Transcr>(
            &proof,
            &verification_key,
            None,
        )
        .unwrap();
        assert!(is_valid, "proof verification failed");

        total_proving += single_prove_time;
    }

    let proving_avg = total_proving / samples;
    let proving_avg = proving_avg.as_millis();

    println!("Average proving time: {:?} ms", proving_avg);
}
