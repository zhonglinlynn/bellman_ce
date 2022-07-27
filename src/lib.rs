#![allow(
    dead_code,
    unused_imports,
    unused_mut,
    unused_variables,
    unused_macros,
    unused_assignments,
    unreachable_patterns
)]

#[macro_use]
extern crate cfg_if;
extern crate bit_vec;
extern crate byteorder;
pub extern crate pairing;
extern crate rand;

use crate::pairing::ff;
pub use ff::*;
pub use pairing::*;

pub mod domain;
pub mod gpu;

pub mod plonk;

#[macro_use]
extern crate lazy_static;

pub mod kate_commitment;

pub mod constants;
mod group;
mod multiexp;
mod prefetch;
mod source;

#[cfg(test)]
mod tests;

mod cs;

mod multicore;
pub mod worker {
    pub use super::multicore::*;
}

pub use self::cs::*;

use std::env;
use std::str::FromStr;

#[macro_use]
mod log;

cfg_if! {
    if #[cfg(not(feature = "nolog"))] {
        fn verbose_flag() -> bool {
            option_env!("BELLMAN_VERBOSE").unwrap_or("0") == "1"
        }
    }
}
