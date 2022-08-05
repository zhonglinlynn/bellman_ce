use fs2::FileExt;
use log::{debug, info, warn};
use std::fs::File;
use std::path::PathBuf;

const GPU_LOCK_NAME: &str = "bellman.gpu.lock";
const PRIORITY_LOCK_NAME: &str = "bellman.priority.lock";
fn tmp_path(filename: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(filename);
    p
}

pub fn get_lock_name_and_gpu_range(all_gpus: usize) -> (String, Range<usize>) {
    let mut num_gpus_per_group = all_gpus;
    let mut group_index = 0usize;

    let maybe_num = env::var("BELLMAN_NUM_GPUS_PER_GROUP");
    let maybe_index = env::var("BELLMAN_GPU_GROUP_INDEX");

    if maybe_num.is_ok() && maybe_index.is_ok() {
        let maybe_num = maybe_num.unwrap().parse();
        let maybe_index = maybe_index.unwrap().parse();
        if maybe_num.is_ok() && maybe_index.is_ok() {
            num_gpus_per_group = maybe_num.unwrap();
            group_index = maybe_index.unwrap();
        }
    }

    assert!(
        (group_index + 1) * num_gpus_per_group <= all_gpus,
        "BELLMAN_GPU_GROUP_INDEX and BELLMAN_NUM_GPUS_PER_GROUP error"
    );

    (
        group_index.to_string(),
        Range {
            start: group_index * num_gpus_per_group,
            end: (group_index + 1) * num_gpus_per_group,
        },
    )
}

/// `GPULock` prevents two kernel objects to be instantiated simultaneously.
#[derive(Debug)]
pub struct GPULock(File);
impl GPULock {
    pub fn lock(index: String) -> GPULock {
        let filename = GPU_LOCK_NAME.to_string() + &index;
        let gpu_lock_file = tmp_path(&filename);
        debug!("Acquiring GPU lock at {:?} ...", &gpu_lock_file);
        let f = File::create(&gpu_lock_file)
            .unwrap_or_else(|_| panic!("Cannot create GPU lock file at {:?}", &gpu_lock_file));
        f.lock_exclusive().unwrap();
        debug!("GPU lock acquired!");
        GPULock(f)
    }
}
impl Drop for GPULock {
    fn drop(&mut self) {
        self.0.unlock().unwrap();
        debug!("GPU lock released!");
    }
}

/// `PrioriyLock` is like a flag. When acquired, it means a high-priority process
/// needs to acquire the GPU really soon. Acquiring the `PriorityLock` is like
/// signaling all other processes to release their `GPULock`s.
/// Only one process can have the `PriorityLock` at a time.
#[derive(Debug)]
pub struct PriorityLock(File);
impl PriorityLock {
    pub fn lock() -> PriorityLock {
        let priority_lock_file = tmp_path(PRIORITY_LOCK_NAME);
        debug!("Acquiring priority lock at {:?} ...", &priority_lock_file);
        let f = File::create(&priority_lock_file).unwrap_or_else(|_| {
            panic!(
                "Cannot create priority lock file at {:?}",
                &priority_lock_file
            )
        });
        f.lock_exclusive().unwrap();
        debug!("Priority lock acquired!");
        PriorityLock(f)
    }
    pub fn wait(priority: bool) {
        if !priority {
            File::create(tmp_path(PRIORITY_LOCK_NAME))
                .unwrap()
                .lock_exclusive()
                .unwrap();
        }
    }
    pub fn should_break(priority: bool) -> bool {
        !priority
            && File::create(tmp_path(PRIORITY_LOCK_NAME))
                .unwrap()
                .try_lock_exclusive()
                .is_err()
    }
}
impl Drop for PriorityLock {
    fn drop(&mut self) {
        self.0.unlock().unwrap();
        debug!("Priority lock released!");
    }
}


use ec_gpu_gen::{EcError as GPUError, EcResult as GPUResult};
use ec_gpu_gen::multiexp::MultiexpKernel;
use ec_gpu_gen::fft::FftKernel;

use crate::domain::create_fft_kernel;
use crate::multiexp::create_multiexp_kernel;
use crate::pairing::Engine;
use std::env;
use std::ops::Range;

macro_rules! locked_kernel {
    ($class:ident, $kern:ident, $func:ident, $name:expr) => {
        pub struct $class<'a, E>
        where
            E: Engine,
        {
            log_d: usize,
            priority: bool,
            kernel: Option<$kern<'a, E>>,
        }

        impl<E> $class<'_, E>
        where
            E: Engine,
        {
            pub fn new(log_d: usize, priority: bool) -> $class<'static, E> {
                $class::<E> {
                    log_d,
                    priority,
                    kernel: None,
                }
            }

            fn init(&mut self) {
                if self.kernel.is_none() {
                    PriorityLock::wait(self.priority);
                    info!("GPU is available for {}!", $name);
                    self.kernel = $func::<E>(self.log_d, self.priority);
                }
            }

            fn free(&mut self) {
                if let Some(_kernel) = self.kernel.take() {
                    warn!(
                        "GPU acquired by a high priority process! Freeing up {} kernels...",
                        $name
                    );
                }
            }

            pub fn with<F, R>(&mut self, mut f: F) -> GPUResult<R>
            where
                F: FnMut(&mut $kern<E>) -> GPUResult<R>,
            {
                if let Ok(flag) = std::env::var("BELLMAN_USE_CPU") {
                    if flag == "1" {
                        return Err(GPUError::Simple("GPUs wrong 1"));
                    }
                }

                self.init();

                loop {
                    if let Some(ref mut k) = self.kernel {
                        match f(k) {
                            Err(GPUError::GPUTaken) => {
                                self.free();
                                self.init();
                            }
                            Err(e) => {
                                warn!("GPU {} failed! Falling back to CPU... Error: {}", $name, e);
                                return Err(e);
                            }
                            Ok(v) => return Ok(v),
                        }
                    } else {
                        return Err(GPUError::Simple("GPUs wrong 2"));
                    }
                }
            }
        }
    };
}

locked_kernel!(
    LockedMultiFFTKernel,
    FftKernel,
    create_fft_kernel,
    "FFT"
);
locked_kernel!(
    LockedMultiexpKernel,
    MultiexpKernel,
    create_multiexp_kernel,
    "Multiexp"
);
