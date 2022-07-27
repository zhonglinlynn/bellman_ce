# bellman_gpu
 
This is a fork of [`bellman_ce`](https://github.com/matter-labs/bellman). We have drastically reduced the time of proving of Plonk(both `PlonkCsWidth4WithNextStep` and `PlonkCsWidth4WithNextStepAndCustomGates`) with GPU and other tricks.

## GPU
We use GPU parallel acceleration to the FFT and Multiexponentation algorithms that is used in the awesome library [`bellperson`](https://github.com/filecoin-project/bellperson). 

### Requirements
- NVIDIA or AMD GPU Graphics Driver
- OpenCL

### Environment variables

The gpu extension contains some env vars that may be set externally to this library.

- `BELLMAN_USE_CPU`

    The default feature is to use GPU. If `BELLMAN_USE_CPU` is set to `1`, then CPU is forced to use.

    ```rust
    // Example
    env::set_var("BELLMAN_USE_CPU", "1");
    ```

- `BELLMAN_NUM_GPUS_PER_GROUP`, `BELLMAN_GPU_GROUP_INDEX`

    The default feature is to use all the GPUs in FFT and Multi-exp. If you want to use a few of them, these two variables just come in handy. You can divide GPUs into several groups. Each group has `BELLMAN_NUM_GPUS_PER_GROUP` GPUs, and you can use the `BELLMAN_GPU_GROUP_INDEX`th group. For example, if `BELLMAN_NUM_GPUS_PER_GROUP=2` and  `BELLMAN_GPU_GROUP_INDEX=0`, it means you can use the first two GPUs.

    ```rust
    Example
    env::set_var("BELLMAN_NUM_GPUS_PER_GROUP", "2");
    env::set_var("BELLMAN_GPU_GROUP_INDEX", "0");
    ```

- `BELLMAN_CUSTOM_GPU`

    Will allow for adding a GPU not in the tested list. This requires researching the name of the GPU device and the number of cores in the format `["name:cores"]`.

    ```rust
    // Example
    env::set_var("BELLMAN_CUSTOM_GPU", "GeForce RTX 2080 Ti:4352, GeForce GTX 1060:1280");
    ```

- `BELLMAN_CPU_UTILIZATION`

    Can be set in the interval [0,1] to designate a proportion of the multiexponenation calculation to be moved to cpu in parallel to the GPU to keep all hardware occupied.

    ```rust
    // Example
    env::set_var("BELLMAN_CPU_UTILIZATION", "0.5");
    ```

- `BELLMAN_NUM_CPUS`
    
    Number of CPUs used to parallel computations.


#### Supported / Tested Cards

Depending on the size of the proof being passed to the gpu for work, certain cards will not be able to allocate enough memory to either the FFT or Multiexp kernel. Below are a list of devices that work for small sets. 

| Device Name            | Cores | Comments       |
|------------------------|-------|----------------|
| Quadro RTX 6000        | 4608  |                |
| TITAN RTX              | 4608  |                |
| Tesla V100             | 5120  |                |
| Tesla P100             | 3584  |                |
| Tesla T4               | 2560  |                |
| Quadro M5000           | 2048  |                |
| GeForce RTX 3090       |10496  |                |
| GeForce RTX 3080       | 8704  |                |
| GeForce RTX 3070       | 5888  |                |
| GeForce RTX 2080 Ti    | 4352  |                |
| GeForce RTX 2080 SUPER | 3072  |                |
| GeForce RTX 2080       | 2944  |                |
| GeForce RTX 2070 SUPER | 2560  |                |
| GeForce GTX 1080 Ti    | 3584  |                |
| GeForce GTX 1080       | 2560  |                |
| GeForce GTX 2060       | 1920  |                |
| GeForce GTX 1660 Ti    | 1536  |                |
| GeForce GTX 1060       | 1280  |                |
| GeForce GTX 1650 SUPER | 1280  |                |
| GeForce GTX 1650       |  896  |                |
|                        |       |                |
| gfx1010                | 2560  | AMD RX 5700 XT |
| gfx906                 | 7400  | AMD RADEON VII |
|------------------------|-------|----------------|

## Performance

The prover's performance depend on mant factors including number of threads, GPUs. The computation we benchmark is our circuit of zkRollup application. We run on Xeon(R) Platinum 8163 @2.5Ghz, 300GB RAM, two NVIDIA T4 Cards, using 32 cores.

| Number of gates   | Proving time using CPU   | Proving time using GPU   | Improvement   |
| :---------------: | :----------------------: | :----------------------: | :-----------: |
|  2^22             | 62 sec                   | 18 sec                   | 3.4x          |
|  2^23             | 107 sec                  | 33 sec                   | 3.2x          |
|  2^24             | 188 sec                  | 62 sec                   | 3.0x          |
|  2^25             | 355 sec                  | 124 sec                  | 2.9x          |
|  2^26             | 690 sec                  | 210 sec                  | 3.3x          |


## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

