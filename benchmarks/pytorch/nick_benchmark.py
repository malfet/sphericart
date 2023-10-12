import argparse
import time

import numpy as np
import torch

import sphericart.torch


docstring = """
Benchmarks for the torch implementation of `sphericart`.

Compares with e3nn and e3nn_jax if those are present and if the comparison is
requested.
"""

try:
    import e3nn

    _HAS_E3NN = True
except ImportError:
    _HAS_E3NN = False

try:
    import jax

    jax.config.update("jax_enable_x64", True)  # enable float64 for jax
    import e3nn_jax
    import jax.numpy as jnp

    _HAS_E3NN_JAX = True
except ImportError:
    _HAS_E3NN_JAX = False


def sphericart_benchmark(
    l_max=10,
    n_samples=10000,
    n_tries=1,
    normalized=False,
    device="cpu",
    dtype=torch.float64,
    compare=False,
    verbose=False,
    warmup=0,
):
    xyz = torch.randn((n_samples, 3), dtype=dtype, device=device)
    sh_calculator = sphericart.torch.SphericalHarmonics(l_max, normalized=normalized)
    omp_threads = sh_calculator.omp_num_threads()
    print(
        f"**** Timings for l_max={l_max}, n_samples={n_samples}, n_tries={n_tries}, "
        + f"dtype={dtype}, device={device}, omp_num_threads={omp_threads} ****"
    )

    time_noderi = np.zeros(n_tries + warmup)
    for i in range(n_tries + warmup):
        elapsed = -time.time()
        sh_sphericart = sh_calculator.compute(xyz)
        elapsed += time.time()
        time_noderi[i] = elapsed

    mean_time = time_noderi[warmup:].mean() / n_samples
    std_time = time_noderi[warmup:].std() / n_samples
    print(
        f" No derivatives: {mean_time * 1e9: 10.1f} ns/sample ± "
        + f"{std_time * 1e9: 10.1f} (std)"
    )
    if verbose:
        print("Warm-up timings / sec.:\n", time_noderi[:warmup])

    time_deri = np.zeros(n_tries + warmup)
    for i in range(n_tries + warmup):
        elapsed = -time.time()
        sh_sphericart, dsh_sphericart = sh_calculator.compute_with_gradients(xyz)
        elapsed += time.time()
        time_deri[i] = elapsed

    mean_time = time_deri[warmup:].mean() / n_samples
    std_time = time_deri[warmup:].std() / n_samples
    print(
        f" Derivatives:    {mean_time * 1e9:10.1f} ns/sample ± "
        + f"{std_time * 1e9:10.1f} (std)"
    )
    if verbose:
        print("Warm-up timings / sec.:\n", time_deri[:warmup])

    # autograd
    xyz = xyz.clone().detach().type(dtype).to(device).requires_grad_()

    print (xyz.requires_grad)

    time_fw = np.zeros(n_tries + warmup)
    time_bw = np.zeros(n_tries + warmup)

    for i in range(n_tries + warmup):
        elapsed = -time.time()
        sh_sphericart = sh_calculator.compute(xyz)
        elapsed += time.time()
        time_fw[i] = elapsed

        sph_sum = torch.sum(sh_sphericart)
        elapsed = -time.time()
        sph_sum.backward()
        elapsed += time.time()
        time_bw[i] = elapsed
        print (xyz.grad)
        
        xyz.grad.zero_()

    print(
        f" Forward:        {time_fw[warmup:].mean()/n_samples*1e9: 10.1f} ns/sample ± \
{time_fw[warmup:].std()/n_samples*1e9: 10.1f} (std)"
    )
    if verbose:
        print("Warm-up timings / sec.:\n", time_fw[:warmup])

    print(
        f" Backward:       {time_bw[warmup:].mean()/n_samples*1e9: 10.1f} ns/sample ± \
{time_bw[warmup:].std()/n_samples*1e9: 10.1f} (std)"
    )
    if verbose:
        print("Warm-up timings / sec.:\n", time_bw[:warmup])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=docstring)

    parser.add_argument("-l", type=int, default=10, help="maximum angular momentum")
    parser.add_argument("-s", type=int, default=10000, help="number of samples")
    parser.add_argument("-t", type=int, default=1, help="number of runs/sample")
    parser.add_argument(
        "-cpu", type=int, default=1, help="print CPU results (0=False, 1=True)"
    )
    parser.add_argument(
        "-gpu", type=int, default=1, help="print GPU results (0=False, 1=True)"
    )
    parser.add_argument(
        "--normalized",
        action="store_true",
        default=False,
        help="compute normalized spherical harmonics",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        default=False,
        help="compare timings with other codes, if installed",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="verbose timing output",
    )
    parser.add_argument(
        "--warmup", type=int, default=16, help="number of warm-up evaluations"
    )

    args = parser.parse_args()

    if torch.cuda.is_available() and args.gpu:
        sphericart_benchmark(
            args.l,
            args.s,
            args.t,
            args.normalized,
            device="cuda",
            dtype=torch.float64,
            compare=args.compare,
            verbose=args.verbose,
            warmup=args.warmup,
        )
        sphericart_benchmark(
            args.l,
            args.s,
            args.t,
            args.normalized,
            device="cuda",
            dtype=torch.float32,
            compare=args.compare,
            verbose=args.verbose,
            warmup=args.warmup,
        )
