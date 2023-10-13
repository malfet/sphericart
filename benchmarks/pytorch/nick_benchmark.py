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
    sh_calculator = sphericart.torch.SphericalHarmonics(l_max, normalized=normalized)
    
    xyz_cpu = torch.randn((n_samples, 3), dtype=dtype, device="cpu", requires_grad=True)
    xyz_gpu = xyz_cpu.clone().detach().type(dtype).to(device).requires_grad_(True)

    #sh_sphericart_gpu = sh_calculator.compute(xyz_gpu)
    #sh_sphericart_cpu = sh_calculator.compute(xyz_cpu)

    sph_cpu, sph_grad_cpu = sh_calculator.compute_with_gradients(xyz_cpu)
    sph_gpu, sph_grad_gpu = sh_calculator.compute_with_gradients(xyz_gpu)
    #print (sph_grad_cpu)
    #print (sph_grad_gpu)
    assert torch.allclose(sph_grad_cpu, sph_grad_gpu.cpu()), ""
    #sph_sum_gpu = torch.sum(sh_sphericart_gpu)
    #sph_sum_cpu = torch.sum(sh_sphericart_cpu)

    #sph_sum_gpu.backward()
    #sph_sum_cpu.backward()

    #print("GPU", xyz_gpu.grad)
    #print("CPU", xyz_cpu.grad)
    
    #xyz_cpu.grad.zero_()
    #xyz_gpu.grad.zero_()

    

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