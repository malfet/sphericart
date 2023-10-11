import torch

import sphericart.torch


torch.manual_seed(0)

def xyz():
    return 6 * torch.randn(100, 3, dtype=torch.float64, requires_grad=True)

def test_cpu_vs_cuda(xyz):
    if torch.cuda.is_available():
        xyz_cuda = xyz.to("cuda")

        calculator = sphericart.torch.SphericalHarmonics(l_max=6, normalized=False)
        sph = calculator.compute(xyz)

        sph_cuda = calculator.compute(xyz_cuda)
        print (sph_cuda)
        print (sph)

        sph, grad_sph = calculator.compute_with_gradients(xyz)
        sph_cuda, grad_sph_cuda = calculator.compute_with_gradients(xyz_cuda)

        print (grad_sph)
        print (grad_sph_cuda)

if __name__ == "__main__":
    test_xyz = xyz()

    test_cpu_vs_cuda(test_xyz)