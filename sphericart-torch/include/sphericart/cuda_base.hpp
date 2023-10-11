#ifndef SPHERICART_TORCH_CUDA_BASE_HPP
#define SPHERICART_TORCH_CUDA_BASE_HPP
#include <vector>
#include <cmath>

namespace sphericart_torch
{

    bool adjust_cuda_shared_memory(
        size_t element_size,
        int64_t l_max,
        int64_t GRID_DIM_X,
        int64_t GRID_DIM_Y,
        bool requires_grad,
        bool requires_hessian);

    template <typename scalar_t>
    void spherical_harmonics_cuda(
        scalar_t * xyz,
        int64_t nsamples,
        scalar_t * prefactors,
        int64_t nprefactors,
        int64_t l_max,
        bool normalize,
        int64_t GRID_DIM_X,
        int64_t GRID_DIM_Y,
        bool gradients,
        bool hessian,
        scalar_t * sph,
        scalar_t * dsph,
        scalar_t * ddsph);

    template <typename scalar_t>
    void spherical_harmonics_backward_cuda(
        scalar_t * dsph,
        scalar_t * sph_grad,
        int64_t nsamples,
        int64_t lmax,
        bool requires_grad,
        scalar_t * xyz_grad);

    template <typename scalar_t>
    void prefactors_cuda(
        int64_t l_max,
        scalar_t *  prefactors);

}

#endif
