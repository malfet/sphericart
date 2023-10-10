#ifndef SPHERICART_TORCH_CUDA_HPP
#define SPHERICART_TORCH_CUDA_HPP
#include <vector>
#include <cmath>

namespace sphericart_torch {

bool adjust_cuda_shared_memory(
    const size_t element_size,
    const int64_t l_max,
    const int64_t GRID_DIM_X,
    const int64_t GRID_DIM_Y,
    const bool requires_grad,
    const bool requires_hessian);


template <typename scalar_t>
void spherical_harmonics_cuda(
    const scalar_t *xyz,
    const int64_t nsamples,
    const scalar_t *prefactors,
    const int64_t nprefactors,
    const int64_t l_max,
    const bool normalize,
    const int64_t GRID_DIM_X,
    const int64_t GRID_DIM_Y,
    const bool gradients,
    const bool hessian,
    scalar_t *const sph,
    scalar_t *const dsph,
    scalar_t *const ddsph);

template <typename scalar_t>
void spherical_harmonics_backward_cuda(
    const scalar_t *dsph,
    const scalar_t *sph_grad,
    const int64_t nsamples,
    const int64_t lmax,
    const bool requires_grad,
    scalar_t *const xyz_grad);

template<typename scalar_t>
void prefactors_cuda(
    const int64_t l_max, 
    const size_t element_size, 
    scalar_t * const prefactors);

}

#endif
