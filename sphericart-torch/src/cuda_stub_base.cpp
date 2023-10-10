#include "sphericart/cuda_base.hpp"

#include <stdexcept>

bool sphericart_torch::adjust_cuda_shared_memory(size_t, int64_t, int64_t, int64_t, bool, bool)
{
    throw std::runtime_error("sphericart_torch was not compiled with CUDA support");
}

template <typename scalar_t>
void sphericart_torch::spherical_harmonics_cuda(
    const scalar_t *,
    const int64_t,
    const scalar_t *,
    const int64_t,
    const int64_t,
    const bool,
    const int64_t,
    const int64_t,
    const bool,
    const bool,
    scalar_t *const,
    scalar_t *const,
    scalar_t *const)
{
    throw std::runtime_error("sphericart_torch was not compiled with CUDA support");
}

template <typename scalar_t>
void sphericart_torch::spherical_harmonics_backward_cuda(
    const scalar_t *,
    const scalar_t *,
    const int64_t,
    const int64_t,
    const bool,
    scalar_t *const)
{
    throw std::runtime_error("sphericart_torch was not compiled with CUDA support");
}

template <typename scalar_t>
void sphericart_torch::prefactors_cuda(
    const int64_t,
    const size_t,
    scalar_t *const)
{
    return at::Tensor();
}
