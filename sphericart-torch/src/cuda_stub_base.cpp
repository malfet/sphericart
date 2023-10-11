#include "sphericart/cuda_base.hpp"

#include <stdexcept>

bool sphericart_torch::adjust_cuda_shared_memory(size_t, int64_t, int64_t, int64_t, bool, bool)
{
    throw std::runtime_error("sphericart_torch was not compiled with CUDA support");
}

template <typename scalar_t>
void sphericart_torch::spherical_harmonics_cuda(
    scalar_t *__restrict__,
    int64_t,
    scalar_t *__restrict__,
    int64_t,
    int64_t,
    bool,
    int64_t,
    int64_t,
    bool,
    bool,
    scalar_t *__restrict__,
    scalar_t *__restrict__,
    scalar_t *__restrict__)
{
    throw std::runtime_error("sphericart_torch was not compiled with CUDA support");
}

template <typename scalar_t>
void sphericart_torch::spherical_harmonics_backward_cuda(
    scalar_t *__restrict__,
    scalar_t *__restrict__,
    int64_t,
    int64_t,
    bool,
    scalar_t *__restrict__)
{
    throw std::runtime_error("sphericart_torch was not compiled with CUDA support");
}

template <typename scalar_t>
void sphericart_torch::prefactors_cuda(
    const int64_t,
    scalar_t *__restrict__)
{
    return at::Tensor();
}
