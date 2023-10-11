
#include <torch/script.h>

#include "sphericart/torch.hpp"
#include "sphericart/autograd_torch.hpp"
#include "sphericart/cuda_base.hpp"

using namespace sphericart_torch;

//extern template void sphericart_torch::prefactors_cuda<float>(int64_t l_max, float * __restrict__ ptr);
//extern template void sphericart_torch::prefactors_cuda<double>(int64_t l_max, double * __restrict__ ptr);

SphericalHarmonics::SphericalHarmonics(int64_t l_max, bool normalized, bool backward_second_derivatives): 
    l_max_(l_max),
    normalized_(normalized),
    backward_second_derivatives_(backward_second_derivatives),
    calculator_double_(l_max_, normalized_),
    calculator_float_(l_max_, normalized_)//,
    //prefactors_cuda_double_(prefactors_cuda(l_max, c10::kDouble)),
    //prefactors_cuda_float_(prefactors_cuda(l_max, c10::kFloat)
    {

    this->prefactors_cuda_double_ = torch::empty({(l_max + 1) * (l_max + 2)}, torch::TensorOptions().device("cpu").dtype(c10::kDouble));
    prefactors_cuda<double>(l_max, this->prefactors_cuda_double_.data_ptr<double>());
    this->prefactors_cuda_double_.to("cuda");

    this->prefactors_cuda_float_ = torch::empty({(l_max + 1) * (l_max + 2)}, torch::TensorOptions().device("cpu").dtype(c10::kFloat));
    prefactors_cuda<float>(l_max, this->prefactors_cuda_float_.data_ptr<float>());
    this->prefactors_cuda_float_.to("cuda");

    this->omp_num_threads_ = calculator_double_.get_omp_num_threads();
}

torch::Tensor SphericalHarmonics::compute(torch::Tensor xyz) {
    return SphericalHarmonicsAutograd::apply(*this, xyz, false, false)[0];
}

std::vector<torch::Tensor> SphericalHarmonics::compute_with_gradients(torch::Tensor xyz) {
    return SphericalHarmonicsAutograd::apply(*this, xyz, true, false);
}

std::vector<torch::Tensor> SphericalHarmonics::compute_with_hessians(torch::Tensor xyz) {
    return SphericalHarmonicsAutograd::apply(*this, xyz, true, true);
}

TORCH_LIBRARY(sphericart_torch, m) {
    m.class_<SphericalHarmonics>("SphericalHarmonics")
        .def(torch::init<int64_t, bool, bool>(), "", {torch::arg("l_max"), torch::arg("normalized") = false, torch::arg("backward_second_derivatives") = false})
        .def("compute", &SphericalHarmonics::compute, "", {torch::arg("xyz")})
        .def("compute_with_gradients", &SphericalHarmonics::compute_with_gradients, "", {torch::arg("xyz")})
        .def("compute_with_hessians", &SphericalHarmonics::compute_with_hessians, "", {torch::arg("xyz")})
        .def("omp_num_threads", &SphericalHarmonics::get_omp_num_threads)
        .def("l_max", &SphericalHarmonics::get_l_max)
        .def("normalized", &SphericalHarmonics::get_normalized_flag);
}
