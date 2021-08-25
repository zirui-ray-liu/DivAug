#include <torch/extension.h>
#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <random>
#include "kmeanpp_common.h"

std::default_random_engine generator;

torch::Tensor _kmeanpp(torch::Tensor z, torch::Tensor dists, int32_t n_clusters, int64_t bias_term) {
    /* z.shape == c x #classes; dists.shape == c x c*/
    CHECK_TENSOR_TYPE(z, torch::kFloat32, 2);
    CHECK_TENSOR_TYPE(dists, torch::kFloat32, 2)
    torch::Tensor centers_indices = torch::empty(n_clusters, torch::dtype(torch::kInt64));
    const int64_t n_samples = z.size(0);
    int64_t center_id = std::rand() % n_samples;
    centers_indices[0] = center_id;
    torch::Tensor closest_dist_sq = dists[center_id];
    float current_pot = closest_dist_sq.sum().item<float>();

    for (int64_t c=1; c < n_clusters; c++){
        torch::Tensor w = closest_dist_sq / current_pot;
        std::vector<float> probs(w.data_ptr<float>(), w.data_ptr<float>() + w.numel());
        std::discrete_distribution<int> d(probs.begin(), probs.end());
        int64_t best_candidate = d(generator);
        torch::Tensor distance_to_candidates = dists[best_candidate];
        torch::Tensor best_dist_sq = torch::minimum(closest_dist_sq, distance_to_candidates);
        current_pot = best_dist_sq.sum().item<float>();
        centers_indices[c] = best_candidate;
        closest_dist_sq = best_dist_sq;
    }
    return centers_indices + bias_term;
}

int multinormal(torch::Tensor p) {
    std::vector<float> probs(p.data_ptr<float>(), p.data_ptr<float>() + p.numel());
    std::discrete_distribution<int> d(probs.begin(), probs.end());
    int32_t number = d(generator);
    return number;
}

torch::Tensor par_kmeanpp(torch::Tensor Z, torch::Tensor Dists, int32_t n_clusters){
    CHECK_TENSOR_TYPE(Z, torch::kFloat32, 3);
    CHECK_TENSOR_TYPE(Dists, torch::kFloat32, 3);
    const int64_t batch_size = Z.size(0);
    torch::Tensor result = torch::empty({batch_size, n_clusters}, torch::dtype(torch::kInt64));
    const int64_t num_candidates = Z.size(1);
    #pragma omp parallel for
    for( int64_t i=0; i < batch_size; i++ ){
        result[i] = _kmeanpp(Z[i], Dists[i], n_clusters, i * num_candidates);
    }
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("_kmeanpp", &_kmeanpp);
  m.def("multinormal", &multinormal);
  m.def("par_kmeanpp", &par_kmeanpp);
}