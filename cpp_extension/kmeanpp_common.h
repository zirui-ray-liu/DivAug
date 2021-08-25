// Helper for type check
#define CHECK_TENSOR_TYPE(name, type, n_dim)                                        \
    TORCH_CHECK(name.dtype() == type, "The type of " #name " is not correct!");     \
    TORCH_CHECK(name.dim() == n_dim, "The dimension of " #name " is not correct!");     \
