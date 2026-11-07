# Third-party notices

The pinned source snapshot is under `solver_attention/vendor_spark_pin_v6/vendor`. The original manifest remains unchanged to preserve its identity in solver proofs. The upstream FlashAttention AUTHORS file is provided alongside the original license.

- FlashAttention, revision [49774b31c8d1](https://github.com/Dao-AILab/flash-attention/tree/49774b31c8d1172a5b28f72e9aed0f322bf4ccba), BSD-3-Clause. See the [license](solver_attention/vendor_spark_pin_v6/vendor/LICENSE) and [authors](solver_attention/vendor_spark_pin_v6/vendor/AUTHORS).
- NVIDIA CUTLASS and CuTe C++ headers, revision [dc4817921edd](https://github.com/NVIDIA/cutlass/tree/dc4817921edda44a549197ff3a9dcf5df0636e7b). See the [license and notices](solver_attention/vendor_spark_pin_v6/vendor/csrc/cutlass/LICENSE.txt).

The prefill and decode generators use and transform portions of the pinned FlashAttention CUDA headers. The FlashAttention BSD-3-Clause terms apply to those portions of their generated output.
