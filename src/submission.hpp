#pragma once

#include <cassert>
#include <cstddef>
#include <cstring>
#include <vector>

class Grid {
private:
  std::size_t rows_;
  std::size_t cols_;
  std::vector<double> data_;

public:
  Grid(std::size_t rows, std::size_t cols)
    : rows_{rows}, cols_{cols}, data_(rows * cols, 0.0) { }

  std::size_t rows() const { return rows_; }
  std::size_t cols() const { return cols_; }

  double* data() { return data_.data(); }
  const double* data() const { return data_.data(); }

  double& operator()(std::size_t i, std::size_t j) { return data_[i * cols_ + j]; }
  double  operator()(std::size_t i, std::size_t j) const { return data_[i * cols_ + j]; }
};

namespace detail {

// Five-point update of one interior row: `out` receives the stencil of `mid`
// with its neighbours `up` and `down`, and the two edge cells copied from `mid`.
//
// This is the only place `__restrict` appears. GCC honours it on function
// parameters; on a local pointer inside the driver loop it was ignored and the
// vectorizer emitted a runtime aliasing check plus a scalar fallback copy of
// the loop. The promise holds here: `out` lives in new_grid's allocation while
// the three inputs live in old_grid's (apply_stencil asserts the grids are
// distinct objects, and each Grid owns its own storage), and the inputs are
// only read.
//
// `cols` is the logical row width, never the storage stride, so the kernel
// cannot reach anything outside the row it was handed. The loop covers only
// the interior 1 .. cols-2 so `mid[j - 1]` and `mid[j + 1]` stay inside the row.
inline void stencil_row(
  const double* __restrict up,
  const double* __restrict mid,
  const double* __restrict down,
  double* __restrict out,
  std::size_t cols
) {
  out[0] = mid[0];
#pragma omp simd
  for (std::size_t j = 1; j < cols - 1; ++j) {
    out[j] = 0.5 * mid[j] +
             0.125 * (up[j] + down[j] + mid[j - 1] + mid[j + 1]);
  }
  out[cols - 1] = mid[cols - 1];
}

}  // namespace detail

inline void apply_stencil(const Grid& old_grid, Grid& new_grid) {
  // The kernel's restrict promises and the row copies below assume two
  // distinct, same-shaped grids. The harness always provides that; the asserts
  // make the precondition visible and cost one comparison per call.
  assert(&old_grid != &new_grid);
  assert(old_grid.rows() == new_grid.rows() && old_grid.cols() == new_grid.cols());

  const std::size_t rows = old_grid.rows();
  const std::size_t cols = old_grid.cols();

  if (rows == 0 || cols == 0) {
    return;
  }

  const double* src = old_grid.data();
  double* dst = new_grid.data();

  if (rows < 3 || cols < 3) {
    std::memcpy(dst, src, rows * cols * sizeof(double));
    return;
  }

  std::memcpy(dst, src, cols * sizeof(double));
  std::memcpy(dst + (rows - 1) * cols, src + (rows - 1) * cols, cols * sizeof(double));

#pragma omp parallel for schedule(static)
  for (std::size_t i = 1; i < rows - 1; ++i) {
    const double* mid = src + i * cols;
    detail::stencil_row(mid - cols, mid, mid + cols, dst + i * cols, cols);
  }
}
