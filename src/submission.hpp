#pragma once

#include <cassert>
#include <cstddef>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

namespace detail {

// Minimal C++17 allocator whose storage starts on an `Alignment`-byte
// boundary. A plain std::vector<double> lands wherever malloc puts it (16 bytes
// into a page on the machines measured), so rows are only 16-byte aligned and
// both grids share the same page offset. With 64-byte storage and a stride
// that is a multiple of 8 doubles, every row starts on its own cache line.
template <typename T, std::size_t Alignment>
struct aligned_allocator {
  static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be a power of two");
  static_assert(Alignment >= alignof(T), "Alignment must be at least alignof(T)");

  using value_type = T;

  template <typename U>
  struct rebind { using other = aligned_allocator<U, Alignment>; };

  aligned_allocator() noexcept = default;

  template <typename U>
  aligned_allocator(const aligned_allocator<U, Alignment>&) noexcept { }

  T* allocate(std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::bad_alloc();
    }
    return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t{Alignment}));
  }

  void deallocate(T* p, std::size_t) noexcept {
    ::operator delete(p, std::align_val_t{Alignment});
  }
};

template <typename T, typename U, std::size_t Alignment>
bool operator==(const aligned_allocator<T, Alignment>&, const aligned_allocator<U, Alignment>&) noexcept {
  return true;
}

template <typename T, typename U, std::size_t Alignment>
bool operator!=(const aligned_allocator<T, Alignment>&, const aligned_allocator<U, Alignment>&) noexcept {
  return false;
}

}  // namespace detail

class Grid {
private:
  static constexpr std::size_t kAlignmentBytes = 64;
  static constexpr std::size_t kAlignmentDoubles = kAlignmentBytes / sizeof(double);

  // Row stride in doubles: `cols` rounded up to a whole cache line. The padded
  // cells at the end of each row are storage only. operator() never reaches
  // them, and the kernel is handed `cols`, never `stride`, so it cannot either.
  static std::size_t padded_stride(std::size_t cols) {
    return (cols + kAlignmentDoubles - 1) / kAlignmentDoubles * kAlignmentDoubles;
  }

  std::size_t rows_;
  std::size_t cols_;
  std::size_t stride_;
  std::vector<double, detail::aligned_allocator<double, kAlignmentBytes>> data_;

public:
  Grid(std::size_t rows, std::size_t cols)
    : rows_{rows}, cols_{cols}, stride_{padded_stride(cols)}, data_(rows * stride_, 0.0) { }

  std::size_t rows() const { return rows_; }
  std::size_t cols() const { return cols_; }
  std::size_t stride() const { return stride_; }

  double* data() { return data_.data(); }
  const double* data() const { return data_.data(); }

  double& operator()(std::size_t i, std::size_t j) { return data_[i * stride_ + j]; }
  double  operator()(std::size_t i, std::size_t j) const { return data_[i * stride_ + j]; }
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
  const std::size_t stride = old_grid.stride();  // same shape, so same stride

  if (rows == 0 || cols == 0) {
    return;
  }

  const double* src = old_grid.data();
  double* dst = new_grid.data();

  if (rows < 3 || cols < 3) {
    // Every cell is boundary, so the result is a copy. Copying the whole
    // storage, padding included, is exactly the traffic of a row-by-row copy
    // without the per-row bookkeeping.
    std::memcpy(dst, src, rows * stride * sizeof(double));
    return;
  }

  std::memcpy(dst, src, cols * sizeof(double));
  std::memcpy(dst + (rows - 1) * stride, src + (rows - 1) * stride, cols * sizeof(double));

#pragma omp parallel for schedule(static)
  for (std::size_t i = 1; i < rows - 1; ++i) {
    const double* mid = src + i * stride;
    detail::stencil_row(mid - stride, mid, mid + stride, dst + i * stride, cols);
  }
}
