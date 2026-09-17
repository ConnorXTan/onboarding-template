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

// Non-owning windows onto a grid's storage. A view carries the stride with
// the pointer, so nothing outside Grid can compute `i * cols` and land on the
// wrong row once rows are padded: `row(i)` is the only way to a row base.
// A third dimension would add a plane stride and a `plane(k)` accessor here;
// the kernel and the driver would not change.
struct ConstGridView {
  const double* data;
  std::size_t rows;
  std::size_t cols;
  std::size_t stride;

  const double* row(std::size_t i) const { return data + i * stride; }
};

struct GridView {
  double* data;
  std::size_t rows;
  std::size_t cols;
  std::size_t stride;

  double* row(std::size_t i) const { return data + i * stride; }

  operator ConstGridView() const { return ConstGridView{data, rows, cols, stride}; }
};

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

  // No allocation and no element copies: a view is four words by value, and
  // it is the only way to the storage, so every caller carries the stride.
  GridView view() { return GridView{data_.data(), rows_, cols_, stride_}; }
  ConstGridView view() const { return ConstGridView{data_.data(), rows_, cols_, stride_}; }

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
// Precondition: cols >= 3, so the row has an interior; apply_stencil routes
// narrower grids to copy_row instead of asking the kernel to reason about them.
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


// Copies the `cols` live cells of one row. Rows are addressed through views,
// so this never touches the padding after a row, and the old whole-buffer
// memcpy of rows * cols doubles (wrong once stride != cols) is not needed.
inline void copy_row(const double* src, double* dst, std::size_t cols) {
  std::memcpy(dst, src, cols * sizeof(double));
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

  const ConstGridView in = old_grid.view();
  const GridView out = new_grid.view();

  if (rows < 3 || cols < 3) {
    // No interior cell exists: every cell is on the boundary ring, so a step
    // is an exact copy of each row. Handling it here lets the kernel assume a
    // real interior instead of reasoning about empty loops on width 1 or 2.
    for (std::size_t i = 0; i < rows; ++i) {
      detail::copy_row(in.row(i), out.row(i), cols);
    }
    return;
  }

  // Boundary rows are copied here; boundary columns are written inside
  // stencil_row. Every output row therefore has exactly one writer, so the
  // parallel loop below shares no row between threads.
  detail::copy_row(in.row(0), out.row(0), cols);
  detail::copy_row(in.row(rows - 1), out.row(rows - 1), cols);

  // Waking the thread pool costs microseconds per call even though the pool
  // persists across calls (about 15 us with libgomp on macOS, a few on Linux).
  // Below about 16k interior cells (a 128x128 grid) the serial loop finishes
  // in a few microseconds, so the small correctness cases run serially and
  // only grids with real work pay for the fork/join.
  // Both are read only by the pragma, which a build without OpenMP drops.
  [[maybe_unused]] constexpr std::size_t kMinInteriorCellsForParallel = std::size_t{1} << 14;
  [[maybe_unused]] const std::size_t interior_cells = (rows - 2) * (cols - 2);

  // schedule(static): every interior row is the same amount of work, so a
  // static split gives each thread one contiguous band of rows. Its three-row
  // input window then walks through its own L1/L2 without other threads'
  // rows interleaved, and there is no scheduler traffic per chunk.
  //
  // Nothing is allocated, resized or looked up inside the region: the views
  // were built once above and stencil_row is pure pointer arithmetic. The
  // loop variable stays std::size_t, which OpenMP 3.0 and later accept.
#pragma omp parallel for schedule(static) if(interior_cells >= kMinInteriorCellsForParallel)
  for (std::size_t i = 1; i < rows - 1; ++i) {
    detail::stencil_row(in.row(i - 1), in.row(i), in.row(i + 1), out.row(i), cols);
  }
}
