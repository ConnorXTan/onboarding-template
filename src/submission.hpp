#pragma once

#include <cstddef>
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

  double& operator()(std::size_t i, std::size_t j) { return data_[i * cols_ + j]; }
  double  operator()(std::size_t i, std::size_t j) const { return data_[i * cols_ + j]; }
};

void apply_stencil(const Grid& old_grid, Grid& new_grid);
