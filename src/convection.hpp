#include <vector>

class Array {
private:
  std::vector<double> m_data;

  int m_n0;

  int m_n1;

  int m_n2;

public:
  Array(int n0, int n1, int n2, int n3)
      : m_data(n0 * n1 * n2 * n3), m_n0(n0), m_n1(n1), m_n2(n2) {}

  Array() = default;

  Array(Array const &rhs) = default;

  Array(Array &&rhs) noexcept = default;

  ~Array() = default;

  Array &operator=(Array const &rhs) = default;

  Array &operator=(Array &&rhs) noexcept = default;

  double &operator()(int i0, int i1, int i2, int i3) noexcept {
    return m_data[i0 + (i1 + (i2 + i3 * m_n2) * m_n1) * m_n0];
  }

  double const &operator()(int i0, int i1, int i2, int i3) const noexcept {
    return m_data[i0 + (i1 + (i2 + i3 * m_n2) * m_n1) * m_n0];
  }
};
