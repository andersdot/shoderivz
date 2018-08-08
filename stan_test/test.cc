#include <iostream>

#include <stan/math.hpp>
#include <stdexcept>
#include <vector>

struct sho_ode {
  template <typename T0, typename T1, typename T2>
  inline std::vector<typename stan::return_type<T1, T2>::type>
  // initial time
  // initial positions
  // parameters
  // double data
  // integer data
  operator()(const T0& t_in, const std::vector<T1>& y_in,
             const std::vector<T2>& theta, const std::vector<double>& x,
             const std::vector<int>& x_int, std::ostream* msgs) const {
    if (y_in.size() != 2)
      throw std::domain_error(
          "this function was called with inconsistent state");

    std::vector<typename stan::return_type<T1, T2>::type> res;
    res.push_back(y_in.at(1));
    res.push_back(-y_in.at(0) - theta.at(0) * y_in.at(1));

    return res;
  }
};

int main () {
  typedef stan::math::var grad_type;

  double t0 = 0.0;

  sho_ode harm_osc;

  std::vector<grad_type> theta;
  theta.push_back(grad_type(0.15));

  std::vector<double> y0;
  y0.push_back(1.0);
  y0.push_back(0.0);

  std::vector<double> ts;
  for (int i = 0; i < 100; i++)
    ts.push_back(t0 + 0.1 * (i + 1));

  std::vector<double> x;
  std::vector<int> x_int;

  //std::vector<std::vector<double> > ode_res_vd
  auto ode_res_vd = stan::math::integrate_ode_rk45(harm_osc, y0, t0, ts, theta, x, x_int);

  std::vector<grad_type> params;
  params.push_back(theta[0]);

  std::vector<double> g;
  ode_res_vd[99][0].grad(params, g);

  std::cout << g[0] << "\n";

  return 0;
}
