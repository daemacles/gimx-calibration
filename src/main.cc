#include <cstdlib>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <cpp_mpl.hpp>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-register"
#include <Eigen/Dense>
#pragma clang diagnostic pop

using cppmpl::NumpyArray;

typedef Eigen::Matrix<NumpyArray::dtype, Eigen::Dynamic, Eigen::Dynamic,
                      Eigen::RowMajor> MatrixXd;

cppmpl::CppMatplotlib MplConnect (void);

int main(int argc, char **argv) {
  // DELETE THESE.  Used to suppress unused variable warnings.
  (void)argc;
  (void)argv;

  auto mpl = MplConnect();

  auto data = MatrixXd(5, 2);
  data <<
    1, 1,
    2, 4,
    3, 9,
    4, 16,
    5, 25;

  auto np_data = NumpyArray("XX", data.data(), 5, 2);

  mpl.SendData(np_data);
  mpl.RunCode("plot(XX[:, 0], XX[:, 1])");

  return 0;
}

cppmpl::CppMatplotlib MplConnect (void) {
  auto config_path = std::getenv("IPYTHON_KERNEL");
  if (config_path == nullptr) {
    std::cerr << "Please export IPYTHON_KERNEL=/path/to/kernel-NNN.json"
      << std::endl;
    std::exit(-1);
  }

  cppmpl::CppMatplotlib mpl{config_path};
  mpl.Connect();

  return std::move(mpl);
}
