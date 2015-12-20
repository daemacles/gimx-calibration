#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdlib>
#include <cstring>
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <flycapture/FlyCapture2.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <cpp_mpl.hpp>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-register"
#include <eigen3/Eigen/Dense>
#pragma clang diagnostic pop

using cppmpl::NumpyArray;

typedef Eigen::Matrix<NumpyArray::dtype, Eigen::Dynamic, Eigen::Dynamic,
                      Eigen::RowMajor> MatrixXd;

cppmpl::CppMatplotlib MplConnect (std::string config_path);


constexpr uint32_t GIMX_UDP_BUF_SIZE = 158;

class GimxController {
public:
  void SerializeTo (uint8_t *buf) const {
    memset(buf, 0, GIMX_UDP_BUF_SIZE-2);
    SerializeTo_((int32_t*)buf);
  }

private:
  virtual void SerializeTo_ (int32_t *axes) const = 0;
};

struct Stick {
  int32_t x = 0;
  int32_t y = 0;
};

class XboneControl : public GimxController {
public:
  Stick left_stick;
  Stick right_stick;

  void SerializeTo_ (int32_t *axes) const override {
    axes[0] = left_stick.x;
    axes[1] = left_stick.y;
    axes[2] = right_stick.x;
    axes[3] = right_stick.y;
  }
};

class GimxConnection {
public:

  GimxConnection (std::string hostname, uint32_t port) :
      hostname_(hostname),
      port_(port),
      sock_fd_(-1)
  {
  }

  ~GimxConnection () {
    if (sock_fd_ != -1) {
      close(sock_fd_);
    }
  }

  bool Connect () {
    struct hostent *host;

    host = gethostbyname(hostname_.c_str());
    if (host == NULL) {
      perror("gethostbyname");
      return false;
    }

    /* initialize socket */
    if ((sock_fd_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) == -1) {
      perror("socket");
      return false;
    }

    /* initialize server addr */
    server_ = {
      .sin_family = AF_INET,
      .sin_port = htons(port_),
      .sin_addr = *((struct in_addr*) host->h_addr)
    };

    return true;
  }

  bool SendControl (const GimxController &ctl) {
    if (sock_fd_ == -1) {
      throw std::runtime_error("GIMX socket not connected");
    }

    uint8_t buf[GIMX_UDP_BUF_SIZE];
    buf[0] = 0xff;
    buf[1] = GIMX_UDP_BUF_SIZE - 2;

    ctl.SerializeTo(&buf[2]);

    /* send message */
    int len = sizeof(server_);
    if (sendto(sock_fd_, buf, GIMX_UDP_BUF_SIZE, 0,
               (struct sockaddr *) &server_, len) == -1) {
      perror("sendto()");
      return false;
    }
    return true;
  }

private:
  std::string hostname_;
  uint32_t port_;
  struct sockaddr_in server_;
  int32_t sock_fd_;
};

int old_main(int argc, char **argv) {
  // DELETE THESE.  Used to suppress unused variable warnings.
  (void)argc;
  (void)argv;


  GimxConnection gimx("localhost", 7799);
  gimx.Connect();

  XboneControl ctl;

  ctl.left_stick.x = 20000;
  gimx.SendControl(ctl);

  usleep(1 * 1e6);

  ctl.left_stick.x = 0;
  gimx.SendControl(ctl);

  //auto mpl = MplConnect();
  //auto data = MatrixXd(5, 2);
  //data <<
  //  1, 1,
  //  2, 4,
  //  3, 9,
  //  4, 16,
  //  5, 25;
  //
  //auto np_data = NumpyArray("XX", data.data(), 5, 2);
  //
  //mpl.SendData(np_data);
  //mpl.RunCode("plot(XX[:, 0], XX[:, 1])");

  return 0;
}

cppmpl::CppMatplotlib MplConnect (std::string config_path="") {
  if (config_path == "") {
    auto config_path_c_str = std::getenv("IPYTHON_KERNEL");
    if (config_path_c_str == nullptr) {
      std::cerr << "Please export IPYTHON_KERNEL=/path/to/kernel-NNN.json"
        << std::endl;
      std::exit(-1);
    }
    config_path = config_path_c_str;
  }

  cppmpl::CppMatplotlib mpl{config_path};
  mpl.Connect();

  return mpl;
}


int main() {
  using namespace FlyCapture2;
  Error error;
  Camera camera;
  CameraInfo camInfo;

  // Connect the camera
  error = camera.Connect(0);
  if ( error != PGRERROR_OK ) {
    std::cout << "Failed to connect to camera" << std::endl;
    return false;
  }

  // Get the camera info and print it out
  error = camera.GetCameraInfo(&camInfo);
  if (error != PGRERROR_OK) {
    std::cout << "Failed to get camera info from camera" << std::endl;
    return false;
  }

  std::cout << camInfo.vendorName << " "
    << camInfo.modelName << " "
    << camInfo.serialNumber << std::endl;

  error = camera.StartCapture();
  if (error == PGRERROR_ISOCH_BANDWIDTH_EXCEEDED) {
    std::cout << "Bandwidth exceeded" << std::endl;
    return false;
  }
  else if (error != PGRERROR_OK) {
    std::cout << "Failed to start image capture" << std::endl;
    return false;
  }

  GimxConnection gimx("localhost", 7799);
  gimx.Connect();

  auto GetImage = [&] () {
    Image rawImage;
    error = camera.RetrieveBuffer( &rawImage );
    if (error != PGRERROR_OK) {
      std::cout << "capture error" << std::endl;
    }

    //     // convert to rgb
    //    Image rgbImage;
    //    rgbImage.SetColorProcessing(IPP);
    //    rawImage.Convert( FlyCapture2::PIXEL_FORMAT_BGR, &rgbImage );

    // Convert to OpenCV Mat
    cv::Mat tmp_image = cv::Mat(rawImage.GetRows(), rawImage.GetCols(),
                                CV_8UC1, rawImage.GetData(),
                                rawImage.GetStride());
    //cv::Mat image(tmp_image, cv::Rect(10, 110, 620, 60));
    cv::Mat image;
    //cv::resize(tmp_image, image, cv::Size(240, 240), 0, 0, cv::INTER_CUBIC);
    cv::resize(tmp_image, image, cv::Size(0, 0), 0.5, 0.5, cv::INTER_CUBIC);
    //image = tmp_image;
    return image;
  };

  cv::Mat image = GetImage();
  cv::Mat prev_image = image.clone();

  char key = 0;
  struct timeval prev_time;
  gettimeofday(&prev_time, NULL);
  double diff_us = 0;
  double counter = 0;
  cv::Mat planes[2];

  std::vector<double> magnitudes;
  std::vector<double> angles;

  // capture loop
  while(key != 'q') {
    struct timeval time;
    gettimeofday(&time, NULL);
    diff_us = 0.5 * diff_us +
      0.5 * ((time.tv_usec + 1000000 * time.tv_sec) -
             (prev_time.tv_usec + 1000000 * prev_time.tv_sec));
    prev_time = time;
    std::cout << "Period: " << 1e6 / diff_us << "Hz" << std::endl;
    counter += diff_us / 1e6;

    XboneControl ctl;
    ctl.right_stick.x = 32000 * sin(counter / 2);
    ctl.right_stick.x = 22000;
    gimx.SendControl(ctl);

    // Get the image
    image = GetImage();

    cv::Mat flow(image.rows, image.cols, CV_32FC2);
    cv::calcOpticalFlowFarneback(prev_image, image, flow,
                                 0.5, // pyr_scale
                                 4,   // pyramid levels
                                 7,   // window size
                                 4,   // iterations
                                 5,   // poly_n expansion (5 or 7 good)
                                 1.1, // poly sigma
                                 0);  // flags

    cv::split(flow, planes);
    cv::Mat mag, angle;
    cv::cartToPolar(planes[0], planes[1], mag, angle, false);
    cv::Mat hsv_parts[3];
    hsv_parts[0] = angle * 180 / M_PI / 2;
    hsv_parts[1] = hsv_parts[0].clone();
    hsv_parts[1] = 255;
    hsv_parts[2] = mag * 1.0;
    //cv::normalize(mag, hsv_parts[2], 0, 255, cv::NORM_MINMAX);

    cv::Mat output_image;
    cv::merge(hsv_parts, 3, output_image);

    //cv::imshow("image", output_image);
    //cv::imshow("image", image);//planes[0]);
    cv::imshow("image", planes[0]);

    double x_min, x_max, y_min, y_max;
    cv::Point x_min_loc, x_max_loc, y_min_loc, y_max_loc;
    cv::minMaxLoc(planes[0], &x_min, &x_max, &x_min_loc, &x_max_loc);
    cv::minMaxLoc(planes[1], &y_min, &y_max, &y_min_loc, &y_max_loc);
    angles.push_back(std::atan2(y_max, x_max)/M_PI * 180);
    //magnitudes.push_back(std::sqrt(x_max*x_max + y_max*y_max));
    magnitudes.push_back(x_max);

    key = cv::waitKey(1) & 0xff;
  }

  gimx.SendControl(XboneControl());

  error = camera.StopCapture();
  if (error != PGRERROR_OK) {
    // This may fail when the camera was removed, so don't show
    // an error message
  }

  auto mpl = MplConnect("/tmp/kernel.json");

  auto sendMat = [&] (const cv::Mat &mat, const std::string &name) {
    cv::Mat output(mat.rows, mat.cols, CV_64F);
    mat.convertTo(output, CV_64F);
    auto np_data = NumpyArray(name, (double*)output.data, output.rows,
                              output.cols);
    mpl.SendData(np_data);
  };

  auto sendVec = [&] (const std::vector<double> &vec, const std::string &name) {
    mpl.SendData(NumpyArray(name, vec));
  };

  sendVec(magnitudes, "magnitudes");
  sendVec(angles, "angles");

  sendMat(planes[0], "F0");
  sendMat(planes[1], "F1");
  sendMat(image, "cur_image");
  sendMat(prev_image, "prev_image");

  //mpl.RunCode("plot(XX[:, 0], XX[:, 1])");

  camera.Disconnect();

  return 0;
}
