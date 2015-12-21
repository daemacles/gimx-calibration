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

#include <algorithm>
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
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

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
    constexpr double SCALE = 1.0;
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
    cv::resize(tmp_image, image, cv::Size(0, 0), SCALE, SCALE, cv::INTER_CUBIC);
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
  auto detdes_ptr = cv::BRISK::create();

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
    ctl.right_stick.x = 32000;
    gimx.SendControl(ctl);

    usleep(0.5 * 1e6);

    // Get the image
    image = GetImage();
    cv::Mat image_1 = image;

    // Skip some frames -- keeps timing accurate
    for (size_t skips = 2; skips != 0; --skips) {
      GetImage();
    }

    cv::Mat image_2 = GetImage();

    std::vector<cv::KeyPoint> keypoints_1;
    detdes_ptr->detect(image_1, keypoints_1);

    cv::Mat descriptor_1;
    detdes_ptr->compute(image_1, keypoints_1, descriptor_1);
    if(descriptor_1.type() != CV_32F) {
      descriptor_1.convertTo(descriptor_1, CV_32F);
    }

    std::vector<cv::KeyPoint> keypoints_2;
    detdes_ptr->detect(image_2, keypoints_2);

    cv::Mat descriptor_2;
    detdes_ptr->compute(image_2, keypoints_2, descriptor_2);
    if(descriptor_2.type() != CV_32F) {
      descriptor_2.convertTo(descriptor_2, CV_32F);
    }

    cv::FlannBasedMatcher matcher;
    std::vector<std::vector<cv::DMatch>> raw_matches;
    matcher.knnMatch(descriptor_1, descriptor_2, raw_matches, 2);

    // Use ratio matching
    std::vector<cv::DMatch> ratio_matches;
    for (size_t idx = 0; idx != raw_matches.size(); ++idx) {
      if (raw_matches[idx][0].distance < 0.45*raw_matches[idx][1].distance) {
        ratio_matches.push_back(raw_matches[idx][0]);
      }
    }

//    // Get rid of impossible matches
//    std::vector<cv::DMatch> filter_matches;
//    std::copy_if(ratio_matches.begin(), ratio_matches.end(),
//                 std::back_inserter(filter_matches),
//      [&](const cv::DMatch &match) {
//      return std::abs(keypoints_1[match.queryIdx].pt.y -
//                      keypoints_2[match.trainIdx].pt.y) < 12;
//      });
//    std::cout << "Removed " << ratio_matches.size() - filter_matches.size() << " matches\n";

    cv::Mat color_image;
    cv::cvtColor(image, color_image, cv::COLOR_GRAY2RGB);
    for (const auto &match : ratio_matches) {
      cv::circle(color_image, keypoints_1[match.queryIdx].pt, 3, {0, 0, 255});
      cv::line(color_image, keypoints_1[match.queryIdx].pt,
               keypoints_2[match.trainIdx].pt, {0, 0, 255});
    }
//    cv::drawMatches(image_1, keypoints_1, image_2, keypoints_2, ratio_matches,
//                    color_image, {0, 0, 255});
    cv::imshow("image", color_image);
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
