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
#include <map>
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

namespace fc = FlyCapture2;
class Flea3 {
public:
  Flea3 () {
  }

  ~Flea3 () {
    fc::Error error = camera_.StopCapture();
    if (error != fc::PGRERROR_OK) {
      // This may fail when the camera was removed, so don't show
      // an error message
    }
    camera_.Disconnect();
  }

  bool Connect () {
    fc::Error error;

    // Connect the camera_
    error = camera_.Connect(0);
    if ( error != fc::PGRERROR_OK ) {
      std::cout << "Failed to connect to camera_" << std::endl;
      return false;
    }

    // Get the camera_ info and print it out
    error = camera_.GetCameraInfo(&camInfo_);
    if (error != fc::PGRERROR_OK) {
      std::cout << "Failed to get camera_ info from camera_" << std::endl;
      return false;
    }

    std::cout << camInfo_.vendorName << " "
      << camInfo_.modelName << " "
      << camInfo_.serialNumber << std::endl;

    error = camera_.StartCapture();
    if (error == fc::PGRERROR_ISOCH_BANDWIDTH_EXCEEDED) {
      std::cout << "Bandwidth exceeded" << std::endl;
      return false;
    }
    else if (error != fc::PGRERROR_OK) {
      std::cout << "Failed to start image capture" << std::endl;
      return false;
    }
    return true;
  }

  cv::Mat GetImage () {
    fc::Error error;

    constexpr double SCALE = 1.0;
    fc::Image rawImage;
    error = camera_.RetrieveBuffer( &rawImage );
    if (error != fc::PGRERROR_OK) {
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
    image.convertTo(image, CV_32F);
    return image;
  }

private:
  fc::Camera camera_;
  fc::CameraInfo camInfo_;
};


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


int main (int argc, char **argv) {
  (void) argc;
  (void) argv;

  GimxConnection gimx("localhost", 7799);
  gimx.Connect();

  Flea3 flea3;
  if (!flea3.Connect()) {
    std::cerr << "Couldn't connect to camera" << std::endl;
    std::exit(1);
  }

  flea3.GetImage();

  XboneControl ctl;
  ctl.right_stick.x = 25000;
  gimx.SendControl(ctl);

  // Compute mask
  std::cout << "Creating motion mask. Takes about 10 seconds." << std::endl;
  std::vector<cv::Mat> images;
  for (size_t capture_count = 0; capture_count != 600; ++capture_count) {
    images.push_back(flea3.GetImage() / 255.0);
    cv::imshow("image", images.back());
    cv::waitKey(1);
    for (size_t skip_count = 0; skip_count != 1; ++skip_count) {
      //flea3.GetImage();
    }
  }

  // Deltas between successive images
  std::cout << "  Computing deltas" << std::endl;
  std::vector<cv::Mat> diffs;
  cv::Mat mean_diff = images[0].clone() * 0;
  for (size_t idx = 1; idx != images.size(); ++idx) {
    cv::Mat diff = images[idx] - images[idx-1];
    mean_diff += diff;
    diffs.push_back(diff);
  }
  mean_diff /= diffs.size();

  // Stdev of the deltas
  std::cout << "  Computing stdev" << std::endl;
  cv::Mat std_diff = diffs[0].clone() * 0;
  for (size_t idx = 0; idx != diffs.size(); ++idx) {
    cv::Mat deviation = diffs[idx] - mean_diff;
    std_diff += deviation.mul(deviation);
  }
  cv::sqrt(std_diff / diffs.size(), std_diff);
  double min, max;
  cv::minMaxLoc(std_diff, &min, &max);
  std_diff = (std_diff - min) / (max - min);

  // Create the mask
  cv::Mat mask;
  cv::GaussianBlur(std_diff, mask, cv::Size(0, 0), 1.0);
  cv::threshold(mask, mask, 0.25, 1.0, cv::THRESH_BINARY);
  cv::erode(mask, mask, cv::getStructuringElement(cv::MORPH_RECT,
                                                  cv::Size(3, 3)));
  mask *= 255;
  mask.convertTo(mask, CV_8UC1);

  cv::imshow("image", mask);
  cv::waitKey(0);
  exit(0);

  // capture loop
  char key = 0;
  struct timeval prev_time;
  gettimeofday(&prev_time, NULL);
  double diff_us = 0;
  double counter = 0;
  auto detdes_ptr = cv::BRISK::create();
  images.clear();
  for (size_t capture_count = 0; key != 'q' && capture_count < 20;
       ++capture_count) {
    struct timeval time;
    gettimeofday(&time, NULL);
    diff_us = 0.5 * diff_us +
      0.5 * ((time.tv_usec + 1000000 * time.tv_sec) -
             (prev_time.tv_usec + 1000000 * prev_time.tv_sec));
    prev_time = time;
    std::cout << "Period: " << 1e6 / diff_us << "Hz" << std::endl;
    counter += diff_us / 1e6;

    // Get the image
    cv::Mat image = flea3.GetImage();
    images.push_back(image);
    cv::imshow("image", image);

    // Skip some frames -- keeps timing accurate
    for (size_t skips = 20; skips != 0; --skips) {
      flea3.GetImage();
    }
    key = cv::waitKey(1) & 0xff;
  }

  // Now extract feature tracks

  // Key is index into the first
  std::map<int, std::vector<cv::KeyPoint>> feature_map;

  std::vector<cv::KeyPoint> keypoints_0;
  cv::Mat descriptor_0;
  detdes_ptr->detectAndCompute(images[0], mask, keypoints_0,
                               descriptor_0, false);
  if(descriptor_0.type() != CV_32F) {
    descriptor_0.convertTo(descriptor_0, CV_32F);
  }

  for (size_t idx = 1; idx != images.size(); ++idx) {

  }


  std::vector<std::array<double, 4>> motion_vecs;
  if (false) {
    cv::Mat image_1 = flea3.GetImage();
    cv::Mat image_2 = flea3.GetImage();

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
    cv::cvtColor(image_1, color_image, cv::COLOR_GRAY2RGB);
    for (const auto &match : ratio_matches) {
      auto pt1 = keypoints_1[match.queryIdx].pt;
      auto pt2 = keypoints_2[match.trainIdx].pt;
      cv::circle(color_image, pt1, 3, {0, 0, 255});
      cv::line(color_image, pt1, pt2, {0, 0, 255});
      if (std::abs(pt2.x - pt1.x) > 4) {
        motion_vecs.push_back({{pt1.x, pt1.y, pt2.x-pt1.x, pt2.y-pt1.y}});
      }
    }
    cv::imshow("image", color_image);
  }

  gimx.SendControl(XboneControl());

  auto mpl = MplConnect("/tmp/kernel.json");

  auto sendMat = [&] (const cv::Mat &mat, const std::string &name) {
    cv::Mat output(mat.rows, mat.cols, CV_64F);
    mat.convertTo(output, CV_64F);
    auto np_data = NumpyArray(name, (double*)output.data, output.rows,
                              output.cols);
    mpl.SendData(np_data);
  };

//  auto sendVec = [&] (const std::vector<double> &vec, const std::string &name) {
//    mpl.SendData(NumpyArray(name, vec));
//  };

  mpl.SendData(NumpyArray("mv", (double*)motion_vecs.data(),
                          motion_vecs.size(), 4));

  sendMat(images.back(), "cur_image");
  mpl.RunCode("images = []");
  for (const auto& image : images) {
    sendMat(image, "tmp_image");
    mpl.RunCode("images.append(tmp_image)");
  }

  //mpl.RunCode("plot(XX[:, 0], XX[:, 1])");

  return 0;
}
