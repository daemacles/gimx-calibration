#include <cassert>
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
#include <fstream>
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
typedef std::vector<cv::Mat> MatVec;
typedef std::vector<cv::KeyPoint> KeyPointVec;

cppmpl::CppMatplotlib MplConnect (std::string config_path);

constexpr uint32_t GIMX_UDP_BUF_SIZE = 158;
constexpr double DISTANCE_THRESHOLD = 750.0;
constexpr double RATIO_THRESHOLD = 75.0;
std::string MASK_IMAGE_FILE = "/tmp/mask.png";


bool FileExists (std::string filename) {
  std::ifstream file_check(filename);
  bool found = file_check.good();
  file_check.close();
  return found;
}


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

    cv::Mat tmp_image;
    if (rawImage.GetPixelFormat() == fc::PIXEL_FORMAT_RAW8) {
     // convert to rgb
      fc::Image rgbImage;
      rgbImage.SetColorProcessing(fc::IPP);
      rawImage.Convert(fc::PIXEL_FORMAT_BGR, &rgbImage);
      tmp_image = cv::Mat(rgbImage.GetRows(), rgbImage.GetCols(), CV_8UC3,
                          rgbImage.GetData(), rgbImage.GetStride());
      cv::cvtColor(tmp_image, tmp_image, cv::COLOR_BGR2GRAY);
    } else {
      // Convert to OpenCV Mat
      tmp_image = cv::Mat(rawImage.GetRows(), rawImage.GetCols(), CV_8UC1,
                          rawImage.GetData(), rawImage.GetStride());
    }
    // Grab window subset
    //cv::Mat image(tmp_image, cv::Rect(10, 110, 620, 60));

    cv::Mat image;
    cv::resize(tmp_image, image, cv::Size(0, 0), SCALE, SCALE, cv::INTER_CUBIC);
    return image;
  }

  cv::Mat GetImageFloat () {
    cv::Mat image = GetImage();
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


namespace std {
template<> struct equal_to<cv::Mat> {
  bool operator()(const cv::Mat& A, const cv::Mat& B) const {
    assert(A.rows == B.rows);
    assert(A.cols == B.cols);
    assert(A.type() == B.type());
    cv::Mat diff = A != B;
    return cv::countNonZero(diff) == 0;
  }
};

template<> struct hash<cv::Mat> {
  size_t operator()(const cv::Mat& A) const {
    return (size_t)A.data;
  }
};

// This specialization lets us use descriptors as keys in a map
template<> struct less<cv::Mat> {
  bool operator()(const cv::Mat& A, const cv::Mat& B) const {
    assert(A.rows == B.rows);
    assert(A.cols == B.cols);
    assert(A.type() == B.type());
    assert(A.channels() == 1);
    assert(B.channels() == 1);

    for (int row = 0; row != A.rows; ++row) {
      for (int col = 0; col != A.cols; ++col) {
        switch (A.type()) {
        case CV_32F:
          if (A.at<float>(row, col) > B.at<float>(row, col))
            return false;
          if (A.at<float>(row, col) < B.at<float>(row, col))
            return true;
          break;

        case CV_64F:
          if (A.at<double>(row, col) > B.at<double>(row, col))
            return false;
          if (A.at<double>(row, col) < B.at<double>(row, col))
            return true;
          break;

        case CV_8UC1:
          if (A.at<uint8_t>(row, col) > B.at<uint8_t>(row, col))
            return false;
          if (A.at<uint8_t>(row, col) < B.at<uint8_t>(row, col))
            return true;
          break;

        default:
          throw std::runtime_error("Unsupported type");
          break;
        }
      }
    }

    // All elements equal, but say we're less to avoid loops?
    return true;
  }
};
}


struct KeyPointNode {
  cv::KeyPoint keypoint;
  int parent;
  bool leaf;
};


cv::Mat MatVecToMat (const MatVec &mv) {
  cv::Mat output (mv.size(), mv[0].cols, mv[0].type());
  for (size_t idx = 0; idx != mv.size(); ++idx) {
    output.row(idx) = mv[idx];
  }
  return output;
}


// See how good any of the new descriptors are by matching them against
// themselves and only selecting those unique enough.
// Can use same variables for output as input.
void GetGoodDescriptors (cv::InputArray descriptors,
                         KeyPointVec &keypoints,
                         cv::OutputArray good_descriptors,
                         KeyPointVec &good_keypoints,
                         double min_distance=DISTANCE_THRESHOLD
                         ) {
  cv::BFMatcher new_matcher;
  std::vector<std::vector<cv::DMatch>> new_raw_matches;
  new_matcher.knnMatch(descriptors, descriptors, new_raw_matches, 3);

  // Use ratio matching
  cv::Mat descriptors_tmp = descriptors.getMat().clone();
  KeyPointVec keypoints_tmp (keypoints);
  MatVec descriptors_vec;
  good_keypoints.clear();
  for (size_t idx = 0; idx != new_raw_matches.size(); ++idx) {
    cv::DMatch best_match = new_raw_matches[idx][1];
    //cv::DMatch second_best_match = new_raw_matches[idx][2];

    //if (best_match.distance < ratio * second_best_match.distance) {
    if (best_match.distance > min_distance) {
      std::cout << best_match.distance << std::endl;
      descriptors_vec.push_back(descriptors_tmp.row(best_match.queryIdx));
      good_keypoints.push_back(keypoints_tmp[best_match.queryIdx]);
    }
  }
  good_descriptors.create(descriptors_vec.size(), descriptors_vec[0].cols,
                          descriptors_vec[0].type());
  cv::Mat output_mat = good_descriptors.getMat();
  for (size_t idx = 0; idx != descriptors_vec.size(); ++idx) {
    output_mat.row(idx) = descriptors_vec[idx];
  }
  std::cout << "Got " << good_keypoints.size() << " keypoints "
    << "(down from " << keypoints_tmp.size() << ")" << std::endl;
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

  cv::Mat mask;
  if (!FileExists(MASK_IMAGE_FILE)) {
    // Compute mask
    std::cout << "Creating motion mask. Takes about 10 seconds." << std::endl;
    MatVec mask_images;
    for (size_t capture_count = 0; capture_count != 600; ++capture_count) {
      mask_images.push_back(flea3.GetImageFloat() / 255.0);
      cv::imshow("image", mask_images.back());
      cv::waitKey(1);
      for (size_t skip_count = 0; skip_count != 0; ++skip_count) {
        flea3.GetImage();
      }
    }

    // Deltas between successive mask_images
    std::cout << "  Computing deltas" << std::endl;
    MatVec diffs;
    cv::Mat mean_diff = mask_images[0].clone() * 0;
    for (size_t idx = 1; idx != mask_images.size(); ++idx) {
      cv::Mat diff = mask_images[idx] - mask_images[idx-1];
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
    cv::GaussianBlur(std_diff, mask, cv::Size(0, 0), 1.0);
    cv::threshold(mask, mask, 0.25, 1.0, cv::THRESH_BINARY);
    cv::erode(mask, mask, cv::getStructuringElement(cv::MORPH_RECT,
                                                    cv::Size(3, 3)));
    mask *= 255;
    mask.convertTo(mask, CV_8UC1);

    cv::imwrite(MASK_IMAGE_FILE, mask);

    cv::imshow("image", mask);
    cv::waitKey(0);
  } else {
    mask = cv::imread(MASK_IMAGE_FILE);
    cv::cvtColor(mask, mask, cv::COLOR_BGR2GRAY);
    std::cout << "Using mask from " << MASK_IMAGE_FILE
      << ". Delete to recreate mask anew." << std::endl;
  }

  // capture loop
  char key = 0;
  struct timeval prev_time;
  gettimeofday(&prev_time, NULL);
  double diff_us = 0;
  double counter = 0;
  auto detdes_ptr = cv::BRISK::create();
  MatVec images;
  for (size_t capture_count = 0; key != 'q' && capture_count < 100;
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
    key = cv::waitKey(0) & 0xff;

    // Skip some frames -- keeps timing accurate
    for (size_t skips = 20; skips != 0; --skips) {
      flea3.GetImage();
    }
  }

  // Now extract feature tracks
  // Pass 1, figure out all descriptors of interest.
  std::cout << "Extracting feature tracks" << std::endl;

  int keypoint_counter = 0;
  MatVec all_descriptors;
  std::vector<KeyPointNode> keypoint_nodes;

  cv::Mat cur_image = images[0];
  KeyPointVec cur_keypoints;
  cv::Mat cur_descriptor;
  detdes_ptr->detectAndCompute(cur_image, mask, cur_keypoints,
                               cur_descriptor, false);

  GetGoodDescriptors(cur_descriptor, cur_keypoints, cur_descriptor, cur_keypoints);

  MatVec prev_descriptor;
  KeyPointVec prev_keypoints;

  for (int row = 0; row != cur_descriptor.rows; ++row) {
    all_descriptors.push_back(cur_descriptor.row(row).clone());

    prev_descriptor.push_back(cur_descriptor.row(row).clone());
    prev_keypoints.push_back(cur_keypoints[row]);

    KeyPointNode kpn;
    kpn.keypoint = cur_keypoints[row];
    kpn.keypoint.class_id = keypoint_counter++;
    kpn.leaf = true;
    kpn.parent = -1;

    keypoint_nodes.push_back(kpn);
  }


  // Track successive keypoints
  std::cout << "Finding tracks" << std::endl;
  for (size_t idx=1; idx != images.size(); ++idx) {
    std::cout << "  Processing image " << idx << std::endl;

    cur_descriptor = cv::Mat();
    cur_keypoints.clear();

    cv::Mat cur_image = images[idx];
    detdes_ptr->detectAndCompute(cur_image, mask, cur_keypoints,
                                 cur_descriptor, false);

    cv::BFMatcher new_matcher;
    std::vector<std::vector<cv::DMatch>> new_raw_matches;
    new_matcher.knnMatch(cur_descriptor, MatVecToMat(prev_descriptor),
                         new_raw_matches, 2);

    // Use ratio matching to detect matches, and threshold to find new tracks
    // starting
    for (size_t idx = 0; idx != new_raw_matches.size(); ++idx) {
      cv::DMatch best_match = new_raw_matches[idx][0];
      cv::DMatch second_best_match = new_raw_matches[idx][1];
      assert(best_match.queryIdx == second_best_match.queryIdx);
      assert(best_match.trainIdx != second_best_match.trainIdx);

      KeyPointNode kpn;
      kpn.keypoint = cur_keypoints[best_match.queryIdx];
      kpn.keypoint.class_id = keypoint_counter++;
      kpn.parent = -1;
      kpn.leaf = true;

      if (best_match.distance < RATIO_THRESHOLD * second_best_match.distance) {
        // Set best match as our parent
        // Find the global index that matches the best match descriptor from
        // the previous image as the parent
        bool eq = false;
        for (size_t pidx = 0; pidx != all_descriptors.size(); ++pidx) {
          cv::Mat diff =
            all_descriptors[pidx] != prev_descriptor[best_match.trainIdx];
          eq = cv::countNonZero(diff) == 0;
          if (eq) {
            kpn.parent = pidx;
            keypoint_nodes[pidx].leaf = false;
            break;
          }
        }

        if (!eq) {
          // Found match was previously unknown because it did not pass the
          // threshold test below, so add it first.
          KeyPointNode kpn_prev;
          kpn_prev.keypoint = prev_keypoints[best_match.trainIdx];
          kpn_prev.leaf = false;
          kpn_prev.parent = -1;
//          keypoint_nodes.push_back(kpn_prev);
//          all_descriptors.push_back(prev_descriptor.row(best_match.trainIdx));

//          kpn.parent = keypoint_nodes.size() - 1;
        } else {
          int row = best_match.queryIdx;
          all_descriptors.push_back(cur_descriptor.row(row));
          keypoint_nodes.push_back(kpn);

          prev_descriptor.push_back(cur_descriptor.row(row).clone());
          prev_keypoints.push_back(cur_keypoints[row]);
        }
      } else if (best_match.distance > DISTANCE_THRESHOLD) {
        // Didn't find a match, but this is a good keypoint on its own, so
        // start a new track.
        int row = best_match.queryIdx;
        keypoint_nodes.push_back(kpn);
        all_descriptors.push_back(cur_descriptor.row(row));

        prev_descriptor.push_back(cur_descriptor.row(row).clone());
        prev_keypoints.push_back(cur_keypoints[row]);
      }
    }
  }

  // Paint the lines
  cv::Mat color_image;
  cv::cvtColor(images[0], color_image, cv::COLOR_GRAY2RGB);
  for (size_t idx=0; idx != keypoint_nodes.size(); ++idx) {
    KeyPointNode kpn = keypoint_nodes[idx];
    size_t count = 0;
    if (kpn.leaf) {
      std::vector<cv::Point> points;
      points.push_back(kpn.keypoint.pt);
      while (kpn.parent >= 0) {
        ++count;
        kpn = keypoint_nodes[kpn.parent];
        points.push_back(kpn.keypoint.pt);
      }

      if (count > 0) {
        for (size_t pidx=0; pidx != points.size()-1; ++pidx) {
          cv::line(color_image, points[pidx], points[pidx+1], {100, 100, 255});
          cv::circle(color_image, points[pidx], 1, {150, 150, 255});

        }
        cv::circle(color_image, points.back(), 3, {0, 0, 255});
      }
    }
  }

  cv::imshow("image", color_image);
  cv::waitKey(0);

  std::vector<std::array<double, 4>> motion_vecs;

  //  cv::Mat color_image;
  //  cv::cvtColor(image_1, color_image, cv::COLOR_GRAY2RGB);
  //  for (const auto &match : ratio_matches) {
  //    auto pt1 = keypoints_1[match.queryIdx].pt;
  //    auto pt2 = keypoints_2[match.trainIdx].pt;
  //    cv::circle(color_image, pt1, 3, {0, 0, 255});
  //    cv::line(color_image, pt1, pt2, {0, 0, 255});
  //    if (std::abs(pt2.x - pt1.x) > 4) {
  //      motion_vecs.push_back({{pt1.x, pt1.y, pt2.x-pt1.x, pt2.y-pt1.y}});
  //    }
  //  }
  //  cv::imshow("image", color_image);

  gimx.SendControl(XboneControl());

  auto mpl = MplConnect("/tmp/kernel.json");

  auto sendMat = [&] (const cv::Mat &mat, const std::string &name) {
    cv::Mat output(mat.rows, mat.cols, CV_64F);
    mat.convertTo(output, CV_64F);
    auto np_data = NumpyArray(name, (double*)output.data, output.rows,
                              output.cols);
    mpl.SendData(np_data);
  };

  auto sendVec = [&] (const std::vector<double> &vec,
                      const std::string &name) {
    mpl.SendData(NumpyArray(name, vec));
  };

  std::vector<double> distances;
  for (const auto &kpn : keypoint_nodes) {
    distances.push_back(kpn.keypoint.response);
  }
  sendVec(distances, "distances");

  std::vector<std::array<double, 2>> feature_points;
  //  for (const auto &kv : feature_map) {
  //    for (const auto &keypoint : kv.second) {
  //      feature_points.push_back({{keypoint.pt.x, keypoint.pt.y}});
  //    }
  //  }
  for (const auto &kpn : keypoint_nodes) {
    feature_points.push_back({{kpn.keypoint.pt.x, kpn.keypoint.pt.y}});
  }

  mpl.SendData(NumpyArray("points", (double*)feature_points.data(),
                          feature_points.size(), 2));

  mpl.SendData(NumpyArray("mv", (double*)motion_vecs.data(),
                          motion_vecs.size(), 4));

  sendMat(images.back(), "cur_image");
  mask.convertTo(mask, CV_64F);
  sendMat(mask, "mask");
  mpl.RunCode("images = []");
  for (const auto& image : images) {
    sendMat(image, "tmp_image");
    mpl.RunCode("images.append(tmp_image)");
  }

  //mpl.RunCode("plot(XX[:, 0], XX[:, 1])");

  return 0;
}
