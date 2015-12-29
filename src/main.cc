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
#include <unordered_map>
#include <unordered_set>
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
//#include <opencv2/xfeatures2d.hpp>
//#include <opencv2/xfeatures2d/nonfree.hpp>

#include <cpp_mpl.hpp>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-register"
#include <eigen3/Eigen/Dense>
#pragma clang diagnostic pop

using cppmpl::NumpyArray;

typedef Eigen::Matrix<NumpyArray::dtype, Eigen::Dynamic, Eigen::Dynamic,
                      Eigen::RowMajor> MatrixXd;
typedef std::vector<cv::Mat> MatVec;
typedef std::unordered_set<cv::Mat> MatSet;
typedef std::vector<cv::KeyPoint> KeyPointVec;
typedef std::unordered_set<cv::KeyPoint> KeyPointSet;

cppmpl::CppMatplotlib MplConnect (std::string config_path);

constexpr uint32_t GIMX_UDP_BUF_SIZE = 158;
constexpr double DISTANCE_THRESHOLD = 750.0;
constexpr double RATIO_THRESHOLD = 0.65;
std::string MASK_IMAGE_FILE = "/tmp/mask.png";
std::string WINDOW = "Figure jimx";

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
  typedef std::shared_ptr<Flea3> ptr_t;

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

    uint32_t *embedded_info = (uint32_t*)rawImage.GetData();

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
    uint32_t* target_info = (uint32_t*)image.data;
    target_info[0] = embedded_info[0];
    target_info[1] = embedded_info[1];

    return image;
  }

  cv::Mat GetImageFloat () {
    cv::Mat image = GetImage();
    image.convertTo(image, CV_32F);
    return image;
  }

  static std::shared_ptr<Flea3> GetInstance (void) {
    return std::make_shared<Flea3>();
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
    cv::Mat diff = A != B;
    return
      cv::countNonZero(diff) == 0 &&
      A.rows == B.rows &&
      A.cols == B.cols &&
      A.type() == B.type();
  }
};

template<> struct hash<cv::Mat> {
  size_t operator()(const cv::Mat& A) const {
    return (size_t)cv::sum(A)[0];
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

template<> struct hash<cv::KeyPoint> {
  size_t operator()(const cv::KeyPoint& k) const {
    return (size_t)(1000*(k.pt.x + k.pt.y));
  }
};

template<> struct equal_to<cv::KeyPoint> {
  bool operator()(const cv::KeyPoint& A, const cv::KeyPoint& B) const {
    return
      A.pt.x == B.pt.x &&
      A.pt.y == B.pt.y &&
      A.response == B.response &&
      A.angle == B.angle &&
      A.size == B.size;
  }
};
}


struct KeyPointNode {
  cv::KeyPoint keypoint;
  int parent;
  bool leaf;
};


cv::Mat MatVecToMat (const MatVec &mv) {
  cv::Mat output(mv.size(), mv[0].cols, mv[0].type());
  for (size_t row=0; row != mv.size(); ++row) {
    output.row(row) = mv[row];
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
  new_matcher.knnMatch(descriptors, descriptors, new_raw_matches, 2);

  // Use ratio matching
  cv::Mat descriptors_tmp = descriptors.getMat().clone();
  KeyPointVec keypoints_tmp (keypoints);
  MatVec descriptors_vec;
  good_keypoints.clear();
  for (size_t idx = 0; idx != new_raw_matches.size(); ++idx) {
    cv::DMatch best_match = new_raw_matches[idx][1];
    //cv::DMatch second_best_match = new_raw_matches[idx][2];

    if (best_match.distance > min_distance) {
      std::cout << best_match.distance << std::endl;
      descriptors_vec.push_back(descriptors_tmp.row(best_match.queryIdx));
      good_keypoints.push_back(keypoints_tmp[best_match.queryIdx]);
    }
  }
  std::cout << "Got " << descriptors_vec.size() << " keypoints "
    << "(down from " << keypoints_tmp.size() << ")" << std::endl;

  assert(descriptors_vec.size() > 0);

  good_descriptors.create(descriptors_vec.size(), descriptors_vec[0].cols,
                          descriptors_vec[0].type());
  cv::Mat output_mat = good_descriptors.getMat();
  for (size_t idx = 0; idx != descriptors_vec.size(); ++idx) {
    output_mat.row(idx) = descriptors_vec[idx];
  }
}


class Measure {
public:
  typedef std::unordered_map<int, KeyPointNode> KpnMap;

  Measure (GimxConnection gimx, Flea3::ptr_t flea3_p) :
      gimx_(gimx),
      flea3_p_(flea3_p),
      metadata_(10)
  {
    gimx_.Connect();
  }

  MatVec GetImages (size_t N) {
    // capture loop
    char key = 0;
    struct timeval prev_time;
    gettimeofday(&prev_time, NULL);
    double diff_us = 0;
    double counter = 0;
    MatVec images;

    XboneControl ctl;
    ctl.right_stick.x = 29250;
    gimx_.SendControl(ctl);
    usleep(1.0e6);

    uint32_t frame_count = 0;
    uint32_t last_frame_count = 0;
    for (size_t capture_count = 0; key != 'q' && capture_count < N;
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
      cv::Mat image = flea3_p_->GetImage();
      images.push_back(image);
      cv::imshow(WINDOW, image);
      key = cv::waitKey(1) & 0xff;
      uint32_t* embedded_info = (uint32_t*)image.data;
      last_frame_count = frame_count;
      frame_count = embedded_info[1];

      // Skip some frames -- keeps timing accurate
      for (size_t skips = 0; skips != 0; --skips) {
        flea3_p_->GetImage();
      }
    }
    metadata_[0] = frame_count - last_frame_count;

    ctl.right_stick.x = 0;
    gimx_.SendControl(ctl);

    return images;
  }

  KpnMap ProcessImages (MatVec &images) {
    // Now extract feature tracks
    // Pass 1, figure out all descriptors of interest.
    std::cout << "Extracting feature tracks" << std::endl;

    int keypoint_counter = 0;
    KpnMap kpn_map;

    cv::Mat cur_image = images[0];

    auto detdes_ptr = cv::BRISK::create();
    cv::Mat cur_descriptors;
    KeyPointVec cur_keypoints;
    detdes_ptr->detectAndCompute(cur_image, GetMask(), cur_keypoints,
                                 cur_descriptors, false);
    cur_descriptors.convertTo(cur_descriptors, CV_32F);

    //  GetGoodDescriptors(cur_descriptors, cur_keypoints,
    //                     cur_descriptors, cur_keypoints);

    for (int row = 0; row != cur_descriptors.rows; ++row) {
      cur_keypoints[row].class_id = keypoint_counter++;
      KeyPointNode kpn;
      kpn.keypoint = cur_keypoints[row];
      kpn.leaf = true;
      kpn.parent = -1;
      kpn_map[kpn.keypoint.class_id] = kpn;
    }

    // Track successive keypoints
    std::cout << "Finding tracks" << std::endl;
    for (size_t idx=1; idx != images.size(); ++idx) {
      std::cout << "  Processing image " << idx << std::endl;
      cur_image = images[idx];

      cv::Mat next_descriptors;
      KeyPointVec next_keypoints;
      detdes_ptr->detectAndCompute(cur_image, GetMask(), next_keypoints,
                                   next_descriptors, false);
      next_descriptors.convertTo(next_descriptors, CV_32F);

      MatSet next_descriptors_set;
      for (int row=0; row != next_descriptors.rows; ++row) {
        next_descriptors_set.insert(next_descriptors.row(row));
      }
      KeyPointSet next_keypoints_set(next_keypoints.begin(),
                                     next_keypoints.end());

      // Use ratio matching to detect matches, and threshold to find new tracks
      // starting
      cv::BFMatcher new_matcher;
      std::vector<std::vector<cv::DMatch>> new_raw_matches;
      new_matcher.knnMatch(next_descriptors, cur_descriptors,
                           new_raw_matches, 4);

      MatVec track_descriptors;
      KeyPointVec track_keypoints;
      for (size_t idx = 0; idx != new_raw_matches.size(); ++idx) {

        cv::DMatch best_match = new_raw_matches[idx][0];
        cv::DMatch second_best_match = new_raw_matches[idx][1];
        assert(best_match.queryIdx == second_best_match.queryIdx);
        assert(best_match.trainIdx != second_best_match.trainIdx);

        if (best_match.distance < RATIO_THRESHOLD * second_best_match.distance) {
          cv::Mat train_desc = cur_descriptors.row(best_match.trainIdx);
          cv::Mat query_desc = next_descriptors.row(best_match.queryIdx);
          cv::KeyPoint &train_keypoint = cur_keypoints[best_match.trainIdx];
          cv::KeyPoint &query_keypoint = next_keypoints[best_match.queryIdx];

          track_descriptors.push_back(query_desc);
          track_keypoints.push_back(query_keypoint);

          next_descriptors_set.erase(query_desc);
          next_keypoints_set.erase(query_keypoint);

          auto& kpn_parent = kpn_map[train_keypoint.class_id];
          kpn_parent.leaf = false;

          // Set best match as our parent
          query_keypoint.class_id = keypoint_counter++;
          KeyPointNode kpn;
          kpn.keypoint = query_keypoint;
          kpn.keypoint.class_id = keypoint_counter++;
          kpn.parent = train_keypoint.class_id;
          kpn.leaf = true;

          kpn_map[query_keypoint.class_id] = kpn;
        } else {
          //        std::cout << best_match.distance <<
          //         " " << second_best_match.distance << std::endl;
        }
      }

      //    MatVec nd (next_descriptors_set.begin(), next_descriptors_set.end());
      //    next_descriptors = MatVecToMat(nd);
      //
      //    next_keypoints = KeyPointVec(next_keypoints_set.begin(),
      //                                 next_keypoints_set.end());
      ////    GetGoodDescriptors(next_descriptors, next_keypoints,
      ////                       next_descriptors, next_keypoints);
      //
      //    // Union
      //    next_descriptors.push_back(MatVecToMat(track_descriptors));
      cur_descriptors = next_descriptors.clone();

      //    next_keypoints.insert(next_keypoints.end(), track_keypoints.begin(),
      //                          track_keypoints.end());
      cur_keypoints = next_keypoints;
    }
    return kpn_map;
  }

  cv::Mat GetMask (void) {
    if (mask_.rows != 0) {
      return mask_;
    }

    cv::Mat mask;
    if (!FileExists(MASK_IMAGE_FILE)) {
      // Compute mask
      std::cout << "Creating motion mask. Takes about 10 seconds." << std::endl;
      XboneControl ctl;
      ctl.right_stick.x = 25000;
      gimx_.SendControl(ctl);

      MatVec mask_images;
      for (size_t capture_count = 0; capture_count != 600; ++capture_count) {
        mask_images.push_back(flea3_p_->GetImageFloat() / 255.0);
        cv::imshow(WINDOW, mask_images.back());
        cv::waitKey(1);
        for (size_t skip_count = 0; skip_count != 0; ++skip_count) {
          flea3_p_->GetImage();
        }
      }

      ctl.right_stick.x = 0;
      gimx_.SendControl(ctl);

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
                                                      cv::Size(3, 3)),
                cv::Point(-1, -1), 8);
      mask *= 255;
      mask.convertTo(mask, CV_8UC1);

      cv::imwrite(MASK_IMAGE_FILE, mask);

      cv::imshow(WINDOW, mask);
      cv::waitKey(0);
    } else {
      mask = cv::imread(MASK_IMAGE_FILE);
      cv::cvtColor(mask, mask, cv::COLOR_BGR2GRAY);
      std::cout << "Using mask from " << MASK_IMAGE_FILE
        << ". Delete to recreate mask anew." << std::endl;
    }

    mask_ = mask;
    return mask;
  }

  std::vector<double> GetMetadata (void) {
    return metadata_;
  }

private:
  GimxConnection gimx_;
  Flea3::ptr_t flea3_p_;
  cv::Mat mask_;

  // 0 - frames between captures
  std::vector<double> metadata_;
};


int main (int argc, char **argv) {
  (void) argc;
  (void) argv;

  GimxConnection gimx("localhost", 7799);
  gimx.Connect();

  Flea3::ptr_t flea3_p = Flea3::GetInstance();
  if (!flea3_p->Connect()) {
    std::cerr << "Couldn't connect to camera" << std::endl;
    std::exit(1);
  }

  flea3_p->GetImage();

  Measure measure(gimx, flea3_p);

  MatVec images = measure.GetImages(50);
  Measure::KpnMap kpn_map = measure.ProcessImages(images);

  // Paint the lines
  cv::Mat color_image;
  cv::cvtColor(images[0], color_image, cv::COLOR_GRAY2RGB);
  std::vector<std::array<double, 4>> motion_vecs;
  for (const auto& kv : kpn_map) {
    KeyPointNode kpn = kv.second;
    size_t count = 0;
    if (kpn.leaf) {
      std::vector<cv::Point> points;
      points.push_back(kpn.keypoint.pt);
      while (kpn.parent >= 0) {
        ++count;
        kpn = kpn_map.at(kpn.parent);
        points.push_back(kpn.keypoint.pt);
      }

      if (count > 0) {
        for (size_t pidx=0; pidx != points.size()-1; ++pidx) {
          const auto &pt1 = points[pidx];
          const auto &pt2 = points[pidx+1];

          cv::line(color_image, pt1, pt2, {100, 100, 255});
          cv::circle(color_image, pt1, 1, {150, 150, 255});
          if (std::abs(pt2.x - pt1.x) > 1) {
            motion_vecs.push_back({{(double)pt1.x, (double)pt1.y,
                                  (double)pt2.x - pt1.x,
                                  (double)pt2.y - pt1.y}});
          }

        }
        cv::circle(color_image, points.back(), 3, {0, 0, 255});
      }
    }
  }

  cv::imshow(WINDOW, color_image);
  cv::waitKey(0);

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
  //  cv::imshow(WINDOW, color_image);

  gimx.SendControl(XboneControl());

  std::cout << "Transferring data to ipython" << std::endl;
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

  sendVec(measure.GetMetadata(), "metadata");

  std::vector<double> distances;
  std::vector<std::array<double, 2>> feature_points;
  for (const auto &kv : kpn_map) {
    distances.push_back(kv.second.keypoint.response);
    feature_points.push_back({{kv.second.keypoint.pt.x,
                               kv.second.keypoint.pt.y}});
  }
  sendVec(distances, "distances");

  mpl.SendData(NumpyArray("points", (double*)feature_points.data(),
                          feature_points.size(), feature_points[0].size()));

  mpl.SendData(NumpyArray("mv", (double*)motion_vecs.data(),
                          motion_vecs.size(), motion_vecs[0].size()));

  sendMat(images.back(), "cur_image");
  cv::Mat mask = measure.GetMask();
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
