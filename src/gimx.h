#ifndef  GIMX_H_
#define  GIMX_H_

#include <cstdlib>
#include <cstring>
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <stdexcept>
#include <string>
#include <vector>

constexpr uint32_t GIMX_UDP_BUF_SIZE = 158;

enum XboneIndices {
  XBONE_LEFT_X  = 0,  // rel_axis_0   [-32768,   32767]
  XBONE_LEFT_Y  = 1,  // rel_axis_1   [-32768,   32767]
  XBONE_RIGHT_X = 2,  // rel_axis_2   [-32768,   32767]
  XBONE_RIGHT_Y = 3,  // rel_axis_3   [-32768,   32767]
  XBONE_VIEW    = 8,  // abs_axis_0   {0,        255}
  XBONE_MENU    = 9,  // abs_axis_1   {0,        255}
  XBONE_GUIDE   = 10, // abs_axis_2   {0,        255}
  XBONE_UP      = 11, // abs_axis_3   {0,        255}
  XBONE_RIGHT   = 12, // abs_axis_4   {0,        255}
  XBONE_DOWN    = 13, // abs_axis_5   {0,        255}
  XBONE_LEFT    = 14, // abs_axis_6   {0,        255}
  XBONE_Y       = 15, // abs_axis_7   {0,        255}
  XBONE_B       = 16, // abs_axis_8   {0,        255}
  XBONE_A       = 17, // abs_axis_9   {0,        255}
  XBONE_X       = 18, // abs_axis_10  {0,        255}
  XBONE_LB      = 19, // abs_axis_11  {0,        255}
  XBONE_RB      = 20, // abs_axis_12  {0,        255}
  XBONE_LT      = 21, // abs_axis_13  [0,        1024]
  XBONE_RT      = 22, // abs_axis_14  [0,        1023]
  XBONE_LS      = 23, // abs_axis_15  {0,        255}
  XBONE_RS      = 24, // abs_axis_16  {0,        255}
  XBONE_NUM_INPUTS
};

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
  std::vector<int> inputs;

  XboneControl () :
      inputs(XBONE_NUM_INPUTS, 0)
  {
  }

  void SetInput (int input, int value) {
    inputs[input] = value;
  }

  void SerializeTo_ (int32_t *axes) const override {
    axes[XBONE_LEFT_X] = left_stick.x;
    axes[XBONE_LEFT_Y] = left_stick.y;
    axes[XBONE_RIGHT_X] = right_stick.x;
    axes[XBONE_RIGHT_Y] = right_stick.y;
    axes[XBONE_LT]     = inputs[XBONE_LT];
    axes[XBONE_RT]     = inputs[XBONE_RT];
    axes[XBONE_VIEW]   = inputs[XBONE_VIEW];
    axes[XBONE_MENU]   = inputs[XBONE_MENU];
    axes[XBONE_GUIDE]  = inputs[XBONE_GUIDE];
    axes[XBONE_UP]     = inputs[XBONE_UP];
    axes[XBONE_RIGHT]  = inputs[XBONE_RIGHT];
    axes[XBONE_DOWN]   = inputs[XBONE_DOWN];
    axes[XBONE_LEFT]   = inputs[XBONE_LEFT];
    axes[XBONE_Y]      = inputs[XBONE_Y];
    axes[XBONE_B]      = inputs[XBONE_B];
    axes[XBONE_A]      = inputs[XBONE_A];
    axes[XBONE_X]      = inputs[XBONE_X];
    axes[XBONE_LB]     = inputs[XBONE_LB];
    axes[XBONE_RB]     = inputs[XBONE_RB];
    axes[XBONE_LS]     = inputs[XBONE_LS];
    axes[XBONE_RS]     = inputs[XBONE_RS];
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

#endif  // #ifndef  GIMX_H_
