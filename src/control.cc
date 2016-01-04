#include <sys/time.h>

#include <iostream>
#include <map>
#include <random>

#include <QtWidgets>

#include "gimx.h"

std::random_device rd;
std::mt19937 rand_gen(rd());

int GetControl (double phi) {
  double a = 0.00136871;
  double c = 0.00017554;
  double d = -0.0057529 + 0.0043841921696713944;
  double inflection = 0.22262857715;
  double fastest = 0.939737022742;

  double sign = std::signbit(phi) ? -1 : 1;
  phi = std::abs(phi);

  if (phi < inflection) {
    return sign*std::log((phi - d) / a) / c;
  } else {
    std::uniform_real_distribution<> dis(inflection, fastest);
    if (true || dis(rand_gen) < phi) {
      int max_control = 32000;
      return sign*max_control;
    } else {
      int exp_control = std::log((inflection - d) / a) / c;
      return sign*exp_control;
    }
  }
}

class AxisControl {
public:
  AxisControl (double upper_thresh, double range) :
      accum_(0.0),
      prev_(0.0),
      upper_thresh_(upper_thresh),
      range_(range),
      exponent_(1.2)
  {
  }

  void Accumulate (double value) {
    double diff = value - prev_;
    prev_ = value;
    if (std::abs(diff) < upper_thresh_) {
      accum_ += diff;
    }
  }

  int GetAxisValue (void) {
    double sign = std::signbit(accum_) ? -1 : 1;

    // value will be between 0 and 1, where range_ represents the maximum
    // expected input
    double value = std::min(std::abs(accum_), range_) / range_;

    // apply acceleration curve and set between 0 and 0.94
    value = sign*std::pow(std::abs(value), exponent_) * 0.94 * 1;

    accum_ = 0;

    return GetControl(value);
  }

private:
  double accum_;
  double prev_;
  double upper_thresh_;
  double range_;
  double exponent_;
};


class GimxControl : public QWidget {
public:
  GimxControl (QWidget *parent = 0);

protected:
  void mouseMoveEvent (QMouseEvent *event) Q_DECL_OVERRIDE;
  void mouseReleaseEvent (QMouseEvent *event) Q_DECL_OVERRIDE;
  void mousePressEvent (QMouseEvent *event) Q_DECL_OVERRIDE;
  void keyPressEvent (QKeyEvent *event) Q_DECL_OVERRIDE;
  void keyReleaseEvent (QKeyEvent *event) Q_DECL_OVERRIDE;
  void timerEvent (QTimerEvent *event) Q_DECL_OVERRIDE;

private:
  bool mouse_grabbed_;
  GimxConnection gimx_;
  AxisControl x_axis_;
  AxisControl y_axis_;
  XboneControl ctl_;
};

GimxControl::GimxControl (QWidget *parent) :
    QWidget(parent),
    mouse_grabbed_(false),
    gimx_("localhost", 7799),
    x_axis_(30, 25),
    y_axis_(30, 25)
{
  this->setMouseTracking(true);
  this->startTimer(4);
  gimx_.Connect();
}


void GimxControl::mouseMoveEvent(QMouseEvent *event) {
  if (mouse_grabbed_) {
    if (event->x() < this->width()/4 ||
        event->x() > 3*this->width()/4) {
      auto cursor = this->cursor();
      cursor.setPos(this->mapToGlobal(QPoint(this->width()/2,
                                             cursor.pos().y())));
      this->setCursor(cursor);
    } else if (event->y() < this->height()/4 ||
               event->y() > 3*this->height()/4) {
      auto cursor = this->cursor();
      cursor.setPos(this->mapToGlobal(QPoint(cursor.pos().x(),
                                             this->height()/2)));
      this->setCursor(cursor);
    }

    x_axis_.Accumulate(event->globalX());
    y_axis_.Accumulate(event->globalY());
  }
}

void GimxControl::mousePressEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    ctl_.SetInput(XBONE_RT, 1000);
  }
}

void GimxControl::mouseReleaseEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    ctl_.SetInput(XBONE_RT, 0);
    std::cout << "Released " << event->x() << std::endl;
  }
}

void GimxControl::keyPressEvent(QKeyEvent *event) {
  if (mouse_grabbed_) {
    switch(event->key()) {
    case Qt::Key_W: ctl_.left_stick.y = -32000; break;
    case Qt::Key_S: ctl_.left_stick.y = 32000; break;
    case Qt::Key_A: ctl_.left_stick.x = -32000; break;
    case Qt::Key_D: ctl_.left_stick.x = 32000; break;

    case Qt::Key_B:       ctl_.SetInput(XBONE_GUIDE,  255); break;
    case Qt::Key_V:       ctl_.SetInput(XBONE_VIEW,   255); break;
    case Qt::Key_N:       ctl_.SetInput(XBONE_MENU,   255); break;
    case Qt::Key_Z:       ctl_.SetInput(XBONE_B,      255); break;
    case Qt::Key_Space:   ctl_.SetInput(XBONE_A,      255); break;
    case Qt::Key_X:       ctl_.SetInput(XBONE_X,      255); break;
    case Qt::Key_C:       ctl_.SetInput(XBONE_Y,      255); break;
    case Qt::Key_Q:       ctl_.SetInput(XBONE_LB,     255); break;
    case Qt::Key_E:       ctl_.SetInput(XBONE_RB,     255); break;
    case Qt::Key_Control: ctl_.SetInput(XBONE_LT,    1000); break;
    case Qt::Key_Up:      ctl_.SetInput(XBONE_UP,     255); break;
    case Qt::Key_Down:    ctl_.SetInput(XBONE_DOWN,   255); break;
    case Qt::Key_Left:    ctl_.SetInput(XBONE_LEFT,   255); break;
    case Qt::Key_Right:   ctl_.SetInput(XBONE_RIGHT,  255); break;
    default: break;
    }
  }
}

void GimxControl::keyReleaseEvent(QKeyEvent *event) {
  switch(event->key()) {
  case Qt::Key_P:
    QCoreApplication::exit(0);
    break;

  case Qt::Key_G:
    if (mouse_grabbed_) {
      this->releaseMouse();
      mouse_grabbed_ = false;
    } else {
      this->grabMouse();
      mouse_grabbed_ = true;
    }
    break;

  default:
    break;
  }

  if (mouse_grabbed_) {
    switch(event->key()) {
    case Qt::Key_W: ctl_.left_stick.y = 0; break;
    case Qt::Key_S: ctl_.left_stick.y = 0; break;
    case Qt::Key_A: ctl_.left_stick.x = 0; break;
    case Qt::Key_D: ctl_.left_stick.x = 0; break;

    case Qt::Key_B:       ctl_.SetInput(XBONE_GUIDE, 0); break;
    case Qt::Key_V:       ctl_.SetInput(XBONE_VIEW,  0); break;
    case Qt::Key_N:       ctl_.SetInput(XBONE_MENU,  0); break;
    case Qt::Key_Z:       ctl_.SetInput(XBONE_B,     0); break;
    case Qt::Key_Space:   ctl_.SetInput(XBONE_A,     0); break;
    case Qt::Key_X:       ctl_.SetInput(XBONE_X,     0); break;
    case Qt::Key_C:       ctl_.SetInput(XBONE_Y,     0); break;
    case Qt::Key_Q:       ctl_.SetInput(XBONE_LB,    0); break;
    case Qt::Key_E:       ctl_.SetInput(XBONE_RB,    0); break;
    case Qt::Key_Control: ctl_.SetInput(XBONE_LT,    0); break;
    case Qt::Key_Up:      ctl_.SetInput(XBONE_UP,    0); break;
    case Qt::Key_Down:    ctl_.SetInput(XBONE_DOWN,  0); break;
    case Qt::Key_Left:    ctl_.SetInput(XBONE_LEFT,  0); break;
    case Qt::Key_Right:   ctl_.SetInput(XBONE_RIGHT, 0); break;

    default: break;
    }
  }
}

void GimxControl::timerEvent (QTimerEvent *event) {
  ctl_.right_stick.x = x_axis_.GetAxisValue();
  ctl_.right_stick.y = y_axis_.GetAxisValue();
  gimx_.SendControl(ctl_);
}

int main (int argc, char *argv[]) {
  QApplication app(argc, argv);
  GimxControl window;
  window.resize(320, 240);
  window.show();
  window.setWindowTitle(QApplication::translate("toplevel",
                                                "Top-level widget"));
  return app.exec();
}
