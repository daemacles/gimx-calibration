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
    int exp_control = std::log((inflection - d) / a) / c;
    int max_control = 32000;
    std::uniform_real_distribution<> dis(inflection, fastest);
    if (dis(rand_gen) < phi) {
      return sign*max_control;
    } else {
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
      exponent_(1.6)
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
    double value = std::min(std::abs(accum_), range_) / range_;
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
  std::map<Qt::Key, bool> keys_pressed_;
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

void GimxControl::mousePressEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    std::cout << "Pressed " << event->x() << std::endl;
  }
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

void GimxControl::mouseReleaseEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    std::cout << "Released " << event->x() << std::endl;
  }
}

void GimxControl::keyPressEvent(QKeyEvent *event) {
  keys_pressed_[event->key()] = true;
}

void GimxControl::keyReleaseEvent(QKeyEvent *event) {
  keys_pressed_[event->key()] = false;

  switch(event->key()) {
  case Qt::Key_Q:
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
}

void GimxControl::timerEvent (QTimerEvent *event) {
  XboneControl ctl;
  ctl.right_stick.x = x_axis_.GetAxisValue();
  ctl.right_stick.y = y_axis_.GetAxisValue();
  gimx_.SendControl(ctl);
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
