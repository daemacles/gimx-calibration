#include <sys/time.h>

#include <iostream>
#include <random>

#include <QtWidgets>

#include "gimx.h"

std::random_device rd;
std::mt19937 rand_gen(rd());

int GetControl (double phi) {
  double a = 0.00136871;
  double c = 0.00017554;
  double d = -0.0057529;
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

class GimxControl : public QWidget {
public:
  GimxControl (QWidget *parent = 0);

protected:
  void mouseMoveEvent (QMouseEvent *event) Q_DECL_OVERRIDE;
  void mouseReleaseEvent (QMouseEvent *event) Q_DECL_OVERRIDE;
  void mousePressEvent (QMouseEvent *event) Q_DECL_OVERRIDE;
  void keyReleaseEvent (QKeyEvent *event) Q_DECL_OVERRIDE;
  void timerEvent (QTimerEvent *event) Q_DECL_OVERRIDE;

private:
  bool mouse_grabbed_;
  GimxConnection gimx_;
  double accum_;
};

GimxControl::GimxControl (QWidget *parent) :
    QWidget(parent),
    mouse_grabbed_(false),
    gimx_("localhost", 7799),
    accum_(0.0)
{
  this->setMouseTracking(true);
  this->startTimer(10);
  gimx_.Connect();
}

void GimxControl::mousePressEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    std::cout << "Pressed " << event->x() << std::endl;
  }
}

void GimxControl::mouseMoveEvent(QMouseEvent *event) {
  static int last_x = 0;

  if (mouse_grabbed_) {
    if (event->x() < this->width()/4 ||
        event->x() > 3*this->width()/4) {
      auto cursor = this->cursor();
      cursor.setPos(this->mapToGlobal(QPoint(this->width()/2,
                                             this->height()/2)));
      this->setCursor(cursor);
    }

    int diff = event->globalX() - last_x;
    last_x = event->globalX();

    if (std::abs(diff) < this->width()/4) {
      accum_ += diff;
    }
  }
}

void GimxControl::mouseReleaseEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    std::cout << "Released " << event->x() << std::endl;
  }
}

void GimxControl::keyReleaseEvent(QKeyEvent *event) {
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
  double sign = std::signbit(accum_) ? -1 : 1;
  double range = 20;
  double value = std::min(std::abs(accum_), range) / range;
  value = sign*std::pow(std::abs(value), 1.6) * 0.94 * 1;

  int x_control = GetControl(value);

  XboneControl ctl;
  ctl.right_stick.x = x_control;
  ctl.right_stick.y = 0;
  gimx_.SendControl(ctl);

  accum_ = 0;
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
