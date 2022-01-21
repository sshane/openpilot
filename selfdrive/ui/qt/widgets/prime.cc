#include "selfdrive/ui/qt/widgets/prime.h"

#include <QDebug>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QPushButton>
#include <QStackedWidget>
#include <QTimer>
#include <QVBoxLayout>

#include <QrCode.hpp>

#include "selfdrive/ui/qt/request_repeater.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/qt_window.h"

using qrcodegen::QrCode;

PairingQRWidget::PairingQRWidget(QWidget* parent) : QWidget(parent) {
  QTimer* timer = new QTimer(this);
  timer->start(5 * 60 * 1000);
  connect(timer, &QTimer::timeout, this, &PairingQRWidget::refresh);
}

void PairingQRWidget::showEvent(QShowEvent *event) {
  refresh();
}

void PairingQRWidget::refresh() {
  if (isVisible()) {
    QString pairToken = CommaApi::create_jwt({{"pair", true}});
    QString qrString = "https://connect.comma.ai/?pair=" + pairToken;
    this->updateQrCode(qrString);
  }
}

void PairingQRWidget::updateQrCode(const QString &text) {
  QrCode qr = QrCode::encodeText(text.toUtf8().data(), QrCode::Ecc::LOW);
  qint32 sz = qr.getSize();
  QImage im(sz, sz, QImage::Format_RGB32);

  QRgb black = qRgb(0, 0, 0);
  QRgb white = qRgb(255, 255, 255);
  for (int y = 0; y < sz; y++) {
    for (int x = 0; x < sz; x++) {
      im.setPixel(x, y, qr.getModule(x, y) ? black : white);
    }
  }

  // Integer division to prevent anti-aliasing
  int final_sz = ((width() / sz) - 1) * sz;
  img = QPixmap::fromImage(im.scaled(final_sz, final_sz, Qt::KeepAspectRatio), Qt::MonoOnly);
}

void PairingQRWidget::paintEvent(QPaintEvent *e) {
  QPainter p(this);
  p.fillRect(rect(), Qt::white);

  QSize s = (size() - img.size()) / 2;
  p.drawPixmap(s.width(), s.height(), img);
}


PairingPopup::PairingPopup(QWidget *parent) : QDialogBase(parent) {
  QHBoxLayout *hlayout = new QHBoxLayout(this);
  hlayout->setContentsMargins(0, 0, 0, 0);
  hlayout->setSpacing(0);

  setStyleSheet("PairingPopup { background-color: #E0E0E0; }");

  // text
  QVBoxLayout *vlayout = new QVBoxLayout();
  vlayout->setContentsMargins(85, 70, 50, 70);
  vlayout->setSpacing(50);
  hlayout->addLayout(vlayout, 1);
  {
    QPushButton *close = new QPushButton(QIcon(":/icons/close.svg"), "", this);
    close->setIconSize(QSize(80, 80));
    close->setStyleSheet("border: none;");
    vlayout->addWidget(close, 0, Qt::AlignLeft);
    QObject::connect(close, &QPushButton::clicked, this, &QDialog::reject);

    vlayout->addSpacing(30);

    QLabel *title = new QLabel("Pair your device to your comma account", this);
    title->setStyleSheet("font-size: 75px; color: black;");
    title->setWordWrap(true);
    vlayout->addWidget(title);

    QLabel *instructions = new QLabel(R"(
      <ol type='1' style='margin-left: 15px;'>
        <li style='margin-bottom: 50px;'>Go to https://connect.comma.ai on your phone</li>
        <li style='margin-bottom: 50px;'>Click "add new device" and scan the QR code on the right</li>
        <li style='margin-bottom: 50px;'>Bookmark connect.comma.ai to your home screen to use it like an app</li>
      </ol>
    )", this);
    instructions->setStyleSheet("font-size: 47px; font-weight: bold; color: black;");
    instructions->setWordWrap(true);
    vlayout->addWidget(instructions);

    vlayout->addStretch();
  }

  // QR code
  PairingQRWidget *qr = new PairingQRWidget(this);
  hlayout->addWidget(qr, 1);
}


PrimeUserWidget::PrimeUserWidget(QWidget* parent) : QWidget(parent) {
  mainLayout = new QVBoxLayout(this);
  mainLayout->setMargin(0);
  mainLayout->setSpacing(30);

  // subscribed prime layout
  QWidget *primeWidget = new QWidget;
  primeWidget->setObjectName("primeWidget");
  QVBoxLayout *primeLayout = new QVBoxLayout(primeWidget);
  primeLayout->setMargin(0);
  primeWidget->setContentsMargins(60, 50, 60, 50);

  subscribed = new QLabel("✕ NOT SUBSCRIBED");
  subscribed->setStyleSheet("font-size: 41px; font-weight: bold; color: #ff4e4e;");
  primeLayout->addWidget(subscribed, 0, Qt::AlignTop);

  primeLayout->addSpacing(60);

  commaPrime = new QLabel("comma prime");
  commaPrime->setStyleSheet("font-size: 75px; font-weight: bold;");
  primeLayout->addWidget(commaPrime, 0, Qt::AlignTop);

  primeLayout->addSpacing(20);

  QLabel* connectUrl = new QLabel("CONNECT.COMMA.AI");
  connectUrl->setStyleSheet("font-size: 41px; font-family: Inter SemiBold; color: #A0A0A0;");
  primeLayout->addWidget(connectUrl, 0, Qt::AlignTop);

  mainLayout->addWidget(primeWidget);

  // comma points layout
  QWidget *pointsWidget = new QWidget;
  pointsWidget->setObjectName("primeWidget");
  QVBoxLayout *pointsLayout = new QVBoxLayout(pointsWidget);
  pointsLayout->setMargin(0);
  pointsWidget->setContentsMargins(60, 50, 60, 50);

  QLabel* commaPoints = new QLabel("COMMA POINTS");
  commaPoints->setStyleSheet("font-size: 41px; font-family: Inter SemiBold;");
  pointsLayout->addWidget(commaPoints, 0, Qt::AlignTop);

  points = new QLabel("0");
  points->setStyleSheet("font-size: 91px; font-weight: bold;");
  pointsLayout->addWidget(points, 0, Qt::AlignTop);

  mainLayout->addWidget(pointsWidget);

  QWidget *thirdWidget = new QWidget;
  thirdWidget->setObjectName("primeWidget");
  QVBoxLayout *thirdLayout = new QVBoxLayout(thirdWidget);
  thirdLayout->setMargin(0);
  thirdWidget->setContentsMargins(60, 50, 60, 50);

//  QLabel* thirdLabel = new QLabel("Happy New Year! \U0001f389");
  sentryToggle = new ToggleControl("Sentry Mode", "", "", Params().getBool("SentryMode"));
  QObject::connect(sentryToggle, &ToggleControl::toggleFlipped, [=](bool enabling) {
    if (enabling && ConfirmationDialog::confirm("Sentry Mode (beta) records video and plays an alert when movement is detected. Currently only Toyota vehicles are supported and it may draw more power when enabled. Are you sure?", this)) {
      Params().putBool("SentryMode", true);
    } else {
      Params().putBool("SentryMode", false);
      if (enabling) {  // declined
        sentryToggle->toggle.togglePosition();
      }
    }
  });

  // Update toggle label with armed status
  QTimer::singleShot(5 * 1000, [=]() {
    if (uiState()->scene.sentry_armed) {
      sentryToggle->setTitle("Sentry Armed");
    } else {
      sentryToggle->setTitle("Sentry Mode");
    }
  });

  sentryToggle->setStyleSheet(R"(
    QPushButton {
      background-color: none;
      font-size: 30px;
      font-family: Inter SemiBold;
      font-weight: 200;
      border-radius: 10px;
    }
  )");
  thirdLayout->addWidget(sentryToggle, 0, Qt::AlignVCenter);

  mainLayout->addWidget(thirdWidget);

  // set up API requests
  if (auto dongleId = getDongleId()) {
    QString url = CommaApi::BASE_URL + "/v1/devices/" + *dongleId + "/owner";
    RequestRepeater *repeater = new RequestRepeater(this, url, "ApiCache_Owner", 6);
    QObject::connect(repeater, &RequestRepeater::requestDone, this, &PrimeUserWidget::replyFinished);
  }
//  setStyleSheet(R"(
//    * {
//      background-color: none;
//      font-size: 50px;
//    }
//  )");
}

void PrimeUserWidget::setPrime(bool hasPrime) {
  subscribed->setText(hasPrime ? "✓ SUBSCRIBED" : "✕ NOT SUBSCRIBED");
  subscribed->setStyleSheet(QString("font-size: 41px; font-weight: bold; color: %1;").arg(hasPrime ? "#86FF4E" : "#ff4e4e"));
  commaPrime->setText(hasPrime ? "comma prime" : "got prime?");
}

void PrimeUserWidget::replyFinished(const QString &response) {
  QJsonDocument doc = QJsonDocument::fromJson(response.toUtf8());
  if (doc.isNull()) {
    qDebug() << "JSON Parse failed on getting points";
    return;
  }

  QJsonObject json = doc.object();
  points->setText(QString::number(json["points"].toInt()));
}

PrimeAdWidget::PrimeAdWidget(QWidget* parent) : QFrame(parent) {
  QVBoxLayout* main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(70, 40, 80, 40);
  main_layout->setSpacing(0);

  QLabel *upgrade = new QLabel("Upgrade Now");
  upgrade->setStyleSheet("font-size: 75px; font-weight: bold;");
  main_layout->addWidget(upgrade, 0, Qt::AlignTop);
  main_layout->addSpacing(35);

  QLabel *description = new QLabel("Become a comma prime member at connect.comma.ai");
  description->setStyleSheet("font-size: 60px; font-weight: light; color: white;");
  description->setWordWrap(true);
  main_layout->addWidget(description, 0, Qt::AlignTop);

  main_layout->addStretch();

  QLabel *features = new QLabel("PRIME FEATURES:");
  features->setStyleSheet("font-size: 41px; font-weight: bold; color: #E5E5E5;");
  main_layout->addWidget(features, 0, Qt::AlignBottom);
  main_layout->addSpacing(20);

  QVector<QString> bullets = {"Remote access", "14 days of storage", "Developer perks"};
  for (auto &b: bullets) {
    const QString check = "<b><font color='#465BEA'>✓</font></b> ";
    QLabel *l = new QLabel(check + b);
    l->setAlignment(Qt::AlignLeft);
    l->setStyleSheet("font-size: 50px; margin-bottom: 15px;");
    main_layout->addWidget(l, 0, Qt::AlignBottom);
  }

  QPushButton *dismiss = new QPushButton("Dismiss");
  dismiss->setFixedHeight(110);
  dismiss->setStyleSheet(R"(
    QPushButton {
      font-size: 55px;
      font-weight: 400;
      border-radius: 10px;
      background-color: #465BEA;
    }
    QPushButton:pressed {
      background-color: #3049F4;
    }
  )");
  QObject::connect(dismiss, &QPushButton::clicked, this, [=]() {
    Params().putBool("PrimeAdDismissed", true);
    emit showPrimeWidget(false);  // dismiss with no prime
  });
  main_layout->addSpacing(20);
  main_layout->addWidget(dismiss);

  setStyleSheet(R"(
    PrimeAdWidget {
      border-radius: 10px;
      background-color: #333333;
    }
  )");
}


SetupWidget::SetupWidget(QWidget* parent) : QFrame(parent) {
  mainLayout = new QStackedWidget;

  // Unpaired, registration prompt layout

  QWidget* finishRegistration = new QWidget;
  finishRegistration->setObjectName("primeWidget");
  QVBoxLayout* finishRegistationLayout = new QVBoxLayout(finishRegistration);
  finishRegistationLayout->setContentsMargins(30, 75, 30, 45);
  finishRegistationLayout->setSpacing(0);

  QLabel* registrationTitle = new QLabel("Finish Setup");
  registrationTitle->setStyleSheet("font-size: 75px; font-weight: bold; margin-left: 55px;");
  finishRegistationLayout->addWidget(registrationTitle);

  finishRegistationLayout->addSpacing(30);

  QLabel* registrationDescription = new QLabel("Pair your device with comma connect (connect.comma.ai) and claim your comma prime offer.");
  registrationDescription->setWordWrap(true);
  registrationDescription->setStyleSheet("font-size: 55px; font-weight: light; margin-left: 55px;");
  finishRegistationLayout->addWidget(registrationDescription);

  finishRegistationLayout->addStretch();

  QPushButton* pair = new QPushButton("Pair device");
  pair->setFixedHeight(220);
  pair->setStyleSheet(R"(
    QPushButton {
      font-size: 55px;
      font-weight: 400;
      border-radius: 10px;
      background-color: #465BEA;
    }
    QPushButton:pressed {
      background-color: #3049F4;
    }
  )");
  finishRegistationLayout->addWidget(pair);

  popup = new PairingPopup(this);
  QObject::connect(pair, &QPushButton::clicked, popup, &PairingPopup::exec);

  mainLayout->addWidget(finishRegistration);

  // build stacked layout
  QVBoxLayout *outer_layout = new QVBoxLayout(this);
  outer_layout->setContentsMargins(0, 0, 0, 0);
  outer_layout->addWidget(mainLayout);

  primeAd = new PrimeAdWidget;
  QObject::connect(primeAd, &PrimeAdWidget::showPrimeWidget, this, &SetupWidget::showPrimeWidget);  // for dismiss button
  mainLayout->addWidget(primeAd);

  primeUser = new PrimeUserWidget;
  mainLayout->addWidget(primeUser);

  mainLayout->setCurrentWidget(primeAd);

  setFixedWidth(750);
  setStyleSheet(R"(
    #primeWidget {
      border-radius: 10px;
      background-color: #333333;
    }
  )");

  // Retain size while hidden
  QSizePolicy sp_retain = sizePolicy();
  sp_retain.setRetainSizeWhenHidden(true);
  setSizePolicy(sp_retain);

  // set up API requests
  if (auto dongleId = getDongleId()) {
    QString url = CommaApi::BASE_URL + "/v1.1/devices/" + *dongleId + "/";
    RequestRepeater* repeater = new RequestRepeater(this, url, "ApiCache_Device", 5);

    QObject::connect(repeater, &RequestRepeater::requestDone, this, &SetupWidget::replyFinished);
  }
  // Only show when first request comes back if not PC
  if (!Hardware::PC()) hide();
}

void SetupWidget::replyFinished(const QString &response, bool success) {
  show();
  if (!success) return;

  QJsonDocument doc = QJsonDocument::fromJson(response.toUtf8());
  if (doc.isNull()) {
    qDebug() << "JSON Parse failed on getting pairing and prime status";
    return;
  }

  QJsonObject json = doc.object();
  if (!json["is_paired"].toBool()) {
    mainLayout->setCurrentIndex(0);
  } else {
    popup->reject();

    bool prime = json["prime"].toBool();

    if (uiState()->has_prime != prime) {
      uiState()->has_prime = prime;
      Params().putBool("HasPrime", prime);
    }

    if (prime || Params().getBool("PrimeAdDismissed")) {
      showPrimeWidget(json["prime"].toBool());
    } else {
      mainLayout->setCurrentWidget(primeAd);
    }
  }
}

void SetupWidget::showPrimeWidget(bool hasPrime) {
  primeUser->setPrime(hasPrime);
  mainLayout->setCurrentWidget(primeUser);
}
