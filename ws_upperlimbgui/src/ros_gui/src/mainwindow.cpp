#include "mainwindow.h"

#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QSplitter>
#include <QGroupBox>
#include <QHeaderView>
#include <QDateTime>
#include <QButtonGroup>
#include <QMutexLocker>
#include <QStyle>

#include <algorithm>
#include <cmath>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    // ================================
    // ROS2 初始化
    // ================================

    node_ = rclcpp::Node::make_shared("qt_gui_node");

    control_command_pub_ =
        node_->create_publisher<upperlimb_robot::msg::ControlCommand>("control_command", 10);

    eeg_frame_pub_ = node_->create_publisher<std_msgs::msg::String>("eeg_test_frame", 10);
    eeg_source_pub_ = node_->create_publisher<std_msgs::msg::Bool>(
        "eeg_debug_source", rclcpp::QoS(1).transient_local());
    std_msgs::msg::Bool initial_source;
    initial_source.data = true;
    eeg_source_pub_->publish(initial_source);
    eeg_status_sub_ = node_->create_subscription<std_msgs::msg::String>(
        "eeg_status", 10,
        std::bind(&MainWindow::eegStatusCallback, this, std::placeholders::_1));

    system_state_sub_ =
        node_->create_subscription<upperlimb_robot::msg::SystemState>(
            "system_state",
            10,
            std::bind(&MainWindow::systemStateCallback, this, std::placeholders::_1));

    robot_state_sub_ =
        node_->create_subscription<upperlimb_robot::msg::RobotState>(
            "robot_state",
            10,
            std::bind(&MainWindow::robotStateCallback, this, std::placeholders::_1));

    // ================================
    // 主窗口
    // ================================
    QWidget *central = new QWidget(this);
    setCentralWidget(central);

    QVBoxLayout *mainLayout = new QVBoxLayout(central);
    mainLayout->setContentsMargins(8, 8, 8, 8);
    mainLayout->setSpacing(8);

    // ================================
    // 顶部状态栏
    // ================================
    QGroupBox *statusGroup = new QGroupBox("系统状态", central);
    statusGroup->setObjectName("statusPanel");
    QHBoxLayout *statusLayout = new QHBoxLayout(statusGroup);
    statusLayout->setContentsMargins(12, 8, 12, 8);
    statusLayout->setSpacing(12);

    // 控制器初始化
    controllerInitIndicator_ = new QLabel(statusGroup);
    controllerInitIndicator_->setObjectName("statusIndicator");
    controllerInitIndicator_->setFixedSize(12, 12);
    controllerInitLabel_ = new QLabel("控制器初始化", statusGroup);
    controllerInitLabel_->setProperty("role", "statusText");

    // 电机初始化
    jointInitIndicator_ = new QLabel(statusGroup);
    jointInitIndicator_->setObjectName("statusIndicator");
    jointInitIndicator_->setFixedSize(12, 12);
    jointInitLabel_ = new QLabel("电机初始化", statusGroup);
    jointInitLabel_->setProperty("role", "statusText");

    // 系统就绪
    systemReadyIndicator_ = new QLabel(statusGroup);
    systemReadyIndicator_->setObjectName("statusIndicator");
    systemReadyIndicator_->setFixedSize(12, 12);
    systemReadyLabel_ = new QLabel("系统就绪", statusGroup);
    systemReadyLabel_->setProperty("role", "statusText");

    // 当前模式
    currentModeIndicator_ = new QLabel(statusGroup);
    currentModeIndicator_->setObjectName("modeIndicator");
    currentModeIndicator_->setFixedSize(12, 12);
    currentModeLabel_ = new QLabel("当前模式: Idle", statusGroup);
    currentModeLabel_->setProperty("role", "modeText");

    // 时间
    timeLabel_ = new QLabel("--", statusGroup);
    timeLabel_->setObjectName("timeLabel");

    // 初始色块样式
    auto initIndicatorStyle = [](QLabel *label)
    {
        label->setProperty("status", "offline");
    };

    initIndicatorStyle(controllerInitIndicator_);
    initIndicatorStyle(jointInitIndicator_);
    initIndicatorStyle(systemReadyIndicator_);
    initIndicatorStyle(currentModeIndicator_);
    currentModeIndicator_->setProperty("status", "active");

    // 控制器初始化
    statusLayout->addWidget(controllerInitIndicator_);
    statusLayout->addWidget(controllerInitLabel_);
    statusLayout->addSpacing(24);

    // 电机初始化
    statusLayout->addWidget(jointInitIndicator_);
    statusLayout->addWidget(jointInitLabel_);
    statusLayout->addSpacing(24);

    // 系统就绪
    statusLayout->addWidget(systemReadyIndicator_);
    statusLayout->addWidget(systemReadyLabel_);
    statusLayout->addSpacing(24);

    // 当前模式
    statusLayout->addWidget(currentModeIndicator_);
    statusLayout->addWidget(currentModeLabel_);

    // 时间推到最右
    statusLayout->addStretch();
    statusLayout->addWidget(timeLabel_);

    // ================================
    // 左右分区
    // ================================
    QSplitter *mainSplitter = new QSplitter(Qt::Horizontal, central);
    mainSplitter->setChildrenCollapsible(false);

    // ================================
    // 左侧控制区
    // ================================
    QGroupBox *controlGroup = new QGroupBox("控制区", mainSplitter);
    controlGroup->setObjectName("sidebarPanel");
    QVBoxLayout *controlLayout = new QVBoxLayout(controlGroup);
    controlLayout->setContentsMargins(12, 12, 12, 12);
    controlLayout->setSpacing(6);

    leftPresetButton_ = new QPushButton("左臂预设位置", controlGroup);
    rightPresetButton_ = new QPushButton("右臂预设位置", controlGroup);
    zeroForceButton_ = new QPushButton("左臂零力", controlGroup);
    rightZeroForceButton_ = new QPushButton("右臂零力", controlGroup);
    bilateralZeroForceButton_ = new QPushButton("双臂零力", controlGroup);
    eegButton_ = new QPushButton("脑电控制模式", controlGroup);
    const QList<QPushButton *> modeButtons = {
        leftPresetButton_, rightPresetButton_, zeroForceButton_,
        rightZeroForceButton_, bilateralZeroForceButton_, eegButton_};
    modeGroup_ = new QButtonGroup(this);
    modeGroup_->setExclusive(true);
    for (QPushButton *button : modeButtons)
    {
        button->setCheckable(true);
        button->setProperty("role", "controlButton");
        button->setCursor(Qt::PointingHandCursor);
        modeGroup_->addButton(button);
        controlLayout->addWidget(button);
    }
    controlLayout->addStretch();
    connect(leftPresetButton_, &QPushButton::clicked, this, &MainWindow::onLeftPresetClicked);
    connect(rightPresetButton_, &QPushButton::clicked, this, &MainWindow::onRightPresetClicked);
    connect(zeroForceButton_, &QPushButton::clicked, this, &MainWindow::onZeroForceClicked);
    connect(rightZeroForceButton_, &QPushButton::clicked, this, &MainWindow::onRightZeroForceClicked);
    connect(bilateralZeroForceButton_, &QPushButton::clicked, this, &MainWindow::onBilateralZeroForceClicked);
    connect(eegButton_, &QPushButton::clicked, this, &MainWindow::onEegClicked);

    QGroupBox *eegGroup = new QGroupBox("脑电控制", mainSplitter);
    eegGroup->setObjectName("experimentPanel");
    QVBoxLayout *eegLayout = new QVBoxLayout(eegGroup);
    sourceCombo_ = new QComboBox(eegGroup);
    sourceCombo_->addItems({"调试：模拟识别结果", "实机：EEG 串口"});
    eegLayout->addWidget(sourceCombo_);
    QLabel *sourceHint = new QLabel("调试按钮代替 EEG 串口发送 TEST 识别结果。先使双臂到预设位，再选择脑电控制模式。", eegGroup);
    sourceHint->setWordWrap(true);
    eegLayout->addWidget(sourceHint);
    const QString names[4] = {"模拟左肩", "模拟右肩", "模拟左肘", "模拟右肘"};
    for (int i = 0; i < 4; ++i)
    {
        testButtons_[i] = new QPushButton(names[i], eegGroup);
        eegLayout->addWidget(testButtons_[i]);
        connect(testButtons_[i], &QPushButton::clicked, this, [this, i]() { sendTestTarget(i + 1); });
    }
    eegActionLabel_ = new QLabel("动作：等待指令", eegGroup);
    serialStatusLabel_ = new QLabel("串口：--", eegGroup);
    eegFrameLabel_ = new QLabel("最近数据：--", eegGroup);
    eegFrameLabel_->setWordWrap(true);
    eegNoteLabel_ = new QLabel("状态：--", eegGroup);
    eegNoteLabel_->setWordWrap(true);
    eegLayout->addWidget(eegActionLabel_);
    eegLayout->addWidget(serialStatusLabel_);
    eegLayout->addWidget(eegFrameLabel_);
    eegLayout->addWidget(eegNoteLabel_);
    eegLayout->addStretch();
    connect(sourceCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this, sourceHint](int index) {
                sourceHint->setText(index == 0 ?
                    "调试按钮代替 EEG 串口发送 TEST 识别结果。先使双臂到预设位，再选择脑电控制模式。" :
                    "实机由控制节点读取 EEG 串口；下方模拟按钮仅在调试来源下使用。");
                std_msgs::msg::Bool source_msg;
                source_msg.data = index == 0;
                eeg_source_pub_->publish(source_msg);
                appendLog(index == 0 ? "调试来源：按钮发送 TEST 模拟识别结果" :
                                       "实机来源：控制节点读取 EEG 串口");
            });

    // ================================
    // 右侧上下分区
    // ================================
    QSplitter *rightSplitter = new QSplitter(Qt::Vertical, mainSplitter);
    rightSplitter->setChildrenCollapsible(false);

    // ================================
    // 右上：数据区
    // ================================
    QWidget *dataWidget = new QWidget(rightSplitter);
    QVBoxLayout *dataMainLayout = new QVBoxLayout(dataWidget);
    dataMainLayout->setContentsMargins(0, 0, 0, 0);
    dataMainLayout->setSpacing(8);

    // 关节状态表：左右臂分开显示，避免 20 行单表需要下翻
    QGroupBox *jointGroup = new QGroupBox("关节状态", dataWidget);
    jointGroup->setObjectName("dataPanel");
    QHBoxLayout *jointLayout = new QHBoxLayout(jointGroup);
    jointLayout->setContentsMargins(8, 8, 8, 8);
    jointLayout->setSpacing(8);

    auto createJointTable = [](QWidget *parent) -> QTableWidget *
    {
        QTableWidget *table = new QTableWidget(parent);
        table->setColumnCount(5);
        table->setHorizontalHeaderLabels(
            QStringList() << "轴号" << "位置(deg)" << "速度(deg/s)" << "力矩(Nm)" << "使能");
        table->setRowCount(10);

        table->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Fixed);
        table->setColumnWidth(0, 45);

        table->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
        table->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
        table->horizontalHeader()->setSectionResizeMode(3, QHeaderView::Stretch);

        table->horizontalHeader()->setSectionResizeMode(4, QHeaderView::Fixed);
        table->setColumnWidth(4, 60);

        table->horizontalHeader()->setDefaultAlignment(Qt::AlignCenter);
        table->verticalHeader()->setVisible(false);
        table->setEditTriggers(QAbstractItemView::NoEditTriggers);
        table->setSelectionMode(QAbstractItemView::NoSelection);
        table->setAlternatingRowColors(true);
        table->setFocusPolicy(Qt::NoFocus);
        table->verticalHeader()->setDefaultSectionSize(30);

        for (int i = 0; i < 10; ++i)
        {
            QTableWidgetItem *axisItem = new QTableWidgetItem(QString::number(i + 1));
            axisItem->setTextAlignment(Qt::AlignCenter);
            table->setItem(i, 0, axisItem);

            QTableWidgetItem *posItem = new QTableWidgetItem("--");
            posItem->setTextAlignment(Qt::AlignCenter);
            table->setItem(i, 1, posItem);

            QTableWidgetItem *velItem = new QTableWidgetItem("--");
            velItem->setTextAlignment(Qt::AlignCenter);
            table->setItem(i, 2, velItem);

            QTableWidgetItem *torqueItem = new QTableWidgetItem("--");
            torqueItem->setTextAlignment(Qt::AlignCenter);
            table->setItem(i, 3, torqueItem);

            QTableWidgetItem *enableItem = new QTableWidgetItem("--");
            enableItem->setTextAlignment(Qt::AlignCenter);
            table->setItem(i, 4, enableItem);
        }

        return table;
    };

    QGroupBox *leftJointGroup = new QGroupBox("左臂关节", jointGroup);
    QVBoxLayout *leftJointLayout = new QVBoxLayout(leftJointGroup);
    leftJointLayout->setContentsMargins(6, 6, 6, 6);

    QGroupBox *rightJointGroup = new QGroupBox("右臂关节", jointGroup);
    QVBoxLayout *rightJointLayout = new QVBoxLayout(rightJointGroup);
    rightJointLayout->setContentsMargins(6, 6, 6, 6);

    leftJointTable_ = createJointTable(leftJointGroup);
    rightJointTable_ = createJointTable(rightJointGroup);

    for (int i = 0; i < 10; ++i)
    {
        leftJointTable_->item(i, 0)->setText(QString("L%1").arg(i + 1));
        rightJointTable_->item(i, 0)->setText(QString("R%1").arg(i + 1));
    }

    leftJointLayout->addWidget(leftJointTable_);
    rightJointLayout->addWidget(rightJointTable_);

    jointLayout->addWidget(leftJointGroup);
    jointLayout->addWidget(rightJointGroup);

    // 六维力区
    QGroupBox *forceGroup = new QGroupBox("六维力信息", dataWidget);
    forceGroup->setObjectName("dataPanel");
    QGridLayout *forceLayout = new QGridLayout(forceGroup);
    forceLayout->setContentsMargins(12, 8, 12, 8);
    forceLayout->setHorizontalSpacing(20);
    forceLayout->setVerticalSpacing(6);

    QLabel *rightHandTitle = new QLabel("右手传感器", forceGroup);
    QLabel *rightArmTitle = new QLabel("右臂传感器", forceGroup);
    QLabel *leftHandTitle = new QLabel("左手传感器", forceGroup);
    QLabel *leftArmTitle = new QLabel("左臂传感器", forceGroup);

    for (QLabel *title : {rightHandTitle, rightArmTitle, leftHandTitle, leftArmTitle})
    {
        title->setProperty("role", "cardTitle");
    }

    rightHandFxLabel_ = new QLabel("Fx: --", forceGroup);
    rightHandFyLabel_ = new QLabel("Fy: --", forceGroup);
    rightHandFzLabel_ = new QLabel("Fz: --", forceGroup);
    rightHandMxLabel_ = new QLabel("Mx: --", forceGroup);
    rightHandMyLabel_ = new QLabel("My: --", forceGroup);
    rightHandMzLabel_ = new QLabel("Mz: --", forceGroup);

    rightArmFxLabel_ = new QLabel("Fx: --", forceGroup);
    rightArmFyLabel_ = new QLabel("Fy: --", forceGroup);
    rightArmFzLabel_ = new QLabel("Fz: --", forceGroup);
    rightArmMxLabel_ = new QLabel("Mx: --", forceGroup);
    rightArmMyLabel_ = new QLabel("My: --", forceGroup);
    rightArmMzLabel_ = new QLabel("Mz: --", forceGroup);

    leftHandFxLabel_ = new QLabel("Fx: --", forceGroup);
    leftHandFyLabel_ = new QLabel("Fy: --", forceGroup);
    leftHandFzLabel_ = new QLabel("Fz: --", forceGroup);
    leftHandMxLabel_ = new QLabel("Mx: --", forceGroup);
    leftHandMyLabel_ = new QLabel("My: --", forceGroup);
    leftHandMzLabel_ = new QLabel("Mz: --", forceGroup);

    leftArmFxLabel_ = new QLabel("Fx: --", forceGroup);
    leftArmFyLabel_ = new QLabel("Fy: --", forceGroup);
    leftArmFzLabel_ = new QLabel("Fz: --", forceGroup);
    leftArmMxLabel_ = new QLabel("Mx: --", forceGroup);
    leftArmMyLabel_ = new QLabel("My: --", forceGroup);
    leftArmMzLabel_ = new QLabel("Mz: --", forceGroup);

    const QList<QLabel *> forceValueLabels = {
        rightHandFxLabel_, rightHandFyLabel_, rightHandFzLabel_, rightHandMxLabel_, rightHandMyLabel_, rightHandMzLabel_,
        rightArmFxLabel_, rightArmFyLabel_, rightArmFzLabel_, rightArmMxLabel_, rightArmMyLabel_, rightArmMzLabel_,
        leftHandFxLabel_, leftHandFyLabel_, leftHandFzLabel_, leftHandMxLabel_, leftHandMyLabel_, leftHandMzLabel_,
        leftArmFxLabel_, leftArmFyLabel_, leftArmFzLabel_, leftArmMxLabel_, leftArmMyLabel_, leftArmMzLabel_};
    for (QLabel *label : forceValueLabels)
    {
        label->setProperty("role", "metric");
    }

    // 布局：上排右手/右臂，下排左手/左臂
    forceLayout->addWidget(rightHandTitle, 0, 0, 1, 2);
    forceLayout->addWidget(rightHandFxLabel_, 1, 0);
    forceLayout->addWidget(rightHandFyLabel_, 1, 1);
    forceLayout->addWidget(rightHandFzLabel_, 2, 0);
    forceLayout->addWidget(rightHandMxLabel_, 2, 1);
    forceLayout->addWidget(rightHandMyLabel_, 3, 0);
    forceLayout->addWidget(rightHandMzLabel_, 3, 1);

    forceLayout->addWidget(rightArmTitle, 0, 2, 1, 2);
    forceLayout->addWidget(rightArmFxLabel_, 1, 2);
    forceLayout->addWidget(rightArmFyLabel_, 1, 3);
    forceLayout->addWidget(rightArmFzLabel_, 2, 2);
    forceLayout->addWidget(rightArmMxLabel_, 2, 3);
    forceLayout->addWidget(rightArmMyLabel_, 3, 2);
    forceLayout->addWidget(rightArmMzLabel_, 3, 3);

    forceLayout->addWidget(leftHandTitle, 4, 0, 1, 2);
    forceLayout->addWidget(leftHandFxLabel_, 5, 0);
    forceLayout->addWidget(leftHandFyLabel_, 5, 1);
    forceLayout->addWidget(leftHandFzLabel_, 6, 0);
    forceLayout->addWidget(leftHandMxLabel_, 6, 1);
    forceLayout->addWidget(leftHandMyLabel_, 7, 0);
    forceLayout->addWidget(leftHandMzLabel_, 7, 1);

    forceLayout->addWidget(leftArmTitle, 4, 2, 1, 2);
    forceLayout->addWidget(leftArmFxLabel_, 5, 2);
    forceLayout->addWidget(leftArmFyLabel_, 5, 3);
    forceLayout->addWidget(leftArmFzLabel_, 6, 2);
    forceLayout->addWidget(leftArmMxLabel_, 6, 3);
    forceLayout->addWidget(leftArmMyLabel_, 7, 2);
    forceLayout->addWidget(leftArmMzLabel_, 7, 3);

    dataMainLayout->addWidget(jointGroup, 3);
    dataMainLayout->addWidget(forceGroup, 1);

    // ================================
    // 右下：日志区
    // ================================
    QGroupBox *logGroup = new QGroupBox("日志", rightSplitter);
    logGroup->setObjectName("dataPanel");
    QVBoxLayout *logLayout = new QVBoxLayout(logGroup);

    logTextEdit_ = new QTextEdit(logGroup);
    logTextEdit_->setReadOnly(true);
    logLayout->addWidget(logTextEdit_);

    // splitter 比例
    mainSplitter->addWidget(controlGroup);
    mainSplitter->addWidget(eegGroup);
    mainSplitter->addWidget(rightSplitter);
    mainSplitter->setStretchFactor(0, 0);
    mainSplitter->setStretchFactor(1, 0);
    mainSplitter->setStretchFactor(2, 1);
    mainSplitter->setSizes(QList<int>() << 210 << 190 << 1000);

    rightSplitter->addWidget(dataWidget);
    rightSplitter->addWidget(logGroup);
    rightSplitter->setStretchFactor(0, 7);
    rightSplitter->setStretchFactor(1, 3);

    mainLayout->addWidget(statusGroup);
    mainLayout->addWidget(mainSplitter, 1);

    setWindowTitle("UpperLimb GUI");
    resize(1400, 900);

    // ================================
    // UI 定时刷新
    // ================================
    uiTimer_ = new QTimer(this);
    connect(uiTimer_, &QTimer::timeout, this, &MainWindow::refreshUi);
    uiTimer_->start(100); // 10 Hz 刷新界面，先稳一些

    // ================================
    // 启动 ROS spin 线程
    // ================================
    executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
    executor_->add_node(node_);

    ros_spin_thread_ = std::thread([this]()
                                   { executor_->spin(); });

    appendLog("GUI 启动完成");
}

MainWindow::~MainWindow()
{
    if (executor_)
    {
        executor_->cancel();
        executor_->remove_node(node_);
    }

    if (ros_spin_thread_.joinable())
    {
        ros_spin_thread_.join();
    }
}

void MainWindow::publishCommand(int cmd)
{
    upperlimb_robot::msg::ControlCommand msg;
    msg.command = cmd;
    control_command_pub_->publish(msg);

    {
        QMutexLocker locker(&dataMutex_);
        requested_mode_ = cmd;
        mode_pending_ = true;
    }

    appendLog(QString("发送控制命令: %1").arg(modeToString(cmd)));
}

void MainWindow::onLeftPresetClicked()
{
    publishCommand(MODE_LEFT_PRESET_POSITION);
}

void MainWindow::onRightPresetClicked()
{
    publishCommand(MODE_RIGHT_PRESET_POSITION);
}

void MainWindow::onZeroForceClicked() { publishCommand(MODE_ZERO_FORCE); }
void MainWindow::onRightZeroForceClicked() { publishCommand(MODE_RIGHT_ZERO_FORCE); }
void MainWindow::onBilateralZeroForceClicked() { publishCommand(MODE_BILATERAL_ZERO_FORCE); }
void MainWindow::onEegClicked() { publishCommand(MODE_EEG_CONTROL); }

void MainWindow::sendTestTarget(int target)
{
    if (sourceCombo_->currentIndex() != 0) return;
    std_msgs::msg::String msg;
    msg.data = QString("RESULT,%1,TEST,%2,15.00\r\n")
                   .arg(++testSequence_).arg(target).toStdString();
    eeg_frame_pub_->publish(msg);
    {
        QMutexLocker locker(&dataMutex_);
        eeg_target_ = target;
        eeg_frame_ = QString::fromStdString(msg.data).trimmed();
        eeg_note_ = "已发送模拟识别结果，等待控制节点反馈";
        if (!eeg_status_seen_)
            serial_status_ = "调试来源，无需脑电串口；控制节点未连接";
    }
    appendLog(QString("发送模拟帧：%1").arg(QString::fromStdString(msg.data).trimmed()));
}

void MainWindow::systemStateCallback(const upperlimb_robot::msg::SystemState::SharedPtr msg)
{
    QMutexLocker locker(&dataMutex_);
    controller_init_ = msg->controller_init;
    joint_init_ = msg->joint_init;
    system_ready_ = msg->system_ready;
}

void MainWindow::robotStateCallback(const upperlimb_robot::msg::RobotState::SharedPtr msg)
{
    QMutexLocker locker(&dataMutex_);
    position_ = msg->position;
    velocity_ = msg->velocity;
    torque_ = msg->torque;
    enabled_ = msg->enabled;
    force_sensor_ = msg->force_sensor;
}

void MainWindow::eegStatusCallback(const std_msgs::msg::String::SharedPtr msg)
{
    const QStringList fields = QString::fromStdString(msg->data).split('|');
    if (fields.size() < 7) return;
    QMutexLocker locker(&dataMutex_);
    eeg_phase_ = fields[0].toInt();
    eeg_target_ = fields[1].toInt();
    eeg_running_ = fields[2] == "1";
    eeg_status_seen_ = true;
    last_eeg_status_time_ = std::chrono::steady_clock::now();
    serial_status_ = fields[3];
    eeg_note_ = fields[4];
    requested_mode_ = fields[5].toInt();
    mode_pending_ = false;
    eeg_frame_ = fields[6].trimmed();
}

void MainWindow::refreshUi()
{
    QMutexLocker locker(&dataMutex_);
    if (eeg_status_seen_ &&
        std::chrono::steady_clock::now() - last_eeg_status_time_ > std::chrono::seconds(2))
    {
        eeg_status_seen_ = false;
        eeg_running_ = false;
        serial_status_ = "控制节点未连接";
        eeg_note_ = "控制节点反馈中断，模拟消息仍可发送";
    }

    updateIndicator(controllerInitIndicator_, controllerInitLabel_, controller_init_, "控制器初始化");
    updateIndicator(jointInitIndicator_, jointInitLabel_, joint_init_, "电机初始化");
    updateIndicator(systemReadyIndicator_, systemReadyLabel_, system_ready_, "系统就绪");

    // 当前模式文字
    currentModeLabel_->setText(QString("当前模式: %1%2")
        .arg(modeToString(requested_mode_),
             mode_pending_ ? "（等待控制节点确认）" : ""));
    switch (requested_mode_)
    {
    case MODE_LEFT_PRESET_POSITION: leftPresetButton_->setChecked(true); break;
    case MODE_RIGHT_PRESET_POSITION: rightPresetButton_->setChecked(true); break;
    case MODE_ZERO_FORCE: zeroForceButton_->setChecked(true); break;
    case MODE_RIGHT_ZERO_FORCE: rightZeroForceButton_->setChecked(true); break;
    case MODE_BILATERAL_ZERO_FORCE: bilateralZeroForceButton_->setChecked(true); break;
    case MODE_EEG_CONTROL: eegButton_->setChecked(true); break;
    default:
        if (modeGroup_->checkedButton())
        {
            modeGroup_->setExclusive(false);
            modeGroup_->checkedButton()->setChecked(false);
            modeGroup_->setExclusive(true);
        }
        break;
    }

    // 右侧时间
    timeLabel_->setText(QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss"));

    QString now = QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss");
    timeLabel_->setText(now);

    // 更新关节表
    // 默认映射：
    // 0~9   左臂
    // 10~19 右臂
    // 如果你的 RobotState 顺序相反，只需要交换下面两个 offset。
    auto updateJointTable = [this](QTableWidget *table, int offset)
    {
        for (int row = 0; row < 10; ++row)
        {
            const int idx = offset + row;

            QString posText = "--";
            QString velText = "--";
            QString torText = "--";
            QString enText = "--";

            if (idx >= 0 && idx < static_cast<int>(position_.size()))
            {
                posText = QString::number(position_[idx], 'f', 2);
            }

            if (idx >= 0 && idx < static_cast<int>(velocity_.size()))
            {
                velText = QString::number(velocity_[idx], 'f', 2);
            }

            if (idx >= 0 && idx < static_cast<int>(torque_.size()))
            {
                torText = QString::number(torque_[idx], 'f', 3);
            }

            if (idx >= 0 && idx < static_cast<int>(enabled_.size()))
            {
                enText = enabled_[idx] ? "ON" : "OFF";
            }

            table->item(row, 1)->setText(posText);
            table->item(row, 2)->setText(velText);
            table->item(row, 3)->setText(torText);
            table->item(row, 4)->setText(enText);
        }
    };

    updateJointTable(leftJointTable_, 10);
    updateJointTable(rightJointTable_, 0);

    // force_sensor 顺序：
    // 0~5   右手
    // 6~11  右臂
    // 12~17 左手
    // 18~23 左臂
    auto getForce = [this](int idx) -> QString
    {
        if (idx >= 0 && idx < static_cast<int>(force_sensor_.size()))
        {
            return QString::number(force_sensor_[idx], 'f', 3);
        }
        return "--";
    };

    // 右手：0~5
    rightHandFxLabel_->setText("Fx: " + getForce(0));
    rightHandFyLabel_->setText("Fy: " + getForce(1));
    rightHandFzLabel_->setText("Fz: " + getForce(2));
    rightHandMxLabel_->setText("Mx: " + getForce(3));
    rightHandMyLabel_->setText("My: " + getForce(4));
    rightHandMzLabel_->setText("Mz: " + getForce(5));

    // 右臂：6~11
    rightArmFxLabel_->setText("Fx: " + getForce(6));
    rightArmFyLabel_->setText("Fy: " + getForce(7));
    rightArmFzLabel_->setText("Fz: " + getForce(8));
    rightArmMxLabel_->setText("Mx: " + getForce(9));
    rightArmMyLabel_->setText("My: " + getForce(10));
    rightArmMzLabel_->setText("Mz: " + getForce(11));

    // 左手：12~17
    leftHandFxLabel_->setText("Fx: " + getForce(12));
    leftHandFyLabel_->setText("Fy: " + getForce(13));
    leftHandFzLabel_->setText("Fz: " + getForce(14));
    leftHandMxLabel_->setText("Mx: " + getForce(15));
    leftHandMyLabel_->setText("My: " + getForce(16));
    leftHandMzLabel_->setText("Mz: " + getForce(17));

    // 左臂：18~23
    leftArmFxLabel_->setText("Fx: " + getForce(18));
    leftArmFyLabel_->setText("Fy: " + getForce(19));
    leftArmFzLabel_->setText("Fz: " + getForce(20));
    leftArmMxLabel_->setText("Mx: " + getForce(21));
    leftArmMyLabel_->setText("My: " + getForce(22));
    leftArmMzLabel_->setText("Mz: " + getForce(23));

    const QString actionNames[5] = {"无", "左肩", "右肩", "左肘", "右肘"};
    const QString name = eeg_target_ >= 0 && eeg_target_ <= 4 ? actionNames[eeg_target_] : "未知";
    const QString phases[4] = {"等待动作", "去程", "返程", "故障"};
    const QString phase = eeg_phase_ >= 0 && eeg_phase_ <= 3 ? phases[eeg_phase_] : "未知";
    eegActionLabel_->setText(QString("动作：%1 / %2").arg(name, phase));
    serialStatusLabel_->setText("串口：" + serial_status_);
    eegFrameLabel_->setText("最近数据：" + eeg_frame_);
    eegNoteLabel_->setText("状态：" + eeg_note_);
    for (QPushButton *button : testButtons_)
        button->setEnabled(sourceCombo_->currentIndex() == 0 && !eeg_running_);
    sourceCombo_->setEnabled(!eeg_running_);
    if (shown_note_ != eeg_note_ && !eeg_note_.isEmpty())
    { shown_note_ = eeg_note_; appendLog(eeg_note_); }

}

QString MainWindow::modeToString(int mode) const
{
    switch (mode)
    {
    case MODE_IDLE: return "未选择模式";
    case MODE_LEFT_PRESET_POSITION: return "左臂预设位置";
    case MODE_RIGHT_PRESET_POSITION: return "右臂预设位置";
    case MODE_ZERO_FORCE: return "左臂零力";
    case MODE_RIGHT_ZERO_FORCE: return "右臂零力";
    case MODE_BILATERAL_ZERO_FORCE: return "双臂零力";
    case MODE_EEG_CONTROL: return "脑电控制";
    default: return "未知";
    }
}

QString MainWindow::boolToText(bool value) const
{
    return value ? "是" : "否";
}

void MainWindow::updateStatusColor(QLabel *label, bool ok, const QString &prefix)
{
    label->setText(QString("%1: %2").arg(prefix, ok ? "正常" : "未就绪"));
    if (ok)
    {
        label->setStyleSheet("QLabel { color: green; font-weight: bold; }");
    }
    else
    {
        label->setStyleSheet("QLabel { color: red; font-weight: bold; }");
    }
}

void MainWindow::appendLog(const QString &text)
{
    const QString now = QDateTime::currentDateTime().toString("HH:mm:ss");
    logTextEdit_->append(QString("[%1] %2").arg(now, text));
}

void MainWindow::updateIndicator(QLabel *indicator, QLabel *textLabel, bool ok, const QString &text)
{
    textLabel->setText(text);

    const char *status = ok ? "online" : "offline";
    if (indicator->property("status").toString() == status)
    {
        return;
    }

    indicator->setProperty("status", status);
    indicator->style()->unpolish(indicator);
    indicator->style()->polish(indicator);
}
