#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QTextEdit>
#include <QTableWidget>
#include <QTimer>
#include <QMutex>
#include <QComboBox>
#include <QButtonGroup>
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/bool.hpp"

#include <chrono>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>

#include "upperlimb_robot/msg/control_command.hpp"
#include "upperlimb_robot/msg/system_state.hpp"
#include "upperlimb_robot/msg/robot_state.hpp"

// ================================
// 控制模式枚举
// 数值必须与控制端一致
// ================================
enum ControlMode : int
{
    MODE_IDLE = 0,
    MODE_LEFT_PRESET_POSITION = 1,
    MODE_RIGHT_PRESET_POSITION = 2,
    MODE_ZERO_FORCE = 4,
    MODE_RIGHT_ZERO_FORCE = 8,
    MODE_BILATERAL_ZERO_FORCE = 9,
    MODE_EEG_CONTROL = 30
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onLeftPresetClicked();
    void onRightPresetClicked();
    void onZeroForceClicked();
    void onRightZeroForceClicked();
    void onBilateralZeroForceClicked();
    void onEegClicked();
    void sendTestTarget(int target);

    void refreshUi();

private:
    // ================================
    // ROS 回调
    // ================================
    void systemStateCallback(const upperlimb_robot::msg::SystemState::SharedPtr msg);
    void robotStateCallback(const upperlimb_robot::msg::RobotState::SharedPtr msg);
    void eegStatusCallback(const std_msgs::msg::String::SharedPtr msg);

    void publishCommand(int cmd);
    void appendLog(const QString &text);
    QString modeToString(int mode) const;
    QString boolToText(bool value) const;
    void updateStatusColor(QLabel *label, bool ok, const QString &prefix);
    void updateIndicator(QLabel *indicator, QLabel *textLabel, bool ok, const QString &text);

private:
    // ================================
    // ROS2
    // ================================
    rclcpp::Node::SharedPtr node_;
    rclcpp::Publisher<upperlimb_robot::msg::ControlCommand>::SharedPtr control_command_pub_;
    rclcpp::Subscription<upperlimb_robot::msg::SystemState>::SharedPtr system_state_sub_;
    rclcpp::Subscription<upperlimb_robot::msg::RobotState>::SharedPtr robot_state_sub_;
    rclcpp::executors::SingleThreadedExecutor::SharedPtr executor_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr eeg_frame_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr eeg_source_pub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr eeg_status_sub_;

    std::thread ros_spin_thread_;

    // ================================
    // UI
    // ================================
    QLabel *controllerInitIndicator_ = nullptr;
    QLabel *controllerInitLabel_ = nullptr;

    QLabel *jointInitIndicator_ = nullptr;
    QLabel *jointInitLabel_ = nullptr;

    QLabel *systemReadyIndicator_ = nullptr;
    QLabel *systemReadyLabel_ = nullptr;

    QLabel *currentModeIndicator_ = nullptr;
    QLabel *currentModeLabel_ = nullptr;

    QLabel *timeLabel_ = nullptr;

    QButtonGroup *modeGroup_ = nullptr;
    QPushButton *leftPresetButton_ = nullptr;
    QPushButton *rightPresetButton_ = nullptr;
    QPushButton *zeroForceButton_ = nullptr;
    QPushButton *rightZeroForceButton_ = nullptr;
    QPushButton *bilateralZeroForceButton_ = nullptr;
    QPushButton *eegButton_ = nullptr;
    QPushButton *testButtons_[4]{};
    QComboBox *sourceCombo_ = nullptr;
    QLabel *eegActionLabel_ = nullptr;
    QLabel *serialStatusLabel_ = nullptr;
    QLabel *eegFrameLabel_ = nullptr;
    QLabel *eegNoteLabel_ = nullptr;
    std::uint64_t testSequence_ = 0;

    // 左右臂关节状态表分开显示
    QTableWidget *leftJointTable_ = nullptr;
    QTableWidget *rightJointTable_ = nullptr;

    // 右手六维力
    QLabel *rightHandFxLabel_ = nullptr;
    QLabel *rightHandFyLabel_ = nullptr;
    QLabel *rightHandFzLabel_ = nullptr;
    QLabel *rightHandMxLabel_ = nullptr;
    QLabel *rightHandMyLabel_ = nullptr;
    QLabel *rightHandMzLabel_ = nullptr;

    // 右臂六维力
    QLabel *rightArmFxLabel_ = nullptr;
    QLabel *rightArmFyLabel_ = nullptr;
    QLabel *rightArmFzLabel_ = nullptr;
    QLabel *rightArmMxLabel_ = nullptr;
    QLabel *rightArmMyLabel_ = nullptr;
    QLabel *rightArmMzLabel_ = nullptr;

    // 左手六维力
    QLabel *leftHandFxLabel_ = nullptr;
    QLabel *leftHandFyLabel_ = nullptr;
    QLabel *leftHandFzLabel_ = nullptr;
    QLabel *leftHandMxLabel_ = nullptr;
    QLabel *leftHandMyLabel_ = nullptr;
    QLabel *leftHandMzLabel_ = nullptr;

    // 左臂六维力
    QLabel *leftArmFxLabel_ = nullptr;
    QLabel *leftArmFyLabel_ = nullptr;
    QLabel *leftArmFzLabel_ = nullptr;
    QLabel *leftArmMxLabel_ = nullptr;
    QLabel *leftArmMyLabel_ = nullptr;
    QLabel *leftArmMzLabel_ = nullptr;

    QTextEdit *logTextEdit_ = nullptr;

    QTimer *uiTimer_ = nullptr;

    // ================================
    // 共享缓存（ROS线程写，Qt线程读）
    // ================================
    QMutex dataMutex_;

    bool controller_init_ = false;
    bool joint_init_ = false;
    bool system_ready_ = false;

    std::vector<double> position_;
    std::vector<double> velocity_;
    std::vector<double> torque_;
    std::vector<bool> enabled_;
    std::vector<double> force_sensor_;

    int requested_mode_ = MODE_IDLE;
    bool mode_pending_ = false;
    bool eeg_status_seen_ = false;
    std::chrono::steady_clock::time_point last_eeg_status_time_{};
    int eeg_phase_ = 0;
    int eeg_target_ = 0;
    bool eeg_running_ = false;
    QString serial_status_ = "调试来源，无需脑电串口";
    QString eeg_note_ = "请选择脑电模式，再点击模拟动作";
    QString eeg_frame_;
    QString shown_note_;
};

#endif // MAINWINDOW_H
