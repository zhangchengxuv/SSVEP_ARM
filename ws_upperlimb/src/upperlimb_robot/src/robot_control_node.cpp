// Copyright 2024 Zhang Chenxu
// Licensed under the Apache License, Version 2.0

#include "rclcpp/rclcpp.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"

#include "upperlimb_robot/msg/control_command.hpp"
#include "upperlimb_robot/msg/system_state.hpp"
#include "upperlimb_robot/msg/robot_state.hpp"

#include "upperlimb_robot/gxn.h"
#include "upperlimb_robot/control_algorithms.hpp"

#include <array>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "serial/serial.h"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/bool.hpp"

#include "sensor_msgs/msg/joint_state.hpp"

using namespace std::chrono_literals;


// ===================== GTN 底层库调用串行化 =====================
//
// Linux 版 GTN/EtherCAT 库在状态采集线程和控制线程中被同时调用时，
// 容易出现轴模式读数异常、PT 下发失败、同步失败等问题。
// 下面这些宏会把所有 GTN_* 调用统一转到 RobotControlNode::callGtn()，
// 由同一把 gtn_mutex_ 保护，保证任意时刻只有一个线程访问底层库。
//
#define GTN_Open(...) callGtn([&]() { return ::GTN_Open(__VA_ARGS__); })
#define GTN_Close(...) callGtn([&]() { return ::GTN_Close(__VA_ARGS__); })
#define GTN_LoadConfig(...) callGtn([&]() { return ::GTN_LoadConfig(__VA_ARGS__); })
#define GTN_ClrSts(...) callGtn([&]() { return ::GTN_ClrSts(__VA_ARGS__); })
#define GTN_InitEcatComm(...) callGtn([&]() { return ::GTN_InitEcatComm(__VA_ARGS__); })
#define GTN_IsEcatReady(...) callGtn([&]() { return ::GTN_IsEcatReady(__VA_ARGS__); })
#define GTN_StartEcatComm(...) callGtn([&]() { return ::GTN_StartEcatComm(__VA_ARGS__); })
#define GTN_TerminateEcatComm(...) callGtn([&]() { return ::GTN_TerminateEcatComm(__VA_ARGS__); })
#define GTN_AxisOn(...) callGtn([&]() { return ::GTN_AxisOn(__VA_ARGS__); })
#define GTN_AxisOff(...) callGtn([&]() { return ::GTN_AxisOff(__VA_ARGS__); })
#define GTN_GetSts(...) callGtn([&]() { return ::GTN_GetSts(__VA_ARGS__); })
#define GTN_GetEcatEncPos(...) callGtn([&]() { return ::GTN_GetEcatEncPos(__VA_ARGS__); })
#define GTN_SetPrfPos(...) callGtn([&]() { return ::GTN_SetPrfPos(__VA_ARGS__); })
#define GTN_SetEncPos(...) callGtn([&]() { return ::GTN_SetEncPos(__VA_ARGS__); })
#define GTN_SynchAxisPos(...) callGtn([&]() { return ::GTN_SynchAxisPos(__VA_ARGS__); })
#define GTN_GetEcatRawData(...) callGtn([&]() { return ::GTN_GetEcatRawData(__VA_ARGS__); })
#define GTN_GetEcatAxisAtlTorque(...) callGtn([&]() { return ::GTN_GetEcatAxisAtlTorque(__VA_ARGS__); })
#define GTN_GetEncVel(...) callGtn([&]() { return ::GTN_GetEncVel(__VA_ARGS__); })
#define GTN_Stop(...) callGtn([&]() { return ::GTN_Stop(__VA_ARGS__); })
#define GTN_SetEcatAxisPT(...) callGtn([&]() { return ::GTN_SetEcatAxisPT(__VA_ARGS__); })
#define GTN_SetEcatAxisMode(...) callGtn([&]() { return ::GTN_SetEcatAxisMode(__VA_ARGS__); })
#define GTN_GetEcatAxisMode(...) callGtn([&]() { return ::GTN_GetEcatAxisMode(__VA_ARGS__); })
#define GTN_PrfTrap(...) callGtn([&]() { return ::GTN_PrfTrap(__VA_ARGS__); })
#define GTN_SetTrapPrm(...) callGtn([&]() { return ::GTN_SetTrapPrm(__VA_ARGS__); })
#define GTN_SetVel(...) callGtn([&]() { return ::GTN_SetVel(__VA_ARGS__); })
#define GTN_SetPos(...) callGtn([&]() { return ::GTN_SetPos(__VA_ARGS__); })
#define GTN_Update(...) callGtn([&]() { return ::GTN_Update(__VA_ARGS__); })

class RobotControlNode : public rclcpp::Node
{
public:
  RobotControlNode() : Node("robot_control_node")
  {
    // 保持原控制节点的默认行为：启动时连接机器人。
    // 仅在明确传入 hardware_enabled:=false 时运行无硬件调试。
    hardware_enabled_ = this->declare_parameter<bool>("hardware_enabled", true);
    serial_device_ = this->declare_parameter<std::string>("serial_device", "/dev/ttyUSB0");
    eeg_velocity_deg_s_ = this->declare_parameter<double>("eeg.velocity_deg_s", 5.0);
    eeg_acceleration_deg_s2_ = this->declare_parameter<double>("eeg.acceleration_deg_s2", 20.0);
    for (int i = 0; i < 4; ++i)
    {
      const std::string name = std::string("eeg.delta_") + std::to_string(i + 1);
      eeg_delta_deg_[i] = this->declare_parameter<double>(name, 3.0);
      // 仅限制本次软件示例运动范围，不代表真实机械安全限位。
      const double preset_deg = i >= 2 ? 45.0 : 0.0;
      eeg_min_deg_[i] = this->declare_parameter<double>("eeg.min_" + std::to_string(i + 1), preset_deg - 3.0);
      eeg_max_deg_[i] = this->declare_parameter<double>("eeg.max_" + std::to_string(i + 1), preset_deg + 3.0);
    }
    RCLCPP_INFO(this->get_logger(), "机器人节点启动，hardware_enabled=%d", hardware_enabled_);

    control_command_sub_ =
        this->create_subscription<upperlimb_robot::msg::ControlCommand>(
            "control_command",
            10,
            std::bind(&RobotControlNode::controlCommandCallback, this, std::placeholders::_1));

    eeg_status_pub_ = this->create_publisher<std_msgs::msg::String>("eeg_status", 10);
    eeg_frame_sub_ = this->create_subscription<std_msgs::msg::String>(
        "eeg_test_frame", 10,
        [this](std_msgs::msg::String::SharedPtr msg) { handleEegFrame(msg->data, false); });
    eeg_source_sub_ = this->create_subscription<std_msgs::msg::Bool>(
        "eeg_debug_source", rclcpp::QoS(1).transient_local(),
        [this](std_msgs::msg::Bool::SharedPtr msg) {
          std::lock_guard<std::mutex> lock(eeg_mutex_);
          if (action_running_) return;
          eeg_debug_source_.store(msg->data);
          eeg_note_ = msg->data ? "调试来源：等待 GUI TEST 指令" : "实机来源：等待串口 EEG 指令";
        });
    serial_timer_ = this->create_wall_timer(20ms, [this]() { readSerial(); });
    debug_timer_ = this->create_wall_timer(50ms, [this]() { updateDebugAction(); });
    eeg_status_timer_ = this->create_wall_timer(100ms, [this]() { publishEegStatus(); });

    system_pub_ =
        this->create_publisher<upperlimb_robot::msg::SystemState>("system_state", 10);

    robot_state_pub_ =
        this->create_publisher<upperlimb_robot::msg::RobotState>("robot_state", 10);

    joint_state_pub_ =
        this->create_publisher<sensor_msgs::msg::JointState>("/joint_states", 10);

    state_timer_ = this->create_wall_timer(
        500ms, std::bind(&RobotControlNode::publishSystemState, this));

    if (!hardware_enabled_)
    {
      RCLCPP_WARN(this->get_logger(), "无硬件调试启动：不初始化 EtherCAT，不打开真实串口，不下发电机命令");
      publishSystemState();
      return;
    }

    if (!initController())
    {
      RCLCPP_ERROR(this->get_logger(), "控制器初始化失败");
      publishSystemState();
      return;
    }
    controller_init_ = true;
    publishSystemState();

    if (!initMotors())
    {
      RCLCPP_ERROR(this->get_logger(), "电机初始化失败");
      publishSystemState();
      return;
    }
    joint_init_ = true;

    system_ready_ = controller_init_ && joint_init_;
    publishSystemState();

    RCLCPP_INFO(this->get_logger(), "硬件初始化完成，系统准备就绪");

    running_.store(true);

    state_read_thread_ = std::thread(&RobotControlNode::stateReadLoop, this);
    RCLCPP_INFO(this->get_logger(), "状态采集线程已启动");

    control_thread_ = std::thread(&RobotControlNode::controlLoop, this);
    RCLCPP_INFO(this->get_logger(), "控制线程已启动");
  }

  ~RobotControlNode() override
  {
    shutdownHardware();
  }

  void shutdownHardware()
  {
    if (shutdown_done_)
    {
      return;
    }
    shutdown_done_ = true;

    running_.store(false);
    closeSerial();
    if (!hardware_enabled_) return;

    if (state_read_thread_.joinable())
    {
      state_read_thread_.join();
    }

    if (control_thread_.joinable())
    {
      control_thread_.join();
    }

    fprintf(stderr, "正在安全关闭机器人系统...\n");
    fflush(stderr);

    clearAllTorqueCommands();
    restoreAllAxesToPositionMode();
    const bool axes_ok = stopAllAxesRaw();

    if (axes_ok)
    {
      fprintf(stderr, "所有电机已安全关闭。\n");
    }
    else
    {
      fprintf(stderr, "部分电机关闭失败，请检查上面的错误信息。\n");
    }
    fflush(stderr);

    if (ecat_started_)
    {
      const short rtn = GTN_TerminateEcatComm(core_id_);
      if (rtn == 0)
      {
        fprintf(stderr, "EtherCAT通信已关闭。\n");
      }
      else
      {
        fprintf(stderr, "EtherCAT通信关闭失败，错误码=%d\n", rtn);
      }
      fflush(stderr);
      ecat_started_ = false;
    }

    if (controller_opened_)
    {
      const short rtn = GTN_Close();
      if (rtn == 0)
      {
        fprintf(stderr, "控制器已关闭。\n");
      }
      else
      {
        fprintf(stderr, "控制器关闭失败，错误码=%d\n", rtn);
      }
      fflush(stderr);
      controller_opened_ = false;
    }
  }

private:
  enum EegPhase { EEG_WAITING = 0, EEG_OUT = 1, EEG_BACK = 2, EEG_FAULT = 3 };
  bool hardware_enabled_ = true;
  std::atomic<bool> eeg_debug_source_{false};
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr eeg_source_sub_;
  std::string serial_device_;
  serial::Serial serial_;
  std::atomic<bool> serial_connected_{false};
  std::string serial_buffer_;
  std::chrono::steady_clock::time_point last_serial_retry_{};
  rclcpp::TimerBase::SharedPtr serial_timer_;
  rclcpp::TimerBase::SharedPtr debug_timer_;
  rclcpp::TimerBase::SharedPtr eeg_status_timer_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr eeg_status_pub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr eeg_frame_sub_;
  std::mutex eeg_mutex_;
  std::string eeg_last_frame_;
  std::string eeg_note_ = "等待脑电指令";
  std::uint64_t last_eeg_sequence_ = 0;
  std::uint64_t last_test_sequence_ = 0;
  bool eeg_sequence_seen_ = false;
  bool test_sequence_seen_ = false;
  int eeg_last_target_ = 0;
  int eeg_target_ = 0;
  int pending_target_ = 0;
  bool action_running_ = false;
  EegPhase eeg_phase_ = EEG_WAITING;
  int stable_count_ = 0;
  std::chrono::steady_clock::time_point last_action_time_{};
  std::chrono::steady_clock::time_point phase_start_{};
  std::atomic<bool> eeg_mode_ready_{false};
  std::atomic<bool> eeg_preset_ok_{false};
  std::atomic<bool> eeg_fault_latched_{false};
  std::array<double, 4> eeg_delta_deg_{};
  std::array<double, 4> eeg_min_deg_{};
  std::array<double, 4> eeg_max_deg_{};
  double eeg_velocity_deg_s_ = 5.0;
  double eeg_acceleration_deg_s2_ = 20.0;
  std::array<bool, upperlimb_robot::AXIS_NUM> feedback_valid_{};
  std::chrono::steady_clock::time_point last_feedback_time_{};
  // ===================== GTN 调用互斥封装 =====================
  //
  // 所有 GTN_* 底层库函数都通过宏转到这里。
  // 目的不是提高性能，而是先验证 Linux 版 GTN 库在多线程访问下是否会出现
  // 非法模式值、PT 失败、同步失败等异常。
  //
  template <typename Fn>
  auto callGtn(Fn &&fn) -> decltype(fn())
  {
    std::lock_guard<std::mutex> lock(gtn_mutex_);
    return fn();
  }



  // ===================== 模式进入失败回退 =====================
  //
  // onModeEnter() 内部不能直接修改 applied_mode_，但可以请求状态机回退。
  // 状态机随后会把 applied_mode_ 和 current_mode_ 统一改到 fallback_mode，
  // 防止进入失败后仍短暂执行失败模式的控制输出。
  //
  void requestModeEnterFallback(int fallback_mode)
  {
    mode_enter_failed_ = true;
    mode_enter_fallback_mode_ = fallback_mode;
  }

  // ===================== 主动控制轴过滤 =====================
  //
  // 机械结构中每条臂最后 3 个关节暂不参与控制：
  // 右臂 joint18~joint20 -> 控制层索引 7~9 / 物理轴 8~10
  // 左臂 joint8~joint10  -> 控制层索引 17~19 / 物理轴 18~20
  //
  // 当前执行层只控制 14 个主动关节：
  // 右臂 joint11~joint17 -> 控制层索引 0~6
  // 左臂 joint1~joint7   -> 控制层索引 10~16
  //
  bool isControlledAxisIndex(std::size_t i) const
  {
    return (i <= 6) || (i >= 10 && i <= 16);
  }


  void closeSerial()
  {
    try { if (serial_.isOpen()) serial_.close(); }
    catch (const std::exception &e) { RCLCPP_WARN(this->get_logger(), "关闭串口失败: %s", e.what()); }
    serial_connected_.store(false);
    serial_buffer_.clear();
  }

  bool openSerial()
  {
    try
    {
      closeSerial();
      serial_.setPort(serial_device_);
      serial_.setBaudrate(115200);
      serial_.setBytesize(serial::eightbits);
      serial_.setParity(serial::parity_none);
      serial_.setStopbits(serial::stopbits_one);
      serial_.setFlowcontrol(serial::flowcontrol_none);
      serial::Timeout timeout = serial::Timeout::simpleTimeout(0);
      serial_.setTimeout(timeout);
      serial_.open();
      serial_connected_.store(serial_.isOpen());
      return serial_connected_.load();
    }
    catch (const std::exception &e)
    {
      RCLCPP_WARN(this->get_logger(), "打开串口%s失败: %s", serial_device_.c_str(), e.what());
      closeSerial();
      return false;
    }
  }

  // ROS 定时回调读已到达字节；不阻塞机器人控制线程。
  void readSerial()
  {
    if (!hardware_enabled_) return;
    if (eeg_debug_source_.load())
    {
      if (serial_connected_.load()) closeSerial();
      return;
    }
    if (current_mode_.load() != upperlimb_robot::MODE_EEG_CONTROL)
    {
      if (serial_connected_.load()) closeSerial();
      return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (!serial_connected_.load())
    {
      if (last_serial_retry_.time_since_epoch().count() != 0 && now - last_serial_retry_ < 1s) return;
      last_serial_retry_ = now;
      if (!openSerial()) { std::lock_guard<std::mutex> lock(eeg_mutex_); eeg_note_ = "串口未连接"; return; }
    }
    try
    {
      for (int attempt = 0; attempt < 4; ++attempt)
      {
        const std::size_t ready = serial_.available();
        if (ready == 0) return;
        const std::string data = serial_.read(std::min<std::size_t>(ready, 256));
        serial_buffer_ += data;
        if (serial_buffer_.size() > 512)
        { serial_buffer_.clear(); std::lock_guard<std::mutex> lock(eeg_mutex_); eeg_note_ = "串口缓冲区溢出"; return; }
        std::size_t end = 0;
        while ((end = serial_buffer_.find("\r\n")) != std::string::npos)
        {
          const std::string frame = serial_buffer_.substr(0, end + 2);
          serial_buffer_.erase(0, end + 2);
          handleEegFrame(frame, true);
        }
      }
    }
    catch (const std::exception &e)
    {
      RCLCPP_WARN(this->get_logger(), "串口读取失败: %s", e.what());
      closeSerial();
    }
  }

  void handleEegFrame(const std::string &frame, bool from_serial)
  {
    upperlimb_robot::EegResult result;
    const bool valid = upperlimb_robot::parseEegResult(frame, result);
    std::lock_guard<std::mutex> lock(eeg_mutex_);
    eeg_last_frame_ = frame;
    if (!valid) { eeg_note_ = "串口数据格式错误"; return; }
    std::uint64_t &last = result.test ? last_test_sequence_ : last_eeg_sequence_;
    bool &seen = result.test ? test_sequence_seen_ : eeg_sequence_seen_;
    if (seen && result.sequence <= last) { eeg_note_ = "重复或过期序号，已忽略"; return; }
    last = result.sequence;
    seen = true;
    eeg_last_target_ = result.target;
    if (result.target == 0) { eeg_note_ = "无有效目标"; return; }
    if (!eeg_mode_ready_.load()) { eeg_note_ = "未进入脑电模式"; return; }
    if (eeg_fault_latched_.load() || eeg_phase_ == EEG_FAULT)
    { eeg_note_ = "故障锁定，请检查并重启节点"; return; }
    if (action_running_) { eeg_note_ = "当前动作未完成，忽略新指令"; return; }
    if (hardware_enabled_ && (!system_ready_ || !eeg_preset_ok_.load()))
    { eeg_note_ = "机器人未到达双臂预设位置"; return; }
    if (result.test && (from_serial || !eeg_debug_source_.load()))
    { eeg_note_ = "当前未选择 GUI 调试来源"; return; }
    if (!result.test && eeg_debug_source_.load())
    { eeg_note_ = "GUI 调试来源已忽略串口 EEG 指令"; return; }
    if (!result.test && !from_serial) { eeg_note_ = "EEG 只能由串口触发"; return; }
    const auto now = std::chrono::steady_clock::now();
    if (last_action_time_.time_since_epoch().count() != 0 && now - last_action_time_ < 5s)
    { eeg_note_ = "距上次动作不足5秒，已忽略"; return; }
    pending_target_ = result.target;
    eeg_target_ = result.target;
    action_running_ = true;
    last_action_time_ = now;
    phase_start_ = now;
    eeg_phase_ = EEG_OUT;
    eeg_note_ = hardware_enabled_ ? "指令已接受" : "调试指令已接受（无电机输出）";
    RCLCPP_INFO(this->get_logger(), "EEG 接受目标%d 序号%llu", result.target,
                static_cast<unsigned long long>(result.sequence));
  }

  void updateDebugAction()
  {
    if (hardware_enabled_) return;
    std::lock_guard<std::mutex> lock(eeg_mutex_);
    if (!action_running_ || eeg_phase_ == EEG_FAULT) return;
    const auto elapsed = std::chrono::steady_clock::now() - phase_start_;
    if (eeg_phase_ == EEG_OUT && elapsed >= 500ms)
    { eeg_phase_ = EEG_BACK; phase_start_ = std::chrono::steady_clock::now(); eeg_note_ = "无硬件：模拟返程阶段"; }
    else if (eeg_phase_ == EEG_BACK && elapsed >= 500ms)
    { eeg_phase_ = EEG_WAITING; action_running_ = false; eeg_note_ = "无硬件：仅验证指令流程，未验证运动"; }
  }

  void publishEegStatus()
  {
    std_msgs::msg::String msg;
    std::lock_guard<std::mutex> lock(eeg_mutex_);
    msg.data = std::to_string(static_cast<int>(eeg_phase_)) + "|" +
               std::to_string(eeg_target_) + "|" +
               (action_running_ ? "1" : "0") + "|" +
               (eeg_debug_source_.load() ? "GUI 调试来源，无需脑电串口" : hardware_enabled_ ?
                    (std::string(serial_connected_.load() ? "串口已连接 " : "串口未连接 ") +
                     serial_device_ + " 115200 8N1") : "无硬件调试") + "|" +
               eeg_note_ + "|" + std::to_string(status_mode_.load()) + "|" + eeg_last_frame_;
    eeg_status_pub_->publish(msg);
  }

  // ===================== 控制命令回调 =====================
  //
  // 这里仅记录“请求模式”。
  // 真正的模式进入/退出动作统一放在控制线程里的状态机中处理，
  // 这样可以避免在 ROS 回调线程里直接改硬件状态。
  //
  void controlCommandCallback(const upperlimb_robot::msg::ControlCommand::SharedPtr msg)
  {
    if (msg->command != upperlimb_robot::MODE_IDLE &&
        msg->command != upperlimb_robot::MODE_LEFT_PRESET_POSITION &&
        msg->command != upperlimb_robot::MODE_RIGHT_PRESET_POSITION &&
        msg->command != upperlimb_robot::MODE_ZERO_FORCE &&
        msg->command != upperlimb_robot::MODE_RIGHT_ZERO_FORCE &&
        msg->command != upperlimb_robot::MODE_BILATERAL_ZERO_FORCE &&
        msg->command != upperlimb_robot::MODE_EEG_CONTROL) return;
    if (msg->command == upperlimb_robot::MODE_EEG_CONTROL && eeg_fault_latched_.load()) return;
    {
      std::lock_guard<std::mutex> lock(eeg_mutex_);
      if (action_running_ && msg->command != upperlimb_robot::MODE_IDLE &&
          msg->command != upperlimb_robot::MODE_EEG_CONTROL)
      { eeg_note_ = "动作进行中，拒绝切换模式"; return; }
    }
    if (msg->command != upperlimb_robot::MODE_EEG_CONTROL) eeg_mode_ready_.store(false);
    if (!hardware_enabled_)
    {
      applied_mode_ = msg->command;
      status_mode_.store(msg->command);
      eeg_mode_ready_.store(msg->command == upperlimb_robot::MODE_EEG_CONTROL);
      if (msg->command == upperlimb_robot::MODE_IDLE)
      {
        std::lock_guard<std::mutex> lock(eeg_mutex_);
        pending_target_ = 0;
        action_running_ = false;
        eeg_phase_ = EEG_WAITING;
        eeg_note_ = "软件停止：调试动作已取消";
      }
    }
    current_mode_.store(msg->command);
    RCLCPP_INFO(this->get_logger(), "收到控制命令=%d", msg->command);
  }

  // ===================== 模式状态机更新 =====================
  //
  // current_mode_ : 外部请求的新模式
  // applied_mode_ : 已经完成进入动作的当前模式
  //
  // 每轮控制循环开始时调用一次：
  // 1. 如果模式没变，直接返回
  // 2. 先执行旧模式退出动作
  // 3. 再执行新模式进入动作
  // 4. 最后更新 applied_mode_
  //
  void updateModeStateMachine()
  {
    const int requested_mode = current_mode_.load();

    if (requested_mode == applied_mode_)
    {
      return;
    }

    const int old_mode = applied_mode_;

    onModeExit(old_mode, requested_mode);

    // onModeEnter() 可能因为轴模式切换失败而要求回退。
    // 进入失败时不能把 applied_mode_ 仍然设置成 requested_mode，
    // 否则控制循环会短暂以失败模式继续计算/下发，造成一帧错误 PT。
    mode_enter_failed_ = false;
    mode_enter_fallback_mode_ = requested_mode;

    onModeEnter(old_mode, requested_mode);

    if (mode_enter_failed_)
    {
      applied_mode_ = mode_enter_fallback_mode_;
      status_mode_.store(applied_mode_);
      current_mode_.store(mode_enter_fallback_mode_);

      RCLCPP_WARN(
          this->get_logger(),
          "模式进入失败: old=%d requested=%d fallback=%d",
          old_mode,
          requested_mode,
          mode_enter_fallback_mode_);
      return;
    }

    applied_mode_ = requested_mode;
    status_mode_.store(applied_mode_);

    RCLCPP_INFO(
        this->get_logger(),
        "模式切换完成: old=%d -> new=%d",
        old_mode,
        requested_mode);
  }

  // ===================== 模式退出动作 =====================
  //
  // 设计目的：
  // 1. 退出扭矩相关模式时，先清零 PT，再恢复到位置模式
  // 3. 下一次进入预设位置模式时，保证位置命令和 Trap 参数会重新完整下发
  //
  void onModeExit(int old_mode, int new_mode)
  {
    (void)new_mode;
    if (old_mode == upperlimb_robot::MODE_ZERO_FORCE ||
        old_mode == upperlimb_robot::MODE_RIGHT_ZERO_FORCE ||
        old_mode == upperlimb_robot::MODE_BILATERAL_ZERO_FORCE)
    {
      clearAllTorqueCommands();
      stopAllAxesMotionOnly();
      restoreAllAxesToPositionMode();
    }
    if (old_mode == upperlimb_robot::MODE_EEG_CONTROL)
    {
      stopAllAxesMotionOnly();
      eeg_mode_ready_.store(false);
      std::lock_guard<std::mutex> lock(eeg_mutex_);
      pending_target_ = 0;
      action_running_ = false;
      eeg_phase_ = EEG_WAITING;
    }
    invalidateAllPositionCommandCache();
  }

  void onModeEnter(int old_mode, int new_mode)
  {
    (void)old_mode;
    switch (new_mode)
    {
    case upperlimb_robot::MODE_ZERO_FORCE:
    case upperlimb_robot::MODE_RIGHT_ZERO_FORCE:
      if (!captureForceSensorBias())
        requestModeEnterFallback(upperlimb_robot::MODE_IDLE);
      break;
    case upperlimb_robot::MODE_BILATERAL_ZERO_FORCE:
    {
      upperlimb_robot::ControlOutputCache first_output{};
      const bool bias_ok = captureForceSensorBias();
      const bool precompute_ok = bias_ok && precomputeBilateralZeroForceBeforeSwitch(first_output);
      if (!precompute_ok || !enterDualArmTorqueModeOnce(&first_output))
        requestModeEnterFallback(upperlimb_robot::MODE_IDLE);
      break;
    }
    case upperlimb_robot::MODE_EEG_CONTROL:
      // 只在控制线程切换驱动。动作还须通过预设位及反馈检查。
      for (std::size_t i = 0; i < upperlimb_robot::AXIS_NUM; ++i)
      {
        if (!isControlledAxisIndex(i)) continue;
        unsigned short actual_mode = 0;
        const int axis = axis_ids_[i];
        if (!axis_enabled_[i] ||
            GTN_GetEcatAxisMode(core_id_, axis, &actual_mode) != 0)
        {
          requestModeEnterFallback(upperlimb_robot::MODE_IDLE);
          return;
        }
        if (actual_mode != 8 && !ensureAxisInPositionMode(i))
        {
          requestModeEnterFallback(upperlimb_robot::MODE_IDLE);
          return;
        }
        if (GTN_GetEcatAxisMode(core_id_, axis, &actual_mode) != 0 || actual_mode != 8)
        {
          requestModeEnterFallback(upperlimb_robot::MODE_IDLE);
          return;
        }
      }
      eeg_mode_ready_.store(true);
      break;
    default:
      break;
    }
  }

  // ===================== 双臂扭矩模式辅助函数 =====================
  //
  // 双臂零力模式同时控制 14 个主动关节。
  // 这类模式不建议在 90 Hz 控制循环内反复执行
  // “位置模式 <-> 扭矩模式”切换。
  //
  // 策略：
  // 1. 进入模式时一次性切 14 个主动轴到扭矩模式。
  // 2. 控制循环中只发送 PT 扭矩。
  // 3. 算法本轮无效时，只清零 PT，不立刻切回位置模式。
  // 4. 退出模式时再统一恢复位置模式。
  //
  bool isDualArmTorqueMode(int mode) const
  {
    return mode == upperlimb_robot::MODE_BILATERAL_ZERO_FORCE;
  }

  std::array<std::size_t, 14> dualArmActiveTorqueIndices() const
  {
    return {
        // 右臂 joint11~joint17 -> 控制层索引 0~6
        0, 1, 2, 3, 4, 5, 6,
        // 左臂 joint1~joint7 -> 控制层索引 10~16
        10, 11, 12, 13, 14, 15, 16};
  }

  // ===================== 双臂零力：切换扭矩模式前预计算 =====================
  //
  // 在任何主动轴切入扭矩模式之前，先让算法完成 Pinocchio 懒初始化、
  // 重力与传感器力矩映射计算，并检查14轴输出的有效性。
  // 此函数不发送 PT、不停止电机、不切换关节控制模式。
  // 结果交由切换函数，供各轴确认CST后立即发送首次PT；
  // 注意：驱动器内部的模式切换力矩空档仍有待实机验证。
  bool precomputeBilateralZeroForceBeforeSwitch(
      upperlimb_robot::ControlOutputCache &precomputed)
  {
    precomputed = upperlimb_robot::ControlOutputCache{};
    if (!state_ready_.load())
    {
      RCLCPP_ERROR(this->get_logger(), "双臂零力预计算失败：关节状态尚未准备完成");
      return false;
    }

    upperlimb_robot::MotorStateCache state;
    upperlimb_robot::SensorStateCache sensor_state;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      state = latest_motor_state_;
    }
    {
      std::lock_guard<std::mutex> lock(sensor_mutex_);
      sensor_state = latest_sensor_state_;
    }

    if (!sensor_state.valid)
    {
      RCLCPP_ERROR(this->get_logger(), "双臂零力预计算失败：六维力传感器数据无效");
      return false;
    }

    for (const std::size_t i : dualArmActiveTorqueIndices())
    {
      if (!std::isfinite(state.position[i]) ||
          !std::isfinite(state.velocity[i]))
      {
        RCLCPP_ERROR(this->get_logger(),
                     "双臂零力预计算失败：控制索引%zu关节位置或速度非有限值", i);
        return false;
      }
    }

    const auto t_begin = std::chrono::steady_clock::now();
    try
    {
      upperlimb_robot::runBilateralZeroForceMode(
          state, sensor_state, precomputed);
    }
    catch (const std::exception &e)
    {
      RCLCPP_ERROR(this->get_logger(),
                   "双臂零力预计算异常：%s", e.what());
      return false;
    }

    if (!precomputed.valid)
    {
      RCLCPP_ERROR(this->get_logger(),
                   "双臂零力预计算失败：Pinocchio初始化或控制输出无效");
      return false;
    }

    for (const std::size_t i : dualArmActiveTorqueIndices())
    {
      if (!precomputed.use_torque[i] ||
          !std::isfinite(precomputed.target_torque[i]))
      {
        RCLCPP_ERROR(this->get_logger(),
                     "双臂零力预计算失败：控制索引%zu输出力矩无效", i);
        return false;
      }
    }

    const auto precompute_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - t_begin).count();
    RCLCPP_INFO(this->get_logger(),
                "双臂零力预计算通过：用时=%lld us，右1轴=%.3f Nm，左11轴=%.3f Nm；尚未切换模式或下发PT",
                static_cast<long long>(precompute_us),
                precomputed.target_torque[0],
                precomputed.target_torque[10]);

    // 首次PT使用此处的有效预计算结果，降低已进入扭矩模式但尚未
    // 收到补偿力矩的等待时间。正式控制周期仍根据最新状态重新计算。
    return true;
  }

  // ===================== 14轴扭矩模式切换和首次PT力矩交接 =====================
  // 无初始力矩参数时保留其他主动/弹簧模式原有行为。
  // 传入 initial_output 仅用于双臂零力：
  // 1. 先为全部14轴验证PT数值、单位换算及short范围（不触碰驱动器）；
  // 2. 每根轴确认实际模式10后，立即发送该轴预计算的首条PT；
  // 3. 首帧PT成功后更新缓存，避免正常控制周期误判为从未发送；
  // 4. 缓存物理轴1、11的关键时间，全部切换结束后再输出日志。
  // 注意：驱动器从位置模式切到CST时仍可能存在内部力矩空档；
  // 这不是无缝力矩交接的安全保证，必须在机械防坠保护下测试。
  bool enterDualArmTorqueModeOnce(
      const upperlimb_robot::ControlOutputCache *initial_output = nullptr)
  {
    const auto indices = dualArmActiveTorqueIndices();
    std::array<short, 14> initial_pt_raw{};

    // 所有验证必须先于第一次调用 ensureAxisInTorqueMode()。
    if (initial_output != nullptr)
    {
      for (std::size_t k = 0; k < indices.size(); ++k)
      {
        const std::size_t i = indices[k];
        if (i >= upperlimb_robot::AXIS_NUM ||
            !isControlledAxisIndex(i) ||
            !initial_output->valid ||
            !initial_output->use_torque[i] ||
            !std::isfinite(initial_output->target_torque[i]))
        {
          RCLCPP_ERROR(this->get_logger(),
                       "双臂零力PT预检查失败：索引%zu的初始力矩无效", i);
          return false;
        }

        const int axis = axis_ids_[i];
        const double raw = upperlimb_robot::torqueNmToRaw(
            axis, initial_output->target_torque[i]);
        const double raw_per_nm = upperlimb_robot::torqueNmToRaw(axis, 1.0);
        if (!std::isfinite(raw) || !std::isfinite(raw_per_nm) ||
            raw_per_nm == 0.0 ||
            raw < static_cast<double>(std::numeric_limits<short>::min()) ||
            raw > static_cast<double>(std::numeric_limits<short>::max()))
        {
          RCLCPP_ERROR(this->get_logger(),
                       "双臂零力PT预检查失败：物理轴%d Nm/raw换算错误或超short范围", axis);
          return false;
        }
        initial_pt_raw[k] = static_cast<short>(std::lround(raw));
      }
    }

    using Clock = std::chrono::steady_clock;
    struct FirstAxisTrace
    {
      int axis{0};
      double target_nm{0.0};
      short raw{0};
      long long mode_us{0};
      long long pt_us{0};
      long long total_us{0};
      bool recorded{false};
    };
    std::array<FirstAxisTrace, 2> traces{};
    const auto start_time = Clock::now();
    bool all_ok = true;

    for (std::size_t k = 0; k < indices.size(); ++k)
    {
      const std::size_t i = indices[k];
      if (i >= upperlimb_robot::AXIS_NUM)
      {
        all_ok = false;
        if (initial_output != nullptr)
          return false;
        continue;
      }

      const auto before_mode = Clock::now();
      if (!ensureAxisInTorqueMode(i))
      {
        all_ok = false;
        if (initial_output != nullptr)
        {
          RCLCPP_ERROR(this->get_logger(),
                       "双臂零力逐轴交接中止：物理轴%d切入扭矩模式失败", axis_ids_[i]);
          return false;
        }
        continue;
      }
      const auto mode_ready = Clock::now();

      if (initial_output != nullptr)
      {
        // 关键：不等待另外13根轴，当前轴模式10确认后立刻写入首条PT。
        const int axis = axis_ids_[i];
        const short rtn_pt = GTN_SetEcatAxisPT(core_id_, axis, initial_pt_raw[k]);
        const auto pt_done = Clock::now();
        if (rtn_pt != 0)
        {
          RCLCPP_ERROR(this->get_logger(),
                       "双臂零力逐轴交接中止：物理轴%d首条PT发送失败，错误码=%d，目标=%.3f Nm，raw=%d",
                       axis, rtn_pt, initial_output->target_torque[i],
                       static_cast<int>(initial_pt_raw[k]));
          return false;
        }

        last_target_torque_[i] = initial_output->target_torque[i];

        if (axis == 1 || axis == 11)
        {
          FirstAxisTrace &trace = traces[axis == 1 ? 0 : 1];
          trace.axis = axis;
          trace.target_nm = initial_output->target_torque[i];
          trace.raw = initial_pt_raw[k];
          trace.mode_us = std::chrono::duration_cast<std::chrono::microseconds>(
                              mode_ready - before_mode).count();
          trace.pt_us = std::chrono::duration_cast<std::chrono::microseconds>(
                            pt_done - mode_ready).count();
          trace.total_us = std::chrono::duration_cast<std::chrono::microseconds>(
                               pt_done - start_time).count();
          trace.recorded = true;
        }
      }
      else
      {
        // 其他调用此函数的模式保持原有行为。
        last_target_torque_[i] = 0.0;
      }

      command_sent_[i] = false;
      position_profile_configured_[i] = false;
    }

    if (initial_output != nullptr && all_ok)
    {
      for (const FirstAxisTrace &trace : traces)
      {
        if (!trace.recorded)
          continue;
        RCLCPP_INFO(this->get_logger(),
                    "双臂零力首条PT交接：物理轴%d，目标=%.3f Nm，raw=%d，单轴切模式=%lld us，模式确认至PT返回=%lld us，从开始切轴至PT返回=%lld us",
                    trace.axis, trace.target_nm, static_cast<int>(trace.raw),
                    trace.mode_us, trace.pt_us, trace.total_us);
      }
      const auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                Clock::now() - start_time).count();
      RCLCPP_INFO(this->get_logger(),
                  "双臂零力14轴逐轴PT交接完成：合计=%lld us；日志记录的是接口调用时间，不等于驱动实际力矩建立时间",
                  static_cast<long long>(total_us));
    }

    return all_ok;
  }

  void clearDualArmTorqueCommandsNoModeSwitch()
  {
    for (const std::size_t i : dualArmActiveTorqueIndices())
    {
      if (i >= upperlimb_robot::AXIS_NUM)
      {
        continue;
      }

      const int axis = axis_ids_[i];
      const short rtn = GTN_SetEcatAxisPT(core_id_, axis, 0);
      if (rtn != 0)
      {
        RCLCPP_ERROR_THROTTLE(
            this->get_logger(), *this->get_clock(), 2000,
            "双臂扭矩模式清零 PT 失败，轴%d，错误码=%d", axis, rtn);
      }

      last_target_torque_[i] = 0.0;
    }
  }

  // ===================== 控制器初始化 =====================
  bool initController()
  {
    short rtn = GTN_Open();
    if (rtn != 0)
    {
      RCLCPP_ERROR(this->get_logger(), "GTN_Open 打开控制器失败，错误码=%d", rtn);
      return false;
    }
    controller_opened_ = true;
    RCLCPP_INFO(this->get_logger(), "GTN_Open 成功");

    std::string config_path;
    try
    {
      const std::string pkg_share =
          ament_index_cpp::get_package_share_directory("upperlimb_robot");
      config_path = pkg_share + "/config/Gecat.eni";
    }
    catch (const std::exception &e)
    {
      RCLCPP_ERROR(this->get_logger(), "获取包配置目录失败: %s", e.what());
      return false;
    }

    rtn = GTN_LoadConfig(core_id_, const_cast<char *>(config_path.c_str()));
    if (rtn != 0)
    {
      RCLCPP_ERROR(
          this->get_logger(),
          "GTN_LoadConfig 加载配置文件失败，错误码=%d, 路径=%s",
          rtn, config_path.c_str());
      return false;
    }
    RCLCPP_INFO(this->get_logger(), "GTN_LoadConfig 成功");

    rtn = GTN_ClrSts(core_id_, 1, static_cast<short>(upperlimb_robot::AXIS_NUM));
    if (rtn != 0)
    {
      RCLCPP_WARN(this->get_logger(), "GTN_ClrSts 更新配置状态失败，错误码=%d", rtn);
    }

    rtn = GTN_InitEcatComm(core_id_);
    if (rtn != 0)
    {
      RCLCPP_ERROR(this->get_logger(), "EtherCAT初始化失败，错误码=%d", rtn);
      return false;
    }

    short ready = 0;
    int retry = 0;
    constexpr int max_retry = 100;

    do
    {
      rtn = GTN_IsEcatReady(core_id_, &ready);
      if (rtn != 0)
      {
        RCLCPP_ERROR(this->get_logger(), "检测EtherCAT状态失败，错误码=%d", rtn);
        return false;
      }

      if (ready == 1)
      {
        break;
      }

      rclcpp::sleep_for(50ms);
      retry++;
    } while (retry < max_retry);

    if (ready != 1)
    {
      RCLCPP_ERROR(this->get_logger(), "EtherCAT启动超时");
      return false;
    }

    rtn = GTN_StartEcatComm(core_id_);
    if (rtn != 0)
    {
      RCLCPP_ERROR(this->get_logger(), "启动EtherCAT通信失败，错误码=%d", rtn);
      return false;
    }

    ecat_started_ = true;
    RCLCPP_INFO(this->get_logger(), "EtherCAT通信启动成功");

    return true;
  }

  // ===================== 电机初始化 =====================
  bool initMotors()
  {
    short rtn = 0;

    for (std::size_t i = 0; i < upperlimb_robot::AXIS_NUM; ++i)
    {
      const int axis = axis_ids_[i];

      if (!isControlledAxisIndex(i))
      {
        enc_pos_[i] = 0;
        axis_enabled_[i] = false;
        RCLCPP_INFO(
            this->get_logger(),
            "轴%d 为末端非控制关节，初始化阶段跳过使能、同步和模式控制",
            axis);
        continue;
      }

      rtn = GTN_GetEcatEncPos(core_id_, axis, &enc_pos_[i]);
      if (rtn != 0)
      {
        RCLCPP_ERROR(this->get_logger(), "读取轴%d编码器位置失败，错误码=%d", axis, rtn);
        return false;
      }

      rtn = GTN_SetPrfPos(core_id_, axis, enc_pos_[i]);
      if (rtn != 0)
      {
        RCLCPP_ERROR(this->get_logger(), "设置轴%d规划位置失败，错误码=%d", axis, rtn);
        return false;
      }

      rtn = GTN_SetEncPos(core_id_, axis, enc_pos_[i]);
      if (rtn != 0)
      {
        RCLCPP_ERROR(this->get_logger(), "设置轴%d编码器位置失败，错误码=%d", axis, rtn);
        return false;
      }

      // ===================== 同步 axis / profile / encoder =====================
      rtn = GTN_SynchAxisPos(core_id_, static_cast<long>(1L << (axis - 1)));
      if (rtn != 0)
      {
        RCLCPP_ERROR(this->get_logger(), "同步轴%d位置失败，错误码=%d", axis, rtn);
        return false;
      }

      rtn = GTN_AxisOn(core_id_, axis);
      if (rtn != 0)
      {
        RCLCPP_ERROR(this->get_logger(), "轴%d使能失败，错误码=%d", axis, rtn);
        return false;
      }

      axis_enabled_[i] = true;

      RCLCPP_INFO(this->get_logger(), "轴%d初始化完成，编码器位置=%d", axis, enc_pos_[i]);
    }

    return true;
  }

  // ===================== 系统状态发布 =====================
  void publishSystemState()
  {
    upperlimb_robot::msg::SystemState msg;
    msg.joint_init = joint_init_;
    msg.controller_init = controller_init_;
    msg.system_ready = system_ready_;
    system_pub_->publish(msg);
  }

  // ===================== 读取 EtherCAT 原始 float 数据 =====================
  bool readFloatRawData(int offset, double &value)
  {
    int raw = 0;
    const short rtn =
        GTN_GetEcatRawData(core_id_, offset, 4, reinterpret_cast<unsigned char *>(&raw));

    if (rtn != 0)
    {
      value = 0.0;
      return false;
    }

    float temp = 0.0f;
    std::memcpy(&temp, &raw, sizeof(float));
    value = static_cast<double>(temp);
    return true;
  }

  // ===================== 读取单个六维力传感器 =====================
  bool readSingleForceSensor(int base_offset, upperlimb_robot::ForceSensorData &data)
  {
    bool ok = true;

    ok &= readFloatRawData(base_offset + 0, data.fx);
    ok &= readFloatRawData(base_offset + 4, data.fy);
    ok &= readFloatRawData(base_offset + 8, data.fz);
    ok &= readFloatRawData(base_offset + 12, data.mx);
    ok &= readFloatRawData(base_offset + 16, data.my);
    ok &= readFloatRawData(base_offset + 20, data.mz);

    return ok;
  }

  // ===================== 读取全部四个六维力传感器 =====================
  bool readAllForceSensors(upperlimb_robot::SensorStateCache &sensor_state)
  {
    bool ok = true;

    ok &= readSingleForceSensor(8, sensor_state.sensor[upperlimb_robot::SENSOR_RIGHT_HAND]);
    ok &= readSingleForceSensor(40, sensor_state.sensor[upperlimb_robot::SENSOR_RIGHT_ARM]);
    ok &= readSingleForceSensor(72, sensor_state.sensor[upperlimb_robot::SENSOR_LEFT_HAND]);
    ok &= readSingleForceSensor(104, sensor_state.sensor[upperlimb_robot::SENSOR_LEFT_ARM]);

    sensor_state.valid = ok;
    return ok;
  }

  // ===================== 记录当前传感器零偏 =====================
  //
  // 只有在 latest_sensor_state_.valid == true 时，才允许记录零偏。
  // 这样可以避免把未初始化值、读失败值或脏值写进 bias。
  //
  bool captureForceSensorBias()
  {
    std::lock_guard<std::mutex> lock(sensor_mutex_);

    if (!latest_sensor_state_.valid)
    {
      return false;
    }

    latest_sensor_state_.bias = latest_sensor_state_.sensor;
    return true;
  }

  // ===================== 六维力去偏 =====================
  upperlimb_robot::ForceSensorData subtractBias(
      const upperlimb_robot::ForceSensorData &raw,
      const upperlimb_robot::ForceSensorData &bias) const
  {
    upperlimb_robot::ForceSensorData out;
    out.fx = raw.fx - bias.fx;
    out.fy = raw.fy - bias.fy;
    out.fz = raw.fz - bias.fz;
    out.mx = raw.mx - bias.mx;
    out.my = raw.my - bias.my;
    out.mz = raw.mz - bias.mz;
    return out;
  }

  // ===================== 传感器状态转成数组 =====================
  // 四个六维力传感器，共 24 维。
  std::vector<double> buildForceSensorArray(
      const upperlimb_robot::SensorStateCache &sensor_state) const
  {
    std::vector<double> out;
    out.reserve(24);

    const auto right_hand = subtractBias(
        sensor_state.sensor[upperlimb_robot::SENSOR_RIGHT_HAND],
        sensor_state.bias[upperlimb_robot::SENSOR_RIGHT_HAND]);

    const auto right_arm = subtractBias(
        sensor_state.sensor[upperlimb_robot::SENSOR_RIGHT_ARM],
        sensor_state.bias[upperlimb_robot::SENSOR_RIGHT_ARM]);

    const auto left_hand = subtractBias(
        sensor_state.sensor[upperlimb_robot::SENSOR_LEFT_HAND],
        sensor_state.bias[upperlimb_robot::SENSOR_LEFT_HAND]);

    const auto left_arm = subtractBias(
        sensor_state.sensor[upperlimb_robot::SENSOR_LEFT_ARM],
        sensor_state.bias[upperlimb_robot::SENSOR_LEFT_ARM]);

    auto fill_one = [&](const upperlimb_robot::ForceSensorData &s)
    {
      out.push_back(s.fx);
      out.push_back(s.fy);
      out.push_back(s.fz);
      out.push_back(s.mx);
      out.push_back(s.my);
      out.push_back(s.mz);
    };

    fill_one(right_hand);
    fill_one(right_arm);
    fill_one(left_hand);
    fill_one(left_arm);

    return out;
  }

  // ===================== 发布 RobotState =====================
  void publishRobotState(
      const upperlimb_robot::MotorStateCache &state,
      const upperlimb_robot::SensorStateCache &sensor_state)
  {
    upperlimb_robot::msg::RobotState msg;

    msg.position.assign(state.position.begin(), state.position.end());
    msg.velocity.assign(state.velocity.begin(), state.velocity.end());
    msg.torque.assign(state.torque.begin(), state.torque.end());
    msg.enabled.assign(state.enable.begin(), state.enable.end());

    const auto force_array = buildForceSensorArray(sensor_state);
    msg.force_sensor.assign(force_array.begin(), force_array.end());

    robot_state_pub_->publish(msg);
    joint_state_pub_->publish(convertToJointStateMsg(state));
  }

  // ===================== 状态采集线程 =====================
  void stateReadLoop()
  {
    std::uint64_t timeout_count = 0;

    while (running_.load())
    {
      const int state_hz = state_hz_.load();

      if (state_hz <= 0)
      {
        RCLCPP_ERROR_THROTTLE(
            this->get_logger(), *this->get_clock(), 2000,
            "state_hz 非法=%d，必须大于 0", state_hz);

        std::this_thread::sleep_for(100ms);
        continue;
      }

      const auto period = std::chrono::microseconds(1000000 / state_hz);
      auto cycle_start = std::chrono::steady_clock::now();

      upperlimb_robot::MotorStateCache local_state;
      upperlimb_robot::SensorStateCache local_sensor_state;
      std::array<bool, upperlimb_robot::AXIS_NUM> local_feedback_valid{};

      const bool do_read_status =
          (++status_read_divider_count_ >= status_read_divider_);

      if (do_read_status)
      {
        status_read_divider_count_ = 0;
      }

      for (std::size_t i = 0; i < upperlimb_robot::AXIS_NUM; ++i)
      {
        const int axis = axis_ids_[i];

        if (!isControlledAxisIndex(i))
        {
          local_state.enable[i] = false;
          local_state.position[i] = 0.0;
          local_state.velocity[i] = 0.0;
          local_state.torque[i] = 0.0;
          continue;
        }

        short rtn = 0;
        int32_t sts = 0;
        int32_t enc_pos = 0;
        double vel = 0.0;
        short torque_raw = 0;

        if (do_read_status)
        {
          rtn = GTN_GetSts(core_id_, axis, &sts, 1);
          if (rtn == 0)
          {
            local_state.enable[i] = (sts != 0);
            local_feedback_valid[i] = true;
          }
          else
          {
            local_state.enable[i] = axis_enabled_[i];
            local_feedback_valid[i] = false;
          }
        }
        else
        {
          local_state.enable[i] = axis_enabled_[i];
        }

        if (!do_read_status) local_feedback_valid[i] = true;
        rtn = GTN_GetEcatEncPos(core_id_, axis, &enc_pos);
        if (rtn == 0)
        {
          local_state.position[i] =
              upperlimb_robot::pulseToJointDeg(axis, static_cast<double>(enc_pos));
        }
        else
        {
          local_state.position[i] = std::numeric_limits<double>::quiet_NaN();
          local_feedback_valid[i] = false;
        }

        rtn = GTN_GetEcatAxisAtlTorque(core_id_, axis, &torque_raw);
        if (rtn == 0)
        {
          local_state.torque[i] =
              upperlimb_robot::torqueRawToNm(axis, static_cast<double>(torque_raw));
        }
        else
        {
          local_state.torque[i] = 0.0;
        }

        rtn = GTN_GetEncVel(core_id_, axis, &vel, 1);
        if (rtn == 0)
        {
          local_state.velocity[i] =
              upperlimb_robot::pulsePerMsToJointDegPerSec(axis, vel);
        }
        else
        {
          local_state.velocity[i] = std::numeric_limits<double>::quiet_NaN();
          local_feedback_valid[i] = false;
        }
      }

      {
        std::lock_guard<std::mutex> lock(sensor_mutex_);
        local_sensor_state.bias = latest_sensor_state_.bias;
      }

      readAllForceSensors(local_sensor_state);

      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        latest_motor_state_ = local_state;
        feedback_valid_ = local_feedback_valid;
        last_feedback_time_ = std::chrono::steady_clock::now();
        state_ready_.store(true);
      }

      {
        std::lock_guard<std::mutex> lock(sensor_mutex_);
        latest_sensor_state_.sensor = local_sensor_state.sensor;
        latest_sensor_state_.bias = local_sensor_state.bias;
        latest_sensor_state_.valid = local_sensor_state.valid;
      }

      publish_divider_count_++;
      if (publish_divider_count_ >= publish_divider_)
      {
        publish_divider_count_ = 0;
        publishRobotState(local_state, local_sensor_state);
      }

      auto cycle_end = std::chrono::steady_clock::now();
      auto elapsed =
          std::chrono::duration_cast<std::chrono::microseconds>(cycle_end - cycle_start);

      if (elapsed < period)
      {
        std::this_thread::sleep_for(period - elapsed);
      }
      else
      {
        timeout_count++;

        if (timeout_count % 100 == 0)
        {
          double actual_hz = 1000000.0 / static_cast<double>(elapsed.count());

          RCLCPP_WARN(
              this->get_logger(),
              "状态采集循环超时，目标频率=%d Hz，实际频率=%.1f Hz，耗时=%ld us",
              state_hz_.load(),
              actual_hz,
              elapsed.count());
        }
      }
    }
  }

  // ===================== 控制线程 =====================
  void controlLoop()
  {
    std::uint64_t timeout_count = 0;

    while (running_.load())
    {
      const int control_hz = control_hz_.load();
      if (control_hz <= 0)
      {
        RCLCPP_ERROR_THROTTLE(
            this->get_logger(), *this->get_clock(), 2000,
            "control_hz 非法=%d，必须大于 0", control_hz);
        std::this_thread::sleep_for(100ms);
        continue;
      }

      const auto period = std::chrono::microseconds(1000000 / control_hz);
      auto cycle_start = std::chrono::steady_clock::now();

      if (!system_ready_)
      {
        std::this_thread::sleep_for(period);
        continue;
      }

      if (!state_ready_.load())
      {
        std::this_thread::sleep_for(period);
        continue;
      }

      upperlimb_robot::MotorStateCache local_state;
      upperlimb_robot::SensorStateCache local_sensor_state;

      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        local_state = latest_motor_state_;
      }

      {
        std::lock_guard<std::mutex> lock(sensor_mutex_);
        local_sensor_state = latest_sensor_state_;
      }

      const int previous_mode = applied_mode_;
      updateModeStateMachine();

      // 双臂零力首帧必须读取切换后最新的关节数据与新记录的传感器零偏。
      // 否则上一模式的缓存零偏可能造成首帧 J^T F 力矩脉冲。
      if (previous_mode != applied_mode_ &&
          applied_mode_ == upperlimb_robot::MODE_BILATERAL_ZERO_FORCE)
      {
        {
          std::lock_guard<std::mutex> lock(state_mutex_);
          local_state = latest_motor_state_;
        }
        {
          std::lock_guard<std::mutex> lock(sensor_mutex_);
          local_sensor_state = latest_sensor_state_;
        }
      }

      const int mode = applied_mode_;
      double actual_dt_s = 1.0 / static_cast<double>(control_hz);
      if (last_control_time_valid_)
      {
        actual_dt_s =
            std::chrono::duration<double>(cycle_start - last_control_time_).count();
      }
      last_control_time_ = cycle_start;
      last_control_time_valid_ = true;

      upperlimb_robot::ControlOutputCache output =
          computeControlOutput(mode, local_state, local_sensor_state, actual_dt_s);


      if (output.valid)
      {
        applyControlOutput(output);
      }
      else
      {
        if (applied_mode_ == upperlimb_robot::MODE_EEG_CONTROL)
        {
          // 预设未确认或故障时，不生成新的位置命令。
        }
        else if (isDualArmTorqueMode(applied_mode_))
        {
          // 纯14轴扭矩模式单帧无效时只清零PT，避免循环中反复切轴模式。
          clearDualArmTorqueCommandsNoModeSwitch();
        }
        else
        {
          clearAllTorqueCommands();
          restoreAllAxesToPositionMode();
        }
      }

      auto cycle_end = std::chrono::steady_clock::now();
      auto elapsed =
          std::chrono::duration_cast<std::chrono::microseconds>(cycle_end - cycle_start);

      if (elapsed < period)
      {
        std::this_thread::sleep_for(period - elapsed);
      }
      else
      {
        timeout_count++;
        if (timeout_count % 100 == 0)
        {
          double actual_hz = 1000000.0 / static_cast<double>(elapsed.count());

          RCLCPP_WARN(
              this->get_logger(),
              "控制循环超时，目标频率=%d Hz，实际频率=%.1f Hz，耗时=%ld us",
              control_hz_.load(),
              actual_hz,
              elapsed.count());
        }
      }
    }
  }

  void setEegFault(const std::string &reason)
  {
    stopAllAxesMotionOnly();
    eeg_fault_latched_.store(true);
    std::lock_guard<std::mutex> lock(eeg_mutex_);
    eeg_phase_ = EEG_FAULT;
    eeg_note_ = reason;
    pending_target_ = 0;
    // 故障时保留占用，普通脑电指令不能重新启动。
    action_running_ = true;
    RCLCPP_ERROR(this->get_logger(), "EEG 故障: %s", reason.c_str());
  }

  // 控制线程按真实编码器位置和速度推进去程、返程。
  void computeEegOutput(const upperlimb_robot::MotorStateCache &state,
                        upperlimb_robot::ControlOutputCache &output)
  {
    upperlimb_robot::ControlOutputCache preset;
    upperlimb_robot::runEegPresetPosition(state, preset);
    bool preset_ok = true;
    bool feedback_ok = true;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      feedback_ok = std::chrono::steady_clock::now() - last_feedback_time_ < 200ms;
      for (std::size_t i = 0; i < upperlimb_robot::AXIS_NUM; ++i)
      {
        if (!isControlledAxisIndex(i)) continue;
        if (!feedback_valid_[i] || !state.enable[i] ||
            !std::isfinite(state.position[i]) || !std::isfinite(state.velocity[i]))
          feedback_ok = false;
        if (std::abs(state.position[i] - preset.target_position[i]) > 1.0 ||
            std::abs(state.velocity[i]) > 1.0)
          preset_ok = false;
      }
    }
    eeg_preset_ok_.store(feedback_ok && preset_ok);
    if (eeg_fault_latched_.load()) return;
    int target = 0;
    EegPhase phase;
    std::chrono::steady_clock::time_point phase_started;
    {
      std::lock_guard<std::mutex> lock(eeg_mutex_);
      phase = eeg_phase_;
      target = eeg_target_;
      if (phase == EEG_OUT && pending_target_ != 0)
      {
        if (!feedback_ok || !preset_ok)
        { pending_target_ = 0; action_running_ = false; eeg_phase_ = EEG_WAITING;
          eeg_note_ = "机器人未到达双臂预设位置"; return; }
        pending_target_ = 0;
        stable_count_ = 0;
        phase_start_ = std::chrono::steady_clock::now();
      }
      phase_started = phase_start_;
    }
    if (!feedback_ok) { setEegFault("位置或速度反馈异常"); return; }
    if (phase == EEG_WAITING)
    {
      // 预设位置未确认时，绝不从任意姿态下发预设目标。
      if (preset_ok) output = preset;
      else
      {
        std::lock_guard<std::mutex> lock(eeg_mutex_);
        if (eeg_note_ == "等待脑电指令" || eeg_note_ == "往返完成，保持预设位置")
          eeg_note_ = "机器人未到达双臂预设位置";
      }
      return;
    }
    if (phase == EEG_FAULT) return;
    const int index = upperlimb_robot::eegTargetAxisIndex(target);
    const int k = target - 1;
    if (index < 0 || k < 0 || k >= 4 ||
        !std::isfinite(eeg_min_deg_[k]) || !std::isfinite(eeg_max_deg_[k]) ||
        !std::isfinite(eeg_delta_deg_[k]) || std::abs(eeg_delta_deg_[k]) > 10.0 ||
        eeg_min_deg_[k] >= eeg_max_deg_[k] ||
        !upperlimb_robot::runEegAction(state, target, eeg_delta_deg_[k],
                    phase == EEG_BACK, eeg_velocity_deg_s_, eeg_acceleration_deg_s2_, output))
    { setEegFault("运动参数或关节限位未配置"); return; }
    const double goal = output.target_position[index];
    if (goal < eeg_min_deg_[k] || goal > eeg_max_deg_[k])
    { output.valid = false; setEegFault("目标角度超出已配置限位"); return; }
    if (std::chrono::steady_clock::now() - phase_started > 8s)
    { output.valid = false; setEegFault("关节运动超时"); return; }
    if (std::abs(state.position[index] - goal) <= 0.5 &&
        std::abs(state.velocity[index]) <= 1.0)
      ++stable_count_;
    else
      stable_count_ = 0;
    if (stable_count_ >= 3)
    {
      std::lock_guard<std::mutex> lock(eeg_mutex_);
      stable_count_ = 0;
      phase_start_ = std::chrono::steady_clock::now();
      if (phase == EEG_OUT)
      { eeg_phase_ = EEG_BACK; eeg_note_ = "去程到位，开始返程"; }
      else
      { eeg_phase_ = EEG_WAITING; action_running_ = false;
        eeg_note_ = "往返完成，保持预设位置"; }
    }
  }

  // ===================== 控制算法总入口 =====================
  upperlimb_robot::ControlOutputCache computeControlOutput(
      int mode, const upperlimb_robot::MotorStateCache &state,
      const upperlimb_robot::SensorStateCache &sensor_state, double actual_dt_s)
  {
    (void)actual_dt_s;
    upperlimb_robot::ControlOutputCache output{};
    switch (mode)
    {
    case upperlimb_robot::MODE_LEFT_PRESET_POSITION:
      upperlimb_robot::runLeftPresetPosition(state, output); break;
    case upperlimb_robot::MODE_RIGHT_PRESET_POSITION:
      upperlimb_robot::runRightPresetPosition(state, output); break;
    case upperlimb_robot::MODE_ZERO_FORCE:
      upperlimb_robot::runZeroForceMode(state, sensor_state, output); break;
    case upperlimb_robot::MODE_RIGHT_ZERO_FORCE:
      upperlimb_robot::runRightZeroForceMode(state, sensor_state, output); break;
    case upperlimb_robot::MODE_BILATERAL_ZERO_FORCE:
      upperlimb_robot::runBilateralZeroForceMode(state, sensor_state, output); break;
    case upperlimb_robot::MODE_EEG_CONTROL:
      computeEegOutput(state, output); break;
    default:
      upperlimb_robot::runIdle(state, output); break;
    }
    for (std::size_t i = 0; i < upperlimb_robot::AXIS_NUM; ++i)
    {
      if (isControlledAxisIndex(i)) continue;
      output.use_position[i] = false;
      output.use_velocity[i] = false;
      output.use_torque[i] = false;
    }
    return output;
  }

  // ===================== 执行控制输出 =====================
  //
  // 普通模式仍沿用原来的位置输出和扭矩输出函数。
  void applyControlOutput(const upperlimb_robot::ControlOutputCache &output)
  {
    applyPositionOutput(output);
    applyTorqueOutput(output);
  }

  bool ensureAxisInPositionMode(std::size_t i)
  {
    if (i >= upperlimb_robot::AXIS_NUM)
    {
      return false;
    }

    if (!isControlledAxisIndex(i))
    {
      return false;
    }

    const int axis = axis_ids_[i];

    const short rtn_stop =
        GTN_Stop(core_id_, static_cast<long>(1L << (axis - 1)), 0);
    if (rtn_stop != 0)
    {
      RCLCPP_ERROR_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000,
          "切回位置模式前停止轴%d失败，错误码=%d", axis, rtn_stop);
      return false;
    }

    bool stopped = false;
    for (int retry = 0; retry < 40; ++retry)
    {
      double raw_velocity = 0.0;
      if (GTN_GetEncVel(core_id_, axis, &raw_velocity, 1) != 0) return false;
      if (std::abs(upperlimb_robot::pulsePerMsToJointDegPerSec(axis, raw_velocity)) < 1.0)
      { stopped = true; break; }
      std::this_thread::sleep_for(5ms);
    }
    if (!stopped)
    {
      RCLCPP_ERROR(this->get_logger(), "轴%d未停止，拒绝重置位置规划", axis);
      return false;
    }

    const short rtn_zero = GTN_SetEcatAxisPT(core_id_, axis, 0);
    if (rtn_zero != 0)
    {
      RCLCPP_ERROR_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000,
          "切回位置模式前清零 PT 失败，轴%d，错误码=%d", axis, rtn_zero);
      return false;
    }

    const short rtn_mode = GTN_SetEcatAxisMode(core_id_, axis, 8);
    if (rtn_mode != 0)
    {
      RCLCPP_ERROR_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000,
          "GTN_SetEcatAxisMode 设置位置模式失败，轴%d，错误码=%d",
          axis, rtn_mode);
      return false;
    }

    unsigned short actual_mode = static_cast<unsigned short>(-1);
    short rtn_get_mode = 0;
    bool mode_ok = false;

    // EtherCAT 轴模式切换不是严格同步完成的。
    // SetEcatAxisMode 返回成功后，立即 GetEcatAxisMode 可能仍读到旧模式。
    // 因此这里短暂轮询，避免把正常的切换延迟误判为失败。
    for (int retry = 0; retry < 8; ++retry)
    {
      rtn_get_mode = GTN_GetEcatAxisMode(core_id_, axis, &actual_mode);
      if (rtn_get_mode == 0 && actual_mode == 8)
      {
        mode_ok = true;
        break;
      }

      std::this_thread::sleep_for(5ms);
    }

    if (!mode_ok)
    {
      if (rtn_get_mode != 0)
      {
        RCLCPP_ERROR_THROTTLE(
            this->get_logger(), *this->get_clock(), 2000,
            "读取轴%d当前模式失败，错误码=%d", axis, rtn_get_mode);
      }
      else
      {
        RCLCPP_ERROR_THROTTLE(
            this->get_logger(), *this->get_clock(), 2000,
            "轴%d 期望切到位置模式，但当前模式=%d", axis, actual_mode);
      }
      return false;
    }

    int32_t enc_pos = 0;
    const short rtn_get_enc = GTN_GetEcatEncPos(core_id_, axis, &enc_pos);
    if (rtn_get_enc != 0)
    {
      RCLCPP_ERROR_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000,
          "读取轴%d编码器位置失败，错误码=%d", axis, rtn_get_enc);
      return false;
    }

    const short rtn_set_prf = GTN_SetPrfPos(core_id_, axis, enc_pos);
    if (rtn_set_prf != 0)
    {
      RCLCPP_ERROR_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000,
          "同步轴%d规划位置失败，错误码=%d", axis, rtn_set_prf);
      return false;
    }

    const short rtn_set_enc = GTN_SetEncPos(core_id_, axis, enc_pos);
    if (rtn_set_enc != 0)
    {
      RCLCPP_ERROR_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000,
          "同步轴%d编码器位置失败，错误码=%d", axis, rtn_set_enc);
      return false;
    }

    // ===================== 新增：同步 axis / profile / encoder =====================
    const short rtn_sync =
        GTN_SynchAxisPos(core_id_, static_cast<long>(1L << (axis - 1)));
    if (rtn_sync != 0)
    {
      RCLCPP_ERROR_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000,
          "同步轴%d axis/profile/encoder 失败，错误码=%d", axis, rtn_sync);
      return false;
    }

    axis_in_torque_mode_[i] = false;
    last_target_torque_[i] = 0.0;
    command_sent_[i] = false;
    position_profile_configured_[i] = false;
    return true;
  }

  // ===================== 单轴切到扭矩模式 =====================
  bool ensureAxisInTorqueMode(std::size_t i)
  {
    if (i >= upperlimb_robot::AXIS_NUM)
    {
      return false;
    }

    if (!isControlledAxisIndex(i))
    {
      return false;
    }

    const int axis = axis_ids_[i];

    const short rtn_stop =
        GTN_Stop(core_id_, static_cast<long>(1L << (axis - 1)), 0);
    if (rtn_stop != 0)
    {
      RCLCPP_ERROR_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000,
          "切到扭矩模式前停止轴%d失败，错误码=%d", axis, rtn_stop);
      return false;
    }

    const short rtn_mode = GTN_SetEcatAxisMode(core_id_, axis, 10);
    if (rtn_mode != 0)
    {
      RCLCPP_ERROR_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000,
          "GTN_SetEcatAxisMode 设置扭矩模式失败，轴%d，错误码=%d",
          axis, rtn_mode);
      return false;
    }

    unsigned short actual_mode = static_cast<unsigned short>(-1);
    short rtn_get_mode = 0;
    bool mode_ok = false;

    // EtherCAT 轴模式切换有生效延迟。
    // 日志中出现“先读到 8，随后恢复时又读到 10”，说明不能只读一次就判失败。
    for (int retry = 0; retry < 8; ++retry)
    {
      rtn_get_mode = GTN_GetEcatAxisMode(core_id_, axis, &actual_mode);
      if (rtn_get_mode == 0 && actual_mode == 10)
      {
        mode_ok = true;
        break;
      }

      std::this_thread::sleep_for(5ms);
    }

    if (!mode_ok)
    {
      if (rtn_get_mode != 0)
      {
        RCLCPP_ERROR_THROTTLE(
            this->get_logger(), *this->get_clock(), 2000,
            "读取轴%d当前模式失败，错误码=%d", axis, rtn_get_mode);
      }
      else
      {
        RCLCPP_ERROR_THROTTLE(
            this->get_logger(), *this->get_clock(), 2000,
            "轴%d 期望切到扭矩模式，但当前模式=%d", axis, actual_mode);
      }
      return false;
    }

    axis_in_torque_mode_[i] = true;
    command_sent_[i] = false;
    position_profile_configured_[i] = false;
    return true;
  }

  // ===================== 清除全部位置命令缓存 =====================
  //
  // 作用：
  // 1. 强制下一轮位置目标重新发送
  // 2. 强制下一轮重新配置 Trap 参数
  //
  void invalidateAllPositionCommandCache()
  {
    for (std::size_t i = 0; i < upperlimb_robot::AXIS_NUM; ++i)
    {
      command_sent_[i] = false;
      position_profile_configured_[i] = false;
    }
  }

  // ===================== 停止全部轴当前运动（不下使能） =====================
  //
  // 作用：
  // 切换位置模式时，先把残余位置运动停掉，
  // 避免旧的正弦轨迹继续影响下一模式。
  //
  void stopAllAxesMotionOnly()
  {
    long mask = 0;

    for (std::size_t i = 0; i < upperlimb_robot::AXIS_NUM; ++i)
    {
      if (!isControlledAxisIndex(i))
      {
        continue;
      }

      if (!axis_enabled_[i])
      {
        continue;
      }

      const int axis = axis_ids_[i];
      mask |= (1L << (axis - 1));
    }

    if (mask == 0)
    {
      return;
    }

    const short rtn = GTN_Stop(core_id_, mask, 0);
    if (rtn != 0)
    {
      RCLCPP_ERROR_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000,
          "GTN_Stop 停止当前运动失败，mask=0x%lx，错误码=%d",
          mask, rtn);
    }
  }

  // ===================== 全部轴恢复到位置模式 =====================
  //
  // 作用：
  // 1. 对当前仍停留在扭矩模式的轴，显式切回位置模式
  // 2. 清除位置命令缓存与 Trap 配置缓存
  // 3. 保证下一次位置模式下发时，目标和参数一定重新发送
  //
  void restoreAllAxesToPositionMode()
  {
    for (std::size_t i = 0; i < upperlimb_robot::AXIS_NUM; ++i)
    {
      if (!isControlledAxisIndex(i))
      {
        axis_in_torque_mode_[i] = false;
        command_sent_[i] = false;
        position_profile_configured_[i] = false;
        continue;
      }

      const int axis = axis_ids_[i];

      unsigned short actual_mode = -1;
      const short rtn_get_mode = GTN_GetEcatAxisMode(core_id_, axis, &actual_mode);
      if (rtn_get_mode == 0)
      {
        axis_in_torque_mode_[i] = (actual_mode == 10);
      }

      const bool need_force_restore = (i == 17 || i == 18 || i == 19);

      if (!axis_in_torque_mode_[i] && !need_force_restore)
      {
        continue;
      }

      (void)ensureAxisInPositionMode(i);
    }

    invalidateAllPositionCommandCache();
  }

  // ===================== 执行位置输出 =====================
  //
  // 设计要点：
  // 1. 不再在每次位置变化时都重复调用 GTN_PrfTrap / GTN_SetTrapPrm / GTN_SetVel
  // 2. 只有在“第一次启用该轴位置控制”或“轨迹参数变化”时，才重新配置 Trap
  // 3. 位置模式第一次接管该轴时，无论该轴此前是否在扭矩模式，
  //    都重新同步当前编码器位置 -> 规划位置，避免规划起点陈旧
  //
  void applyPositionOutput(const upperlimb_robot::ControlOutputCache &output)
  {
    long update_mask = 0;
    const bool dual_arm_torque_mode = isDualArmTorqueMode(applied_mode_);

    for (std::size_t i = 0; i < upperlimb_robot::AXIS_NUM; ++i)
    {
      if (!isControlledAxisIndex(i))
      {
        command_sent_[i] = false;
        position_profile_configured_[i] = false;
        last_target_torque_[i] = 0.0;
        continue;
      }

      // 纯双臂扭矩模式下，算法层只输出14个主动轴扭矩。
      // 末端 fixed 轴 8~10 / 18~20 不在控制循环内做位置保持、模式读取或 PT 清理，
      // 避免继续触发 SetPos / SetTrapPrm / SetVel / SyncAxisPos 类错误。
      if (dual_arm_torque_mode && output.use_torque[i])
      {
        command_sent_[i] = false;
        position_profile_configured_[i] = false;
        continue;
      }

      if (dual_arm_torque_mode && !output.use_position[i])
      {
        command_sent_[i] = false;
        position_profile_configured_[i] = false;
        continue;
      }

      const int axis = axis_ids_[i];

      unsigned short actual_mode = static_cast<unsigned short>(-1);
      const short rtn_get_mode = GTN_GetEcatAxisMode(core_id_, axis, &actual_mode);
      if (rtn_get_mode == 0)
      {
        axis_in_torque_mode_[i] = (actual_mode == 10);
      }

      if (applied_mode_ == upperlimb_robot::MODE_EEG_CONTROL &&
          (rtn_get_mode != 0 || actual_mode != 8))
      {
        setEegFault("EEG 位置输出前驱动模式异常");
        return;
      }

      if (output.use_torque[i])
      {
        command_sent_[i] = false;
        position_profile_configured_[i] = false;
        continue;
      }

      if (!output.use_position[i])
      {
        command_sent_[i] = false;
        position_profile_configured_[i] = false;

        if (axis_in_torque_mode_[i])
        {
          (void)ensureAxisInPositionMode(i);
        }
        continue;
      }

      if (axis_in_torque_mode_[i])
      {
        if (!ensureAxisInPositionMode(i))
        {
          continue;
        }
      }

      const double target_deg = output.target_position[i];
      const double target_pulse =
          upperlimb_robot::jointDegToPulse(axis, target_deg);
      if (applied_mode_ == upperlimb_robot::MODE_EEG_CONTROL &&
          (!std::isfinite(target_pulse) ||
           target_pulse < static_cast<double>(std::numeric_limits<int32_t>::min()) ||
           target_pulse > static_cast<double>(std::numeric_limits<int32_t>::max())))
      {
        setEegFault("EEG 目标位置换算异常");
        return;
      }

      const double vel_deg = output.target_velocity[i];
      const double acc_deg = output.target_acc[i];
      const double dec_deg = output.target_dec[i];
      const short smooth_time = output.target_smooth_time[i];

      bool need_reconfigure_profile = false;

      if (!position_profile_configured_[i])
      {
        need_reconfigure_profile = true;
      }
      else
      {
        if (std::abs(vel_deg - last_profile_velocity_[i]) > 1e-6 ||
            std::abs(acc_deg - last_profile_acc_[i]) > 1e-6 ||
            std::abs(dec_deg - last_profile_dec_[i]) > 1e-6 ||
            smooth_time != last_profile_smooth_time_[i])
        {
          need_reconfigure_profile = true;
        }
      }

      if (need_reconfigure_profile)
      {
        int32_t enc_pos = 0;
        short rtn = GTN_GetEcatEncPos(core_id_, axis, &enc_pos);
        if (rtn != 0)
        {
          RCLCPP_ERROR_THROTTLE(
              this->get_logger(), *this->get_clock(), 2000,
              "读取轴%d编码器位置失败，错误码=%d", axis, rtn);
          continue;
        }

        rtn = GTN_SetPrfPos(core_id_, axis, enc_pos);
        if (rtn != 0)
        {
          RCLCPP_ERROR_THROTTLE(
              this->get_logger(), *this->get_clock(), 2000,
              "同步轴%d规划位置失败，错误码=%d", axis, rtn);
          continue;
        }

        rtn = GTN_SetEncPos(core_id_, axis, enc_pos);
        if (rtn != 0)
        {
          RCLCPP_ERROR_THROTTLE(
              this->get_logger(), *this->get_clock(), 2000,
              "同步轴%d编码器位置失败，错误码=%d", axis, rtn);
          continue;
        }

        // ===================== 新增：同步 axis / profile / encoder =====================
        rtn = GTN_SynchAxisPos(core_id_, static_cast<long>(1L << (axis - 1)));
        if (rtn != 0)
        {
          RCLCPP_ERROR_THROTTLE(
              this->get_logger(), *this->get_clock(), 2000,
              "同步轴%d axis/profile/encoder 失败，错误码=%d", axis, rtn);
          continue;
        }

        rtn = GTN_PrfTrap(core_id_, axis);
        if (rtn != 0)
        {
          RCLCPP_ERROR_THROTTLE(
              this->get_logger(), *this->get_clock(), 2000,
              "GTN_PrfTrap 轴%d失败，错误码=%d", axis, rtn);
          continue;
        }

        TTrapPrm trap{};
        trap.acc =
            upperlimb_robot::jointAccelDegPerSec2ToPulsePerMs2(axis, acc_deg);
        trap.dec =
            upperlimb_robot::jointAccelDegPerSec2ToPulsePerMs2(axis, dec_deg);
        trap.smoothTime = smooth_time;
        trap.velStart = 0;

        rtn = GTN_SetTrapPrm(core_id_, axis, &trap);
        if (rtn != 0)
        {
          RCLCPP_ERROR_THROTTLE(
              this->get_logger(), *this->get_clock(), 2000,
              "GTN_SetTrapPrm 轴%d失败，错误码=%d", axis, rtn);
          continue;
        }

        rtn = GTN_SetVel(
            core_id_,
            axis,
            upperlimb_robot::jointSpeedDegPerSecToPulsePerMs(axis, vel_deg));
        if (rtn != 0)
        {
          RCLCPP_ERROR_THROTTLE(
              this->get_logger(), *this->get_clock(), 2000,
              "GTN_SetVel 轴%d失败，错误码=%d", axis, rtn);
          continue;
        }

        position_profile_configured_[i] = true;
        last_profile_velocity_[i] = vel_deg;
        last_profile_acc_[i] = acc_deg;
        last_profile_dec_[i] = dec_deg;
        last_profile_smooth_time_[i] = smooth_time;
      }

      if (command_sent_[i] &&
          std::abs(target_deg - last_target_position_[i]) < position_epsilon_)
      {
        continue;
      }

      const short rtn_set_pos =
          GTN_SetPos(core_id_, axis, static_cast<int32_t>(target_pulse));
      if (rtn_set_pos != 0)
      {
        RCLCPP_ERROR_THROTTLE(
            this->get_logger(), *this->get_clock(), 2000,
            "GTN_SetPos 轴%d失败，错误码=%d", axis, rtn_set_pos);
        continue;
      }

      update_mask |= (1L << (axis - 1));
      last_target_position_[i] = target_deg;
      command_sent_[i] = true;
    }

    if (update_mask != 0)
    {
      const short rtn = GTN_Update(core_id_, update_mask);
      if (rtn != 0)
      {
        RCLCPP_ERROR_THROTTLE(
            this->get_logger(), *this->get_clock(), 2000,
            "GTN_Update 失败，mask=0x%lx，错误码=%d",
            update_mask, rtn);
      }
    }
  }

  // ===================== 执行扭矩输出 =====================
  //
  // 1. use_torque == true 时，确保切到扭矩模式
  // 2. use_torque == false 时，如果该轴还停留在扭矩模式，则清零并切回位置模式
  //
  void applyTorqueOutput(const upperlimb_robot::ControlOutputCache &output)
  {
    const bool dual_arm_torque_mode = isDualArmTorqueMode(applied_mode_);

    for (std::size_t i = 0; i < upperlimb_robot::AXIS_NUM; ++i)
    {
      if (!isControlledAxisIndex(i))
      {
        axis_in_torque_mode_[i] = false;
        last_target_torque_[i] = 0.0;
        continue;
      }

      // 纯双臂扭矩模式下，非14个主动扭矩轴完全跳过。
      if (dual_arm_torque_mode && !output.use_torque[i])
      {
        last_target_torque_[i] = 0.0;
        continue;
      }

      const int axis = axis_ids_[i];

      unsigned short actual_mode = static_cast<unsigned short>(-1);
      const short rtn_get_mode =
          GTN_GetEcatAxisMode(core_id_, axis, &actual_mode);
      if (rtn_get_mode == 0)
      {
        axis_in_torque_mode_[i] = (actual_mode == 10);
      }

      // 当前轴不再请求扭矩控制时，先清零PT，再恢复位置模式。
      if (!output.use_torque[i])
      {
        if (axis_in_torque_mode_[i])
        {
          const short rtn_zero = GTN_SetEcatAxisPT(core_id_, axis, 0);
          if (rtn_zero != 0)
          {
            RCLCPP_ERROR_THROTTLE(
                this->get_logger(), *this->get_clock(), 2000,
                "GTN_SetEcatAxisPT 清零失败，轴%d，错误码=%d", axis, rtn_zero);
          }

          if (!dual_arm_torque_mode)
          {
            (void)ensureAxisInPositionMode(i);
          }
        }

        last_target_torque_[i] = 0.0;
        continue;
      }

      const double test_raw = upperlimb_robot::torqueNmToRaw(axis, 1.0);
      if (test_raw == 0.0)
      {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *this->get_clock(), 2000,
            "轴%d 当前没有有效扭矩标定系数，跳过扭矩输出", axis);
        continue;
      }

      const double target_torque_nm = output.target_torque[i];
      if (!std::isfinite(target_torque_nm))
      {
        RCLCPP_ERROR_THROTTLE(
            this->get_logger(), *this->get_clock(), 2000,
            "轴%d 目标扭矩为NaN或Inf，本轮不发送", axis);
        continue;
      }

      // 先确保轴已经处于扭矩模式，再判断是否需要重复发送。
      // 这样第一帧即使目标扭矩与缓存值相同，也不会漏掉模式切换。
      bool switched_to_torque_mode = false;

      if (!axis_in_torque_mode_[i])
      {
        if (dual_arm_torque_mode)
        {
          RCLCPP_ERROR_THROTTLE(
              this->get_logger(), *this->get_clock(), 2000,
              "双臂扭矩模式下轴%d 未处于扭矩模式，本轮跳过 PT；不在控制循环内反复切模式",
              axis);
          continue;
        }

        if (!ensureAxisInTorqueMode(i))
        {
          RCLCPP_ERROR_THROTTLE(
              this->get_logger(), *this->get_clock(), 2000,
              "轴%d 切入扭矩模式失败，本轮不发送PT", axis);
          continue;
        }

        switched_to_torque_mode = true;
      }

      if (!switched_to_torque_mode &&
          std::abs(target_torque_nm - last_target_torque_[i]) < torque_epsilon_)
      {
        continue;
      }

      const double torque_raw_double =
          upperlimb_robot::torqueNmToRaw(axis, target_torque_nm);

      if (!std::isfinite(torque_raw_double))
      {
        RCLCPP_ERROR_THROTTLE(
            this->get_logger(), *this->get_clock(), 2000,
            "轴%d 扭矩单位换算结果非法，本轮不发送", axis);
        continue;
      }

      const short torque_raw =
          static_cast<short>(std::lround(torque_raw_double));

      const short rtn_pt = GTN_SetEcatAxisPT(core_id_, axis, torque_raw);
      if (rtn_pt != 0)
      {
        RCLCPP_ERROR_THROTTLE(
            this->get_logger(), *this->get_clock(), 2000,
            "GTN_SetEcatAxisPT 发送扭矩失败，轴%d，目标扭矩=%.4f Nm，raw=%d，错误码=%d",
            axis, target_torque_nm, torque_raw, rtn_pt);
        continue;
      }

      last_target_torque_[i] = target_torque_nm;
    }
  }

  // ===================== 清零全部扭矩命令 =====================
  void clearAllTorqueCommands()
  {
    for (std::size_t i = 0; i < upperlimb_robot::AXIS_NUM; ++i)
    {
      if (!isControlledAxisIndex(i))
      {
        axis_in_torque_mode_[i] = false;
        last_target_torque_[i] = 0.0;
        continue;
      }

      const int axis = axis_ids_[i];

      unsigned short actual_mode = -1;
      const short rtn_get_mode = GTN_GetEcatAxisMode(core_id_, axis, &actual_mode);
      if (rtn_get_mode == 0)
      {
        axis_in_torque_mode_[i] = (actual_mode == 10);
      }

      if (!axis_in_torque_mode_[i])
      {
        continue;
      }
      const short rtn = GTN_SetEcatAxisPT(core_id_, axis, 0);
      if (rtn != 0)
      {
        RCLCPP_ERROR_THROTTLE(
            this->get_logger(), *this->get_clock(), 2000,
            "清零扭矩命令失败，轴%d，错误码=%d", axis, rtn);
      }

      last_target_torque_[i] = 0.0;
    }
  }

  // ===================== 停止全部轴 =====================
  bool stopAllAxesRaw()
  {
    if (!controller_opened_)
    {
      return true;
    }

    bool all_ok = true;
    int failed_axes = 0;

    for (std::size_t i = 0; i < upperlimb_robot::AXIS_NUM; ++i)
    {
      if (!isControlledAxisIndex(i))
      {
        continue;
      }

      const int axis = axis_ids_[i];

      if (!axis_enabled_[i])
      {
        continue;
      }

      short rtn_stop =
          GTN_Stop(core_id_, static_cast<long>(1L << (axis - 1)), 0);
      if (rtn_stop != 0)
      {
        fprintf(stderr, "停止轴%d失败，错误码=%d\n", axis, rtn_stop);
        fflush(stderr);
        all_ok = false;
        failed_axes++;
      }

      short rtn_off = GTN_AxisOff(core_id_, axis);
      if (rtn_off != 0)
      {
        fprintf(stderr, "关闭轴%d使能失败，错误码=%d\n", axis, rtn_off);
        fflush(stderr);
        all_ok = false;
        failed_axes++;
      }
      else
      {
        axis_enabled_[i] = false;
      }
    }

    if (!all_ok)
    {
      fprintf(stderr, "共有 %d 个轴关闭过程出现错误。\n", failed_axes);
      fflush(stderr);
    }

    return all_ok;
  }

  // ===================== JointState 转换 =====================
  //
  // URDF 定义：
  // joint1  ~ joint10  -> 左臂
  // joint11 ~ joint20  -> 右臂
  //
  // 控制层内部状态：
  // state[10~19] -> 左臂
  // state[0~9]   -> 右臂
  //
  sensor_msgs::msg::JointState convertToJointStateMsg(
      const upperlimb_robot::MotorStateCache &state)
  {
    sensor_msgs::msg::JointState msg;
    msg.header.stamp = this->now();

    msg.name = {
        "joint1", "joint2", "joint3", "joint4", "joint5",
        "joint6", "joint7", "joint8", "joint9", "joint10",
        "joint11", "joint12", "joint13", "joint14", "joint15",
        "joint16", "joint17", "joint18", "joint19", "joint20"};

    msg.position.resize(20);
    msg.velocity.resize(20);
    msg.effort.resize(20);

    constexpr double deg_to_rad = M_PI / 180.0;

    for (std::size_t src = 10, dst = 0; src < 20; ++src, ++dst)
    {
      msg.position[dst] = state.position[src] * deg_to_rad;
      msg.velocity[dst] = state.velocity[src] * deg_to_rad;
      msg.effort[dst] = state.torque[src];
    }

    for (std::size_t src = 0, dst = 10; src < 10; ++src, ++dst)
    {
      msg.position[dst] = state.position[src] * deg_to_rad;
      msg.velocity[dst] = state.velocity[src] * deg_to_rad;
      msg.effort[dst] = state.torque[src];
    }

    return msg;
  }

private:
  rclcpp::Subscription<upperlimb_robot::msg::ControlCommand>::SharedPtr control_command_sub_;
  rclcpp::Publisher<upperlimb_robot::msg::SystemState>::SharedPtr system_pub_;
  rclcpp::Publisher<upperlimb_robot::msg::RobotState>::SharedPtr robot_state_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
  rclcpp::TimerBase::SharedPtr state_timer_;

  bool joint_init_ = false;
  bool controller_init_ = false;
  bool system_ready_ = false;

  bool controller_opened_ = false;
  bool ecat_started_ = false;
  bool shutdown_done_ = false;

  std::atomic<bool> running_{false};
  std::atomic<bool> state_ready_{false};

  std::thread state_read_thread_;
  std::thread control_thread_;

  std::chrono::steady_clock::time_point last_control_time_{};
  bool last_control_time_valid_ = false;

  std::atomic<int> current_mode_{upperlimb_robot::MODE_IDLE};
  std::atomic<int> status_mode_{upperlimb_robot::MODE_IDLE};
  int applied_mode_{upperlimb_robot::MODE_IDLE};

  bool mode_enter_failed_ = false;
  int mode_enter_fallback_mode_ = upperlimb_robot::MODE_IDLE;

  std::atomic<int> state_hz_{50};
  std::atomic<int> control_hz_{50};

  int status_read_divider_ = 20;
  int status_read_divider_count_ = 0;

  std::mutex state_mutex_;
  upperlimb_robot::MotorStateCache latest_motor_state_;

  std::mutex sensor_mutex_;
  upperlimb_robot::SensorStateCache latest_sensor_state_;



  // GTN/EtherCAT 底层库互斥锁：状态采集线程和控制线程必须串行访问控制卡库。
  std::mutex gtn_mutex_;

  int publish_divider_ = 10;
  int publish_divider_count_ = 0;

  int core_id_ = 1;

  std::array<int, upperlimb_robot::AXIS_NUM> axis_ids_ = {
      1, 2, 3, 4, 5,
      6, 7, 8, 9, 10,
      11, 12, 13, 14, 15,
      16, 17, 18, 19, 20};

  std::array<int32_t, upperlimb_robot::AXIS_NUM> enc_pos_{};
  std::array<bool, upperlimb_robot::AXIS_NUM> axis_enabled_{};

  // ===================== 位置命令缓存 =====================
  //
  // last_target_position_ :
  // 记录上一次发送的位置目标（单位 deg）
  //
  // command_sent_ :
  // 标记该轴是否已经发送过位置命令
  //
  std::array<double, upperlimb_robot::AXIS_NUM> last_target_position_{};
  std::array<bool, upperlimb_robot::AXIS_NUM> command_sent_{};

  // ===================== 扭矩模式状态缓存 =====================
  //
  // axis_in_torque_mode_ :
  // 标记该轴当前是否处于扭矩模式
  //
  // last_target_torque_ :
  // 记录上一次发送的扭矩目标（单位 Nm）
  //
  std::array<bool, upperlimb_robot::AXIS_NUM> axis_in_torque_mode_{};
  std::array<double, upperlimb_robot::AXIS_NUM> last_target_torque_{};

  // ===================== 位置规划参数缓存 =====================
  //
  // 作用：
  // 1. 避免在连续轨迹模式下，每个周期都重复配置 Trap 参数
  // 2. 只有在速度/加速度/减速度/平滑时间变化时，才重新配置
  //
  std::array<bool, upperlimb_robot::AXIS_NUM> position_profile_configured_{};
  std::array<double, upperlimb_robot::AXIS_NUM> last_profile_velocity_{};
  std::array<double, upperlimb_robot::AXIS_NUM> last_profile_acc_{};
  std::array<double, upperlimb_robot::AXIS_NUM> last_profile_dec_{};
  std::array<short, upperlimb_robot::AXIS_NUM> last_profile_smooth_time_{};

  double position_epsilon_ = 0.05;
  double torque_epsilon_ = 0.01;
};

static std::shared_ptr<RobotControlNode> g_node = nullptr;

void signal_handler(int signum)
{
  fprintf(stderr, "\n收到退出信号(%d)，开始安全关闭...\n", signum);
  fflush(stderr);

  if (g_node)
  {
    g_node->shutdownHardware();
  }

  rclcpp::shutdown();
}

int main(int argc, char const *argv[])
{
  rclcpp::init(argc, argv);

  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  g_node = std::make_shared<RobotControlNode>();

  rclcpp::spin(g_node);

  if (g_node)
  {
    g_node->shutdownHardware();
    g_node.reset();
  }

  return 0;
}
