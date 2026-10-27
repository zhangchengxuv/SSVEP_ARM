#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

namespace upperlimb_robot
{

  // 串口一帧的解析结果；序号是否新鲜由接收端结合上一次序号判断。
  struct EegResult
  {
    std::uint64_t sequence = 0;
    bool test = false;
    int target = 0;
    double frequency_hz = 0.0;
  };

  // 输入为完整的 ASCII 帧，可带 CRLF；失败时不修改 result。
  bool parseEegResult(const std::string &frame, EegResult &result);
  int eegTargetAxisIndex(int target);

  // ===================== 系统常量定义 =====================

  // 总轴数
  // 控制层内部数组索引统一使用 0~19
  static constexpr std::size_t AXIS_NUM = 20;

  // 六维力传感器数量
  // 传感器数组索引统一使用 0~3
  static constexpr std::size_t FORCE_SENSOR_NUM = 4;

  // ===================== 统一重力补偿参数 =====================
  //
  // 所有使用 Pinocchio 重力力矩的控制算法都共用这里的参数。
  // 左右臂分别配置，并且每条臂的 7 个主动关节分别配置。
  //
  // 修正公式：
  //   tau_gravity[i] = gain[i] * tau_gravity_model[i] + bias_nm[i]
  //
  // 参数顺序：
  //   左臂：joint1  ~ joint7
  //   右臂：joint11 ~ joint17
  //
  // 注意：bias_nm 使用 Pinocchio 模型方向，右臂最终发送时的整体负号
  // 仍由各控制算法原有的驱动方向映射负责。
  struct GravityCompensationParameters
  {
    std::array<double, 7> gain{};
    std::array<double, 7> bias_nm{};
  };

  extern const GravityCompensationParameters LEFT_ARM_GRAVITY_COMPENSATION;
  extern const GravityCompensationParameters RIGHT_ARM_GRAVITY_COMPENSATION;

  // ===================== 统一关节摩擦补偿参数 =====================
  //
  // 所有扭矩控制算法共用这里的摩擦补偿参数。
  // 左右臂分别配置，并且每条臂的 7 个主动关节分别配置。
  //
  // 补偿公式：
  //   tau_friction[i]
  //       = coulomb_nm[i] * tanh(dq[i] / velocity_smooth_rad_s[i])
  //       + viscous_nm_per_rad_s[i] * dq[i]
  //
  // 参数含义：
  //   coulomb_nm              -> 低速近似恒定的库仑摩擦补偿，单位 Nm；
  //   viscous_nm_per_rad_s    -> 随速度增加的粘性摩擦补偿，单位 Nm/(rad/s)；
  //   velocity_smooth_rad_s   -> 零速度附近的方向平滑范围，单位 rad/s。
  //
  // tau_friction 与实际关节速度方向相同，用于抵消机构摩擦。
  // 右臂最终发送时的整体负号仍由各控制算法原有的驱动方向映射负责。
  struct FrictionCompensationParameters
  {
    std::array<double, 7> coulomb_nm{};
    std::array<double, 7> viscous_nm_per_rad_s{};
    std::array<double, 7> velocity_smooth_rad_s{};
  };

  extern const FrictionCompensationParameters LEFT_ARM_FRICTION_COMPENSATION;
  extern const FrictionCompensationParameters RIGHT_ARM_FRICTION_COMPENSATION;

  // ===================== 全局六维力传感器交互反馈增益 =====================
  // 所有采用真实力传感器参与控制的算法共用；不改变原始采样、零偏和实验测量值。
  // 1.00 表示不缩放，无量纲。具体数值集中在 control_algorithms.cpp 顶部修改。
  extern const double LEFT_ARM_FORCE_FEEDBACK_GAIN;
  extern const double LEFT_HAND_FORCE_FEEDBACK_GAIN;
  extern const double RIGHT_ARM_FORCE_FEEDBACK_GAIN;
  extern const double RIGHT_HAND_FORCE_FEEDBACK_GAIN;

  // ===================== 控制模式定义 =====================
  //
  // 说明：
  // 1. current_mode_ 表示“请求模式”
  // 2. applied_mode_ 表示“已经完成进入动作的模式”
  // 3. 真正的模式进入/退出动作建议在 onModeEnter()/onModeExit() 中执行
  //
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

  // ===================== 电机状态缓存 =====================
  //
  // 对外统一使用真实物理量：
  // position : deg
  // velocity : deg/s
  // torque   : Nm
  //
  // 注意：
  // 1. 这里的下标是“控制层数组索引 0~19”
  // 2. 不一定等于 GTN 物理轴号 1~20
  // 3. 真正访问驱动时，应通过 axis_ids_[i] 做映射
  //
  struct MotorStateCache
  {
    // 数组索引 0~19
    // 关节位置，单位 deg
    std::array<double, AXIS_NUM> position{};

    // 数组索引 0~19
    // 关节速度，单位 deg/s
    std::array<double, AXIS_NUM> velocity{};

    // 数组索引 0~19
    // 关节扭矩，单位 Nm
    std::array<double, AXIS_NUM> torque{};

    // 数组索引 0~19
    // 轴是否使能
    std::array<bool, AXIS_NUM> enable{};
  };

  // ===================== 单个六维力传感器 =====================
  //
  // 力单位：N
  // 力矩单位：Nm
  //
  struct ForceSensorData
  {
    double fx{0.0};
    double fy{0.0};
    double fz{0.0};
    double mx{0.0};
    double my{0.0};
    double mz{0.0};
  };

  // ===================== 传感器索引定义 =====================
  //
  // sensor[0~3] / bias[0~3] 的索引约定
  //
  enum ForceSensorIndex : std::size_t
  {
    SENSOR_RIGHT_HAND = 0,
    SENSOR_RIGHT_ARM = 1,
    SENSOR_LEFT_HAND = 2,
    SENSOR_LEFT_ARM = 3,
  };

  // ===================== 传感器状态缓存 =====================
  //
  // sensor : 当前实时值
  // bias   : 已记录的零偏
  // valid  : 当前实时值是否有效
  //
  // 说明：
  // valid == true 表示本轮读取成功
  // valid == false 时，不应拿当前 sensor 直接记录 bias
  //
  struct SensorStateCache
  {
    // 数组索引 0~3
    // 当前实时传感器值
    std::array<ForceSensorData, FORCE_SENSOR_NUM> sensor{};

    // 数组索引 0~3
    // 零偏值
    std::array<ForceSensorData, FORCE_SENSOR_NUM> bias{};

    // 当前传感器缓存是否有效
    bool valid{false};
  };

  // ===================== 控制输出缓存 =====================
  //
  // 这是“算法层 -> 执行层”的统一输出接口。
  //
  // 约定：
  // 1. 数组索引统一为 0~19
  // 2. 算法层统一使用真实物理量
  // 3. target_torque 的单位是 Nm，不是 raw
  // 4. 正常写法为：output.target_torque[6] = 2.0;（2.0 Nm）
  // 5. 允许不同轴使用不同控制方式，例如左臂位置模式、右臂扭矩模式；
  //    但同一个轴不能同时把 use_position 和 use_torque 设为 true。
  //
  struct ControlOutputCache
  {
    // 数组索引 0~19
    // 目标位置，单位 deg
    std::array<double, AXIS_NUM> target_position{};

    // 数组索引 0~19
    // 目标速度，单位 deg/s
    std::array<double, AXIS_NUM> target_velocity{};

    // 数组索引 0~19
    // 目标扭矩，单位 Nm
    // 正常写法为：output.target_torque[6] = 2.0;（2.0 Nm）
    std::array<double, AXIS_NUM> target_torque{};

    // 数组索引 0~19
    // 目标加速度，单位 deg/s^2
    std::array<double, AXIS_NUM> target_acc{};

    // 数组索引 0~19
    // 目标减速度，单位 deg/s^2
    std::array<double, AXIS_NUM> target_dec{};

    // 数组索引 0~19
    // 平滑时间，单位由底层驱动接口定义
    std::array<short, AXIS_NUM> target_smooth_time{};

    // 数组索引 0~19
    // 是否对该轴启用位置控制
    std::array<bool, AXIS_NUM> use_position{};

    // 数组索引 0~19
    // 是否显式使用速度字段
    // 当前版本主要作为语义保留
    std::array<bool, AXIS_NUM> use_velocity{};

    // 数组索引 0~19
    // 是否对该轴启用扭矩控制
    std::array<bool, AXIS_NUM> use_torque{};

    // 输出是否有效
    // valid == false 时，执行层应跳过本轮控制输出
    bool valid{false};

  };

  // ===================== 单位换算接口 =====================
  //
  // 注意：
  // 这些换算函数传入的是“真实物理轴号 axis（1~20）”
  // 不是控制层数组索引 i（0~19）
  //
  double jointDegToPulse(int axis, double joint_deg);
  double pulseToJointDeg(int axis, double pulse);

  double jointSpeedDegPerSecToPulsePerMs(int axis, double joint_speed_deg_per_sec);
  double pulsePerMsToJointDegPerSec(int axis, double pulse_per_ms);

  double jointAccelDegPerSec2ToPulsePerMs2(int axis, double joint_acc_deg_per_sec2);
  double pulsePerMs2ToJointDegPerSec2(int axis, double pulse_per_ms2);

  // raw -> Nm
  // 用于“读取驱动反馈扭矩”
  double torqueRawToNm(int axis, double raw);

  // Nm -> raw
  // 用于“下发驱动扭矩命令”
  double torqueNmToRaw(int axis, double torque_nm);

  // 保留原有预设位姿与零力算法；EEG 复用同一位置输出结构。
  void runIdle(const MotorStateCache &state, ControlOutputCache &output);
  void runLeftPresetPosition(const MotorStateCache &state, ControlOutputCache &output);
  void runRightPresetPosition(const MotorStateCache &state, ControlOutputCache &output);
  void runZeroForceMode(const MotorStateCache &state, const SensorStateCache &sensor_state, ControlOutputCache &output);
  void runRightZeroForceMode(const MotorStateCache &state, const SensorStateCache &sensor_state, ControlOutputCache &output);
  void runBilateralZeroForceMode(const MotorStateCache &state, const SensorStateCache &sensor_state, ControlOutputCache &output);
  void runEegPresetPosition(const MotorStateCache &state, ControlOutputCache &output);
  bool runEegAction(const MotorStateCache &state, int target, double delta_deg,
                    bool returning, double velocity_deg_s, double acceleration_deg_s2,
                    ControlOutputCache &output);

} // namespace upperlimb_robot
