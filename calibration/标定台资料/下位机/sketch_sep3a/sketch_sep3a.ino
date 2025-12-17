const int STEP_INTERVAL_MICROS = 500;

struct Axis {
  const char* name;
  int stepPin, dirPin, enPin;
  int cwLimitPin, ccwLimitPin;
  int dir;
  long stepsToMove;
  bool isMoving;

  bool lastLimitState;
  int limitTriggerCount;

  Axis(const char* n, int step, int dirPin, int en, int cw, int ccw)
    : name(n), stepPin(step), dirPin(dirPin), enPin(en),
      cwLimitPin(cw), ccwLimitPin(ccw),
      dir(1), stepsToMove(0), isMoving(false),
      lastLimitState(false), limitTriggerCount(0) {}
};

// 四个轴：X, Y, Z, θ
Axis axes[4] = {
  Axis("X", 8, 9, 10, A0, A1),
  Axis("Y", 2, 3, 4, 11, 12),
  Axis("Z", 5, 6, 7, A3, A4),
  Axis("T", A2, 13, A5, -1, -1) // θ轴无限位
};

String inputLine = "";
bool emergencyStopped = false;

// 通用微调任务
struct AdjustTask {
  bool active = false;
  int dir = 1;
  int pulsesPerCycle = 30;
  int repeatCount = 100;
  int completedCount = 0;
  unsigned long lastActionTime = 0;
  bool isStepping = false;
  long stepsRemaining = 0;
  int intervalMillis = 1000;
  int axisIndex = -1; // 对应哪个轴
};

// 四个轴对应的微调任务
AdjustTask adjustTasks[4];

void setup() {
  Serial.begin(9600);
  for (int i = 0; i < 4; i++) {
    pinMode(axes[i].stepPin, OUTPUT);
    pinMode(axes[i].dirPin, OUTPUT);
    pinMode(axes[i].enPin, OUTPUT);
    if (axes[i].cwLimitPin != -1) pinMode(axes[i].cwLimitPin, INPUT);
    if (axes[i].ccwLimitPin != -1) pinMode(axes[i].ccwLimitPin, INPUT);
    digitalWrite(axes[i].enPin, LOW);  // 启用驱动器
  }
  Serial.println("请输入命令示例：");
  Serial.println("  X 1 2000          // 普通移动");
  Serial.println("  XADJ 1 30 100 500 // X轴微调：方向 每次脉冲数 重复次数 间隔ms");
  Serial.println("  YADJ 0 20 50 200  // Y轴微调");
  Serial.println("  ZADJ 1 30 100 500 // Z轴微调");
  Serial.println("  TADJ 1 40 10 1000 // θ轴微调");
  Serial.println("  STOP              // 急停");
  Serial.println("  RESUME            // 解除急停");
}

void loop() {
  // 读取串口输入
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (inputLine.length() > 0) {
        processCommand(inputLine);
        inputLine = "";
      }
    } else {
      inputLine += c;
    }
  }

  if (emergencyStopped) return;

  // 普通轴运动处理
  for (int i = 0; i < 4; i++) {
    Axis& axis = axes[i];
    if (axis.isMoving && axis.stepsToMove > 0) {
      if (checkLimitTriggered(axis)) {
        axis.isMoving = false;
        Serial.print("⚠️ 限位触发，轴 ");
        Serial.print(axis.name);
        Serial.println(" 已强制停止！");
        continue;
      }
      stepOnce(axis);
      delayMicroseconds(STEP_INTERVAL_MICROS * 2);
    } else if (axis.isMoving && axis.stepsToMove <= 0) {
      axis.isMoving = false;
      Serial.print("✅ 轴 ");
      Serial.print(axis.name);
      Serial.println(" 步数完成，已停止");
    }
  }

  // 微调处理
  for (int i = 0; i < 4; i++) handleAdjust(adjustTasks[i]);
}

// 发出一步脉冲
void stepOnce(Axis& axis) {
  digitalWrite(axis.dirPin, axis.dir);
  digitalWrite(axis.stepPin, HIGH);
  delayMicroseconds(STEP_INTERVAL_MICROS);
  digitalWrite(axis.stepPin, LOW);
  axis.stepsToMove--;
}

// 检查限位
bool checkLimitTriggered(Axis& axis) {
  if (axis.cwLimitPin == -1 || axis.ccwLimitPin == -1) return false; // θ轴不检查
  int limitPin = (axis.dir == 1) ? axis.cwLimitPin : axis.ccwLimitPin;
  bool curr = digitalRead(limitPin) == HIGH;

  if (curr && axis.lastLimitState) {
    axis.limitTriggerCount++;
  } else {
    axis.limitTriggerCount = 0;
  }

  axis.lastLimitState = curr;
  return axis.limitTriggerCount >= 1;
}

// 微调逻辑
void handleAdjust(AdjustTask& task) {
  if (!task.active || emergencyStopped) return;
  unsigned long now = millis();

  if (!task.isStepping && (now - task.lastActionTime >= (unsigned long)task.intervalMillis)) {
    if (task.completedCount >= task.repeatCount) {
      task.active = false;
      Serial.print("✅ ");
      Serial.print(axes[task.axisIndex].name);
      Serial.println(" 轴微调完成");
      return;
    }
    Axis& a = axes[task.axisIndex];
    a.dir = task.dir;
    a.stepsToMove = task.pulsesPerCycle;
    a.isMoving = true;
    task.stepsRemaining = task.pulsesPerCycle;
    task.isStepping = true;

    Serial.print("🔄 ");
    Serial.print(a.name);
    Serial.print(" 轴微调 第 ");
    Serial.print(task.completedCount + 1);
    Serial.println(" 次开始");
  }

  if (task.isStepping) {
    Axis& a = axes[task.axisIndex];
    if (a.stepsToMove > 0) {
      stepOnce(a);
      delayMicroseconds(STEP_INTERVAL_MICROS * 2);
    } else {
      a.isMoving = false;
      task.isStepping = false;
      task.lastActionTime = millis();
      task.completedCount++;
    }
  }
}

// 处理串口命令
void processCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  if (cmd.equalsIgnoreCase("STOP")) {
    emergencyStopped = true;
    for (int i = 0; i < 4; i++) {
      axes[i].isMoving = false;
      adjustTasks[i].active = false;
    }
    Serial.println("🛑 急停触发，所有轴停止！");
    return;
  }

  if (cmd.equalsIgnoreCase("RESUME")) {
    emergencyStopped = false;
    Serial.println("✅ 已解除急停，可以继续控制");
    return;
  }

  // 微调命令
  if (cmd.startsWith("XADJ") || cmd.startsWith("YADJ") ||
      cmd.startsWith("ZADJ") || cmd.startsWith("TADJ")) {
    char axisChar;
    int dir, pulses, repeat, interval;
    int parsed = sscanf(cmd.c_str(), "%cADJ %d %d %d %d", &axisChar, &dir, &pulses, &repeat, &interval);
    if (parsed != 5 || (dir != 0 && dir != 1) || pulses <= 0 || repeat <= 0 || interval < 10) {
      Serial.println("❌ 格式错误，应为：<X/Y/Z/T>ADJ <0/1> <脉冲数> <次数> <间隔毫秒>");
      return;
    }
    axisChar = toupper(axisChar);
    int axisIndex = (axisChar=='X')?0:(axisChar=='Y')?1:(axisChar=='Z')?2:(axisChar=='T')?3:-1;
    if (axisIndex == -1) return;

    AdjustTask& t = adjustTasks[axisIndex];
    t.active = true;
    t.dir = dir;
    t.pulsesPerCycle = pulses;
    t.repeatCount = repeat;
    t.intervalMillis = interval;
    t.completedCount = 0;
    t.isStepping = false;
    t.lastActionTime = millis();
    t.axisIndex = axisIndex;

    Serial.print("⚙️ 开始 ");
    Serial.print(axes[axisIndex].name);
    Serial.print(" 轴微调：方向 ");
    Serial.print(dir);
    Serial.print("，每次 ");
    Serial.print(pulses);
    Serial.print(" 脉冲，重复 ");
    Serial.print(repeat);
    Serial.print(" 次，间隔 ");
    Serial.print(interval);
    Serial.println(" 毫秒");
    return;
  }

  // 普通XYZT命令
  char axisChar;
  int dir;
  long steps;
  int parsed = sscanf(cmd.c_str(), "%c %d %ld", &axisChar, &dir, &steps);
  if (parsed != 3) {
    Serial.println("❌ 格式错误！请输入如：X 1 2000");
    return;
  }

  axisChar = toupper(axisChar);
  if (axisChar != 'X' && axisChar != 'Y' && axisChar != 'Z' && axisChar != 'T') {
    Serial.println("❌ 轴名必须为 X/Y/Z/T");
    return;
  }

  if ((dir != 0 && dir != 1) || steps <= 0) {
    Serial.println("❌ 方向需为0或1，步数必须为正");
    return;
  }

  for (int i = 0; i < 4; i++) {
    if (axes[i].name[0] == axisChar) {
      axes[i].dir = dir;
      axes[i].stepsToMove = steps;
      axes[i].isMoving = true;
      axes[i].limitTriggerCount = 0;
      axes[i].lastLimitState = false;
      Serial.print("▶️ 轴 ");
      Serial.print(axisChar);
      Serial.print(" 开始 ");
      Serial.print(dir == 1 ? "顺时针" : "逆时针");
      Serial.print(" 移动 ");
      Serial.print(steps);
      Serial.println(" 步");
      break;
    }
  }
}
