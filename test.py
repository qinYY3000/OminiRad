import swanlab
import random

# 初始化一个新的swanlab run类来跟踪这个脚本
swanlab.init(
  # 设置将记录此次运行的项目信息
  project="ominirad",
  workspace="qinYY3000",
  # 跟踪超参数和运行元数据
  config={
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10
  }
)

# 模拟训练
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
  acc = 1 - 2 ** -epoch - random.random() / epoch - offset
  loss = 2 ** -epoch + random.random() / epoch + offset

  # 向swanlab上传训练指标
  swanlab.log({"acc": acc, "loss": loss})

# [可选] 完成训练，这在notebook环境中是必要的
swanlab.finish()

if __name__ == '__main__':
    swanlab.run()
