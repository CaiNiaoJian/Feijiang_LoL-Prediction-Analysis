import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from paddle.io import DataLoader, Dataset

# 自定义数据集类
class LOLDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)

# 加载训练数据
train_data = pd.read_csv('train.csv')

# 训练集
X_train = train_data.drop(['id', 'win'], axis=1)
y_train = train_data['win']

# 标准化处理
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# 将数据转换为 float32 类型
X_train = X_train.astype('float32')
y_train = y_train.values.astype('float32')

# 创建数据集和数据加载器
train_dataset = LOLDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

class SimpleNN(nn.Layer):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# 初始化模型
input_size = X_train.shape[1]
model = SimpleNN(input_size)

criterion = nn.BCELoss()
optimizer = optim.Adam(learning_rate=0.001, parameters=model.parameters())

# 训练模型
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # 前向传播
        outputs = model(data)
        loss = criterion(outputs, target.reshape([-1, 1]))

        # 反向传播和优化
        optimizer.clear_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 加载测试数据
test_data = pd.read_csv('test.csv')

# 测试集
X_test = test_data.drop(['id'], axis=1)

# 标准化处理
X_test = scaler.transform(X_test)

# 将数据转换为 float32 类型
X_test = X_test.astype('float32')

# 将测试数据转换为PaddlePaddle张量
X_test = paddle.to_tensor(X_test, dtype='float32')

# 预测
model.eval()
y_pred = model(X_test)
y_pred = (y_pred > 0.5).astype('int32')

# 将预测结果保存到测试数据中
test_data['win'] = y_pred.numpy().flatten()

# 保存预测结果
test_data[['id', 'win']].to_csv('submission.csv', index=False)