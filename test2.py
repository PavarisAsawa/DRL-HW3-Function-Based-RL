import torch
import torch.nn as nn
import torch.optim as optim

# 1. สร้างโมเดลง่าย ๆ (MLP)
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# 2. สร้างอินสแตนซ์โมเดลและ optimizer
model = SimpleNet()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# 3. สร้างข้อมูลตัวอย่าง
#    x: batch_size=4, feature=10
x = torch.randn(4, 10)
#    y: target scalar
y = torch.randn(4, 1)

# 4. ฝึกโมเดล 1 รอบ (1 step)
model.train()
optimizer.zero_grad()
pred = model(x)
loss = criterion(pred, y)
loss.backward()
optimizer.step()
print(f"Loss after 1 step: {loss.item():.4f}")

# 5. บันทึก state_dict ของโมเดล
torch.save(model.state_dict(), "simple_model.pth")
print("Model weights saved to simple_model.pth")

# 6. โหลด state_dict ไปยังโมเดลใหม่
model2 = SimpleNet()
model2.load_state_dict(torch.load("simple_model.pth"))
model2.eval()  # เปลี่ยนเป็นโหมด inference

# 7. ตรวจสอบว่า prediction เหมือนกัน
with torch.no_grad():
    pred1 = model(x)
    pred2 = model2(x)
print("Difference between original and loaded model outputs:",
      torch.norm(pred1 - pred2).item())
print(model.state_dict())