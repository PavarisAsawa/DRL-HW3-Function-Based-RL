import numpy as np
import torch

class DummyIsaacEnv:
    def __init__(self):
        # กำหนดมิติของ action และ state
        self.action_dim = 3    # ตัวอย่าง สมมุติมี action 3 มิติ
        self.state_dim = 10    # สมมุติมี state เป็นเวกเตอร์ขนาด 10

        # กำหนดขอบเขตของ action space (continuous)
        self.action_low = -1.0
        self.action_high = 1.0

        # กำหนดขอบเขตของ state space (dummy สามารถปรับให้เหมาะสม)
        self.state_low = -10.0
        self.state_high = 10.0

        self.max_steps = 200  # จำนวนก้าวสูงสุดต่อหนึ่ง episode
        self.current_step = 0

    def reset(self):
        """Reset environment โดยคืนค่า state เริ่มต้นและข้อมูลเพิ่มเติม (info) เป็น dict ว่าง"""
        self.current_step = 0
        # สร้าง state เริ่มต้นแบบสุ่มในช่วง state space
        state = np.random.uniform(self.state_low, self.state_high, size=(self.state_dim,))
        info = {}
        return state, info

    def step(self, action):
        """
        รับ action (เป็น numpy array หรือ list) แล้วคำนวณ:
          - next_state: state ถัดไป (dummy สร้างแบบสุ่ม)
          - reward: คำนวณจาก action (เช่นใช้ -||action||^2)
          - done: Boolean ระบุว่า episode สิ้นสุดหรือไม่
          - info: dict ข้อมูลเพิ่มเติม (สามารถเพิ่มรายละเอียดได้ตามต้องการ)
        """
        self.current_step += 1
        
        # สร้าง next state แบบสุ่ม
        next_state = np.random.uniform(self.state_low, self.state_high, size=(self.state_dim,))
        
        # ตัวอย่าง reward: ให้น้ำหนักลบกับ norm ของ action
        action = np.array(action)  # ให้แน่ใจว่าเป็น numpy array
        reward = -np.sum(action ** 2)
        
        # ตรวจสอบว่า episode สิ้นสุดหรือไม่
        done = self.current_step >= self.max_steps
        
        info = {}
        return next_state, reward, done, info

    def render(self):
        """สำหรับการ render (dummy environment จึงไม่ต้องแสดงผลใด ๆ)"""
        pass

    def close(self):
        """สำหรับการปิด environment (dummy environment ไม่จำเป็นต้องทำอะไร)"""
        pass

# ทดสอบการทำงานของ DummyIsaacEnv
if __name__ == '__main__':
    env = DummyIsaacEnv()
    
    # เริ่มต้น environment ด้วย reset()
    state, info = env.reset()
    print("Initial state:", state)
    
    # สมมุติ loop ทำ actions จำนวน 5 ครั้ง
    for _ in range(5):
        # สุ่ม action จากขอบเขตที่กำหนด
        action = np.random.uniform(env.action_low, env.action_high, size=(env.action_dim,))
        next_state, reward, done, info = env.step(action)
        
        print("\nAction:", action)
        print("Next state:", next_state)
        print("Reward:", reward)
        print("Done:", done)
        
        if done:
            break
