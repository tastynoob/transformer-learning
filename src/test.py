

import mnist
from np.numpyNN import *

import numpy as np
import tqdm


mnist.init()


if __name__ == "__main__":
    class BPNetWork:
        optimizer = AdaptiveGDopt(0.01)
        layer1 = Linear(784, 64, optimizer, True)
        layer2 = ReLU()
        layer3 = ResidualConnection(Layers([
            Linear(64, 64, optimizer, True),
            ReLU(),
        ]))
        layer4 = Layers([
            Linear(64, 10, optimizer, True),
            Softmax(),
        ])

        def forward(self, X):
            X = self.layer1.forward(X)
            X = self.layer2.forward(X)
            X = self.layer3.forward(X)
            X = self.layer4.forward(X)
            return X
        
        def backward(self, prev_G):
            prev_G = self.layer4.backward(prev_G)
            prev_G = self.layer3.backward(prev_G)
            prev_G = self.layer2.backward(prev_G)
            prev_G = self.layer1.backward(prev_G)
            return prev_G
        
        def train(self, X, Y):
            Y_hat = self.forward(X)
            loss, G = absolute_loss(Y, Y_hat)
            self.backward(G)
            self.optimizer.update()
            return loss
    nn = BPNetWork()

    def train():
        train_data = mnist.getTrainData()
        test_data = mnist.getTestData()
        print("Dataset load successfully")
        try:
            for epoch in range(20):
                print(f"Epoch {epoch + 1}")
                total_loss = 0
                #shuffle the training data
                np.random.shuffle(train_data)
                for i in tqdm.tqdm(range(len(train_data))):
                    img, label_index = train_data[i]
                    # to one-hot encoding [0, 0, ..., 1, 0, ...]
                    label = np.zeros((1, 10))
                    label[0, label_index] = 1  # One-hot encoding
                    
                    loss = nn.train(img, label)
                    total_loss += loss
                print(f"Total loss: {total_loss / len(train_data)}")
        except KeyboardInterrupt:
            print("Training interrupted by user")

        # test the model
        correct = 0
        for i in tqdm.tqdm(range(len(test_data))):
            img, label_index = test_data[i]
            # to one-hot encoding [0, 0, ..., 1, 0, ...]
            label = np.zeros((1, 10))
            label[0, label_index] = 1
            Y_hat = nn.forward(img) 
            prediction = np.argmax(Y_hat, axis=1)[0]
            if prediction == label_index:
                correct += 1
        print(f"Test accuracy: {correct / len(test_data) * 100:.2f}%")
        print("Training completed successfully")

    def Inference(img):
        img = img.reshape(1, 784)  # Reshape to (1, 784)
        Y_hat = nn.forward(img)
        print(Y_hat)
        return np.argmax(Y_hat, axis=1)[0]  # Return the index of the maximum value

    set_training_mode(True)  # Set training mode to True
    train()
    set_training_mode(False)  # Set training mode to False

    from PIL import Image, ImageTk
    import tkinter as tk
    from tkinter import ttk
    import io

    class DrawingApp:
        def __init__(self, root):
            self.root = root
            self.root.title("画图程序")
            self.root.geometry("600x600")
            
            # 创建主框架
            main_frame = ttk.Frame(root)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # 创建画布
            self.canvas = tk.Canvas(main_frame, bg='white', width=400, height=400, relief=tk.SUNKEN, borderwidth=2)
            self.canvas.pack(side=tk.TOP, pady=(0, 10))
            
            # 创建控制框架
            control_frame = ttk.Frame(main_frame)
            control_frame.pack(side=tk.BOTTOM, fill=tk.X)
            
            # 创建按钮
            self.button = ttk.Button(control_frame, text="点击我", command=self.on_button_click)
            self.button.pack(side=tk.LEFT, padx=(0, 10))

            self.clear_button = ttk.Button(control_frame, text="清除画布", command=lambda: self.canvas.delete("all"))
            self.clear_button.pack(side=tk.LEFT, padx=(0, 10))
            
            # 创建文本框
            self.text_box = tk.Text(control_frame, height=3, width=50)
            self.text_box.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            # 绑定鼠标事件以便在画布上绘图
            self.canvas.bind("<Button-1>", self.start_draw)
            self.canvas.bind("<B1-Motion>", self.draw)
            
            # 记录上一个鼠标位置
            self.last_x = None
            self.last_y = None
        
        def on_button_click(self):
            """按钮点击事件处理函数"""
            # 获取画布图像
            ps_data = self.canvas.postscript(colormode='color')
            # 将PostScript数据转换为PIL图像对象
            img = Image.open(io.BytesIO(ps_data.encode('latin-1')))
            img = img.convert('L')  # 转为灰度图像
            img = img.resize((28, 28))

            # 处理图像：二值化
            img = img.point(lambda x: 1 if x < 255 else 0, '1')

            # 将PIL图像转换为numpy数组
            img_array = np.array(img).reshape(1, 784)  # 转为 (1, 784) 的形状
            # 进行推理
            prediction = Inference(img_array)

            # # 将PIL图像转换为PhotoImage对象
            # self.photo_img = ImageTk.PhotoImage(img.resize((200, 200), Image.NEAREST))  # 放大显示
            # # 重新在画布显示
            # self.canvas.delete("all")
            # self.canvas.create_image(300, 200, image=self.photo_img)  # 居中显示


            # 显示预测结果
            self.text_box.delete(1.0, tk.END)
            self.text_box.insert(tk.END, f"预测结果: {prediction}\n")
        
        def start_draw(self, event):
            """开始绘图"""
            self.last_x = event.x
            self.last_y = event.y
        
        def draw(self, event):
            """绘图函数"""
            if self.last_x and self.last_y:
                self.canvas.create_line(
                    self.last_x, self.last_y, event.x, event.y,
                    width=1, fill='black', capstyle=tk.ROUND, smooth=tk.TRUE
                )
            self.last_x = event.x
            self.last_y = event.y

    def main():
        root = tk.Tk()
        app = DrawingApp(root)
        root.mainloop()

    if __name__ == "__main__":
        main()