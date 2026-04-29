from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
import io
import numpy as np

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

        img_array = np.array(img).reshape(1, 784)  # 转为 (1, 784) 的形状

        def printImage(img):
            for i in range(28):
                for j in range(28):
                    print(int(img[i * 28 + j]), end=' ')
                print()
            print()

        printImage(img_array[0])  # 打印图像数据

        
        # 将PIL图像转换为PhotoImage对象
        self.photo_img = ImageTk.PhotoImage(img.resize((200, 200), Image.NEAREST))  # 放大显示
        
        # 重新在画布显示
        self.canvas.delete("all")
        self.canvas.create_image(300, 200, image=self.photo_img)  # 居中显示

        self.text_box.insert(tk.END, "图像已处理并显示\n")
        self.text_box.see(tk.END)  # 滚动到最新内容
    
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