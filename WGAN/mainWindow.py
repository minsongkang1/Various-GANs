


import cv2
import time
import torch
import tkinter
import numpy as np
from PIL import Image,ImageTk
import matplotlib.pyplot as plt
from net.Generator import Generator

#显示的图片大小
img_width = 300
img_height = 300

#创建主窗口
root = tkinter.Tk()
root.title('WGAN-GP随机生成图片')
root.geometry("520x520")
#root.iconphoto(True,tkinter.PhotoImage(file = 'hometown.png'))


def Canvas_(root):
    # 创建画布
    canvas = tkinter.Canvas(root, bg='white', width=img_width, height=img_height)
    canvas.place(x=111, y=50)
    # 创建标签
    label = tkinter.Label(root, text='随机生成图片', font=('黑体', 14), width=15, height=1)
    # `anchor=nw`则是把图片的左上角作为锚定点
    label.place(x=190, y=20, anchor='nw')
    return canvas

def showImage(frame):
    """
    :param root: 主窗口
    :return:
    """
    # 摄像头翻转
    # frame=cv2.flip(frame,1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    image = image.astype(np.uint8)
    PILimage = Image.fromarray(image)
    PILimage = PILimage.resize((img_width, img_height), Image.LANCZOS)
    try:
        tkImage = ImageTk.PhotoImage(image=PILimage)
    except:
        return 0
    return tkImage


#这里需要将tkImage设置为全局变量，不然显示不图片
tkImage=''
def set_BackGround(image_path, root = root):
    global tkImage
    canvas = Canvas_(root)
    img = cv2.imread(image_path)
    tkImage = showImage(img)
    canvas.create_image(0, 0, anchor='nw', image=tkImage)

#关闭主界面时弹出的确认窗口
def closeEvent(root):
    # root.withdraw()  # 仅显示对话框，隐藏主窗口
    btn_close=tkinter.Button(root,text='退出',font=('黑体', 14), height=1, width=29,command=root.destroy)
    btn_close.place(x=115,y=400)
    return True

def generagedImage(model):
    """
    :param model:
    :return:
    """
    random_noise = torch.randn(size=(16, 100,1,1), device='cpu')
    genImg = model(random_noise).detach().cpu()
    # --------------------------------------------------------------
    # vutilsImg = vutils.make_grid(result, padding=2, normalize=True)
    # fig = plt.figure(figsize=(4, 4))
    # plt.imshow(np.transpose(vutilsImg, (1, 2, 0)))
    # plt.axis('off')
    # plt.show()
    # --------------------------------------------------------------
    result = np.squeeze(genImg.numpy())
    fig = plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(np.transpose(result[i], (1, 2, 0)))
        plt.axis('off')
    plt.savefig('cur.png')

def loadModel(model,root):
    """
    :param model:
    :return:
    """
    # 注意这里的变量，一定要声明为全局变量，不然后面显示不出来图片
    global  tkImage
    #生成图片
    generagedImage(model)
    time.sleep(1)
    #从保存的生成器生成图片中读取该图片并显示出来
    canvas = Canvas_(root)
    img = cv2.imread('cur.png')
    tkImage = showImage(img)
    canvas.create_image(0, 0, anchor='nw', image=tkImage)

#点击生成图片按钮
def ButtonImage(model,root):
    # root.withdraw()  # 仅显示对话框，隐藏主窗口
    btn_gen = tkinter.Button(root, text='生成图片', font=('黑体', 14), height=1, width=29,
                               command = lambda : loadModel(model,root))
    btn_gen.place(x=115, y=450)

if __name__ == '__main__':
    #image_path是一张背景图
    image_path = 'cat.png'
    set_BackGround(image_path=image_path,root = root)
    model = torch.load('models/gen.pth', map_location=lambda storage, loc: storage)
    while True:
        ButtonImage(model,root)
        flag = closeEvent(root)
        if flag:
            break
    root.mainloop()

