# 참고자료 : https://funnypani.com/1533
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.widgets import Button
from tkinter  import *

win = Tk()
win.geometry("500x500")
win.title("AI vs Human")
win['bg'] = '#23201f'
win.option_add("*Font","메이플스토리  25")
win.resizable(False,False)

lab_im = Label(win)
img = PhotoImage(file="./fileimgs/1-6.png", master= win)
img2 = PhotoImage(file="./fileimgs/doggie.png", master=win)
win.iconphoto(True, img2)
img = img.subsample(2)
lab_im.config(image= img)
lab_im['bg'] = '#23201f'
lab_im.pack()

lab = Label(win)
lab.config(text="숫자를 입력해주세요", fg='white')
lab['bg'] = '#23201f'
lab.pack()

#(1) 데이터
paths = glob.glob("./imgs/*/*.jpg") # 28X28 로 픽셀조정한 8가지 종류의 이미지 종류별 30장
paths = np.random.permutation(paths) 

independent = np.array([plt.imread(paths[i]) for i in range(len(paths))])
dependent = np.array([paths[i].split('\\')[-2] for i in range(len(paths))])
#print(independent.shape, dependent.shape) #(238, 28, 28, 3) (238,)

dependent = pd.get_dummies(dependent)
#print(independent.shape, dependent.shape) # (238, 28, 28, 3) (238, 8) : 원핫 성공

#(2) 모델
X = tf.keras.layers.Input(shape=[28, 28, 3])

H = tf.keras.layers.Conv2D(6, kernel_size=5, padding='same', activation='swish')(X) #컨벌루션 mask1적용
H = tf.keras.layers.MaxPool2D()(H) #풀링1
H = tf.keras.layers.Conv2D(16, kernel_size=5, activation='swish')(H) #컨벌루션 mask2적용
H = tf.keras.layers.MaxPool2D()(H) #풀링2

H = tf.keras.layers.Flatten()(H) #평탄화추가
H = tf.keras.layers.Dense(120, activation='swish')(H) #기존 X였던거를 평탄화작업이 추가되면서 H로 변경함
H = tf.keras.layers.Dense(84, activation='swish')(H)
Y = tf.keras.layers.Dense(8, activation='softmax')(H)
model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')

#모델유형 분석
model.summary()

# #(3) 학습
model.fit(independent, dependent, epochs=100)

# #(4) 검증( 예측값 : 원본값 )
print("< 판단값 >")
pre = model.predict(independent[0:10])
print(pd.DataFrame(pre).round(2))

print("< 실제값 >")
print(dependent[0:10])

ent = Entry(win)
ent.place(x=30,y=320)
index = ent.get()
btn = Button(win, text='시작')
btn.config(width=6,height=1)
btn.place(x=185,y=400)

def showImage(a):
    global answer
    answer = ent.get()
    answer = int(answer)
    plt.imshow(independent[answer], cmap='gray')
    plt.show()
    ent.delete(0,len(ent.get()))
    
btn2 = Button(win, text='시작', command=lambda: showImage(ent.get()))
btn2.config(width=6,height=1)
btn2.place(x=185,y=400)

win.mainloop()