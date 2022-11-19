import numpy as np
import matplotlib.pyplot as plt

d = 0
h = 0
s = 0
n = 0
sd = 0
a = 0
f = 0
max_value = 0

data1 = ["Disgusted", "Happy", "Surprised", "Neutral", "Sad", "Angry", "Fearful"]
data2 = ['disgusted','happy','happy','neutral','angry','sad','sad','surprised','fearful','sad']#sent by hamzanaiwrita peeps in an array

for str in data2:
    if str == 'disgusted':
        d += 1
    if str == 'happy':
        h += 1
    if str == 'surprised':
        s += 1
    if str == 'neutral':
        n += 1
    if str == 'sad':
        sd += 1
    if str == 'angry':
        a += 1
    if str == 'fearful':
        f += 1
        
data3 = [d, h, s, n, sd, a, f]
max_value = max(data3)

if d == max_value:
    plt.figure(facecolor='chartreuse')
    ax = plt.axes()
    ax.set_facecolor('lightsteelblue')
    p1 = plt.bar(data1,data3, color=['chartreuse', 'yellow', 'dodgerblue', 'dimgray', 'darkgray', 'tomato', 'darkviolet'])

    plt.xlabel('Emotions')
    plt.ylabel('Amount')
    plt.title('Emotion Detector')

    plt.legend()

    plt.show()
    
elif h == max_value:
    plt.figure(facecolor='yellow')
    ax = plt.axes()
    ax.set_facecolor('lightsteelblue')
    p1 = plt.bar(data1, data3, color=['chartreuse', 'yellow', 'dodgerblue', 'dimgray', 'darkgray', 'tomato', 'darkviolet'])

    plt.xlabel('Emotions')
    plt.ylabel('Amount')
    plt.title('Emotion Detector')

    plt.legend()

    plt.show()
    
elif s == max_value:
    plt.figure(facecolor='dodgerblue')
    ax = plt.axes()
    ax.set_facecolor('lightsteelblue')
    p1 = plt.bar(data1, data3, color=['chartreuse', 'yellow', 'dodgerblue', 'dimgray', 'darkgray', 'tomato', 'darkviolet'])

    plt.xlabel('Emotions')
    plt.ylabel('Amount')
    plt.title('Emotion Detector')

    plt.legend()

    plt.show()
    
elif n == max_value:
    plt.figure(facecolor='dimgray')
    ax = plt.axes()
    ax.set_facecolor('lightsteelblue')
    p1 = plt.bar(data1, data3, color=['chartreuse', 'yellow', 'dodgerblue', 'dimgray', 'darkgray', 'tomato', 'darkviolet'])

    plt.xlabel('Emotions')
    plt.ylabel('Amount')
    plt.title('Emotion Detector')

    plt.legend()

    plt.show()
    
elif sd == max_value:
    plt.figure(facecolor='darkgray')
    ax = plt.axes()
    ax.set_facecolor('lightsteelblue')
    p1 = plt.bar(data1, data3, color=['chartreuse', 'yellow', 'dodgerblue', 'dimgray', 'darkgray', 'tomato', 'darkviolet'])

    plt.xlabel('Emotions')
    plt.ylabel('Amount')
    plt.title('Emotion Detector')

    plt.legend()

    plt.show()
    
elif a == max_value:
    plt.figure(facecolor='tomato')
    ax = plt.axes()
    ax.set_facecolor('lightsteelblue')
    p1 = plt.bar(data1, data3, color=['chartreuse', 'yellow', 'dodgerblue', 'dimgray', 'darkgray', 'tomato', 'darkviolet'])

    plt.xlabel('Emotions')
    plt.ylabel('Amount')
    plt.title('Emotion Detector')

    plt.legend()

    plt.show()

else:
    plt.figure(facecolor='darkviolet')
    ax = plt.axes()
    ax.set_facecolor('lightsteelblue')
    p1 = plt.bar(data1, data3, color=['chartreuse', 'yellow', 'dodgerblue', 'dimgray', 'darkgray', 'tomato', 'darkviolet'])

    plt.xlabel('Emotions')
    plt.ylabel('Amount')
    plt.title('Emotion Detector')

    plt.legend()

    plt.show()














