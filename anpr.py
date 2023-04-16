
from tkinter import *
from tkinter import filedialog
#import tkFileDialog as filedialog
from PIL import ImageTk, Image
master = Tk()
master.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
global outstring
outstring = " "
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
with open('Results.txt', 'w') as f:
    for j1 in range(1, 109):
        imgs = str(j1) + ".jpg"
        img = cv2.imread(master.filename, 0)
        img = cv2.bilateralFilter(img, 5, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        _, img_dilation = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        img_dilation = cv2.erode(img_dilation, kernel, iterations=1)
        laplacian = cv2.Sobel(img_dilation, cv2.CV_64F, 1, 0, ksize=-1)
        height, width = laplacian.shape
        count = temp = 0
        i, x1 = (height // 2) - 50, 0
        j, x2 = 100, 0
        set_height, set_width = 50, 215
        while i < height:
            if height - i < set_height:
                break
            j = 100
            while j < width:
                if width - j < set_width + 100:
                    break
                j1 = j
                i1 = i
                count = np.count_nonzero(laplacian[i1:i1 + set_height, j1:j1 + set_width])
                if count < temp // 4:
                    j += 4
                    continue
                count = np.count_nonzero(laplacian[i1:i1 + set_height, j1:j1 + set_width])
                if temp < count:
                    temp = count
                    x1, x2 = i, j
                j += 4
            i += 3
        c_img = img_dilation[x1:x1 + set_height, x2:x2 + set_width]
        img = c_img
        height, width = img.shape
        org = np.zeros((height, width, 1), np.uint8)
        i, j = 0, 0
        while i < height:
            while j < width:
                org[i][j] = img[i][j]
                j += 1
            i += 1
        kernel = np.ones((3, 3), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)
        sobelx = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=-1)
        sobely = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=-1)
        _, sobelx = cv2.threshold(sobelx, 128, 255, cv2.THRESH_BINARY)
        _, sobely = cv2.threshold(sobely, 128, 255, cv2.THRESH_BINARY)

    i=0
    count=0
    ar=[]
    xar=[]

    i=0 
    j=0
    while i<width:
        j=0
        count=0
        while j<height:
            if sobelx[j][i]==255:
                count+=1
            j+=1
        ar.append(count)
        i+=1


    prev_val_stored=0
    first_val_stored=0
    count=0
    i=0
    j=0
    k=0
    tall=0
    first_point=0
    first_val=0
    prev_val=0
    val=0
    after=49
    last_point=0
    first_point_var=0
    while i<height:
        j=0
        count=0
        while j<width:
            if sobelx[i][j]==255:
                count+=1
            j+=1
        xar.append(count)
        val=count
        if i==0:
            first_val=count
            prev_val=count
            first_point_var=i
        else:
            if val<prev_val:
                temp=prev_val-first_val
                if temp>tall:
                    tall=temp
                    prev_val_stored=prev_val
                    first_val_stored=first_val
                    last_point=i-1
                    first_point=first_point_var
                k=0
            if val>prev_val:
                if k==0:
                    k=1
                    first_val=prev_val
                    first_point_var=i-1
            prev_val=val
        i+=1
    #print "first point->",first_point," start->",first_val_stored,"last point->",last_point," end->",prev_val_stored
    xx=first_point
    yy=last_point

    prev_val_stored=0
    first_val_stored=0
    count=0
    i=0
    j=0
    k=0
    tall=0
    first_point=0
    first_val=0
    prev_val=0
    val=0
    after=49
    last_point=0
    first_point_var=0

    while i<height:
        j=0
        count=0
        while j<width:
            if sobelx[i][j]==255:
                count+=1
            j+=1
        #xar.append(count)
        val=count
        if i==0:
            first_val=count
            prev_val=count
            first_point_var=i
        else:
            if val<prev_val:
                temp=prev_val-first_val
                if temp>tall and first_point_var!=xx and yy!=i-1:
                    tall=temp
                    prev_val_stored=prev_val
                    first_val_stored=first_val
                    last_point=i-1
                    first_point=first_point_var
                k=0
            if val>prev_val:
                if k==0:
                    k=1
                    first_val=prev_val
                    first_point_var=i-1
            prev_val=val
        i+=1
    #print "first point->",first_point," start->",first_val_stored,"last point->",last_point," end->",prev_val_stored

    prev_val_stored=0
    first_val_stored=0
    count=0
    i=height-1
    j=0
    k=1
    tall=0
    first_point1=0
    first_val=0
    prev_val=0
    val=0
    after=49
    last_point1=0
    first_point_var=0
    while i>=0:
        j=0
        count=0
        while j<width:
            if sobelx[i][j]==255:
                count+=1
            j+=1
        #xar.append(count)
        val=count
        if i==height-1:
            first_val=count
            prev_val=count
            first_point_var=i
        else:
            if val<prev_val:
                temp=prev_val-first_val
                if temp>tall:
                    tall=temp
                    prev_val_stored=prev_val
                    first_val_stored=first_val
                    if i!=0:
                        last_point1=i+1
                    else:
                        last_point1=0
                    first_point1=first_point_var
                k=0
            if val>prev_val:
                if k==0:
                    k=1
                    first_val=prev_val
                    first_point_var=i+1
            prev_val=val
        i-=1
    #print "first point->",first_point1," start->",first_val_stored,"last point->",last_point1," end->",prev_val_stored
    xx1=first_point1
    yy1=last_point1

    prev_val_stored=0
    first_val_stored=0
    count=0
    i=height-1
    j=0 
    k=1
    tall=0
    first_point1=0
    first_val=0
    prev_val=0
    val=0
    after=49
    last_point1=0
    first_point_var=0
    while i>=0:
        j=0
        count=0
        while j<width:
            if sobelx[i][j]==255:
                count+=1
            j+=1
        #xar.append(count)
        val=count
        if i==height-1:
            first_val=count
            prev_val=count
            first_point_var=i
        else:
            if val<prev_val:
                temp=prev_val-first_val
                if temp>tall and first_point_var!=xx1 and yy1!=i+1:
                    tall=temp
                    prev_val_stored=prev_val
                    first_val_stored=first_val
                    if i!=0:
                        last_point1=i+1
                    else:
                        last_point1=0
                    first_point1=first_point_var
                k=0
            if val>prev_val:
                if k==0:
                    k=1
                    first_val=prev_val
                    first_point_var=i+1
            prev_val=val
        i-=1
    lp1=last_point1
    lp=last_point
    fp1=first_point1
    fp=first_point
    starting1=0
    ending1=0


    if yy1-yy<last_point1-last_point and ((yy>lp and yy>lp1 and yy1>lp and yy1>lp1)or (yy<lp and yy<lp1 and yy1<lp and yy1<lp1)):
        if (lp+fp)//2<lp-3:
            var=lp-(lp+fp)//2
        else:
            var=3
        if (lp1+fp1)//2<lp1+3:
            var1=(lp1+fp1)//2-lp1
        else:
            var1=3
        starting=(last_point-var)
        ending=(last_point1+var1)
        starting1=(((fp+lp)//2)+fp)//2#(last_point-var)
        ending1=(((fp1+lp1)//2)+fp1)//2#(last_point1+var1
    else:
        if (xx+yy)//2<yy-3:
            var=yy-(xx+yy)//2
        else:
            var=3
        if (xx1+yy1)//2<yy1+3:
            var1=(xx1+yy1)//2-yy1
        else:
            var1=3  
        starting=(yy-var)
        ending=(yy1+var1)
        starting1=(((xx+yy)//2)+xx)//2#(last_point-var)
        ending1=(((xx1+yy1)//2)+xx1)//2#(last_point1+var1


    if starting>=ending:
        j1111=j1111+1
        continue

    if starting<0:
        starting=0
    if ending>=height:
        ending=height-1
    heightss, widthss = img.shape
    c_img=img[starting:ending,0:0+widthss] 
    
    img=img[starting1:ending1,0:0+widthss]
    heightss, widthss = img.shape

    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

    kernel = np.ones((3,3), np.uint8)
    c_img = cv2.dilate(c_img, kernel, iterations=1)
    _,c_img = cv2.threshold(c_img,128, 255, cv2.THRESH_BINARY)

    heightz,widthz= c_img.shape 
    xary=[]
    count1=0
    i=0
    j=0


    while i<widthz:
        j=0
        count1=0
        while j<heightz:
            if c_img[j][i]==0:
                count1+=1
            j+=1
        xary.append(count1)
        i+=1
    varia_returns=0
    i=0
    j=0
    k=0
    z=0
    ct=0
    seg_img=[]
    variable=0
    lit=[]


    org=img
    while i<widthz:
        if (xary[i]<3 and k==0):
            k=1
            z=i
            while xary[z]<3:
                z+=1
            i=z
            if i!=0:
                j=i-1
            else:
                j=i
        elif xary[i]<3 and k==1:
            
            variable=0
            ia=i
            ja=j
            if j>=2:
                ja=j-2
            if i<=(width-1)-2:
                ia=i+2
            
            lit.append(org[0:heightss,ja:ia])
            ct+=1
            z=i
            while z<widthz and xary[z]<3:
                z+=1
            i=z
            j=i-1
            continue
        i+=1
    j=0
    i=0

    while i<len(lit):
        lit[i]=cv2.cvtColor(lit[i],cv2.COLOR_BGR2GRAY)
        i+=1

    train=[]
    i=1
    j=0
    # loop through directories containing images and append to train list
    for letter in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n']:
        for i in range(1, 50):
            k = letter + str(i) + ".jpg"
            name = ''.join(k)
            img = cv2.imread(name, 0)
            img = cv2.resize(img, (20, 20))
            a, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
            train.append(img)


    train=np.array(train)
    train=train.reshape(-1,400).astype(np.float32)

    l=np.arange(10)
    label1=np.repeat(l,30)

    label2=(31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,36,36,36,36,36,36,36,36,36,36,36,36,36,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,40,40,40,40,40,40,40,40,40,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,42,42,42,42,42,42,42,42,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,44,44,44,44,44,44,44,44,44,44,44,44,44,45,45,45,45,45,45,45,45,45,45,45,45,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,48,48,48,48,48,48,48,48,48,48,48,48,48,48,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,)
    label2=np.array(label2)

    labels=np.concatenate((label1,label2),axis=0)
    labels=labels[:,np.newaxis]

    if j1111==1:
        
        knn=cv2.ml.KNearest_create()
        
        forest = RandomForestClassifier(n_estimators = 130)#
        
        svm= SVC(kernel='poly')

        nn = MLPClassifier(hidden_layer_sizes=(270,150, ),activation='logistic',max_iter=500,verbose=False)

        print("training kNN model")
        knn.train(train, cv2.ml.ROW_SAMPLE, np.ravel(labels))
        print("training Random Forest model")
        forest = forest.fit( train, np.ravel(labels))#
        print("training Neural Network model")
        nn.fit(train,np.ravel(labels))
        print("training SVM model")
        svm.fit(train,np.ravel(labels))

    outp=[]
    outp1=[]
    outp2=[]
    outp3=[]
    for test in lit:
        ht,wd=test.shape
        if ht<15:
            continue 

        a,test=cv2.threshold(test,128,255,cv2.THRESH_BINARY)
        i=0
        j=0
        count=0
        while i<ht:
            j=0
            while j<wd:
                if test[i][j]==0:
                    count+=1
                j+=1
            i+=1

        if count<95:
            continue


# Load the image and convert it to grayscale
img = cv2.imread("path/to/image.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Resize the image and threshold it
test = cv2.resize(gray, (20, 20))
_, test = cv2.threshold(test, 128, 255, cv2.THRESH_BINARY)

# Reshape the image and convert it to float32
test = np.array(test)
test = test.reshape(-1, 400).astype(np.float32)

# Use k-NN, Random Forest, and SVM to predict the number plate characters
_, result, _, dist = knn.findNearest(test, k=3)
result1 = forest.predict(test)
result2 = nn.predict(test)
result3 = svm.predict(test)

# Define a function to convert the predicted number to a character
def get_letter(num):
    if num == 51.0:
        return 'Z'
    elif num > 30 and num < 52:
        return chr(int(num) + 64)
    else:
        return str(int(num))

# Create a list to store the output characters
outp = [[], [], []]

# Loop through the predicted results and add the characters to the output list
for i, result in enumerate([result, result1, result2]):
    if (dist[0][0] > 5500000 and dist[0][1] > 5500000) or \
       (dist[0][0] > 5500000 and dist[0][2] > 5500000) or \
       (dist[0][1] > 5500000 and dist[0][2] > 5500000):
        continue
    else:
        letter = get_letter(result[0])
        if isinstance(letter, str):
            outp[i].append(letter)
        else:
            outp[i].append(str(letter))

# Join the output characters into strings
out1 = ''.join(map(str, outp[0]))
out2 = ''.join(map(str, outp[1]))
out3 = ''.join(map(str, outp[2]))

# Load the image again and display it with the predicted number plate characters
image = Image.open("path/to/image.jpg").resize((160, 120), Image.ANTIALIAS)
panel = Label(master, image=ImageTk.PhotoImage(image))
panel.pack()
panel.place(x=200, y=70)

# Set up the GUI window and display the input image path and output characters
master.minsize(600, 500)
master.geometry("320x100")
master.configure(background="#FFE4B5")

Label(master, text="AUTOMATIC NUMBER PLATE RECOGNITION SYSTEM", font='Helvetica 22 bold', borderwidth=2, relief="groove").place(x=12, y=10)
Label(master, text="Input image path", font='Helvetica 17', borderwidth=5, relief="groove").place(x=35, y=240)
Label(master, text="Output", font='Helvetica 17', borderwidth=5, relief="groove").place(x=35, y=280)
Label(master, text="path/to/image.jpg", borderwidth=2, font='Helvetica 17 bold', relief="groove").place(x=200, y=240)
Label(master, text=out1, borderwidth=2, font='Helvetica 17 bold', relief="groove").place(x=200, y
