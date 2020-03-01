from PIL import Image, ImageDraw, ImageFont, ImageTk
import numpy as np
import sys
import math
import tkinter as tk
from tkinter import filedialog, messagebox

top = tk.Tk()                   # creates window
top.geometry('600x300+400+300') #window width, height and starting position
top.title('Catch All ImageZ')   #window title
filename= ''

def main():

    global v
    global m
    select_image_but = tk.Button(top, text="Select Image", command=ButtonEvent).place(x=50,y=50)           #Select Image Button
    find_char_but = tk.Button(top, text="Find Characters", command=ButtonEvent2).place(x=410,y=250)     #Find Characters Button

    v = tk.IntVar()
    v.set(1)  # initializing the choice, i.e. Hu Moment
    radio1= tk.Radiobutton(text='Hu Moment',variable=v,value=1).place(x=430,y=50)             #Hu Moment Button
    radio2= tk.Radiobutton(text='R Moment',variable=v, value=2).place(x=430, y=80)            #R Moment Button
    radio3= tk.Radiobutton(text='Zernike Moment',variable=v, value=3).place(x=430,y=110)      #Zernike Moment Button

    m = tk.IntVar()
    m.set(1)  # initializing the choice, i.e. First
    radio4= tk.Radiobutton(text='Comparison Method 1',variable=m,value=1).place(x=400,y=160)  #Comparison Method 1 Button
    radio5= tk.Radiobutton(text='Comparison Method 2',variable=m,value=2).place(x=400,y=190)  #Comparison Method 2 Button
    radio6= tk.Radiobutton(text='Comparison Method 3',variable=m,value=3).place(x=400,y=220)  #Comparison Method 3 Button
    
    top.mainloop()

def ButtonEvent():


   global img1
   global filename
   #Opening Filedialog
   top.filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                             filetypes=(("png files", "*.png"),("jpeg files", "*.jpg"),("all files", "*.*")))
   filename = top.filename
   print(filename)
   image = Image.open(filename)
   width, height = image.size
    
   #Image fitting
   if width > height:

       if width > 370:
         coef = 370/width
         width = 370
         height *=coef
   elif height>width:
     if height >180:
         coef =180/height
         height=180
         width *=coef

   image = image.resize((round(width),round(height)), Image.ANTIALIAS) #Image resize
   img1 = ImageTk.PhotoImage(image)  #image

   panel = tk.Label(top, image=img1).place(x=20,y=100)

def ButtonEvent2():

    try:
        img = Image.open(filename)  #opening image
    except AttributeError:
        messagebox.showinfo( "Warning", "Please select image.")
    img_gray = img.convert('L')  #converts the image to grayscale image

    ONE = 255
    a = np.asarray(img_gray)                        # from PIL to np array
    a_bin = threshold(a, 100, ONE, 0)
    im = Image.fromarray(a_bin)                     # from np array to PIL format
    label = blob_coloring_8_connected(a_bin, ONE)   #finds label
    new_img2 = np2PIL_color(label)
    labelDict = add_dict_labelled(label)              #stores labels into dictionary with their indexes
    recAtt = find_rectangle(label, labelDict)        #find coordinates of every labels
    drawing_rect(img, recAtt)                        #draws rectangle
    numberOfLabels = len(labelDict)
    type = ""
    
    if v.get() == 1: #Hu Moment
        type = "Hu"
        numberOfMoments = 7  #There is 7 moment for Hu
        featureVectors = np.empty(shape=(numberOfLabels, numberOfMoments),
                                  dtype=np.double)  # FeatureVectors Array Initialization
        for i in range(numberOfLabels):
            minx = recAtt[0][i]
            miny = recAtt[1][i]
            maxx = recAtt[2][i]
            maxy = recAtt[3][i]

            resizedIm = resize_rec(im, minx, miny, maxx, maxy)  # crops and resizes every character

            # CALCULATION HU MOMENTS
            currentMoments = calc_moments_hu(resizedIm)  # calculates HU moments of character
            featureVectors[i] = currentMoments  # stores hu moments of character

    if v.get() == 2: #R Moment
        type = "R"
        numberOfMoments = 10  #There is 10 moment for R
        featureVectors = np.empty(shape=(numberOfLabels, numberOfMoments),
                                  dtype=np.double)  # FeatureVectors Array Initialization
        for i in range(numberOfLabels):
            minx = recAtt[0][i]
            miny = recAtt[1][i]
            maxx = recAtt[2][i]
            maxy = recAtt[3][i]

            resizedIm = resize_rec(im, minx, miny, maxx, maxy)  # crops and resizes every character

            # CALCULATION HU MOMENTS
            currentMoments = calc_moments_r(resizedIm)  # calculates R moments of character
            featureVectors[i] = currentMoments  # stores R moments of character

    if v.get() == 3: #Zernike Moment
        type = "Zernike"
        numberOfMoments = 12 #There is 12 moment for Zernike
        featureVectors = np.empty(shape=(numberOfLabels, numberOfMoments),
                                  dtype=np.double)  # FeatureVectors Array Initialization

        for i in range(numberOfLabels):
            minx = recAtt[0][i]
            miny = recAtt[1][i]
            maxx = recAtt[2][i]
            maxy = recAtt[3][i]

            resizedIm = resize_rec(im, minx, miny, maxx, maxy)  # crops and resizes every character

            # CALCULATION HU MOMENTS
            currentMoments = calc_moments_zernike(resizedIm)  # calculates Zernike moments of character
            featureVectors[i] = currentMoments  # stores Zernike moments of character

    results=0
    if m.get() == 1: #Comparison Method 1
        results = multiple_comparison(featureVectors, recAtt, type)
    if m.get() == 2: #Comparison Method 2
        results = multiple_comparison2(featureVectors, recAtt, type)
    if m.get() == 3: #Comparison Method 3
        results = multiple_comparison3(featureVectors, recAtt, type)
    write_assumptions(recAtt, img, results) #writes all estimations into image

def np2PIL(im):
    img = Image.fromarray(im, 'RGB')
    return img

def np2PIL_color(im):
    img = Image.fromarray(np.uint8(im))
    return img

def threshold(im,T, LOW, HIGH):
    (nrows, ncols) = im.shape
    im_out = np.zeros(shape = im.shape)
    for i in range(nrows):
        for j in range(ncols):
            if abs(im[i][j]) <  T :
                im_out[i][j] = LOW
            else:
                im_out[i][j] = HIGH
    return im_out

def blob_coloring_8_connected(bim, ONE):
    max_label = int(10000)
    nrow = bim.shape[0]
    ncol = bim.shape[1]
    im = np.zeros(shape=(nrow,ncol), dtype = int)
    a = np.zeros(shape=max_label, dtype = int)
    a = np.arange(0,max_label, dtype = int)
    color_map = np.zeros(shape = (max_label,3), dtype= np.uint8)
    color_im = np.zeros(shape = (nrow, ncol,3), dtype= np.uint8)

    for i in range(max_label):
        np.random.seed(i)
        color_map[i][0] = np.random.randint(0,255,1,dtype = np.uint8)
        color_map[i][1] = np.random.randint(0,255,1,dtype = np.uint8)
        color_map[i][2] = np.random.randint(0,255,1,dtype = np.uint8)


    k = 0
    for i in range(nrow):
        for j in range(ncol):
            im[i][j] = max_label
    for i in range(1, nrow - 1):
        for j in range(1, ncol - 1):
                c   = bim[i][j]
                l   = bim[i -1][j]
                u   = bim[i][j - 1]
                d   = bim[i - 1 ][j - 1]
                r   = bim[i + 1][j - 1]

                label_l = im[i - 1][j]
                label_u  = im[i][j - 1]
                label_d  = im[i - 1][j - 1]
                label_r  = im[i + 1][j - 1]

                im[i][j] = max_label
                if c == ONE:
                    min_label = min( label_u, label_l, label_d, label_r)
                    if min_label == max_label:
                        k += 1
                        im[i][j] = k
                    else:
                        im[i][j] = min_label
                        if min_label != label_u and label_u != max_label  :
                         update_array(a, min_label, label_u)

                        if min_label != label_l and label_l != max_label  :
                            update_array(a, min_label, label_l)

                        if min_label != label_d and label_d != max_label  :
                            update_array(a, min_label, label_d)

                        if min_label != label_r and label_r != max_label  :
                            update_array(a, min_label, label_r)

                else :
                    im[i][j] = max_label

    # final reduction in label array
    for i in range(k+1):
        index = i
        while a[index] != index:
            index = a[index]
        a[i] = a[index]


    #second pass to resolve labels and show label colors
    for i in range(nrow):
        for j in range(ncol):
            if bim[i][j] == ONE:
                im[i][j] = a[im[i][j]]
                if im[i][j] == max_label:
                    color_im[i][j][0] = 0
                    color_im[i][j][1] = 0
                    color_im[i][j][2] = 0
                color_im[i][j][0] = color_map[im[i][j],0]
                color_im[i][j][1] = color_map[im[i][j],1]
                color_im[i][j][2] = color_map[im[i][j],2]

    return color_im

def rgb_to_hash(rgb_array): #creates a unique number from RGB values
    r = rgb_array[0] * 1000000
    g = rgb_array[1] * 1000
    b = rgb_array[2]

    return r+g+b

def update_array(a, label1, label2) :
    index = lab_small = lab_large = 0
    if label1 < label2 :
        lab_small = label1
        lab_large = label2
    else :
        lab_small = label2
        lab_large = label1
    index = lab_large
    while index > 1 and a[index] != lab_small:
        if a[index] < lab_small:
            temp = index
            index = lab_small
            lab_small = a[temp]
        elif a[index] > lab_small:
            temp = a[index]
            a[index] = lab_small
            index = temp
        else: #a[index] == lab_small
            break

    return

def add_dict_labelled(color_im):
    nrow = color_im.shape[0]
    ncol = color_im.shape[1]

    labelDict = {}
    for i in range(nrow):
        for j in range(ncol):
            if color_im[i][j][0] != 0 and color_im[i][j][1] != 0 and color_im[i][j][2] != 0:
                labelDict[rgb_to_hash(color_im[i][j])] = [i, j]

    labelDict = add_dict_labelled2(labelDict)
    return labelDict

def add_dict_labelled2(label_dict):
    label_dict2 = {}
    number_of_labels = 0
    for i in label_dict:
        label_dict2[i] = number_of_labels
        number_of_labels += 1

    return label_dict2


def find_rectangle(color_im, label_dict):  #Finds every label coordinates
    number_of_labels = len(label_dict)
    rec_att = np.zeros(shape=(4, number_of_labels))

    nrow = color_im.shape[0]
    ncol = color_im.shape[1]

    for t in range(number_of_labels):
        rec_att[0][t]=10000
        rec_att[1][t]=10000

    for i in range(nrow):
        for j in range(ncol):
            if color_im[i][j][0] != 0 and color_im[i][j][1] != 0 and color_im[i][j][2] != 0:
                current = rgb_to_hash(color_im[i][j])
                currentIndex = label_dict[current]

                #min i
                if i < rec_att[0][currentIndex]:
                    rec_att[0][currentIndex] = i
                #min j
                if j < rec_att[1][currentIndex]:
                    rec_att[1][currentIndex] = j
                #max i
                if i > rec_att[2][currentIndex]:
                    rec_att[2][currentIndex] = i
                #max j
                if j > rec_att[3][currentIndex]:
                    rec_att[3][currentIndex] = j
    return rec_att

def drawing_rect(img, rec_att):  #Draws rectange into image
    nrow = rec_att.shape[0]
    ncol = rec_att.shape[1]

    for j in range(ncol):
        draw = ImageDraw.Draw(img)
        shape = [(rec_att[3][j], rec_att[0][j]), (rec_att[1][j], rec_att[2][j])]
        # create rectangle image
        img1 = ImageDraw.Draw(img)
        img1.rectangle(shape, outline="red", width=1)

def resize_rec(im, min_x, min_y, max_x, max_y): #crops and resizes every label
    im2 = im.crop((min_y, min_x, max_y, max_x))
    im3 = im2.resize((21, 21))
    return im3

def write_assumptions(rec_att, img, results): #writes estimations into image
    nrow = rec_att.shape[0]
    ncol = rec_att.shape[1]
    for j in range(ncol):
        x1 = rec_att[1][j]
        y1 = rec_att[0][j]
        x2 = rec_att[3][j]
        y2 = rec_att[2][j]
        # create rectangle image
        txt = Image.new('RGBA', img.size, (255, 255, 255, 0))
        fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 25)
        d = ImageDraw.Draw(txt)
        d.text(((x1 + x2 -15) / 2, y1-22), str(results[j]), font=fnt, fill=(0, 0, 0, 255))
        img = Image.alpha_composite(img, txt)

    img.show()

def calc_moments_hu(resized_image): #calculates hu moments for every resized label
    f = np.asarray(resized_image)
    nrow = f.shape[0]
    ncol = f.shape[1]

    raw_moments = [[0, 0], [0, 0]]

    for i in range(2):
        for j in range(2):
            for x in range(nrow):
                for y in range(ncol):
                    raw_moments[i][j] += pow(x, i) * pow(y, j) * f[x][y]

    central_moments = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    xZero = raw_moments[1][0] / raw_moments[0][0]
    yZero = raw_moments[0][1] / raw_moments[0][0]

    for i in range(4):
        for j in range(4):
            for x in range(nrow):
                for y in range(ncol):
                    central_moments[i][j] += pow(x - xZero, i) * pow(y - yZero, j) * f[x][y]

    normalized_central = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    for i in range(4):
        for j in range(4):
            for x in range(nrow):
                for y in range(ncol):
                    normalized_central[i][j] = central_moments[i][j] / pow(central_moments[0][0], (1 + ((i + j) / 2)))

    H1 = normalized_central[2][0] + normalized_central[0][2]
    H2 = pow((normalized_central[2][0] - normalized_central[0][2]), 2) + (4 * pow((normalized_central[1][1]), 2))
    H3 = pow((normalized_central[3][0] - (3 * normalized_central[1][2])), 2) + pow(
        ((3 * normalized_central[2][1]) - normalized_central[0][3]), 2)
    H4 = pow((normalized_central[3][0] + normalized_central[1][2]), 2) + pow(
        (normalized_central[2][1] + normalized_central[0][3]), 2)
    H5 = (normalized_central[3][0] - (3 * normalized_central[1][2])) * (
    (normalized_central[3][0] + normalized_central[1][2])) * \
         ((pow((normalized_central[3][0] + normalized_central[1][2]), 2)) - (
                     3 * (pow((normalized_central[2][1] + normalized_central[0][3]), 2)))) + \
         ((3 * normalized_central[2][1]) - normalized_central[0][3]) * (
                     normalized_central[2][1] + normalized_central[0][3]) * \
         ((3 * (pow((normalized_central[3][0] + normalized_central[1][2]), 2))) - pow(
             (normalized_central[2][1] + normalized_central[0][3]), 2))
    H6 = (normalized_central[2][0] - normalized_central[0][2]) * \
         (pow((normalized_central[3][0] + normalized_central[1][2]), 2) - pow(
             (normalized_central[2][1] + normalized_central[0][3]), 2)) + \
         (4 * normalized_central[1][1]) * (normalized_central[3][0] + normalized_central[1][2]) * (
                     normalized_central[2][1] + normalized_central[0][3])
    H7 = -(((3 * normalized_central[2][1]) - normalized_central[0][3]) * (
                normalized_central[3][0] + normalized_central[1][2]) * \
           (pow((normalized_central[3][0] + normalized_central[1][2]), 2) - (
                       3 * pow(normalized_central[2][1] + normalized_central[0][3], 2))) - \
           ((normalized_central[3][0] - (3 * normalized_central[1][2]))) * (
                       normalized_central[2][1] + normalized_central[0][3]) * \
           ((3 * pow((normalized_central[3][0] + normalized_central[1][2]), 2)) - (
           (pow((normalized_central[2][1] + normalized_central[0][3]), 2)))))

    return [H1, H2, H3, H4, H5, H6, H7]

def calc_moments_r(resized_image): #calculates R moments for every resized label
    hu_moments = calc_moments_hu(resized_image)

    r1 = math.sqrt(hu_moments[1])/hu_moments[0]
    r2 = (hu_moments[0] + math.sqrt(hu_moments[1]))/(hu_moments[0] - math.sqrt(hu_moments[1]))
    r3 = math.sqrt(hu_moments[2])/math.sqrt(hu_moments[3])
    r4 = math.sqrt(hu_moments[2])/math.sqrt(abs(hu_moments[4]))
    r5 = math.sqrt(hu_moments[3])/math.sqrt(abs(hu_moments[4]))
    r6 = abs(hu_moments[5])/(hu_moments[0]*hu_moments[2])
    r7 = abs(hu_moments[5])/(hu_moments[0]*math.sqrt(abs(hu_moments[4])))
    r8 = abs(hu_moments[5])/(hu_moments[2]*math.sqrt(abs(hu_moments[1])))
    r9 = abs(hu_moments[5])/math.sqrt(hu_moments[1]*abs(hu_moments[4]))
    r10 = abs(hu_moments[4])/(hu_moments[2]*hu_moments[3])

    return [r1,r2,r3,r4,r5,r6,r7,r8,r9,r10]

def calc_moments_zernike(resized_image): #calculates Zernike moments for every resized label
    z11 = math.sqrt(pow(zernike_ZRnm(resized_image, 1, 1), 2) + pow(zernike_ZInm(resized_image, 1, 1), 2))
    z22 = math.sqrt(pow(zernike_ZRnm(resized_image, 2, 2), 2) + pow(zernike_ZInm(resized_image, 2, 2), 2))
    z31 = math.sqrt(pow(zernike_ZRnm(resized_image, 3, 1), 2) + pow(zernike_ZInm(resized_image, 3, 1), 2))
    z33 = math.sqrt(pow(zernike_ZRnm(resized_image, 3, 3), 2) + pow(zernike_ZInm(resized_image, 3, 3), 2))
    z42 = math.sqrt(pow(zernike_ZRnm(resized_image, 4, 2), 2) + pow(zernike_ZInm(resized_image, 4, 2), 2))
    z44 = math.sqrt(pow(zernike_ZRnm(resized_image, 4, 4), 2) + pow(zernike_ZInm(resized_image, 4, 4), 2))
    z51 = math.sqrt(pow(zernike_ZRnm(resized_image, 5, 1), 2) + pow(zernike_ZInm(resized_image, 5, 1), 2))
    z53 = math.sqrt(pow(zernike_ZRnm(resized_image, 5, 3), 2) + pow(zernike_ZInm(resized_image, 5, 3), 2))
    z55 = math.sqrt(pow(zernike_ZRnm(resized_image, 5, 5), 2) + pow(zernike_ZInm(resized_image, 5, 5), 2))
    z62 = math.sqrt(pow(zernike_ZRnm(resized_image, 6, 3), 2) + pow(zernike_ZInm(resized_image, 6, 2), 2))
    z64 = math.sqrt(pow(zernike_ZRnm(resized_image, 6, 4), 2) + pow(zernike_ZInm(resized_image, 6, 4), 2))
    z66 = math.sqrt(pow(zernike_ZRnm(resized_image, 6, 6), 2) + pow(zernike_ZInm(resized_image, 6, 6), 2))

    return [z11,z22,z31,z33,z42,z44,z51,z53,z55,z62,z64,z66]

def zernike_rnm(n, m, pij): #calculates Rnm for Zernike moment calculations
    rnm = 0
    for i in range(int((n-abs(m))/2)):
        rnm += (pow(-1,i)*pow(pij,(n-2*i))*math.factorial(n-i))/(math.factorial(i)*math.factorial(int(((n+abs(m))/2)-i))*math.factorial(int(((n-abs(m))/2))-i))

    return rnm

def zernike_ZRnm(resized_image, n, m): #calculates ZRnm for Zernike moment calculations
    f = np.asarray(resized_image)
    nrow = f.shape[0]
    ncol = f.shape[1]

    zr = 0

    for i in range(nrow):
        for j in range(ncol):
            xi = ((math.sqrt(2)/(nrow-1))*i) - 1/math.sqrt(2)
            yj = ((math.sqrt(2)/(nrow-1))*j) - 1/math.sqrt(2)
            pij = math.sqrt(pow(xi,2)+pow(yj,2))
            try:
                qij = math.atan(yj/xi)
            except ZeroDivisionError:
                qij = 0

            zr += f[i][j] * zernike_rnm(n, m, pij) * math.cos(m * qij) * pow(2 / (nrow * math.sqrt(2)), 2)
    zr *= (n+1)/math.pi

    return zr

def zernike_ZInm(resized_image, n, m): #calculates ZInm for Zernike moment calculations
    f = np.asarray(resized_image)
    nrow = f.shape[0]
    ncol = f.shape[1]

    zi = 0

    for i in range(nrow):
        for j in range(ncol):
            xi = ((math.sqrt(2) / (nrow - 1)) * i) - 1 / math.sqrt(2)
            yj = ((math.sqrt(2) / (nrow - 1)) * j) - 1 / math.sqrt(2)
            pij = math.sqrt(pow(xi,2)+pow(yj,2))
            try:
                qij = math.atan(yj/xi)
            except ZeroDivisionError:
                qij = 0
            zi += f[i][j] * zernike_rnm(n, m, pij) * math.sin(m * qij) * pow(2 / (nrow * math.sqrt(2)), 2)
    zi *= -(n+1)/math.pi

    return zi

def multiple_comparison(feature_vectors, recAtt, type): #Comparison Method 1
    numberOfSource = 10
    number_of_label = feature_vectors.shape[0]
    number_of_moment = feature_vectors.shape[1]
    alignment = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # LOG Transformation
    # Scaling Moments with Log
    for k in range(number_of_label):
        for l in range(number_of_moment):
            if feature_vectors[k][l] != 0:
                feature_vectors[k][l] = -np.sign(feature_vectors[k][l]) * np.log10(np.abs(feature_vectors[k][l]))

    most_relevant = np.empty(shape=(number_of_label), dtype=int)
    for i in range(number_of_label):
        distances = np.empty(shape=(number_of_label,numberOfSource), dtype=np.double)

        for j in range(numberOfSource):
            filename = "Database/source" + type + str(j) + ".npy"
            momentSource = np.load(filename)  # SOURCE LOAD
            characterOfSource = momentSource.shape[0]
            numberOfMoments = momentSource.shape[1]

            #LOG Transformation
            for k in range(characterOfSource):
                for l in range(numberOfMoments):
                    if momentSource[k][l] != 0:
                        momentSource[k][l] = -np.sign(momentSource[k][l]) * np.log10(np.abs(momentSource[k][l]))

            distances0 = np.zeros(shape=(characterOfSource), dtype=np.double)

            for k in range(characterOfSource):
                if type == "Hu":
                    #Distance Calculation
                    dis1 = pow(momentSource[k][0] - feature_vectors[i][0], 2)
                    dis2 = pow(momentSource[k][1] - feature_vectors[i][1], 2)
                    dis3 = pow(momentSource[k][2] - feature_vectors[i][2], 2)
                    dis4 = pow(momentSource[k][3] - feature_vectors[i][3], 2)
                    dis5 = pow(momentSource[k][4] - feature_vectors[i][4], 2)
                    dis6 = pow(momentSource[k][5] - feature_vectors[i][5], 2)
                    dis7 = pow(momentSource[k][6] - feature_vectors[i][6], 2)

                    totalDis = dis1 + dis2 + dis3 + dis4 + dis5 + dis6 + dis7
                    totalDis = math.sqrt(totalDis)
                    distances0[k] = totalDis

                if type == "R":
                    #Distance Calculation
                    dis1 = pow(momentSource[k][0] - feature_vectors[i][0], 2)
                    dis2 = pow(momentSource[k][1] - feature_vectors[i][1], 2)
                    dis3 = pow(momentSource[k][2] - feature_vectors[i][2], 2)
                    dis4 = pow(momentSource[k][3] - feature_vectors[i][3], 2)
                    dis5 = pow(momentSource[k][4] - feature_vectors[i][4], 2)
                    dis6 = pow(momentSource[k][5] - feature_vectors[i][5], 2)
                    dis7 = pow(momentSource[k][6] - feature_vectors[i][6], 2)
                    dis8 = pow(momentSource[k][7] - feature_vectors[i][7], 2)
                    dis9 = pow(momentSource[k][8] - feature_vectors[i][8], 2)
                    dis10 = pow(momentSource[k][9] - feature_vectors[i][9], 2)

                    totalDis = dis1 + dis2 + dis3 + dis4 + dis5 + dis6 + dis7 + dis8 + dis9 + dis10
                    totalDis = math.sqrt(totalDis)
                    distances0[k] = totalDis


                if type == "Zernike":
                    #Distance Calculation
                    dis1 = pow(momentSource[k][0] - feature_vectors[i][0], 2)
                    dis2 = pow(momentSource[k][1] - feature_vectors[i][1], 2)
                    dis3 = pow(momentSource[k][2] - feature_vectors[i][2], 2)
                    dis4 = pow(momentSource[k][3] - feature_vectors[i][3], 2)
                    dis5 = pow(momentSource[k][4] - feature_vectors[i][4], 2)
                    dis6 = pow(momentSource[k][5] - feature_vectors[i][5], 2)
                    dis7 = pow(momentSource[k][6] - feature_vectors[i][6], 2)
                    dis8 = pow(momentSource[k][7] - feature_vectors[i][7], 2)
                    dis9 = pow(momentSource[k][8] - feature_vectors[i][8], 2)
                    dis10 = pow(momentSource[k][9] - feature_vectors[i][9], 2)
                    dis11 = pow(momentSource[k][10] - feature_vectors[i][10], 2)
                    dis12 = pow(momentSource[k][11] - feature_vectors[i][11], 2)

                    totalDis = dis1 + dis2 + dis3 + dis4 + dis5 + dis6 + dis7 + dis8 + dis9 + dis10 + dis11 + dis12
                    totalDis = math.sqrt(totalDis)
                    distances0[k] = totalDis

            distances[i][j] = sum(distances0)/len(distances0) #Average calculation for every source
        most_relevant[i] = alignment[np.argmin(distances[i])] #Takes minumum average to find prediction for every label
    return most_relevant

def multiple_comparison2(feature_vectors, rec_att, type): #Comparison Method 2
    number_of_source = 10
    number_of_label = feature_vectors.shape[0]
    number_of_moment = feature_vectors.shape[1]
    alignment = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # LOG Transformation
    # Scaling Moments with Log
    for k in range(number_of_label):
        for l in range(number_of_moment):
            if feature_vectors[k][l] != 0:
                feature_vectors[k][l] = -np.sign(feature_vectors[k][l]) * np.log10(np.abs(feature_vectors[k][l]))

    most_relevant = np.empty(shape=(number_of_label), dtype=int)
    for i in range(number_of_label):
        distances = np.empty(shape=(number_of_label,number_of_source), dtype=np.double)

        for j in range(number_of_source):
            filename = "Database/source" + type + str(j) + ".npy"
            momentSource = np.load(filename)  # SOURCE LOAD
            characterOfSource = momentSource.shape[0]
            numberOfMoments = momentSource.shape[1]

            #LOG Transformation
            for k in range(characterOfSource):
                for l in range(numberOfMoments):
                    if momentSource[k][l] != 0:
                        momentSource[k][l] = -np.sign(momentSource[k][l]) * np.log10(np.abs(momentSource[k][l]))

            distance1 = sys.maxsize
            for k in range(characterOfSource):
                if type == "Hu":
                    #Distance Calculation
                    dis1 = pow(momentSource[k][0] - feature_vectors[i][0], 2)
                    dis2 = pow(momentSource[k][1] - feature_vectors[i][1], 2)
                    dis3 = pow(momentSource[k][2] - feature_vectors[i][2], 2)
                    dis4 = pow(momentSource[k][3] - feature_vectors[i][3], 2)
                    dis5 = pow(momentSource[k][4] - feature_vectors[i][4], 2)
                    dis6 = pow(momentSource[k][5] - feature_vectors[i][5], 2)
                    dis7 = pow(momentSource[k][6] - feature_vectors[i][6], 2)

                    totalDis = dis1 + dis2 + dis3 + dis4 + dis5 + dis6 + dis7
                    totalDis = math.sqrt(totalDis)
                    if totalDis < distance1: #Find minimum distance
                        distance1=totalDis

                if type == "R":
                    #Distance Calculation
                    dis1 = pow(momentSource[k][0] - feature_vectors[i][0], 2)
                    dis2 = pow(momentSource[k][1] - feature_vectors[i][1], 2)
                    dis3 = pow(momentSource[k][2] - feature_vectors[i][2], 2)
                    dis4 = pow(momentSource[k][3] - feature_vectors[i][3], 2)
                    dis5 = pow(momentSource[k][4] - feature_vectors[i][4], 2)
                    dis6 = pow(momentSource[k][5] - feature_vectors[i][5], 2)
                    dis7 = pow(momentSource[k][6] - feature_vectors[i][6], 2)
                    dis8 = pow(momentSource[k][7] - feature_vectors[i][7], 2)
                    dis9 = pow(momentSource[k][8] - feature_vectors[i][8], 2)
                    dis10 = pow(momentSource[k][9] - feature_vectors[i][9], 2)

                    totalDis = dis1 + dis2 + dis3 + dis4 + dis5 + dis6 + dis7 + dis8 + dis9 + dis10
                    totalDis = math.sqrt(totalDis)
                    if totalDis < distance1: #Find minimum distance
                        distance1=totalDis

                if type == "Zernike":
                    #Distance Calculation
                    dis1 = pow(momentSource[k][0] - feature_vectors[i][0], 2)
                    dis2 = pow(momentSource[k][1] - feature_vectors[i][1], 2)
                    dis3 = pow(momentSource[k][2] - feature_vectors[i][2], 2)
                    dis4 = pow(momentSource[k][3] - feature_vectors[i][3], 2)
                    dis5 = pow(momentSource[k][4] - feature_vectors[i][4], 2)
                    dis6 = pow(momentSource[k][5] - feature_vectors[i][5], 2)
                    dis7 = pow(momentSource[k][6] - feature_vectors[i][6], 2)
                    dis8 = pow(momentSource[k][7] - feature_vectors[i][7], 2)
                    dis9 = pow(momentSource[k][8] - feature_vectors[i][8], 2)
                    dis10 = pow(momentSource[k][9] - feature_vectors[i][9], 2)
                    dis11 = pow(momentSource[k][10] - feature_vectors[i][10], 2)
                    dis12 = pow(momentSource[k][11] - feature_vectors[i][11], 2)

                    totalDis = dis1 + dis2 + dis3 + dis4 + dis5 + dis6 + dis7 + dis8 + dis9 + dis10 + dis11 + dis12
                    totalDis = math.sqrt(totalDis)
                    if totalDis < distance1: #Find minimum distance
                        distance1 = totalDis

            distances[i][j] = distance1 #Assign minimum distance
        most_relevant[i] = alignment[np.argmin(distances[i])] #Find minimum distance from every source
    return most_relevant


def multiple_comparison3(feature_vectors, rec_att, type): #Comparison Method 3
    number_of_source = 10
    number_of_label = feature_vectors.shape[0]
    number_of_moment = feature_vectors.shape[1]
    alignment = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # LOG Transformation
    # Scaling Moments with Log
    for k in range(number_of_label):
        for l in range(number_of_moment):
            if feature_vectors[k][l] != 0:
                feature_vectors[k][l] = -np.sign(feature_vectors[k][l]) * np.log10(np.abs(feature_vectors[k][l]))

    most_relevant = np.empty(shape=(number_of_label), dtype=int)
    for i in range(number_of_label):
        distances = np.empty(shape=(number_of_label,number_of_source), dtype=np.double)

        for j in range(number_of_source):
            filename = "Database/source" + type + str(j) + ".npy"
            momentSource = np.load(filename)  # SOURCE LOAD
            characterOfSource = momentSource.shape[0]
            numberOfMoments = momentSource.shape[1]

            #LOG Transformation
            for k in range(characterOfSource):
                for l in range(numberOfMoments):
                    if momentSource[k][l] != 0:
                        momentSource[k][l] = -np.sign(momentSource[k][l]) * np.log10(np.abs(momentSource[k][l]))

            distance1 = sys.maxsize
            for k in range(characterOfSource):
                if type == "Hu":
                    #Ratio calculation
                    ratio1 = abs(((momentSource[k][0] - feature_vectors[i][0]) / momentSource[j][0]) * 100)
                    ratio2 = abs(((momentSource[k][1] - feature_vectors[i][1]) / momentSource[j][1]) * 100)
                    ratio3 = abs(((momentSource[k][2] - feature_vectors[i][2]) / momentSource[j][2]) * 100)
                    ratio4 = abs(((momentSource[k][3] - feature_vectors[i][3]) / momentSource[j][3]) * 100)
                    ratio5 = abs(((momentSource[k][4] - feature_vectors[i][4]) / momentSource[j][4]) * 100)
                    ratio6 = abs(((momentSource[k][5] - feature_vectors[i][5]) / momentSource[j][5]) * 100)
                    ratio7 = abs(((momentSource[k][6] - feature_vectors[i][6]) / momentSource[j][6]) * 100)

                    totalRatio = ratio1 + ratio2 + ratio3 + ratio4 + ratio5 + ratio6 + ratio7
                    if totalRatio < distance1: #Find minimum ratio
                        distance1=totalRatio

                if type == "R":
                    #Ratio calculation
                    ratio1 = abs(((momentSource[k][0] - feature_vectors[i][0]) / momentSource[j][0]) * 100)
                    ratio2 = abs(((momentSource[k][1] - feature_vectors[i][1]) / momentSource[j][1]) * 100)
                    ratio3 = abs(((momentSource[k][2] - feature_vectors[i][2]) / momentSource[j][2]) * 100)
                    ratio4 = abs(((momentSource[k][3] - feature_vectors[i][3]) / momentSource[j][3]) * 100)
                    ratio5 = abs(((momentSource[k][4] - feature_vectors[i][4]) / momentSource[j][4]) * 100)
                    ratio6 = abs(((momentSource[k][5] - feature_vectors[i][5]) / momentSource[j][5]) * 100)
                    ratio7 = abs(((momentSource[k][6] - feature_vectors[i][6]) / momentSource[j][6]) * 100)
                    ratio8 = abs(((momentSource[k][7] - feature_vectors[i][7]) / momentSource[j][7]) * 100)
                    ratio9 = abs(((momentSource[k][8] - feature_vectors[i][8]) / momentSource[j][8]) * 100)
                    ratio10 = abs(((momentSource[k][9] - feature_vectors[i][9]) / momentSource[j][9]) * 100)

                    totalRatio = ratio1 + ratio2 + ratio3 + ratio4 + ratio5 + ratio6 + ratio7 + ratio8 + ratio9 + ratio10
                    if totalRatio < distance1:#Find minimum ratio
                        distance1 = totalRatio

                if type == "Zernike":
                    #Ratio calculation
                    ratio1 = 0.0 #Zero according to Zernike Moment
                    ratio2 = 0.0 #Zero according to Zernike Moment
                    ratio3 = abs(((momentSource[k][2] - feature_vectors[i][2]) / momentSource[j][2]) * 100)
                    ratio4 = 0.0 #Zero according to Zernike Moment
                    ratio5 = abs(((momentSource[k][4] - feature_vectors[i][4]) / momentSource[j][4]) * 100)
                    ratio6 = 0.0 #Zero according to Zernike Moment
                    ratio7 = abs(((momentSource[k][6] - feature_vectors[i][6]) / momentSource[j][6]) * 100)
                    ratio8 = abs(((momentSource[k][7] - feature_vectors[i][7]) / momentSource[j][7]) * 100)
                    ratio9 = 0.0 #Zero according to Zernike Moment
                    ratio10 = abs(((momentSource[k][9] - feature_vectors[i][9]) / momentSource[j][9]) * 100)
                    ratio11 = abs(((momentSource[k][10] - feature_vectors[i][10]) / momentSource[j][10]) * 100)
                    ratio12 = 0.0 #Zero according to Zernike Moment

                    totalRatio = ratio1 + ratio2 + ratio3 + ratio4 + ratio5 + ratio6 + ratio7 + ratio8 + ratio9 + ratio10 + ratio11 + ratio12
                    if totalRatio < distance1:#Find minimum ratio
                        distance1 = totalRatio

            distances[i][j] = distance1
        most_relevant[i] = alignment[np.argmin(distances[i])]
    return most_relevant

if __name__=='__main__':
    main()
