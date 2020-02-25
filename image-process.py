from PIL import *
from PIL import Image, ImageDraw
import numpy as np
import cv2
import sys
import math

def main():

    img = Image.open('font1.png') #fotoğrafı açıyor
    img_gray = img.convert('L') # converts the image to grayscale image

    ONE = 255
    a = np.asarray(img_gray)  # from PIL to np array
    a_bin = threshold(a, 100, ONE, 0)
    im = Image.fromarray(a_bin)  # from np array to PIL format

    #RENKLENDİRME VE LABEL
    label = blob_coloring_8_connected(a_bin, ONE)
    new_img2 = np2PIL_color(label)
    new_img2.show()

    #DİKDÖRTGEN
    labelDict = addDictLabelled(label)
    #print("LabelDict")
    #print(labelDict)
    recAtt = findRectangle(label, labelDict)
    #print("-**-*-*-*-*-*-*-*-*-*")
    #print("RecAtt")
    #print(recAtt)

    #DİKDÖRTGENLERİ ÇİZ
    drawingRect(img,recAtt)

    #CROP AND RESIZE
    numberOfLabels = len(labelDict)
    numberOfMoments = 7 #HU İÇİN 7
    featureVectors = np.empty(shape=(numberOfLabels, numberOfMoments), dtype=np.double)  # FeatureVectors Array Initialization
    for i in range(numberOfLabels):
        minx = recAtt[0][i]
        miny = recAtt[1][i]
        maxx = recAtt[2][i]
        maxy = recAtt[3][i]

        resizedIm = resizeRec(im, minx, miny, maxx, maxy)  #crops and resizes every character

        #CALCULATION HU MOMENTS
        currentMoments = calcMomentsHu(resizedIm)  #calculates HU moments of character
        featureVectors[i] = currentMoments  #stores hu moments of character
        np.set_printoptions(threshold=sys.maxsize)
        #print(featureVectors)


        #huMomentSource = loadHuSources()

    #np.save('font7hu',featureVectors) #SOURCE KAYIT

    #print("num: ", numberOfLabels)


    #MANUAL COMPARING
    huMomentSource = np.load('font2hu.npy')  #SOURCE LOAD
    a = compareMomentsVER2(featureVectors, huMomentSource, recAtt)
    print(a)
    print("*********")


    multipleComparison(featureVectors,recAtt)










def np2PIL(im):
    print("size of arr: ",im.shape)
    img = Image.fromarray(im, 'RGB')
    return img

def np2PIL_color(im):
    print("size of arr: ",im.shape)
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
    #print("nrow, ncol", nrow, ncol)
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

def rgbToHash(rgbArray):
    r = rgbArray[0]*1000000
    g = rgbArray[1]*1000
    b = rgbArray[2]

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

def addDictLabelled(color_im):
    nrow = color_im.shape[0]
    ncol = color_im.shape[1]

    labelDict = {}
    for i in range(nrow):
        for j in range(ncol):
            if color_im[i][j][0] != 0 and color_im[i][j][1] != 0 and color_im[i][j][2] != 0:
                labelDict[rgbToHash(color_im[i][j])] = [i,j]

    labelDict = addDictLabelled2(labelDict)
    return labelDict

def addDictLabelled2(labelDict):
    labelDict2 = {}
    numberOfLabels = 0
    for i in labelDict:
        #print("i: ", i)
        labelDict2[i] = numberOfLabels
        numberOfLabels += 1

    return labelDict2

def findRectangle(color_im, labelDict):
    numberOfLabels = len(labelDict)
    recAtt = np.zeros(shape=(4, numberOfLabels))

    nrow = color_im.shape[0]
    ncol = color_im.shape[1]

    for t in range(numberOfLabels):
        recAtt[0][t]=10000
        recAtt[1][t]=10000

    for i in range(nrow):
        for j in range(ncol):
            if color_im[i][j][0] != 0 and color_im[i][j][1] != 0 and color_im[i][j][2] != 0:
                current = rgbToHash(color_im[i][j])
                currentIndex = labelDict[current]

                #min i
                if i < recAtt[0][currentIndex]:
                    recAtt[0][currentIndex] = i
                #min j
                if j < recAtt[1][currentIndex]:
                    recAtt[1][currentIndex] = j
                #max i
                if i > recAtt[2][currentIndex]:
                    recAtt[2][currentIndex] = i
                #max j
                if j > recAtt[3][currentIndex]:
                    recAtt[3][currentIndex] = j
    return recAtt

def drawingRect(img,recAtt):
    nrow = recAtt.shape[0]
    ncol = recAtt.shape[1]

    for j in range(ncol):
        draw = ImageDraw.Draw(img)
        shape = [(recAtt[3][j], recAtt[0][j]),(recAtt[1][j], recAtt[2][j])]
        #print(shape)
        # create rectangle image
        img1 = ImageDraw.Draw(img)
        img1.rectangle(shape, outline="red", width=1)
    img.show()

def resizeRec(im,minx,miny,maxx,maxy):
    im2 = im.crop((miny, minx, maxy, maxx))
    im3 = im2.resize((21, 21))
    return im3

def calcMomentsHu(resizedIm):
    f = np.asarray(resizedIm)
    nrow = f.shape[0]
    ncol = f.shape[1]

    rawMoments = [[0, 0], [0, 0]]

    for i in range(2):
        for j in range(2):
            for x in range(nrow):
                for y in range(ncol):
                    rawMoments[i][j] += pow(x, i) * pow(y, j) * f[x][y]

    centralMoments = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    xZero = rawMoments[1][0] / rawMoments[0][0]
    yZero = rawMoments[0][1] / rawMoments[0][0]

    for i in range(4):
        for j in range(4):
            for x in range(nrow):
                for y in range(ncol):
                    centralMoments[i][j] += pow(x - xZero, i) * pow(y - yZero, j) * f[x][y]

    normalizedCentral = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    for i in range(4):
        for j in range(4):
            for x in range(nrow):
                for y in range(ncol):
                    normalizedCentral[i][j] = centralMoments[i][j] / pow(centralMoments[0][0], (1 + ((i + j) / 2)))

    H1 = normalizedCentral[2][0] + normalizedCentral[0][2]
    H2 = pow((normalizedCentral[2][0] - normalizedCentral[0][2]), 2) + (4 * pow((normalizedCentral[1][1]), 2))
    H3 = pow((normalizedCentral[3][0] - (3 * normalizedCentral[1][2])), 2) + pow(
        ((3 * normalizedCentral[2][1]) - normalizedCentral[0][3]), 2)
    H4 = pow((normalizedCentral[3][0] + normalizedCentral[1][2]), 2) + pow(
        (normalizedCentral[2][1] + normalizedCentral[0][3]), 2)
    H5 = (normalizedCentral[3][0] - (3 * normalizedCentral[1][2])) * (
    (normalizedCentral[3][0] + normalizedCentral[1][2])) * \
         ((pow((normalizedCentral[3][0] + normalizedCentral[1][2]), 2)) - (
                     3 * (pow((normalizedCentral[2][1] + normalizedCentral[0][3]), 2)))) + \
         ((3 * normalizedCentral[2][1]) - normalizedCentral[0][3]) * (
                     normalizedCentral[2][1] + normalizedCentral[0][3]) * \
         ((3 * (pow((normalizedCentral[3][0] + normalizedCentral[1][2]), 2))) - pow(
             (normalizedCentral[2][1] + normalizedCentral[0][3]), 2))
    H6 = (normalizedCentral[2][0] - normalizedCentral[0][2]) * \
         (pow((normalizedCentral[3][0] + normalizedCentral[1][2]), 2) - pow(
             (normalizedCentral[2][1] + normalizedCentral[0][3]), 2)) + \
         (4 * normalizedCentral[1][1]) * (normalizedCentral[3][0] + normalizedCentral[1][2]) * (
                     normalizedCentral[2][1] + normalizedCentral[0][3])
    # Negative ???
    H7 = -(((3 * normalizedCentral[2][1]) - normalizedCentral[0][3]) * (
                normalizedCentral[3][0] + normalizedCentral[1][2]) * \
           (pow((normalizedCentral[3][0] + normalizedCentral[1][2]), 2) - (
                       3 * pow(normalizedCentral[2][1] + normalizedCentral[0][3], 2))) - \
           ((normalizedCentral[3][0] - (3 * normalizedCentral[1][2]))) * (
                       normalizedCentral[2][1] + normalizedCentral[0][3]) * \
           ((3 * pow((normalizedCentral[3][0] + normalizedCentral[1][2]), 2)) - (
           (pow((normalizedCentral[2][1] + normalizedCentral[0][3]), 2)))))

    '''print("H1 : " , H1)
    print("H2 : " , H2)
    print("H3 : " , H3)
    print("H4 : " , H4)
    print("H5 : " , H5)
    print("H6 : " , H6)
    print("H7 : " , H7)

    moments = cv2.moments(f)
    huMoments = cv2.HuMoments(moments)

    print("----------------")
    print(huMoments)'''

    return [H1, H2, H3, H4, H5, H6, H7]

def compareMoments(featureVectors, huMomentSource,recAtt,img):
    numberOfSource = huMomentSource.shape[0]
    numberOfMoments = huMomentSource.shape[2]
    characterOfSource = huMomentSource.shape[1]
    numberOfLabel = featureVectors.shape[0]
    numberOfHuMoment = featureVectors.shape[1]

    # Scaling Hu Moments with Log
    for i in range(numberOfSource):
        for j in range(characterOfSource):
            for k in range(numberOfMoments):
                #huMomentSource[i][j][k] = -np.sign(huMomentSource[i][j][k])* np.log10(np.abs(huMomentSource[i][j][k]))
                huMomentSource[i][j][k] = np.log10(abs(huMomentSource[i][j][k]))


    for i in range(numberOfLabel):
        for j in range(numberOfMoments):
            #featureVectors[i][j] = -np.sign(featureVectors[i][j])* np.log10(np.abs(featureVectors[i][j]))
            featureVectors[i][j] = np.log10(abs(featureVectors[i][j]))

    '''# Scaling Hu Moments with Log
    for i in range(numberOfSource):
        for j in range(characterOfSource):
            for k in range(numberOfMoments):
                huMoments[i] = -1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i])))'''


    #for a in range(numberOfLabel): #for feature vectors
    comparisonResults2 = np.zeros(shape=(numberOfLabel), dtype=int)
    for a in range(numberOfLabel): #for feature vectors
        comparisonResults = np.zeros(shape=(characterOfSource), dtype=int)
        distances = np.empty(shape=(characterOfSource), dtype=np.double)
        comparisonResults1 = np.zeros(shape=(characterOfSource), dtype=int)
        minIndex = -1
        maxIndex = -1
        for i in range(numberOfSource): #for font file number
            for j in range(characterOfSource): # for characters features
                print("*****************")
                print("a: ", a," i: ", i," j: ", j)
                huMoments = featureVectors[a] #karşılaştırmak istediğimiz label'ın tüm hu'ları burada [h1,h2,h3,h4,h5,h6,h7]
                sourceCharacters = huMomentSource[i] #font number i'deki tüm characterler
                sourceCharMoments = sourceCharacters[j] #bir source karakterinin momentleri
                '''print("huMoments[0]: " ,huMoments[0])
                print("huMoments[1]: " ,huMoments[1])
                print("sourceCharMoments[0]: ", sourceCharMoments[0])
                print("sourceCharMoments[1]: ", sourceCharMoments[1])'''

                dis1 = pow(huMoments[0] - sourceCharMoments[0],2)
                dis2 = pow(huMoments[1] - sourceCharMoments[1],2)
                dis3 = pow(huMoments[2] - sourceCharMoments[2],2)
                dis4 = pow(huMoments[3] - sourceCharMoments[3],2)
                dis5 = pow(huMoments[4] - sourceCharMoments[4],2)
                dis6 = pow(huMoments[5] - sourceCharMoments[5],2)
                dis7 = pow(huMoments[6] - sourceCharMoments[6],2)
                '''print("Dis1: ",dis1)
                print("Dis2: ",dis2)
                print("Dis3: ",dis3)
                print("Dis4: ",dis4)
                print("Dis5: ",dis5)
                print("Dis6: ",dis6)
                print("Dis7: ",dis7)'''


                totalDis = dis1+dis2+dis3+dis4+dis5+dis6+dis6
                totalDis = math.sqrt(totalDis)
                distances[j] = totalDis
                #print("bulunan distance:  " , distances[j])
            #find lowest distance and put this into comparison result array

            min = distances[0]
            minIndex = 0
            for p in range(characterOfSource-1):
                if distances[p+1] < min:
                    min = distances[p+1]
                    minIndex = p+1
            #comparisonResults[j] = minIndex

            #print("En küçük distance: ", min, " index'i ", minIndex)

            #print("Comparison Results: ")
            #print(comparisonResults)
            #print("**************")


            #find most frequent number and it assign to comparison result2 array with index a
            comparisonResults1[minIndex] = 1 + comparisonResults1[minIndex]

        #for t in range(characterOfSource):
        #    comparisonResults1[comparisonResults[t]] = 1 + comparisonResults1[comparisonResults[t]]
        '''print("**************")
        print("comparisonResults1 array. Sıklık:")
        print(comparisonResults1)'''
        #find maximum number and its index will be found character
        max = comparisonResults1[0]
        maxIndex = 0
        for l in range(characterOfSource):
            if comparisonResults1[l] > max:
                max = comparisonResults1[l]
                maxIndex = l
        comparisonResults2[a] = maxIndex

    print(comparisonResults2)
    #ekrana yazı yaz
    alignment = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    nrow = recAtt.shape[0]
    ncol = recAtt.shape[1]

    for j in range(ncol):
        x1=recAtt[2][j]
        y1=recAtt[1][j]
        x2=recAtt[4][j]
        y2=recAtt[3][j]
        # print(shape)
        # create rectangle image
        txt = Image.new('RGBA', img.size, (255, 255, 255, 0))
        #img1.rectangle(shape, outline="red")
        #font = ImageFont.load("arial.tff")
        #font = ImageFont.truetype("arial.tff", 15)
        fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 40)
        d = ImageDraw.Draw(txt)
        d.text(((x1+x2-20)/2, y1-20), str(alignment[comparisonResults2[j]]), font=fnt, fill=(0, 0, 0, 255))
        #img1.draw.text((x1+x2)/2, (y1+y2)/2, comparisonResults2[j], font=font)
        img = Image.alpha_composite(img, txt)

    img.show()



    return comparisonResults2


def multipleComparison(featureVectors,recAtt):
    numberOfSource = 7
    allAssumptions = np.zeros(shape=(numberOfSource,featureVectors.shape[0]), dtype=int)
    for i in range(numberOfSource):
        filename = "font" + str(i+1) + "hu.npy"
        print("filename is. " ,filename)
        huMomentSource = np.load(filename)  # SOURCE LOAD
        print("source")
        print(huMomentSource)
        print("**")

        '''# LOG Transformation
        # Scaling Hu Moments with Log
        for k in range(huMomentSource.shape[0]):
            for l in range(huMomentSource.shape[1]):
                #huMomentSource[k][l] = np.log10(abs(huMomentSource[k][l]))
                huMomentSource[k][l] = -np.sign(huMomentSource[k][l])* np.log10(np.abs(huMomentSource[k][l]))

        for k in range(featureVectors.shape[0]):
            for l in range(featureVectors.shape[1]):
                #featureVectors[k][l] = np.log10(abs(featureVectors[k][l]))
                featureVectors[k][l] = -np.sign(featureVectors[k][l])* np.log10(np.abs(featureVectors[k][l]))'''

        oneFontResults = compareMomentsVER2(featureVectors, huMomentSource, recAtt) #sadece 1 source karşılaşma sonuçları
        print(oneFontResults)
        allAssumptions[i] = oneFontResults

    print(allAssumptions)


def compareMomentsVER2(featureVectors, huMomentSource,recAtt):
    #numberOfSource = huMomentSource.shape[0]
    #print("number of source: ", numberOfSource)
    characterOfSource = huMomentSource.shape[0]
    #print("number of characters in source: " , characterOfSource)
    numberOfMoments = huMomentSource.shape[1]
    #print("number of moments in source: ", numberOfMoments)
    numberOfLabel = featureVectors.shape[0]
    #print("number of label: ", numberOfLabel)
    numberOfHuMoment = featureVectors.shape[1]
    #print("number of hu moment: ",numberOfHuMoment)
    alignment = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    mostRelevant = np.empty(shape=(numberOfLabel), dtype=int)

    '''#LOG Transformation
    # Scaling Hu Moments with Log
    for i in range(characterOfSource):
        for j in range(numberOfMoments):
            #huMomentSource[i][j] = np.log10(abs(huMomentSource[i][j]))
            huMomentSource[i][j] = -np.sign(huMomentSource[i][j])* np.log10(np.abs(huMomentSource[i][j]))

    for i in range(numberOfLabel):
        for j in range(numberOfHuMoment):
            #featureVectors[i][j] = np.log10(abs(featureVectors[i][j]))
            featureVectors[i][j] = -np.sign(featureVectors[i][j])* np.log10(np.abs(featureVectors[i][j]))'''

    for i in range(numberOfLabel):
        distances = np.empty(shape=(characterOfSource), dtype=np.double)
        for j in range(characterOfSource):
            '''print("*****************")
            print(" i: ", i, " j: ", j)'''
            '''print("huMomentSource[0]: " ,huMomentSource[j][0])
            print("huMomentSource[1]: " ,huMomentSource[j][1])
            print("sourceCharMoments[0]: ", featureVectors[i][0])
            print("sourceCharMoments[1]: ", featureVectors[i][1])'''

            '''dis1 = pow(huMomentSource[j][0] - featureVectors[i][0], 2)
            dis2 = pow(huMomentSource[j][1] - featureVectors[i][1], 2)
            dis3 = pow(huMomentSource[j][2] - featureVectors[i][2], 2)
            dis4 = pow(huMomentSource[j][3] - featureVectors[i][3], 2)
            dis5 = pow(huMomentSource[j][4] - featureVectors[i][4], 2)
            dis6 = pow(huMomentSource[j][5] - featureVectors[i][5], 2)
            dis7 = pow(huMomentSource[j][6] - featureVectors[i][6], 2)'''

            ratio1 = ((huMomentSource[j][0] - featureVectors[i][0])/huMomentSource[j][0])*100
            ratio2 = ((huMomentSource[j][1] - featureVectors[i][1])/huMomentSource[j][1])*100
            ratio3 = ((huMomentSource[j][2] - featureVectors[i][2])/huMomentSource[j][2])*100
            ratio4 = ((huMomentSource[j][3] - featureVectors[i][3])/huMomentSource[j][3])*100
            ratio5 = ((huMomentSource[j][4] - featureVectors[i][4])/huMomentSource[j][4])*100
            ratio6 = ((huMomentSource[j][5] - featureVectors[i][5])/huMomentSource[j][5])*100
            ratio7 = ((huMomentSource[j][6] - featureVectors[i][6])/huMomentSource[j][6])*100


            '''print("Dis1: ",dis1)
            print("Dis2: ",dis2)
            print("Dis3: ",dis3)
            print("Dis4: ",dis4)
            print("Dis5: ",dis5)
            print("Dis6: ",dis6)
            print("Dis7: ",dis7)'''

            ''''totalDis = dis1 + dis2 + dis3 + dis4 + dis5 + dis6 + dis6
            totalDis = math.sqrt(totalDis)'''
            #print("total distance is: ",totalDis)
            #distances[j] = totalDis

            totalRatio= ratio1+ratio2+ratio3+ratio4+ratio5+ratio6+ratio7
            distances[j] = totalRatio


        #bir i değeri için tüm j değerleri gezildi.

        print(distances)
        #find lowest distance and put this into comparison result array
        min = 10000
        minIndex = -1
        for p in range(characterOfSource):
            if distances[p] < min:
                min = distances[p]
                minIndex = p
        #print("minindex: ", minIndex)

        mostRelevant[i] = alignment[minIndex]

    print(".-*-.-*-.-*-.-*")
    print(mostRelevant)
    print(".-*-.-*-.-*-.-*")

    return mostRelevant


'''def writeAssumptions(recAtt,img, TAHMİNLERARRAY):
    nrow = recAtt.shape[0]
    ncol = recAtt.shape[1]
    for j in range(ncol):
        x1 = recAtt[2][j]
        y1 = recAtt[1][j]
        x2 = recAtt[4][j]
        y2 = recAtt[3][j]
        # print(shape)
        # create rectangle image
        txt = Image.new('RGBA', img.size, (255, 255, 255, 0))
        # img1.rectangle(shape, outline="red")
        # font = ImageFont.load("arial.tff")
        # font = ImageFont.truetype("arial.tff", 15)
        fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 40)
        d = ImageDraw.Draw(txt)
        d.text(((x1 + x2 - 20) / 2, y1 - 20), ????, font=fnt, fill=(0, 0, 0, 255))
        # img1.draw.text((x1+x2)/2, (y1+y2)/2, comparisonResults2[j], font=font)
        img = Image.alpha_composite(img, txt)

    img.show()'''


if __name__=='__main__':
    main()
