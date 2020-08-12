# 1920x1080 resimler icin
# resimleri tek tek alıyor
# her saniye text dosyasını güncelliyor
from __future__ import division

import threading
import numpy as np
import math
import time
import sys
import argparse
import os
import os.path as osp
import pickle as pkl
import pandas as pd
import time
from collections import Counter

import torch
from torch.autograd import Variable
import cv2

from util.parser import load_classes
from util.model import Darknet
from util.image_processor import prep_image
from util.utils import non_max_suppression


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 detection Module')

    parser.add_argument("--images", dest='images', help="Image / Directory containing images to perform detection upon",
                        default="imgs", type=str)
    parser.add_argument("--outputs", dest='outputs', help="Image / Directory to store detections", default="outputs",
                        type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help="Config file", default="config/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile", default="weights/yolov3A.weights", type=str)
    parser.add_argument("--reso", dest='reso',
                        help="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="448", type=str)

    return parser.parse_args()


args = arg_parse()
images = args.images
outputs_names = args.outputs
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()

classes = load_classes("data/coco.names")

counter = 0
# Set up the neural network
print("Loading network.....")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.hyperparams["height"] = args.reso
inp_dim = int(model.hyperparams["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32

num_classes = model.num_classes

# If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()

# Set the model in evaluation mode
model.eval()

read_dir = time.time()
# detection phase

integer = 1
indeks = 0
trafikteKalma = 1  # araclarin trafikte kaldiklari her frame icin kalma sureleri 0.25 artmali
trafikteKalma2 = 0  # araclarin trafikte kaldiklari her frame icin kalma sureleri 0.25 artmali
trafikteKalma3 = 0  # araclarin trafikte kaldiklari her frame icin kalma sureleri 0.25 artmali
trafikteKalma4 = 0  # araclarin trafikte kaldiklari her frame icin kalma sureleri 0.25 artmali
trafikteKalma5 = 1  # araclarin trafikte kaldiklari her frame icin kalma sureleri 0.25 artmali
counter = 0
counter2 = 0
counter3 = 0  # line ve line2 arasindaki arac sayisi
counterAracImg = 0  # mevcut framedeki arac sayisi
counterAracPrev = 0  # onceki framedeki arac sayisi
ilkImg = 0  # gelen frame ilk framese 0 degilse 1

line = [(875, 320), (1040, 320)]  # yol giris   saymada kullanilan      counter(+)
line2 = [(750, 900), (1160, 900)]  # yol cikis   saymada kullanilan      counter2(-)
# line3 = [(840, 550), (1070, 550)]  # line2 ile bu çizginin arasindaki arac sayisini tutmak icin counter3 kullaniyor

lineIlk = [(875, 400), (1040, 400)]  # 20br yeterli
lineIki = [(840, 555), (1070, 555)]  # 20br yeterli
lineUc = [(800, 700), (1050, 700)]  # 20br yeterli
lineDort = [(750, 955), (1160, 955)]  # 20br yeterli
prev = None

t0 = time.time()  # trafigin baslama suresi
t1 = time.time()  # trafigin baslama suresinden beri gecen sure

t2 = time.time()  # birinci cizgideki araclarin kalma suresi
t3 = time.time()  # ikinci cizgideki araclarin kalma suresi
t4 = time.time()  # ucuncu cizgideki araclarin kalma suresi
t5 = time.time()  # dorduncu cizgideki araclarin kalma suresi

t2_2 = time.time()
t3_2 = time.time()
t4_2 = time.time()
t5_2 = time.time()

total = 0  # trafigin toplam suresi
total2 = 0  # ilk cizgideki trafik suresi
total3 = 0  # ikinci cizgideki trafik suresi
total4 = 0  # ucuncu cizgideki trafik suresi
total5 = 0  # dorduncu cizgideki trafik suresi
totalOrtalama = 0

imgSayisi = 1001
toplamSure = 0
green = 0
t2var = 0  # ikinci cizgide arac var mi varsa 1
t3var = 0  # ucuncu cizgide arac var mi varsa 1
t4var = 0  # dorduncu cizgide arac var mi varsa 1

# 1. satır satır okuma
# 2. son frame bişey yazmaması

totalYedek = 0
total2Yedek = 0
total3Yedek = 0
total4Yedek = 0
image = None
trafikOlusturanAracSayisi = 0;  # en alttaki cizgide bekleyen arac sayisini tutacak
trafikOlusturanAracSayisiYedek = 0;


def yazdir(img):
    global counter
    global counter2
    global counterAracImg
    global counterAracPrev
    global total
    global total2
    global total3
    global total4
    global total5
    global totalOrtalama

    cv2.putText(img, str(counter), (40, 200), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 1)
    cv2.putText(img, str(counter2), (360, 200), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 1)
    cv2.putText(img, str("%.5f" % total), (200, 600), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 1)
    cv2.putText(img, str("%.5f" % total2), (1200, 400), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 1)
    cv2.putText(img, str("%.5f" % total3), (1200, 555), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 1)
    cv2.putText(img, str("%.5f" % total4), (1200, 700), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 1)
    cv2.putText(img, str("%.5f" % total5), (1200, 955), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 1)
    cv2.putText(prev, str("%.5f" % totalOrtalama), (200, 500), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 1)
    cv2.putText(img, str(counterAracImg), (10, 600), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 1)
    cv2.putText(img, str(counterAracPrev), (100, 600), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 1)


def sifirla():
    global counter
    global counter2
    global counter3
    global counterAracImg
    global counterAracPrev
    global t0
    global t1
    global t2
    global t3
    global t4
    global t5
    global t2_2
    global t3_2
    global t4_2
    global t5_2
    global total
    global total2
    global total3
    global total4
    global total5
    global ilkImg
    global trafikteKalma
    global trafikteKalma2
    global trafikteKalma3
    global trafikteKalma4
    global trafikteKalma5
    global imgSayisi
    global integer
    global green
    global t2var
    global t3var
    global t4var
    global totalYedek
    global total2Yedek
    global total3Yedek
    global total4Yedek

    t2var = 0
    t3var = 0
    t4var = 0
    t0 = time.time()
    t1 = time.time()
    t2 = time.time()
    t3 = time.time()
    t4 = time.time()
    t5 = time.time()
    t2_2 = time.time()
    t3_2 = time.time()
    t4_2 = time.time()
    t5_2 = time.time()
    counter3 = 0
    ilkImg = 0
    counterAracImg = 0


def writefunc(x, results):
    global counter
    global counter2
    global counter3
    global counterAracImg
    global counterAracPrev
    global prev
    global t0
    global t1
    global t2
    global t3
    global t4
    global t5
    global t2_2
    global t3_2
    global t4_2
    global t5_2
    global total
    global total2
    global total3
    global total4
    global total5
    global ilkImg
    global trafikteKalma
    global trafikteKalma2
    global trafikteKalma3
    global trafikteKalma4
    global trafikteKalma5
    global imgSayisi
    global integer
    global green
    global t2var
    global t3var
    global t4var
    global totalYedek
    global total2Yedek
    global total3Yedek
    global total4Yedek
    global trafikOlusturanAracSayisi
    global trafikOlusturanAracSayisiYedek
    global totalOrtalama

    if (integer == 1):
        prev = img[0]
        integer = 0
    elif (integer == 2):
        totalYedek = total
        total2Yedek = total2
        total3Yedek = total3
        total4Yedek = total4
        integer = 0

    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]

    if (np.array_equal(prev, img) == False and t2var == 0 and green == 1):  # içerde araç yoksa
        t2_2 = time.time()
        total2 = 0
    if (np.array_equal(prev, img) == False and t3var == 0 and green == 1):
        t3_2 = time.time()
        total3 = 0
    if (np.array_equal(prev, img) == False and t4var == 0 and green == 1):
        t4_2 = time.time()
        total4 = 0

    if (green == 0 and np.array_equal(prev, img) == False):  # kirmizi isiga gecmis
        t0 = time.time()
        t2_2 = time.time()
        t3_2 = time.time()
        t4_2 = time.time()
        t5_2 = time.time()
        total = 0
        total2 = 0
        total3 = 0
        total4 = 0
        total5 = 0
        trafikteKalma = 1
        trafikteKalma2 = 0
        trafikteKalma3 = 0
        trafikteKalma4 = 0
        trafikteKalma5 = 1
        t2var = 0
        t3var = 0
        t4var = 0
    elif (green == 1 and np.array_equal(prev, img) == False):  # yesil isik yanarken
        print("img :", imgSayisi)
        print("t1 : ", t1)
        print("t0 : ", t0)
        total = totalYedek + t1 - t0 + 1
        if (t2var == 1):
            total2 = total2Yedek + t2 - t2_2 + 1
        if (t3var == 1):
            total3 = total3Yedek + t3 - t3_2 + 1
        if (t4var == 1):
            total4 = total4Yedek + t4 - t4_2 + 1
    totalYedek = total
    total2Yedek = total2
    total3Yedek = total3
    total4Yedek = total4
    total5 = total

    if (np.array_equal(prev, img) == False):  # ayni framede degilsen
        trafikOlusturanAracSayisiYedek = trafikOlusturanAracSayisi
        trafikOlusturanAracSayisi = 0
        imgSayisi = imgSayisi + 1
        t2var = 0
        t3var = 0
        t4var = 0
        t0 = time.time()
        t1 = time.time()
        t2 = time.time()
        t3 = time.time()
        t4 = time.time()
        t5 = time.time()
        t2_2 = time.time()
        t3_2 = time.time()
        t4_2 = time.time()
        t5_2 = time.time()
        if (counterAracImg != 0):
            totalOrtalama =  (total + total2 + total3 + total4) / counterAracImg
        else:
            totalOrtalama = 0
        cv2.putText(prev, str(counter), (40, 200), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 1)
        cv2.putText(prev, str(counter2), (360, 200), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 1)
        cv2.putText(prev, str("%.5f" % total), (200, 600), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 1)
        cv2.putText(prev, str("%.5f" % total2), (1200, 400), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 1)
        cv2.putText(prev, str("%.5f" % total3), (1200, 555), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 1)
        cv2.putText(prev, str("%.5f" % total4), (1200, 700), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 1)
        cv2.putText(prev, str("%.5f" % total5), (1200, 955), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 1)
        cv2.putText(prev, str("%.5f" % totalOrtalama), (200, 500), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 1)
        cv2.putText(prev, str(counterAracImg), (10, 600), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 1)
        cv2.putText(prev, str(counterAracPrev), (100, 600), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 1)

    if (np.array_equal(prev, img) == False):  # ayni framede degilsen yani bir sonraki frame'e gecilmis
        ilkImg = 1
        counter3 = 0
        if (counterAracPrev >= counterAracImg + 2):  # trafik acilmissa
            t0 = time.time()
            t1 = time.time()
            t2 = time.time()
            t3 = time.time()
            t4 = time.time()
            t5 = time.time()
            t2_2 = time.time()
            t3_2 = time.time()
            t4_2 = time.time()
            t5_2 = time.time()
            trafikteKalma = 1
            trafikteKalma2 = 0
            trafikteKalma3 = 0
            trafikteKalma4 = 0
            trafikteKalma5 = 1
        counterAracPrev = counterAracImg
        counterAracImg = 0
    elif (ilkImg == 0):
        counterAracPrev = counterAracPrev + 1

    if int(x[0]) not in clss:
        clss[int(x[0])] = []
    cls = int(x[-1])
    color = colors[cls % 100]
    label = "{0}: {1}".format(classes[cls], str(clss[int(x[0])][cls]))
    cv2.rectangle(img, c1, c2, color, 1)
    p0 = (int(c1[0] + (c2[0] - c1[0]) / 2), int(c1[1] + (c2[1] - c1[1]) / 2))
    p1 = (int((c1[0] + (c2[0] - c1[0]) / 2)), int((c1[1] + (c2[1] - c1[1]) / 2)))

    cv2.line(img, p0, p1, [213, 0, 255], 4)
    counterAracImg = counterAracImg + 1

    # arac sayma blogu
    if noktaninDogruyaDikUzakligi(0, 1, -320, p0[0], p0[1]) >= -30 and noktaninDogruyaDikUzakligi(0, 1, -320, p0[0],
                                                                                                  p0[1]) <= 0:
        counter = counter + 1
    elif noktaninDogruyaDikUzakligi(0, 1, -370, p0[0], p0[1]) <= 30 and noktaninDogruyaDikUzakligi(0, 1, -370, p0[0],
                                                                                                   p0[1]) > 0:
        counter = counter + 1
    elif noktaninDogruyaDikUzakligi(0, 1, -900, p0[0], p0[1]) >= -300 and noktaninDogruyaDikUzakligi(0, 1, -900, p0[0],
                                                                                                     p0[1]) <= 0:
        trafikOlusturanAracSayisi = trafikOlusturanAracSayisi + 1;
        counter2 = counter2 - 1
    elif noktaninDogruyaDikUzakligi(0, 1, -900, p0[0], p0[1]) <= 300 and noktaninDogruyaDikUzakligi(0, 1, -900, p0[0],
                                                                                                    p0[1]) > 0:
        trafikOlusturanAracSayisi = trafikOlusturanAracSayisi + 1;
        counter2 = counter2 - 1

    if (noktaninDogruyaDikUzakligi(0, 1, -400, p0[0], p0[1]) >= -50 and noktaninDogruyaDikUzakligi(0, 1, -400, p0[0],
                                                                                                   p0[1]) <= 50):
        t2 = time.time()
        t2var = 1
    if (noktaninDogruyaDikUzakligi(0, 1, -555, p0[0], p0[1]) >= -50 and noktaninDogruyaDikUzakligi(0, 1, -555, p0[0],
                                                                                                   p0[1]) <= 50):
        t3 = time.time()
        t3var = 1
    if (noktaninDogruyaDikUzakligi(0, 1, -700, p0[0], p0[1]) >= -50 and noktaninDogruyaDikUzakligi(0, 1, -700, p0[0],
                                                                                                   p0[1]) <= 50):
        t4 = time.time()
        t4var = 1
    if (noktaninDogruyaDikUzakligi(0, 1, -955, p0[0], p0[1]) >= -50 and noktaninDogruyaDikUzakligi(0, 1, -955, p0[0],
                                                                                                   p0[1]) <= 50):
        t5 = time.time()

    start_point = (940, 30)
    end_point = (960, 10)
    start_point2 = (980, 30)
    end_point2 = (1000, 10)
    colorRed = (0, 0, 255)  # BGR
    colorGreen = (255, 255, 255)  # (0, 255, 0)

    if ((p0[1] <= 970 and p0[1] >= 320)):  # iki cizgi arasindaki arac sayisi counter3
        counter3 = counter3 + 1
    print(trafikOlusturanAracSayisiYedek)
    if (trafikOlusturanAracSayisiYedek >= 2):  # yesil isik yak

        colorGreen = (0, 255, 0)
        colorRed = (255, 255, 255)
        counter = 0
        counter2 = 0
        t1 = time.time()
        green = 1
    else:  # kirmizi isik yak
        colorRed = (0, 0, 255)
        colorGreen = (255, 255, 255)
        counter2 = 0
        green = 0

    cv2.rectangle(img, start_point, end_point, colorRed, -1)
    cv2.rectangle(img, start_point2, end_point2, colorGreen, -1)
    cv2.line(img, line[0], line[1], (0, 255, 255), 1)
    cv2.line(img, line2[0], line2[1], (0, 255, 255), 1)
    cv2.line(img, lineIlk[0], lineIlk[1], (0, 0, 0), 1)
    cv2.line(img, lineIki[0], lineIki[1], (0, 0, 0), 1)
    cv2.line(img, lineUc[0], lineUc[1], (0, 0, 0), 1)
    cv2.line(img, lineDort[0], lineDort[1], (0, 0, 0), 1)
    prev = img
    return img


def noktaninDogruyaDikUzakligi(a, b, c, x, y):
    return int((a * x + b * y + c) / math.sqrt(a * a + b * b))


def outText():
    while (1 == 1):
        try:
            time.sleep(1)
            # print("\noutText\n")
            directory = './Assets/'
            filename = "A.txt"
            file_path = os.path.join(directory, filename)
            fwrite = open(file_path, "w")
            # fwrite.write(str(total))
            # fwrite.write(format(total, '.2f'))
            fwrite.write(format(math.ceil(totalOrtalama), '.0f'))
            fwrite.write("\n")
            fwrite.write(str(counterAracImg))
            print("str :", str(counterAracImg))
            fwrite.close()
        except:
            print("cannot write to file")
            continue


class myThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        outText()


thread1 = myThread()
thread1.daemon = True
thread1.start()

if not os.path.exists(args.outputs):
    os.makedirs(args.outputs)

while (1 == 1):
    ilkImg = 0
    integer = 2
    imlist = []
    try:
        while (1 == 1):
            img = os.listdir(images)
            if (len(img) != 0):
                if (img[0].endswith(".png") == False):  # .png ile bitmiyorsa .png görene kadar sil
                    os.remove(os.path.join(images, img[0]))
                    continue
                else:  # ilk resmi aldık
                    break
            else:
                print("---------------------No image found---------------------")
                time.sleep(0.5)
                continue
        imlist = [osp.join(osp.realpath('.'), images, img[0])]

    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        print ("No file or directory with the name {}".format(images))
        exit()

    loaded_ims = [cv2.imread(x) for x in imlist]

    im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))
    im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

    leftover = 0
    if (len(im_dim_list) % batch_size):
        leftover = 1

    if batch_size != 1:
        num_batches = len(imlist) // batch_size + leftover
        im_batches = [torch.cat((im_batches[i * batch_size: min((i + 1) * batch_size,
                                                                len(im_batches))])) for i in range(num_batches)]

    write = 0

    if CUDA:
        im_dim_list = im_dim_list.cuda()

    # 5 resim için 10sn sürüyor
    start_outputs_loop = time.time()

    # bu for içinde tek tek resim işlemeye çalış
    for i, batch in enumerate(im_batches):
        # load the image

        if CUDA:
            batch = batch.cuda()
        with torch.no_grad():
            prediction = model(Variable(batch))  # bu işlem uzun sürüyor

        prediction = non_max_suppression(prediction, confidence, num_classes, nms_conf=nms_thesh)
        try:
            prediction[:, 0] += i * batch_size  # transform the atribute from index in batch to index in imlist
        except TypeError as e:
            # yazdir(img[0])
            outputs_names = pd.Series(imlist).apply(lambda x: "{}/{}".format(args.outputs, x.split("/")[-1]))
            list(map(cv2.imwrite, outputs_names, loaded_ims))
            os.remove(os.path.join(images, img[0]))
            print ("\nTYPE ERROR\n")
            continue

        if not write:  # If we have't initialised output
            output = prediction
            write = 1
        else:
            output = torch.cat((output, prediction))

        if CUDA:
            torch.cuda.synchronize()

        timeOutputs = time.time()
        toplamSure = timeOutputs - start_outputs_loop
        print("\ntimeOutputs : ", toplamSure, "\n")

        try:
            output
        except NameError:
            print ("No detections were made")
            continue
        im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())

        scaling_factor = torch.min(416 / im_dim_list, 1)[0].view(-1, 1)

        output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
        output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

        output[:, 1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim_list[i, 0])
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim_list[i, 1])

        output_recast = time.time()
        class_load = time.time()
        colors = pkl.load(open("colors/pallete", "rb"))

        draw = time.time()

        clss = {}

        for i in output:
            if int(i[0]) not in clss:
                clss[int(i[0])] = []
            clss[int(i[0])].append(int(i[-1]))

        for key, value in clss.items():
            clss[key] = Counter(value)

        # okunan resimleri kaydet
        print("\n___________________________________________________________________\n")
        listImg = list(map(lambda x: writefunc(x, loaded_ims), output))
        # yazdir(prev)
        yazdir(listImg[len(listImg) - 1])
        outputs_names = pd.Series(imlist).apply(lambda x: "{}/{}".format(args.outputs, x.split("/")[-1]))
        list(map(cv2.imwrite, outputs_names, loaded_ims))
        os.remove(os.path.join(images, img[0]))  # okunan resmi sil

torch.cuda.empty_cache()

