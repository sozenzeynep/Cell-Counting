"""
KÖK HÜCRELERİN GÖRÜNTÜ İŞLEME YÖNTEMLERİ İLE CANLI VE ÖLÜ HÜCRE OLARAK AYRILIP SAYILMASI

ZEYNEP SÖZEN ALKÜ BİLGİSAYAR MÜHENDİSLİĞİ 160204009

BU PROJEDE AMAÇ INPUT OLARAK VERİLEN BİR JPG DOSYASINDAKİ HÜCRELERİN SAYIMINI
WATERSHED ALGORİTMASI İLE HESAPLAYIP, ÇIKTI OLARAK HÜCRE SAYISINI ALMAKTIR.
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import measure, color, io
from skimage.segmentation import clear_border
import glob


pixels_to_um = 0.5 # 1 pixel = 500 nm (got this from the metadata of original image)
propList = ['Area', #Alan
            'equivalent_diameter', #Eş değer Çap
            'orientation', #X ekseni ve ana eksen arasındaki açı.
            'MajorAxisLength', #Anaeksen uzunluğu
            'MinorAxisLength', #Minör eksen uzunluğu
            'Perimeter', #Çevre
            'MinIntensity', #minimum yoğunluk
            'MeanIntensity', #ortalama yoğunluk
            'MaxIntensity']  #maximium yoğunluk

output_file = open('cell_couting.csv', 'w')  #Yazılabilir CSV dosyamız

#İlk satıra Dosya adını, hücre sayısını ve diğer özellikleri ekliyoruz
output_file.write('FileName' + "," + 'Cell Number '+  "," + ",".join(propList) + '\n' + '\n')

path = "hucreler/*.jpg"

for file in glob.glob(path): #Görüntülere ulaşmak için oluşturduğumuz for döngüsü
    print(file)     #Konsola okunan dosya yazılıyor
    img1= cv2.imread(file) #Görüntüyü okuyoruz
    cv2.imshow("Orginal Image", img1)
    img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY) #Gri tonlamaya döndürüyoruz
#    img = img[0:450, :]
#    img = cv2.medianBlur(img, ksize=7)
#    kernel2 = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
#    img = cv2.filter2D(img,-1,kernel2)
#Otsu ile threshold yapıp görüntüyü binary hale getiriyoruz. Tüm pixeller 255 olarak ayarlanıyor.
#Yani hücre sınırlarını hücrelerden ayırmış olduk.
    ret1, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Kernel'ımızı oluşturuyoruz.
    kernel = np.ones((3,3),np.uint8)
# Morfolojik işlemler sayesinde görüntüdeki küçük gürültüleri kaldırıyoruz.
    opening1 = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
# Opening ile görüntüdeki kenarlarda kalan yarım hücreleri siliyoruz.
    opening = clear_border(opening1)
# Erozyon yapmak küçük hücrelerimizi aşındıracağından onu kullanmıyoruz.
    erosion1 = cv2.erode(thresh, kernel, iterations=1)

#Artık biliyoruzki ortaya çıkan yerler hücrelerin merkezi, bu durumda arka plana bakabiliriz.
    sure_bg = cv2.dilate(opening,kernel,iterations=1)


#Birbirlerine dokunan nesneleri ayırmak için distanceTransform kullanıyoruz.
#ve ardından da eşikleme yapıyoruz.
#dist_tranform resmimizle aynı boyutta ama float  değerinde
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)

#threshold limit için %20'yi oran olarak belirliyoruz böylece %20'nin altındakileri kabul etmezzken
#%20 üstündekileri kabul ediyoruz
#dist_transform.max değeri yaklaşık 21,9 verir.
    ret2, sure_fg = cv2.threshold(dist_transform,0.2*dist_transform.max(),255,0)
# Float değeri 8 bitlik yapıyoruz.
    sure_fg = np.uint8(sure_fg)

# Ne olduğunu bilmediğimiz noktalar var. Watershed bu noktaları dolduracak.
    unknown = cv2.subtract(sure_bg,sure_fg)

# Marker yarattık ve kesin bölgelerimizi işaretleyeceğiz.
# Bilinmeyen bölgeler otomatik olarak 0 olarak etiketlenecektir.
    ret3, markers = cv2.connectedComponents(sure_fg)

# Arkaplande 0'lardan oluştuğu için Watershed bunu ayırabilsin diye +10 veriyoruz.
# Yoksa bu bölgeyi bilinmeye olarak kabul eder.
# Orginalde markers 0 oluyor biz bg i le karışmasın diye 10 veriyoruz
#markerlar aslında arraylerdir
    markers = markers+10

# Bilinmeyen eğer bölgeyi sıfırla işaretlenir
    markers[unknown==255] = 0
    #plt.imshow(markers)

# watershed artık doldurma yapar
    markers = cv2.watershed(img1,markers)

# watershede sınırı -1 olarak geçiyor
# marker -1 değeri görünce sarı renk yapmasını diyorum
    img1[markers == -1] = [0,255,255]
#etiketleri rgb olarak boyama yapar
    img2 = color.label2rgb(markers, bg_label=0)

    # Görüntünün farklı hallerinin görüntülenmesi
    cv2.imshow('Overlay on original image', img1)
    cv2.imshow("Thresholded Image",thresh)
    cv2.imshow("Opened Image",opening1)
    cv2.imshow("Clean Borders", opening)
    cv2.imshow("Erosion",erosion1)
    cv2.imshow("Background of Image",sure_bg)
    cv2.imshow("Foreground of Image",sure_fg)
    cv2.imshow("Unknown Part of Image",unknown)
    cv2.imshow('Colored Grains', img2)
    cv2.waitKey(0)

#Özellikleri çıkarma
#Skimage deki regionprops fonksiyonu her objenin özelliklerini ölçmeye yarıyor
    regions = measure.regionprops(markers, intensity_image=img)

# Tespit edilen hücrelerin özelliklerini çıkaracağız
# Regionprops fonksiyonundaki measure her hücre için hesaplama yaoar
    grain_number = 1 #Başlangıç olarak verdiğimiz hücre sayımız çünkü altta hemen 1 yazıyorum ondan dolayı
    for region_props in regions:
        output_file.write(file+",") # Dosya adını hücre sayımı bitene kadar yazdırıyoruz
        output_file.write(str(grain_number)) # Her döngüde okunan hücre sayısı artıyor
        for i,prop in enumerate(propList): # Sırayla hücrenin özellikleri yazılıyor
            if(prop == 'Area'):
                to_print = region_props[prop]*pixels_to_um**2   #pixelleri micro'nun karesine dönüştürme
            elif(prop == 'orientation'):
                to_print = region_props[prop]*57.2958  #Dereceyi radyana dönüştürme
            elif(prop.find('Intensity') < 0):          #İsmi intensity olmayan diğer özellikler için
                to_print = region_props[prop]*pixels_to_um
            else:
                to_print = region_props[prop]     ## Diğer özelliklerde ekstra bir matematiksel işleme ihtiyaç olmadığı için direk sonucu yazıyoruzensity in its name
            output_file.write(',' + str(to_print))
        output_file.write('\n')  # Yeni satıra geçiş.
        grain_number += 1 #Yeni hücreye geçiş ile hücre sayısı artıyor.1
    print(file, "Goruntusunde", grain_number-1, "adet hücre sayildi.")  # Konsol Çıktıları
    output_file.write(file + " GORUNTUSUNDE " + str(grain_number - 1) + " ADET HUCRE SAYILDI. " + '\n' + '\n')  # Konsol Çıktıları

output_file.close()   #Dosyayı kapatıyoruzki read only olmasın