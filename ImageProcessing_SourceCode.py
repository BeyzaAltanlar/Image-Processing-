import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

""" foto=cv2.imread("./ataturk.jpeg")
print(foto.shape)
cv2.imshow("acilan foto :", foto)
cv2.waitKey(0)
cv2.destroyAllWindows()
# foto actik
hist_es_foto=cv2.equalizeHist(foto)
foto2=np.hstack((foto, hist_es_foto))
plt.imshow(foto2, cmap="gray")
plt.show()

"""


path = r'audreyhepburn.jpeg'

img = cv2.imread(path) #img oku
img = cv2.resize(img, (400,400), interpolation = cv2.INTER_AREA)
before = img

cv2.imshow('image',img) #üzerinde çalışılacak görüntü
cv2.waitKey(10000)
cv2.destroyAllWindows()

def show_before_after(img): #öncesi-sonrası
    cv2.imshow('before',before)
    cv2.imshow('after',img)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()


"""img_filename = 'gemi.jpeg'
save_filename = 'output_image.jpeg'
"""

"""img = Image.open(path)

# convert to grayscale
imgray = img.convert(mode='L')

#numpy array e cevir
img_array = np.asarray(imgray)

"""
def remove_salt_pepper_noise(img):
    img= cv2.medianBlur(img, 3)
    show_before_after(img)


#remove_salt_pepper_noise(img)

def erosion(img):
    kernel = np.ones((5, 5), np.uint8)

    #img = cv2.erode(img, np.ones((5, 5)))
    erosion = cv2.erode(img, kernel, iterations=1)
    show_before_after(erosion)

#erosion(img)

def dilation(img):
    img = cv2.dilate(img, np.ones((3, 3)))
    show_before_after(img)

#dilation(img)


def gauss(im):
    kernel = np.array([[0, 1, 0] , [1, -4, 1] , [0, 1, 0]])
    dst2 = cv2.GaussianBlur(img,(5,5),1)
    show_before_after(dst2)


def filtrele(giris, kernel_yada_filtre, yapilacak_islem):
    m, n = kernel_yada_filtre.shape

    yarim_m = m // 2
    yarim_n = n // 2
    # padledik.
    sifir_eklenmis_giris = np.pad(
        giris,
        ((yarim_m, yarim_m), (yarim_n, yarim_n)),
        constant_values=((0, 0), (0, 0))
    )

    sifir_eklenmis_giris = sifir_eklenmis_giris.astype(float)
    M, N = sifir_eklenmis_giris.shape[:2]

    cikis_foto = np.zeros_like(sifir_eklenmis_giris)

    for satir in range(yarim_m, M - yarim_m):
        for sutun in range(yarim_n, N - yarim_n):
            giris_pencere = sifir_eklenmis_giris[satir - yarim_m:satir + yarim_m + 1,
                            sutun - yarim_n:sutun + yarim_n + 1]
            cikis_foto[satir, sutun] = yapilacak_islem(giris_pencere, kernel_yada_filtre)

    show_before_after(cikis_foto.astype(np.uint8))

def medyan_filtre(giris, m, n):
    bos_filtre = np.empty((m, n))
    yapilacak_islem = lambda giris_pencere, kernel: np.median(giris_pencere)
    return filtrele(giris, bos_filtre, yapilacak_islem)

medyan_filtre(img, 10 ,10)





# gauss formul exp(s2+t2 / 2sigma^2) carpi k (keskinlestirme derecem)
def gauss_kernel(m, n, k, sigma):
    mYarisi = m // 2
    nYarisi = n // 2

    gauss_filtre = np.empty((m, n))
    for s in range(-mYarisi, mYarisi+1):
        for t in range(-nYarisi, nYarisi+1):
            rkare = s**2 + t**2
            payda = 2 * sigma**2 # 2sigma2
            u = -(rkare / payda)
            deger = k * np.exp(u)

            ss = mYarisi+ s
            tt = nYarisi + t

            gauss_filtre[ss, tt] = deger
    return gauss_filtre / gauss_filtre.sum()



def kutuFiltre (m, n, deger):
    kutu_filtre = np.full((m, n), deger)
    return kutu_filtre / kutu_filtre.sum()



#HISTOGRAM
def fotografin_histogramini_olustur(foto, L):
    histogram, bins = np.histogram(foto, bins=L, range=(0, L))
    return histogram


def n_histogram_olustur(foto, L): #pr
    histogram = fotografin_histogramini_olustur(foto, L)
    return histogram / foto.size #  M*N

#kumulatif distribution
def kumulatif_dagilim(p_r):
    return np.cumsum(p_r)


def histogram_equalization(foto, L):
    pr = n_histogram_olustur(foto, L)
    dagilim = kumulatif_dagilim(pr)
    donusum_fonk = (L-1) * dagilim  # fonk= ( L-1 ) * toplam sembol( p.r )
    shape = foto.shape
    r= foto.ravel() #array i tek boyut yapiyo elemanlari ekledik ,numpy dan bir metot
    hist_es_foto = np.zeros_like(r)

    for i, pixel in enumerate(r):
        hist_es_foto[i] = donusum_fonk[pixel]
    return hist_es_foto.reshape(shape).astype(np.uint8)



L = 2**8

#hist = histogram_equalization(img, L)


#foto2= np.hstack((img, hist))

#show_before_after(foto2) #histogram icin acilacak


#histogram_equalization(img,L)   #HISTOGRAM ICIN





#UNSharp Masking - 
#once blurla fotoyu
#mask = orjinal-bulanik
#orjinal+maskelenmis
def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape)) # verilen sekilde yeni dizi
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)

    show_before_after(sharpened)

#unsharp_mask(img)


"""def canny_edge_detector (img):
    img = cv2.Canny(img,100,200)
    show_before_after(img)
"""



#Padding and filtering
def padYap(foto, kernel):
    m, n = kernel.shape

    sol = m // 2
    sag = n // 2

    pedlenmis = np.pad( foto,((sol, sol), (sag, sag)),constant_values=((0, 0), (0, 0)))  #padle 0 la. matrisin cevresi kaplandi
    return pedlenmis





def canny_edge_detector (img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, output2) = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
    output2 = cv2.Canny(output2, 180, 255)
    show_before_after(output2)


#canny_edge_detector(img)

"""
def harris_corner_detector(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)

    img[dst>0.01*dst.max()]=[0,0,255] #eşikleme
    show_before_after(img)
"""
















""" stackte sunulan cozumlerden alinti , sonuclar rapora eklenmedi.  
def threshold_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    img_result = np.ones(img.shape) * 255 
    img_result[:, :, 0] = thresh[:, :]  
    img_result[:, :, 2] = thresh[:, :]  
    show_before_after(img_result)
"""


"""from pylab import array, plot, show, axis, arange, figure, uint8
def increase_decrease_contrast(img, increase):
    maxIntensity = 255.0
    x = arange(maxIntensity)

    phi = 1
    theta = 1 #increase contrast
    if (increase == 1):
        # intensity güncellenip pixellerin rengi açılmış oluyor
        newImage0 = (maxIntensity/phi)*(img/(maxIntensity/theta))**0.5
        newImage0 = array(newImage0,dtype=uint8)
        show_before_after(newImage0)
    else:

        y = ((maxIntensity/phi)*(x/(maxIntensity/theta))**0.5)*0.10

        # intensity arttırılıp pixeller daha koyulaştırılıyor
        newImage1 = ((maxIntensity/phi)*(img/(maxIntensity/theta))**2)
        newImage1 = array(newImage1,dtype=uint8)
        show_before_after(newImage1)

#increase_decrease_contrast(img, 0)
#harris_corner_detector(img)
#sobel_edge_detector(img)
"""

