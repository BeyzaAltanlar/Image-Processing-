import cv2
import numpy as np

path = r'noise.jpg'

img = cv2.imread(path) #görüntüyü oku
img = cv2.resize(img, (400,400), interpolation = cv2.INTER_AREA)
before = img

cv2.imshow('image',img) #üzerinde çalışılacak görüntüyü göster
cv2.waitKey(10000)
cv2.destroyAllWindows()

def show_before_after(img): #öncesi-sonrası şeklinde iki görüntüyü yan yana göster
    cv2.imshow('before',before)
    cv2.imshow('after',img)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()


def increase_brightness(img, value=20): #value değeriyle oynanarak farklı sonuçlar denenebilir
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value #görüntü hsv(Hue Saturation Value) formatına çevrilip eşikleniyor
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    show_before_after(img)


def decreasee_brightness(img, value=30): #value değeriyle oynanarak farklı sonuçlar denenebilir
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 + value
    v[v > lim] = 255
    v[v <= lim] -= value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    show_before_after(img)


#decreasee_brightness(img)

#increase_brightness(img)

def remove_salt_pepper_noise(img):
    img= cv2.medianBlur(img, 3) #3 yerine farklı değerler denenerek daha bulanıklaştırılabilir
    show_before_after(img)

def gauss(img):
    img= cv2.GaussianBlur(img,(1,1),1)
    show_before_after(img)


#gauss(img)

remove_salt_pepper_noise(img)

from pylab import array, plot, show, axis, arange, figure, uint8
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

def sobel_edge_detector(img): #yatay ve dikeyde sobel uygulanıyor
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    img = (grad * 255 / grad.max()).astype(np.uint8)
    show_before_after(img)

"""
def sobel_without_opencv(img):
    import skimage.exposure as exposure

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # blur
    blur = cv2.GaussianBlur(gray, (0,0), 1.3, 1.3)

    # apply sobel derivatives
    sobelx = cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(blur,cv2.CV_64F,0,1,ksize=3)

    # optionally normalize to range 0 to 255 for proper display
    sobelx_norm = exposure.rescale_intensity(sobelx, in_range='image', out_range=(0,255)).clip(0,255).astype(np.uint8)
    sobely_norm= exposure.rescale_intensity(sobelx, in_range='image', out_range=(0,255)).clip(0,255).astype(np.uint8)

    # square
    sobelx2 = cv2.multiply(sobelx,sobelx)
    sobely2 = cv2.multiply(sobely,sobely)

    # add together and take square root
    sobel_magnitude = cv2.sqrt(sobelx2 + sobely2)

    # normalize to range 0 to 255 and clip negatives
    sobel_magnitude = exposure.rescale_intensity(sobel_magnitude, in_range='image', out_range=(0,255)).clip(0,255).astype(np.uint8)
    show_before_after(sobel_magnitude)

"""
#sobel_without_opencv(img)


def canny_edge_detector (img):
    img = cv2.Canny(img,100,200)
    show_before_after(img)


""" def canny_without_open_cv(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold
    thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]

    # morphology edgeout = dilated_mask - mask
    # morphology dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    dilate = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

    # get absolute difference between dilate and thresh
    diff = cv2.absdiff(dilate, thresh)

    # invert
    edges = 255 - diff
    show_before_after(edges) """

#canny_without_open_cv(img)
#canny_edge_detector (img)


def harris_corner_detector(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)

    img[dst>0.01*dst.max()]=[0,0,255] #eşikleme
    show_before_after(img)

#harris_corner_detector(img)

def dilate(img):
    img = cv2.dilate(img, np.ones((2, 2))) #2 yerine farklı değerler denenebilir
    show_before_after(img)


dilate(img)

def erosion(img):
    img = cv2.erode(img, np.ones((3, 3))) #3 yerine farklı değerler denenebilir
    show_before_after(img)

#erosion(img)

def histogram_equalization (img):
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # equalize the histogram of the Y channel
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

    # convert back to RGB color-space from YCrCb
    img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
    show_before_after(img)

#histogram_equalization (img)


def histogram_equalization_without_opencv(img):
    import PIL
    from IPython.display import display, Math, Latex
    import matplotlib.pyplot as plt

    img = PIL.Image.open(path)

    # display the image
    plt.imshow(img, cmap='gray')

    img = np.asarray(img)

    # put pixels in a 1D array by flattening out img array
    flat = img.flatten()

    # show the histogram
    plt.hist(flat, bins=50)

    def get_histogram(image, bins):
        # array with size of bins, set to zeros
        histogram = np.zeros(bins)

        # loop through pixels and sum up counts of pixels
        for pixel in image:
            histogram[pixel] += 1

        # return our final result
        return histogram

    # execute our histogram function
    hist = get_histogram(flat, 256)

    def cumsum(a):
        a = iter(a)
        b = [next(a)]
        for i in a:
            b.append(b[-1] + i)
        return np.array(b)

    # execute the fn
    cs = cumsum(hist)

    # display the result
    plt.plot(cs)

    nj = (cs - cs.min()) * 255
    N = cs.max() - cs.min()

    # re-normalize the cumsum
    cs = nj / N

    # cast it back to uint8 since we can't use floating point values in images
    cs = cs.astype('uint8')

    plt.plot(cs)

    img_new = cs[flat]

    # put array back into original shape since we flattened it
    img_new = np.reshape(img_new, img.shape)

    # set up side-by-side image display
    fig = plt.figure()
    fig.set_figheight(15)
    fig.set_figwidth(15)

    fig.add_subplot(1,2,1)
    plt.imshow(img, cmap='gray')

    # display the new image
    fig.add_subplot(1,2,2)
    plt.imshow(img_new, cmap='gray')

    plt.show(block=True)


    #histogram_equalization_without_opencv(img)

    def crop_objects(img):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # invert gray image
        gray = 255 - gray

        # threshold
        thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)[1]

        # apply close and open morphology to fill tiny black and white holes and save as mask
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # get contours (presumably just one around the nonzero pixels)
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        cntr = contours[0]
        x,y,w,h = cv2.boundingRect(cntr)

        # make background transparent by placing the mask into the alpha channel
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        new_img[:, :, 3] = mask

        # then crop it to bounding rectangle
        crop = new_img[y:y+h, x:x+w]
        cv2.imshow("THRESH", thresh)
        cv2.imshow("MASK", mask)
        cv2.imshow("CROP", crop)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    #crop_objects(img)

    def remove_blues(im):

        # mavinin sınırları yazıldı
        BlueMin = np.array([0,  0, 0],np.uint8)
        BlueMax = np.array([20, 20, 20],np.uint8)

        #hsv uzayına geçildi
        HSV  = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(HSV, BlueMin, BlueMax)

        #maskeleme
        im[mask>0] = [255,255,255]
        show_before_after(im)



    def change_color(img):
        from PIL import Image

        im = Image.open('cards.jpg')
        #rgb bileşenleri alındı
        R, G, B = im.split()
        #red - green bileşenlerinin yerleri değiştirildi
        result = Image.merge('RGB',[G,R,B])

        result.save('result.jpg')


    #change_color(img)


    def gauss(im):
        kernel = np.array([[0, 1, 0] , [1, -4, 1] , [0, 1, 0]])
        dst2 = cv2.GaussianBlur(img,(5,5),1)
        show_before_after(dst2)


   # laplacian(img)


    def threshold_image (img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        threshold, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        img_result=np.ones(img.shape)*255 #set all to 255
        img_result[:,:,0]=thresh[:,:] #set red channel to threshold
        img_result[:,:,2]=thresh[:,:] #set blue channel to threshold
        show_before_after(img_result)


#threshold_image (img)

""" def find_triangles(img):
        _, threshold = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
        _, contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
            cv2.drawContours(img, [approx], 0, (0), 5)
            x = approx.ravel()[0]
            y = approx.ravel()[1]
        cv2.putText(img, "Triangle", (x, y), font, 1, (0))
        cv2.imshow("shapes", img)
        cv2.imshow("Threshold", threshold)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    #find_triangles(img)"""