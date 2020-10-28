import numpy as np
import imageio as im
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
MAX_PIXEL=255
#section 3.1
x = np.hstack([np.repeat(np.arange(0,50,2),10)[None,:], np.array([255]*6)[None,:]])
grad = np.tile(x,(256,1))

#section 3.2
def read_image(filename, representation):
    #the function will read an image file and return a normalizes array of
    # its intesitys
    image=im.imread(filename)
    image=image.astype(np.float64)/255
    if representation==1 and image.ndim!=1:#return RGB from RGB file
        return image
    elif representation==2 and image.ndim==3:#return grayscale from RGB file
        return rgb2gray(image)
    elif representation==2 and image.ndim==2: #return grayscale from
        # grayscale file
        return image
#section 3.3
def imdisplay(filename, representation):
    s = read_image(filename, representation)
    if representation==2:
        plt.imshow(s,cmap='gray')
    else:
        plt.imshow(s)
    plt.show()

#section 3.4
def rgb2yiq(imRGB):
    #the transition matrix
    trans_mat=np.array([[0.299,0.587,0.114],[0.596,-0.275,-0.321],[0.212,-0.523,0.311]])
    yiqimg=np.dot(imRGB,trans_mat.T.copy())
    #perform matrix multiplication with the transpose trans mat,i order to
    # coordinate the RGB channels in the multiplaction
    return yiqimg

def yiq2rgb(imYIQ):
    trans_mat =np.linalg.inv(np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321],[0.212, -0.523, 0.311]]))
    # perform matrix multiplication with the transpose trans mat,i order to
    # coordinate the RGB channels in the multiplaction
    yiqimg = np.dot(imYIQ, trans_mat.T.copy())
    return yiqimg
##section 3.5
def histogram_equalize(im_orig):
    if im_orig.ndim==2:#case of grayscale
        im_eq,hist_orig,hist_eq=histogram_eq_helper(im_orig)
    else:#in case of rgb image
        yiq=rgb2yiq(im_orig)
        Y=yiq[:,:,0]
        Y_new,hist_orig,hist_eq=histogram_eq_helper(Y)
        yiq[:,:,0]=Y_new
        im_eq=yiq2rgb(yiq)
    return im_eq,hist_orig,hist_eq

def histogram_eq_helper(im_o):
    ##doing the hist equlazation
    im=np.array(im_o)
    hist_orig, norm_dim = hist_create(im)
    cum_hist = np.cumsum(hist_orig)
    c_m = np.ma.masked_equal(cum_hist, 0)
    T = MAX_PIXEL * (c_m-c_m.min())/((cum_hist.max()-c_m.min()))
    T = np.ma.filled(T, 0).astype('uint8')
    new_im=T[norm_dim]
    hist_eq, hist = hist_create(new_im)
    im_eq = np.reshape(new_im, im_o.shape)/255
    return im_eq,hist_orig,hist_eq


def hist_create(im):
    ##help funtion that create the apropriate histogram
    dim = im.flatten()#make the im to 1 vector
    norm_dim = (dim * 255).astype('uint8')#change the scale
    hist_og, bins = np.histogram(norm_dim,256,[0,255])
    return hist_og, norm_dim


im=read_image('low_contrast.jpg',2)
#im1=rgb2yiq(im)[:,:,0]
#print(im)
new,hist_orig,hist_eq=histogram_eq_helper(im)

#im[:,:,0]=new
#n=yiq2rgb(im)
#print(n)
plt.imshow(new)
plt.show()
plt.imshow(im)
plt.show()
plt.plot(hist_eq)
plt.show()
plt.plot(hist_orig)
plt.show()