import numpy as np
import imageio as im
import matplotlib.pyplot as plt
import math as mt
from skimage.color import rgb2gray
import skimage.color
MAX_PIXEL=255
RGB=3
GRAY_SCALE=2
#section 3.1
x = np.hstack([np.repeat(np.arange(0,50,2),10)[None,:], np.array([255]*6)[None,:]])
grad = np.tile(x,(256,1))

#section 3.2
def read_image(filename, representation):
    #the function will read an image file and return a normalizes array of
    # its intesitys
    image=im.imread(filename).astype(np.float64)
    if np.amax(image)>1:
        image=image.astype(np.float64)/MAX_PIXEL
    if representation==1 and image.ndim!=GRAY_SCALE:#return RGB from RGB file
        return image
    elif representation==2 and image.ndim==RGB:#return grayscale from RGB file
        return rgb2gray(image)
    elif representation==2 and image.ndim==GRAY_SCALE: #return grayscale from
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
    if im_orig.ndim==GRAY_SCALE:#case of grayscale
        im_eq,hist_orig,hist_eq=histogram_eq_helper(im_orig)
    else:#in case of rgb image
        yiq=rgb2yiq(im_orig)
        Y=yiq[:,:,0]
        Y_new,hist_orig,hist_eq=histogram_eq_helper(Y)
        yiq[:,:,0]=Y_new
        im_eq=yiq2rgb(yiq)
    return im_eq.astype('float64'),hist_orig,hist_eq

def histogram_eq_helper(im_o):
    ##doing the hist equlazation
    im=np.array(im_o)
    hist_orig, norm_dim = hist_create(im)
    cum_hist = np.cumsum(hist_orig)
    c_m = np.ma.masked_equal(cum_hist, 0)
    T = MAX_PIXEL*(c_m) / ((cum_hist.max()))
    if cum_hist[0]!=0 and cum_hist[-1]!=MAX_PIXEL:
        T = MAX_PIXEL*(c_m-c_m.min())/((cum_hist.max()-c_m.min()))
    ####add check of min max z
    T = np.ma.filled(T, 0).astype('uint8')
    new_im=T[norm_dim]
    hist_eq, hist = hist_create(new_im)
    im_eq = np.reshape(new_im, im_o.shape)/MAX_PIXEL
    return im_eq,hist_orig,hist_eq


def hist_create(im):
    ##help function that create the histogram
    dim = im.flatten()#make the im to 1 vector
    norm_dim = (dim * MAX_PIXEL).astype('uint8')#change the scale
    hist_og, bins = np.histogram(norm_dim,256,[0,MAX_PIXEL])
    return hist_og, norm_dim

#section 3.6
def quantize (im_orig, n_quant, n_iter):
    if im_orig.ndim==GRAY_SCALE:#in case of grayscale
        [im_quant, error] = quntize_gray(im_orig, n_iter, n_quant)
    else:
        [im_quant, error]=quntize_rgb(im_orig, n_iter, n_quant)
    return [im_quant, np.array(error)]


def quntize_gray(im_orig, n_iter, n_quant):
    hist, norm_dim = hist_create(im_orig)
    z_i_list, num_of_pixels = z_bounds(hist, n_quant)
    q_list, err = find_q_from_z(hist, z_i_list, num_of_pixels)
    error = []
    for i in range(n_iter):
        new_z = find_z_from_q(q_list)
        q_list, err = find_q_from_z(hist, new_z, num_of_pixels)
        error.append(err)
        if np.array_equal(new_z, z_i_list) == True:
            break
        z_i_list = new_z
    q_list = q_list.astype('uint8')
    z_i_list = z_i_list.astype('uint8')
    look_up = look_up_table(q_list, z_i_list)
    im_quant = look_up[norm_dim]
    im_quant = np.reshape(im_quant, im_orig.shape) / MAX_PIXEL
    return [im_quant, np.array(error)]

def quntize_rgb(im_orig, n_iter, n_quant):
    Yiq = rgb2yiq(im_orig)
    Y=Yiq[:,:,0]
    hist, norm_dim = hist_create(Y)
    z_i_list, num_of_pixels = z_bounds(hist, n_quant)
    q_list, err = find_q_from_z(hist, z_i_list, num_of_pixels)
    error = []
    for i in range(n_iter):
        new_z = find_z_from_q(q_list)
        q_list, err = find_q_from_z(hist, new_z, num_of_pixels)
        error.append(err)
        if np.array_equal(new_z, z_i_list) == True:
            break
        z_i_list = new_z
    q_list = q_list.astype('uint8')
    z_i_list = z_i_list.astype('uint8')
    look_up = look_up_table(q_list, z_i_list)
    new_Y = look_up[norm_dim]
    new_Y = np.reshape(new_Y, Y.shape) / MAX_PIXEL
    Yiq[:,:,0]=new_Y
    im_quant=yiq2rgb(Yiq)
    return [im_quant, np.array(error)]


def look_up_table(q_list, z_i_list):
    look_up = np.zeros(256)
    for i in range(len(z_i_list)-1):
        a=z_i_list[i + 1] - z_i_list[i]
        look_up[z_i_list[i]:z_i_list[1 + i]]=[q_list[i]]*a
        #print(look_up)
    look_up[-1]=q_list[-1]
    return look_up


def find_q_from_z(hist,z_i_list,num_of_pixels):
    ##creates a q list from given z list
    q_list=[]
    total_err=0
    for i in range(len(z_i_list)-1):
        z_i=mt.floor(z_i_list[i])+1#set bound
        z_i_next=mt.floor(z_i_list[i+1])
        h_g_vect=np.array([hist[j] for j in range(z_i,z_i_next)])
        g_vect=np.array([j for j in range(z_i,z_i_next)])
        up=np.dot(g_vect,h_g_vect.T)
        down=np.sum(h_g_vect)
        q_i=up/down
        q_list.append(q_i)
        section_err=find_section_err(g_vect, hist, num_of_pixels, q_i)
        total_err+=section_err
    return np.array(q_list),total_err


def find_section_err(g_vect, hist, num_of_pixels, q_i):
    #implement the error calculation function,return a singrl section err
    g_q_err = (g_vect - q_i) ** 2
    prob_vect = hist[g_vect] / num_of_pixels
    section_err = np.dot(g_q_err, prob_vect.T)
    return section_err

def find_z_from_q(q_list):
    ##creates a z list from given q list
    z_i_list=[0]
    for i in range(1,len(q_list)):
        z_i=(q_list[i-1]+q_list[i])/2
        z_i_list.append(z_i)
    z_i_list.append(MAX_PIXEL)
    return np.array(z_i_list)

def z_bounds(hist, n_quant):
    ##this function devide the z axes of the histogram to equal pixels sections
    #and return an z points array and the number of pixels in de image
    hist_cum = np.cumsum(hist)
    dp = int(hist_cum[len(hist_cum)-1]/(n_quant+1))
    z_i_list=[0]
    j=1
    for i in range(MAX_PIXEL):
        if hist_cum[i]>=j*dp:
            z_i_list.append(i)
            j+=1
    #z_i_list.append(MAX_PIXEL)
    return np.array(z_i_list),hist_cum[len(hist_cum)-1]


#s,t=quantize(grad,25,4)
#im=read_image('low_contrast.jpg',2)
im_eq,hist_orig,hist_eq=histogram_equalize(grad/255)
im1,err=quantize(im_eq,10,10)
plt.imshow(im1,cmap='gray')
plt.show()
plt.plot(hist_eq)
plt.show()

####tests
#ims=read_image(r"externals/jerusalem.jpg",1)






images = []
jer_bw = read_image(r"externals/jerusalem.jpg", 1)
images.append((jer_bw, "jerusalem grayscale"))
jer_rgb = read_image(r"externals/jerusalem.jpg", 2)
images.append((jer_rgb, "jerusalem RGB"))
low_bw = read_image(r"externals/low_contrast.jpg", 1)
images.append((low_bw, "low_contrast grayscale"))
low_rgb = read_image(r"externals/low_contrast.jpg", 2)
images.append((low_rgb, "low_contrast RGB"))
monkey_bw = read_image(r"externals/monkey.jpg", 1)
images.append((monkey_bw, "monkey grayscale"))
monkey_rgb = read_image(r"externals/monkey.jpg", 2)
images.append((monkey_rgb, "monkey RGB"))


def test_rgb2yiq_and_yiq2rgb(im, name):
    """
    Tests the rgb2yiq and yiq2rgb functions by comparing them to the built in ones in the skimage library.
    Allows error to magnitude of 1.e-3 (Difference from built in functions can't be bigger than 0.001).
    :param im: The image to test on.
    :param name: Name of image.
    :return: 1 on success, 0 on failure.
    """
    imp = rgb2yiq(im)
    off = skimage.color.rgb2yiq(im)

    if not np.allclose(imp, off, atol=1.e-3):
        print("ERROR: in rgb2yiq on image '%s'" % name)
        return 0
    imp2 = yiq2rgb(imp)
    off2 = skimage.color.yiq2rgb(off)
    if not np.allclose(imp2, off2, atol=1.e-3):
        print("ERROR: in yiq2rgb on image '%s'" % name)
        return 0
    print("passed conversion test on '%s'" % name)
    return 1


for im in images:
    if len(im[0].shape) == 3:
        result = test_rgb2yiq_and_yiq2rgb(im[0], im[1])
        if not result:
            print("=== Failed Conversion Test ===")
            break


def display_all(im, add_bonus):
    if len(im.shape) == 3 and add_bonus:
        fig, a = plt.subplots(nrows=3, ncols=2)
    else:
        fig, a = plt.subplots(nrows=2, ncols=2)

    # adds the regular image
    a[0][0].imshow(im, cmap=plt.cm.gray)
    a[0][0].set_title(r"original image")

    # adds the quantified image
    quant = quantize(im, 3, 10)[0]
    a[0][1].imshow(quant, cmap=plt.cm.gray)
    a[0][1].set_title(r"quantize to 3 levels, 10 iterations")

    # adds the histogram equalized image
    hist = histogram_equalize(im)[0]
    a[1][0].imshow(hist, cmap=plt.cm.gray)
    a[1][0].set_title("histogram equalization")

    # adds quantization on histogram equalized image
    hist_quant = quantize(hist, 6, 10)[0]
    a[1][1].imshow(hist_quant, cmap=plt.cm.gray)
    a[1][1].set_title("quantize on equalization")

    # adds the bonus image
    if len(im.shape) == 3 and add_bonus:
        a[2][0].imshow(quantize_rgb(im, 3))
        a[2][0].set_title(r"bonus quantize_rgb")

    plt.show()


for im in images:
    # change "False" to "True" if you wish to add the bonus task to the print
    display_all(im[0], False)
