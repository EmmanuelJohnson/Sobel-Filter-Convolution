import cv2
import numpy as np
import math

#Read the image using opencv
def get_image(path):
    return cv2.imread(path)

#Read the image in gray scale using opencv
def get_image_gray(path):
    return cv2.imread(path,0)

#Show the resulting image
def show_image(name,image):
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Save the resulting image
def save_image(name,image):
    cv2.imwrite(name,image) 

#Vertical and Horizontal Sobel Matrix defined
def get_sobel(value):
    #Flipped Sobel X and Sobel Y
    sobels = {
        "sobelX": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        "sobelY": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    }
    return sobels[value]

#Create a Matrix with all elements 0
def createZeroMatrix(r, c):
    matrix = []
    for i in range(r):
        row = [0 for j in range(c)]
        matrix.append(row)
    return np.asarray(matrix,dtype="float32")

#Add Zero padding around the edge of the image
def add_padding(img_matrix,padding):
    row = [0 for j in range(img_matrix.shape[1]+(2*padding))]
    tempImgMatrix = img_matrix.tolist()
    for i in range(img_matrix.shape[0]):
        for f in range(padding):
            tempImgMatrix[i].append(0)
            tempImgMatrix[i].insert(0,0)
    for i in range(padding):
        tempImgMatrix.append(row)
        tempImgMatrix.insert(0,row)
    img_padded_matrix = np.array(tempImgMatrix, dtype='float32')
    return img_padded_matrix

#Perform convolution between image and sobel
def convolution(img_matrix,sobelx,sobely):
    #Get image height and width
    ih, iw = img_matrix.shape[0],img_matrix.shape[1]
    print("__Creating 3 Zero Matrix__")
    #Create 3 empty matrix to store the result
    op_mx_x,op_mx_y,op_mx_g = createZeroMatrix(ih, iw),createZeroMatrix(ih, iw),createZeroMatrix(ih, iw)
    
    #Padding width
    pad = sobelx.shape[0]//2

    print("__Adding 0 padding to the image__")
    #Adding zero padding to the image
    img_matrix = add_padding(img_matrix, pad)
    
    #SobelX height and width
    sxh, sxw = sobelx.shape[0],sobelx.shape[1]
    #SobelY height and width
    syh, syw = sobely.shape[0],sobely.shape[1]
    print("__Convolving SobelX and SobelY with the image__")
    for i in range(pad, ih + pad):
        for j in range(pad, iw + pad):

            sx,sy,sg = 0,0,0
            #The submatrix from the image on which convolution is done
            #Dimension is based on the dimension of the kernel being applied
            submatrix = img_matrix[i-pad:i-pad+sxh,j-pad:j-pad+sxw]
            
            for si in range(sxh):
                for sj in range(sxw):
                    sx = sx + (sobelx[si][sj] * submatrix[si][sj])
                    sy = sy + (sobely[si][sj] * submatrix[si][sj])
            #Saving the convolved output to a new matrix
            op_mx_x[i - pad, j - pad] = sx
            op_mx_y[i - pad, j - pad] = sy
            #Finding the gradient of the two sobels
            sg = math.sqrt((sx * sx) + (sy * sy))
            op_mx_g[i-1][j-1] = sg
    return op_mx_x,op_mx_y,op_mx_g

def main():
    print("__Reading the given image : task1.png__")
    img = get_image_gray('task1.png')
    sobelX = np.asarray(get_sobel('sobelX'),dtype='float32')
    sobelY = np.asarray(get_sobel('sobelY'),dtype='float32')
    sobelX_img,sobelY_img,gradient_img = convolution(img,sobelX,sobelY)
    print("__Finished Convolution__")
    print("__Saving the results__")
    save_image('SobelX.png',sobelX_img)
    save_image('SobelY.png',sobelY_img)
    save_image('Gradient.png',gradient_img)
    

if __name__ == '__main__':
    main()
