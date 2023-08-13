"""
===============================================================================
ENGR 13300 Fall 2021

Program Description

Assignment Information
    Assignment:     Team HW5 - Py4, Task 2
    Author:         Pranesh Monda, pmonda@purdue.edu
    Team ID:
    LC2 - 05 (e.g. LC1 - 01; for section LC1, team 01)

Contributor: Sijie Zhang, zhan2355@purdue.edu
    My contributor(s) helped me:
    [ ] understand the assignment expectations without
        telling me how they will approach it.
    [ ] understand different ways to think about a solution
        without helping me plan my solution.
    [ ] think through the meaning of a specific error or
        bug present in my code without looking at my code.
    Note that if you helped somebody else with their code, you
    have to list that person as a contributor here as well.

ACADEMIC INTEGRITY STATEMENT
I have not used source code obtained from any other unauthorized
source, either modified or unmodified. Neither have I provided
access to my code to another. The project I am submitting
is my own original work.
===============================================================================
"""

import random
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def decrypt_image(image, keyArr):
    decrypted = np.zeros(image.shape, dtype=np.uint8) #1152*1304*4, set with dimensions of original image's shape to make appropriately sized key array

    for i in range(len(image)):
        for j in range(len(image[0])):
            decrypted[i][j][0] = (keyArr[i][j]) ^ (image[i][j][0]) #decryption of the red color channel
            decrypted[i][j][1] = (keyArr[i][j]) ^ (image[i][j][1]) #decryption of the green color channel
            decrypted[i][j][2] = (keyArr[i][j]) ^ (image[i][j][2]) #decryption of the blue color channel

    return decrypted

def grayScale(image): #Part 3
    imgFile = plt.imread(image)
    imArr = np.array(imgFile)
    gray = np.zeros(np.shape(imArr), dtype=np.float64) #a numpy array with zeros that will eventually store the grayscale image
    for row in range(len(imArr)):
        for col in range(len(imArr[0])):
            gray[row][col] = int(0.2126 * imArr[row][col][0] + 0.7152 * imArr[row][col][1] + 0.0722 * imArr[row][col][2]) #this equation is applied to each gray array index to take color pixel values and turn them into grayscale

    gray = np.array(gray, dtype=np.uint8) #now the grayscale array has type uint8
    plt.imsave('gray.tiff', gray)
    return gray

def sobel_edge_detection(decryptedGray):
    gau_filter = np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]]) #this is the smoothing kernel
    imgSmoothed = signal.convolve2d(decryptedGray, gau_filter) #the scipy signal function applies the gaussian filter to smooth out the image and reduce noise in the grayscale image that is passed into the function
    plt.imsave("plainSmooth.tiff", imgSmoothed, cmap="gray")


    kernelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) #this kernel array finds edges in the x direction
    kernely = np.array([[1,2,1], [0,0,0], [-1,-2,-1]]) #this kernal array finds edges in the y direction
    g_x = signal.convolve2d(imgSmoothed, kernelx) #by using convolve2d, the edges in x-direction are discovered
    g_y = signal.convolve2d(imgSmoothed, kernely) #by using convolve2d, the edges in y-direction are discovered

    edgeImg = np.hypot(g_x,g_y) #in order to get the full picture, taking the square root of the squares of the x and y direction deterrmines provides the average of what the image looks like as whole with edges detected
    plt.imshow(edgeImg, cmap='gray')
    plt.show()
    plt.imsave("gradientMap.tiff", edgeImg, cmap="gray")
    earthRow, earthCol = findEarth(edgeImg)
    earthRow = earthRow[0] #this variable includes only the value of the row number of Earth's pixel
    earthCol = earthCol[0] #this variable includes only the value of the column number of Earth's pixel
    print(f"Row of Earth: {earthRow}, Column of Earth: {earthCol}")
    cropEarth("gradientMap.tiff", earthRow, earthCol)

def findEarth(edgeImg):
    return (np.where(edgeImg == np.max(edgeImg))[0], np.where(edgeImg == np.max(edgeImg))[1]) #one return statement returns exactly the earth's row and column as element one then another element

def cropEarth(imgEdgeName, x, y):
    edges = plt.imread(imgEdgeName) #this is the grayscale picture that has edges detected
    h=100 #height cropping area
    w=101 #width cropping area
    croppedImg = edges[x-(int(w/2)):x+(int(w/2)), y-(int(h/2)):y+(int((h+1)/2)), :] #adding and subtracting w and h values respectively in order to capture 50 pixels/51 pixels above and below the earth to produce a cropped 100 x 101 image, values are
    plt.imshow(croppedImg)
    plt.show()
    plt.imsave("cropped.tiff", croppedImg[:, :, 0:3])

def generateKey(decryptedImage):
    inputKey = input("Enter initial key: ")
    numLetters = 0
    for letter in inputKey:
        if(letter != " "):
            numLetters+=1 #counts number of letters in entered keystring (exludes spaces)
    rand_nums = np.zeros((decryptedImage.shape[0],decryptedImage.shape[1]), dtype=np.uint8) #new array that will hold keys for encryption, set with proper shape requirements
    for i in range(len(rand_nums)):
        for j in range(len(rand_nums[0])):
            encryptScal= (i+j) % numLetters #this scalar value adds the row and column number and finds the remainder when divided by the number of letters in the key string
            key = (encryptScal*(random.randint(1,100))) #key is scalar multiple types a random integer from 1 to 100.
            key *= 255 #multiplication by 255, ensures random values fit in the appropriate values taken on by an image array
            rand_nums[i][j] = key

    print(rand_nums)
    return rand_nums

def encryptImage(decryptedImg ,keyArr):
    encrypted = np.zeros(decryptedImg.shape, dtype=np.uint8)  # 1152*1304*4

    for i in range(len(decryptedImg)):
        for j in range(len(decryptedImg[0])):
            encrypted[i][j][0] = (keyArr[i][j]) ^ (decryptedImg[i][j][0])
            encrypted[i][j][1] = (keyArr[i][j]) ^ (decryptedImg[i][j][1])
            encrypted[i][j][2] = (keyArr[i][j]) ^ (decryptedImg[i][j][2])

    return encrypted


def main():
    imageFileName = ""
    invalidInput = True
    while(invalidInput == True):
        imageFileName = input("Enter name of color image file: ")  # retrieving user-inputted image file name
        if (imageFileName.__contains__(".tiff")):
            image = plt.imread(imageFileName)  # read in the image file using plt.imread()
            image = (image).astype(np.uint8)  # image as uint8 type
            invalidInput = False
        else:
            print("Please enter a valid file name")

    #Part 2
    row, column, num_layers = image.shape # (1152,1304,4)


    for row_index in range(len(image)):
        for column_index in range(len(image[0])):
            r = image[row_index, column_index, 0]
            g = image[row_index, column_index, 1]
            b = image[row_index, column_index, 2]

    keyString = 'COME AND GET YOUR LOVE'
    countLetters = 0

    for letter in keyString:
        if (letter != ' '):
            countLetters += 1
    dim = (image.shape[0],image.shape[1]) #(1152, 1304)
    keyArr = np.zeros(dim) #2D array of dimensions of the image only rows and columns

    for i in range(row):
        for j in range(column):
            A = (i * j) % countLetters #here % finds the remainder of the quotient
            key = A * ((2 ** 8) // countLetters) #A acts as a scalar multiple of the fraction: 2 to the power of 8 floor divided by number of letters in keyString (floor division brings down division to nearest lower integer)
            keyArr[i][j] = key

    keyArr = keyArr.astype(np.uint8) #key array needs to be type uint8 in order to apply the XOR cipher
    decryptedArr = decrypt_image(image, keyArr) #this calls the function decrypt_image which does XOR operations on each layer/channel of the image
    decryptedArrProper = decryptedArr[:,:,0:3]
    plt.imsave('plain.tiff', decryptedArrProper)

    #Part 3
    decryptedGray = grayScale('plain.tiff')
    sobel_edge_detection(decryptedGray[:,:,0])

    #Part 4

    plt.hist(decryptedArrProper[:,:,0].reshape(decryptedArrProper.shape[0]*decryptedArrProper.shape[1]), bins=np.arange((2**8)+1))
    plt.savefig('rhist.tiff')
    plt.show()
    plt.hist(decryptedArrProper[:,:,1].reshape(decryptedArrProper.shape[0]*decryptedArrProper.shape[1]), bins=np.arange((2**8)+1))
    plt.savefig('ghist.tiff')
    plt.show()
    plt.hist(decryptedArrProper[:,:,2].reshape(decryptedArrProper.shape[0]*decryptedArrProper.shape[1]), bins=np.arange((2**8)+1))
    plt.savefig('bhist.tiff')
    plt.show()

    keyArrEncrypt = generateKey(decryptedArrProper)
    encryptedArr = encryptImage(decryptedArrProper, keyArrEncrypt)



###################################################
#For developing two encrypted images to be placed into Box Drive
    # testImageForEncryption = plt.imread("purduebelltower.tiff")
    # keyArrEncrypt = generateKey(testImageForEncryption[:, :, 0:3])
    # encryptedArr = encryptImage(testImageForEncryption[:, :, 0:3], keyArrEncrypt)
    # plt.imshow(encryptedArr)
    # plt.show()
    # plt.imsave('LC2_05_Image_1.tiff', encryptedArr)
    #
    # testImageForEncryption2 = plt.imread("enggfountain.tiff")
    # keyArrEncrypt = generateKey(testImageForEncryption2[:, :, 0:3])
    # encryptedArr = encryptImage(testImageForEncryption2[:, :, 0:3], keyArrEncrypt)
    # plt.imshow(encryptedArr)
    # plt.show()
    #
    # plt.imsave('LC2_05_Image_2.tiff', encryptedArr)
##############################################################

    plt.hist(encryptedArr[:, :, 0].reshape(encryptedArr.shape[0] * encryptedArr.shape[1]),
             bins=np.arange((2 ** 8) + 1))
    plt.savefig('rhistEncrypted.tiff')
    plt.show()
    plt.hist(encryptedArr[:, :, 1].reshape(encryptedArr.shape[0] * encryptedArr.shape[1]),
             bins=np.arange((2 ** 8) + 1))
    plt.savefig('ghistEncrypted.tiff')
    plt.show()

    plt.hist(encryptedArr[:, :, 2].reshape(encryptedArr.shape[0] * encryptedArr.shape[1]),
             bins=np.arange((2 ** 8) + 1))
    plt.savefig('bhistEncrpyted.tiff')
    plt.show()

if __name__ == "__main__":
    main()