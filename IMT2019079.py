import cv2
import numpy as np

#hough line implementation algortihm.
def HoughLinesp(CannyImage,image,threshold):
    image1 = image.copy()
    width = CannyImage.shape[1]
    height = CannyImage.shape[0]
    Thetas = np.deg2rad(np.arange(0.0, 180.0))
    
    #Initializing the Accumulator.
    DiagonalLength = int(np.around(np.sqrt(width*width + height*height),0))
    Accumulator = np.zeros((2*DiagonalLength,len(Thetas)))
    Rhos = np.linspace(-DiagonalLength,DiagonalLength,2*DiagonalLength)
    
    for i in range(0,height):
        for j in range(0,width):
            for k in range(0,len(Thetas)):
                Rho = int(np.around(i*(np.cos(Thetas[k])) + j*(np.sin(Thetas[k])),0)) 
                Accumulator[Rho+DiagonalLength,k] += 1
          
    linesList = []
    #Thresholding the votes.
    for i in range(0,Accumulator.shape[0]):
        for j in range(0,Accumulator.shape[1]):
            if Accumulator[i,j] > threshold:
                linesList.append([Rhos[i],Thetas[j]])
                
    #Plotting the lines on given Image.
    for i in range(0,len(linesList)):
        for i in range(0, len(linesList)):
            a = np.cos(np.deg2rad(linesList[i][1]))
            b = np.sin(np.deg2rad(linesList[i][1]))
            x0 = a*linesList[i][0]
            y0 = b*linesList[i][0]
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            cv2.line(image1, (x1, y1), (x2, y2), (255, 0, 0), 5)

    return image1


#Hough circle implementation.
def HoughCircle(CannyImage,image,radius):
    image1 = image.copy()
    width  = CannyImage.shape[1]
    height = CannyImage.shape[0]
    Thetas = np.deg2rad(np.arange(0.0, 180.0))
    
    #Initializing the Accumulator.
    list1 = []
    list2 = []
    accumulator = np.zeros((2*width,2*width))
    for i in range(0,height):
        for j in range(0,width):
            for k in range(0,len(Thetas)):
                a = int(np.around(i - radius*np.cos(Thetas[k]),0)) 
                b = int(np.around(j - radius*np.sin(Thetas[k]),0)) 
                list1.append(a)
                list2.append(b)
                accumulator[a+width,b+width] += 1
    
    circle = []
    #Thresholding the votes.
    for i in range(0,accumulator.shape[0]):
        for j in range(0,accumulator.shape[1]):
            if accumulator[i,j] > 96:
                circle.append([i-width,j-width])

    #Plotting the circles on given Image.
    for center in circle:
        temp = center
        x = temp[0]
        y = temp[1]
        cv2.circle(image1, (x, y), radius, (0, 0, 0), 3)
        cv2.circle(image1, (x, y), 2, (0, 0, 255), 3)

    return image1

def processing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50,150)
    return canny

def Masking(image):
    height = image.shape[0]
    polygons = np.array([
        [(0, 60), (0, height), (594, height), (594, 70), (310, 0)]
        ]) 
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    maskedImage = cv2.bitwise_and(image, mask)
    return maskedImage


#PART1, LANE DETECTION
lanesimage = cv2.imread('lanes.jpg')
finalL = processing(lanesimage)
croppedImage = Masking(finalL)
lanes = HoughLinesp(croppedImage, lanesimage, 50)

cv2.imshow("Final Lanes", lanes)
# cv2.imwrite('Final Lanes.jpg', lanes)



#PART1, COIN DETECTION
coinsimage = cv2.imread('coins.jpg')
finalC = processing(coinsimage)
coins = HoughCircle(finalC, coinsimage, 15)

cv2.imshow("Final Coins", coins)
# cv2.imwrite('Final Coins.jpg', coins)



cv2.waitKey(0)
cv2.destroyAllWindows




