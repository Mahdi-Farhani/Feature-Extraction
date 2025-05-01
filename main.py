import cv2
import math

def computeHistogramFeatures(imagePath):
    img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    h,w = img.shape

    histogram= [0]*256
    for i in range(h):
        for j in range(w):
            intensity= img[i][j]
            histogram[img[i][j]] += 1
    totalPixels=h*w
    probs= [count/totalPixels for count in histogram]
    mean=sum(i * probs[i] for i in range(256))
    variance=sum((i-mean)**2 * probs[i] for i in range(256))
    stddev=math.sqrt(variance)
    skewness=sum((i-mean)**3 * probs[i] for i in range(256)) / (stddev**3) if stddev != 0 else 0
    energy=sum(i**2 * probs[i] for i in range(256))
    entropy=-sum(probs[i] * math.log2(probs[i]) for i in range(256) if probs[i] > 0)
    return mean, stddev, skewness, energy, entropy

if __name__ == "__main__":
    imagePath = "07DZW.png"  
    mean, stddev, skewness, energy, entropy = computeHistogramFeatures(imagePath)
    print(f"Mean: {mean}")
    print(f"Standard Deviation: {stddev}")
    print(f"Skewness: {skewness}")
    print(f"Energy: {energy}")
    print(f"Entropy: {entropy}")