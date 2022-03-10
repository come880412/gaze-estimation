'''
Modified Date: 2022/01/13
Author: Gi-Luen Huang
mail: come880412@gmail.com
'''

import random
import cv2
from PIL import ImageOps, ImageFilter

class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img

def data_augmentation():
    from torchvision import transforms
    from PIL import Image

    img = cv2.imread("0000047.png")
    cv2.imwrite('./visual/ori.jpg', img)
    
    img = Image.open("0000047.png").convert("RGB")
    img = transforms.RandomResizedCrop((200, 320))(img)

    img10 = transforms.RandomHorizontalFlip()(img)
    img10.save('./visual/Horizontal.jpg')

    imgB = transforms.ColorJitter(brightness=0.5)(img)
    imgB.save('./visual/brightness.jpg')
    imgC = transforms.ColorJitter(contrast=0.5)(img)
    imgC.save('./visual/contrast.jpg')
    imgS = transforms.ColorJitter(saturation=0.5)(img)
    imgS.save('./visual/saturation.jpg')
    imgH = transforms.ColorJitter(hue=0.5)(img)
    imgH.save('./visual/hue.jpg')
    imgall = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)(img)
    imgall.save('./visual/ColorJitter.jpg')

    img_gaussian = transforms.GaussianBlur((23, 23), (0.1, 2.0))(img)
    img_gaussian.save('./visual/gaussian.jpg')

    img_solor = Solarization(1)(img)
    img_solor.save('./visual/solar.jpg')

    img_gray = transforms.RandomGrayscale(1)(img)
    img_gray.save('./visual/grayscale.jpg')

if __name__ in "__main__":
    data_augmentation() # Visualize the image after data augmentation
