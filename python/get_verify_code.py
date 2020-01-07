import time
from io import BytesIO

import requests
from matplotlib import pyplot as plt

base_url = "https://jxfw.gdut.edu.cn/"


def get_pic():
    pic_url = base_url + "/yzm?d=%d" % time.time()
    r = requests.get(pic_url)
    return plt.imread(BytesIO(r.content), "jpg")


def show_pic(image_data):
    plt.imshow(image_data)
    plt.show()


def main():
    data = get_pic()
    print(data.shape)
    print(data)
    show_pic(data)


if __name__ == '__main__':
    main()
