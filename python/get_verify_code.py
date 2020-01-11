import time
from io import BytesIO

import requests
from matplotlib import pyplot as plt

base_url = "https://jxfw.gdut.edu.cn/"
headers = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36"
}


def get_pic():
    pic_url = base_url + "/yzm?d=%d" % time.time()
    r = requests.get(pic_url, headers=headers, verify=False)
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
