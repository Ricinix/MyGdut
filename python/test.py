from matplotlib import pyplot as plt

import captcha_generate

if __name__ == '__main__':
    gen = captcha_generate.GenerateCaptcha()
    X, y = gen.gen_test_captcha()
    print(X.shape)
    plt.imshow(X.view(60, 140, 3))
    print(gen.decode_captcha(y))
    plt.show()
