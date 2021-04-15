from sklearn.preprocessing import MinMaxScaler
from PIL import Image
from torchvision import transforms


class PriceToImgae:
    def __init__(self, days, width, height, price_area_height, volume_area_height, pixelwidth, pixelheight):
        self.days = days
        self.width = width
        self.height = height
        self.price_area_height = price_area_height
        self.volume_area_height = volume_area_height
        self.pixelwidth = pixelwidth
        self.pixelheight = pixelheight
        self.price_area_logical_height = int(self.price_area_height / self.pixelheight)
        self.volume_area_logical_height = int(self.volume_area_height / self.pixelheight)

    def __drawPixel(self, x, y, pixel):
        logical_height = int(self.height / self.pixelheight)
        for i in range(self.pixelwidth):
            for j in range(self.pixelheight):
                self.img.putpixel((self.pixelwidth * x + i, self.pixelheight * (logical_height - 1 - y) + j), pixel)

    def __drawPrice(self, index, price, moving_average, volume, pixel):
        open_price = price[0]
        high_price = price[1]
        low_price = price[2]
        close_price = price[3]

        # 画OHLC表
        self.__drawPixel(3 * index + 0, self.volume_area_logical_height + 1 + open_price, pixel)

        for i in range(high_price - low_price + 1):
            self.__drawPixel(3 * index + 1, self.volume_area_logical_height + 1 + low_price + i, pixel)
        self.__drawPixel(3 * index + 2, self.volume_area_logical_height + 1 + close_price, pixel)

        # 画MA线
        self.__drawPixel(3 * index + 0, self.volume_area_logical_height + 1 + moving_average, pixel)
        self.__drawPixel(3 * index + 1, self.volume_area_logical_height + 1 + moving_average, pixel)
        self.__drawPixel(3 * index + 2, self.volume_area_logical_height + 1 + moving_average, pixel)

        # 画成交量柱
        for i in range(volume):
            self.__drawPixel(3 * index + 1, i, pixel)

    def getImg(self, price_array, moving_average_array, volume_array, background_pixel, color_pixel):
        self.img = Image.new("RGB", (self.width, self.height), background_pixel)
        for i in range(price_array.shape[0]):
            self.__drawPrice(i, price_array[i], moving_average_array[i], volume_array[i], color_pixel)
        return self.img


def image_loader(image):
    transform = transforms.Compose([
        transforms.ToTensor()])
    image = transform(image).squeeze(0)
    return image


def tensor_to_PIL(tensor):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = unloader(image)
    return image


def get_image_with_price(price):
    '''
    price [time, feature]
    feature: open, close, low, high, vol, MA
    '''
    # 像素大小
    PIXEL_WIDTH = 1
    PIXEL_HEIGHT = 1
    # 宽度是时间序列的三倍长
    WIDTH = 3 * price.shape[0] * PIXEL_WIDTH

    # 价格占高度2/3，vol占1/3
    PRICE_LOGICAL_HEIGHT = 2 * price.shape[0]
    VOLUME_LOGICAL_HEIGHT = price.shape[0]

    # 计算区域各区域大小
    PRICE_AREA_HEIGHT = PRICE_LOGICAL_HEIGHT * PIXEL_HEIGHT
    V0LUME_AREA_HEIGHT = VOLUME_LOGICAL_HEIGHT * PIXEL_HEIGHT

    # 总高度还是加一个pixel大小分割
    HEIGHT = PRICE_AREA_HEIGHT + V0LUME_AREA_HEIGHT + PIXEL_HEIGHT

    # 放缩
    sclr1 = MinMaxScaler((0, PRICE_LOGICAL_HEIGHT - 1))
    sclr2 = MinMaxScaler((1, VOLUME_LOGICAL_HEIGHT))
    price_minmax = sclr1.fit_transform(price[:, :-1].reshape(-1, 1)).reshape(price.shape[0], -1).astype(int)
    volume_minmax = sclr2.fit_transform(price[:, -1].reshape(-1, 1)).reshape(price.shape[0]).astype(int)

    # 时间序列长度
    days = price_minmax.shape[0]

    # 转图片
    p2i = PriceToImgae(days, WIDTH, HEIGHT, PRICE_AREA_HEIGHT, V0LUME_AREA_HEIGHT, PIXEL_WIDTH, PIXEL_HEIGHT)
    background_pixel = (0, 0, 0, 100)
    color_pixel = (255, 255, 255, 100)
    image = p2i.getImg(price_minmax[:, :-1], price_minmax[:, -1], volume_minmax, background_pixel, color_pixel)
    # 转成黑白像素
    image = image.convert('1')
    return image