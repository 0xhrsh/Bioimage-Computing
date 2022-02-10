from skimage import io, color
import numpy as np

IMAGE_NAME = "Q1"
IMAGE_URL = IMAGE_NAME + ".png"
NUMBER_OF_ITERATIONS = 10

DEFAULT_K = 100
DEFAULT_M = 20


class Cluster(object):
    cluster_index = 1

    def __init__(self, h, w, l, a, b):
        self.no = self.cluster_index
        Cluster.cluster_index += 1
        self.pixels = []
        self.update(h, w, l, a, b)

    def update(self, h, w, l, a, b):
        self.h = h
        self.w = w
        self.l = l
        self.a = a
        self.b = b


class SLICProcessor(object):
    def __init__(self, filename, K, M):
        print("SLIC Processor initiated with K = {}, M = {}".format(K, M))
        self.K = K
        self.M = M

        rgbImg = io.imread(filename)
        self.data = color.rgb2lab(rgbImg)

        self.image_height = self.data.shape[0]
        self.image_width = self.data.shape[1]

        self.N = self.image_height * self.image_width
        self.S = int((self.N / self.K)**0.5)

        self.dis = np.full((self.image_height, self.image_width), np.inf)

        self.label = {}
        self.clusters = []
        

    def init_clusters(self):
        h = self.S / 2
        w = self.S / 2
        while h < self.image_height:
            while w < self.image_width:
                ih = int(h)
                iw = int(w)
                new_cluster = Cluster(
                    ih, iw, self.data[ih][iw][0], self.data[ih][iw][1], self.data[ih][iw][2])
                self.clusters.append(new_cluster)
                w += self.S
            w = self.S / 2
            h += self.S

    def update_cluster(self):
        for cluster in self.clusters:
            sum_h = 0
            sum_w = 0
            number = 0
            for p in cluster.pixels:
                number += 1

                sum_h += p[0]
                sum_w += p[1]

            _h = int(sum_h / number)
            _w = int(sum_w / number)
            cluster.update(
                _h, _w, self.data[_h][_w][0], self.data[_h][_w][1], self.data[_h][_w][2])

    def move_clusters(self):
        for cluster in self.clusters:
            cluster_gradient = self.find_gradient(cluster.h, cluster.w)
            for dh in range(-1, 2):
                for dw in range(-1, 2):
                    _h = cluster.h + dh
                    _w = cluster.w + dw
                    new_gradient = self.find_gradient(_h, _w)
                    if cluster_gradient > new_gradient:
                        cluster_gradient = new_gradient
                        cluster.update(
                            _h, _w, self.data[_h][_w][0], self.data[_h][_w][1], self.data[_h][_w][2])

    def find_gradient(self, h, w):
        if h + 1 >= self.image_height:
            h = self.image_height - 2
        if w + 1 >= self.image_width:
            w = self.image_width - 2

        grad = 0
        for t in range(3):
            grad += self.data[h + 1][w + 1][t] - self.data[h][w][t]

        return grad

    def assign(self):
        for cluster in self.clusters:
            for h in range(cluster.h - 2 * self.S, cluster.h + 2 * self.S):
                if h >= 0 and h < self.image_height:
                    for w in range(cluster.w - 2 * self.S, cluster.w + 2 * self.S):
                        if w >= 0 and w < self.image_width:
                            L, A, B = self.data[h][w]
                            Dc = ((L - cluster.l) ** 2 +
                                  (A - cluster.a) ** 2 + (B - cluster.b) ** 2)**0.5

                            Ds = ((h - cluster.h) ** 2 +
                                  (w - cluster.w) ** 2)**0.5

                            D = ((Dc / self.M) ** 2 + (Ds / self.S) ** 2)**0.5
                            if D < self.dis[h][w]:
                                if (h, w) in self.label:
                                    self.label[(h, w)].pixels.remove((h, w))
                                    self.label[(h, w)] = cluster
                                    cluster.pixels.append((h, w))
                                else:
                                    self.label[(h, w)] = cluster
                                    cluster.pixels.append((h, w))

                                self.dis[h][w] = D

    def iterate(self, n):
        self.init_clusters()
        self.move_clusters()
        for i in range(n):
            print("Iteration: {}/{}".format(i, n))
            self.assign()
            self.update_cluster()
            name = IMAGE_NAME + \
                '_M{}_K{}_{}.png'.format(self.M, self.K, i)
            self.save_image(name)

    def save_image(self, name):
        image_arr = np.copy(self.data)

        for cluster in self.clusters:
            for p in cluster.pixels:
                image_arr[p[0]][p[1]][0] = cluster.l
                image_arr[p[0]][p[1]][1] = cluster.a
                image_arr[p[0]][p[1]][2] = cluster.b
            for t in range(3):
                image_arr[cluster.h][cluster.w][t] = 0

        rgb_arr = color.lab2rgb(image_arr)
        io.imsave(name, rgb_arr)


if __name__ == '__main__':
    p = SLICProcessor(IMAGE_URL, DEFAULT_K, DEFAULT_M)
    p.iterate(NUMBER_OF_ITERATIONS)
