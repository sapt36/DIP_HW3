import sys
import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QSpinBox,
                             QHBoxLayout, QWidget, QScrollArea, QFileDialog)
from PyQt5.QtGui import QImage, QPixmap

class LocalEnhancementApp(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('局部增強方法')
        self.setGeometry(100, 100, 1200, 800)

        # 按鈕和輸入欄位
        self.loadButton = QPushButton('載入影像')
        self.loadButton.clicked.connect(self.load_image)

        self.neighborhoodSizeLabel = QLabel('鄰域大小 (Sxy):')
        self.neighborhoodSizeSpinBox = QSpinBox()
        self.neighborhoodSizeSpinBox.setRange(1, 20)
        self.neighborhoodSizeSpinBox.setValue(3)

        self.applyLocalButton = QPushButton('套用局部增強')
        self.applyLocalButton.clicked.connect(self.apply_local_enhancement)

        self.applyHistButton = QPushButton('套用直方圖均衡化')
        self.applyHistButton.clicked.connect(self.apply_histogram_equalization)

        self.applyLocalHistStatButton = QPushButton('局部直方圖統計增強')
        self.applyLocalHistStatButton.clicked.connect(self.apply_local_histogram_enhancement)

        self.imageLabel = QLabel()
        self.localResultLabel = QLabel()  # 局部增強結果顯示
        self.histEqualizationResultLabel = QLabel()  # 直方圖均衡化結果顯示
        self.finalResultLabel = QLabel()  # 局部直方圖統計增強結果顯示

        # 添加三個結果分開顯示局部增強、直方圖均衡化和局部直方圖統計增強
        imageLayout = QHBoxLayout()
        imageLayout.addWidget(QLabel('原始影像:'))
        imageLayout.addWidget(self.imageLabel)
        imageLayout.addWidget(QLabel('局部增強結果:'))
        imageLayout.addWidget(self.localResultLabel)
        imageLayout.addWidget(QLabel('均衡化結果:'))
        imageLayout.addWidget(self.histEqualizationResultLabel)

        finalLayout = QHBoxLayout()
        finalLayout.addWidget(QLabel('局部直方圖統計增強結果:'))
        finalLayout.addWidget(self.finalResultLabel)

        SXYLayout = QHBoxLayout()
        SXYLayout.addWidget(self.neighborhoodSizeLabel)
        SXYLayout.addWidget(self.neighborhoodSizeSpinBox)

        layout = QVBoxLayout()
        layout.addWidget(self.loadButton)
        layout.addLayout(SXYLayout)
        layout.addWidget(self.applyLocalButton)
        layout.addWidget(self.applyHistButton)
        layout.addWidget(self.applyLocalHistStatButton)
        layout.addLayout(imageLayout)
        layout.addLayout(finalLayout)

        # 設置滾動區域
        scrollArea = QScrollArea()
        scrollArea.setWidgetResizable(True)
        container = QWidget()
        container.setLayout(layout)
        scrollArea.setWidget(container)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(scrollArea)

    def load_image(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp);;All Files (*)", options=options)
        if fileName:
            self.image = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
            self.show_image(self.image, self.imageLabel)

    def show_image(self, image, label):
        height, width = image.shape
        bytesPerLine = width
        qimage = QImage(image.data, width, height, bytesPerLine, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        label.setPixmap(pixmap.scaled(256, 256, Qt.KeepAspectRatio))

    def apply_local_enhancement(self):
        if self.image is None:
            return

        neighborhood_size = self.neighborhoodSizeSpinBox.value()
        result_image = self.local_enhancement(self.image, neighborhood_size)
        self.show_image(result_image, self.localResultLabel)

    def apply_histogram_equalization(self):
        if self.image is None:
            return

        result_image = self.manual_histogram_equalization(self.image)
        self.show_image(result_image, self.histEqualizationResultLabel)

    def apply_local_histogram_enhancement(self):
        if self.image is None:
            return

        result_image = self.local_histogram_statistics(self.image)
        self.show_image(result_image, self.finalResultLabel)

    def local_enhancement(self, image, sxy):
        image_float = image.astype(np.float32)
        result_image = np.zeros_like(image_float)
        padded_image = np.pad(image_float, sxy//2, mode='reflect')

        global_mean = 161  # 全局均值
        global_std = 103   # 全局標準差
        C = 22.8  # 增強係數

        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                local_region = padded_image[y:y + sxy, x:x + sxy]
                local_mean = np.mean(local_region)
                local_std = np.std(local_region)

                if local_mean < global_mean:  # 增強暗部區域
                    k = C * ((global_mean - local_mean) / (global_mean + local_mean))
                    result_image[y, x] = image_float[y, x] + k * (image_float[y, x] - local_mean)

        result_image = np.clip(result_image, 0, 255).astype(np.uint8)
        return result_image

    def manual_histogram_equalization(self, image):
        # Step 1: 計算直方圖
        hist, bins = np.histogram(image.flatten(), 256, [0, 256])

        # Step 2: 計算累積分佈函數 (CDF) 並進行正規化
        cdf = hist.cumsum()  # 計算累積分佈函數
        cdf_normalized = cdf * (255 / cdf[-1])  # 將 CDF 正規化到 [0, 255]

        # Step 3: 應用 CDF 來轉換像素值
        equalized_image = np.interp(image.flatten(), bins[:-1], cdf_normalized).reshape(image.shape)

        # Step 4: 降噪 (Denoising) - 使用簡單的均值濾波器進行降噪
        denoised_image = self.mean_filter(equalized_image)

        return denoised_image.astype(np.uint8)

    def mean_filter(self, image):
        """ 手動實作的均值濾波器 (Mean Filter)，用來降噪 """
        kernel = np.ones((3, 3), np.float32) / 9
        padded_image = np.pad(image, ((1, 1), (1, 1)), mode='constant', constant_values=0)
        filtered_image = np.zeros_like(image)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                filtered_image[i, j] = np.sum(padded_image[i:i + 3, j:j + 3] * kernel)

        return filtered_image

    def local_histogram_statistics(self, image):
        image_float = image.astype(np.float32)
        result_image = np.zeros_like(image_float)

        global_mean = np.mean(image)
        global_std = np.std(image)

        C = 22.8
        k0, k1 = 0, 0.1
        k2, k3 = 0, 0.1

        sxy = self.neighborhoodSizeSpinBox.value()
        padded_image = np.pad(image_float, sxy//2, mode='reflect')

        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                local_region = padded_image[y:y + sxy, x:x + sxy]

                local_mean = np.mean(local_region)
                local_std = np.std(local_region)

                if k0 * global_mean <= local_mean <= k1 * global_mean and k2 * global_std <= local_std <= k3 * global_std:
                    result_image[y, x] = C * image_float[y, x]
                else:
                    result_image[y, x] = image_float[y, x]

        result_image = np.clip(result_image, 0, 255).astype(np.uint8)
        return result_image


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = LocalEnhancementApp()
    ex.show()
    sys.exit(app.exec_())
