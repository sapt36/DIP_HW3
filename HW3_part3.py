import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QFileDialog, QSlider, QWidget, \
    QComboBox, QSpinBox, QHBoxLayout, QScrollArea, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QFont


class EdgeDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('邊緣檢測：Marr-Hildreth 和 Sobel')
        self.setGeometry(100, 100, 1200, 800)

        # 設置字體
        self.setFont(QFont('微軟正黑體', 12, QFont.Bold))

        # UI 元件
        self.loadButton = QPushButton('載入影像')
        self.loadButton.clicked.connect(self.load_image)

        self.edgeMethodLabel = QLabel('邊緣檢測方法:')
        self.edgeMethodCombo = QComboBox()
        self.edgeMethodCombo.addItems(['Marr-Hildreth', 'Sobel'])
        self.edgeMethodCombo.currentTextChanged.connect(self.update_method_layout)

        self.zeroCrossingThresholdLabel = QLabel('Marr-Hildreth 零交叉閾值:')
        self.zeroCrossingThresholdSpinBox = QSpinBox()
        self.zeroCrossingThresholdSpinBox.setRange(1, 100)
        self.zeroCrossingThresholdSpinBox.setValue(5)

        self.applyButton = QPushButton('套用邊緣檢測')
        self.applyButton.clicked.connect(self.apply_edge_detection)

        self.imageLabel = QLabel()
        self.resultLabel = QLabel()

        # 水平佈局來顯示原始圖像和處理後圖像
        imageLayout = QHBoxLayout()
        imageLayout.addWidget(QLabel('原始影像:'))
        imageLayout.addWidget(self.imageLabel)
        imageLayout.addWidget(QLabel('處理後影像:'))
        imageLayout.addWidget(self.resultLabel)
        imageLayout.setAlignment(Qt.AlignCenter)

        methodLayout = QHBoxLayout()
        methodLayout.addWidget(self.edgeMethodLabel)
        methodLayout.addWidget(self.edgeMethodCombo)
        methodLayout.addWidget(self.zeroCrossingThresholdLabel)
        methodLayout.addWidget(self.zeroCrossingThresholdSpinBox)

        # 佈局設置
        layout = QVBoxLayout()
        layout.addWidget(self.loadButton)
        layout.addLayout(methodLayout)
        layout.addWidget(self.applyButton)
        layout.addLayout(imageLayout)

        self.setLayout(layout)

        # 設置滾動區域
        scrollArea = QScrollArea()
        scrollArea.setWidgetResizable(True)
        container = QWidget()
        container.setLayout(layout)
        scrollArea.setWidget(container)
        self.setLayout(QVBoxLayout())  # 清空主界面
        self.layout().addWidget(scrollArea)  # 將滾動區域添加到主界面

    def update_method_layout(self):
        method = self.edgeMethodCombo.currentText()

        if method == 'Marr-Hildreth':
            self.zeroCrossingThresholdLabel.show()
            self.zeroCrossingThresholdSpinBox.show()
        else:
            self.zeroCrossingThresholdLabel.hide()
            self.zeroCrossingThresholdSpinBox.hide()

    # 讀取影像
    def load_image(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp);;All Files (*)")
        if fileName:
            self.image = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
            self.show_image(self.image, self.imageLabel)

    # 將影像顯示在 QLabel 上
    def show_image(self, image, label):
        height, width = image.shape
        bytesPerLine = width
        qimage = QImage(image.data, width, height, bytesPerLine, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        label.setPixmap(pixmap.scaled(256, 256))

    # 影像處理主邏輯
    def apply_edge_detection(self):
        if self.image is None:
            QMessageBox.warning(self, '警告', '請先載入影像！')
            return

        method = self.edgeMethodCombo.currentText()

        if method == 'Marr-Hildreth':
            # Marr-Hildreth 邊緣檢測
            threshold = self.zeroCrossingThresholdSpinBox.value()
            processed_image = self.marr_hildreth_edge_detection(self.image, threshold)
        elif method == 'Sobel':
            # Sobel 邊緣檢測
            processed_image = self.sobel_edge_detection(self.image)

        self.show_image(processed_image, self.resultLabel)

    # Marr-Hildreth 邊緣檢測
    def marr_hildreth_edge_detection(self, image, zero_crossing_threshold):
        # 步驟1：使用高斯平滑影像
        blurred_image = self.gaussian_smoothing(image)

        # 步驟2：使用拉普拉斯運算子
        laplacian = self.laplacian_of_gaussian(blurred_image)

        # 步驟3：零交叉檢測
        zero_crossing_image = self.zero_crossing_detection(laplacian, zero_crossing_threshold)

        return zero_crossing_image

    # 手動高斯平滑
    def gaussian_smoothing(self, image, sigma=1.0):
        # 產生高斯核
        size = int(2 * np.ceil(3 * sigma) + 1)
        gauss_kernel = self.generate_gaussian_kernel(size, sigma)
        return self.convolve(image, gauss_kernel)

    # 產生高斯核
    def generate_gaussian_kernel(self, size, sigma):
        kernel = np.fromfunction(
            lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2) / (2*sigma**2)),
            (size, size)
        )
        return kernel / np.sum(kernel)

    # 手動拉普拉斯運算
    def laplacian_of_gaussian(self, image):
        # 使用拉普拉斯卷積核
        laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        return self.convolve(image, laplacian_kernel)

    # 零交叉檢測
    def zero_crossing_detection(self, laplacian, threshold):
        h, w = laplacian.shape
        zero_crossings = np.zeros_like(laplacian, dtype=np.uint8)

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                patch = laplacian[i-1:i+2, j-1:j+2]
                if np.min(patch) < 0 and np.max(patch) > 0:
                    if np.abs(np.max(patch) - np.min(patch)) > threshold:
                        zero_crossings[i, j] = 255

        return zero_crossings

    # Sobel 邊緣檢測（手動卷積）
    def sobel_edge_detection(self, image):
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        gx = self.convolve(image, sobel_x)
        gy = self.convolve(image, sobel_y)

        gradient_magnitude = np.sqrt(gx ** 2 + gy ** 2)
        return np.clip(gradient_magnitude, 0, 255).astype(np.uint8)

    # 手動卷積函數
    def convolve(self, image, kernel):
        h, w = image.shape
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2

        # 填充圖像以應對邊緣情況
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

        result = np.zeros_like(image, dtype=np.float64)

        for i in range(h):
            for j in range(w):
                result[i, j] = np.sum(padded_image[i:i+kh, j:j+kw] * kernel)

        return result


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = EdgeDetectionApp()
    window.show()
    sys.exit(app.exec_())
