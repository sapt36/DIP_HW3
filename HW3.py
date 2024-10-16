import sys
import cv2
import numpy as np
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QGridLayout, QWidget, QPushButton,
                             QLineEdit, QFileDialog, QMessageBox, QSpinBox, QComboBox, QHBoxLayout, QScrollArea, QTabWidget)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt

class SpatialFilteringApp(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Spatial Filtering Operations')
        self.setGeometry(100, 100, 1200, 800)

        # 設置字體
        self.setFont(QFont('微軟正黑體', 12, QFont.Bold))

        # Load and Mask UI
        self.loadButton = QPushButton('載入影像')
        self.loadButton.clicked.connect(self.load_image)

        self.maskTypeLabel = QLabel('遮罩類型:')
        self.maskTypeCombo = QComboBox()
        self.maskTypeCombo.addItems(['Box', 'Gaussian', 'Other'])
        self.maskTypeCombo.currentTextChanged.connect(self.update_mask_inputs)
        self.maskTypeCombo.currentTextChanged.connect(self.update_sigma_inputs)

        self.maskSizeLabel = QLabel('遮罩大小:')
        self.maskSizeInput = QSpinBox()
        self.maskSizeInput.setRange(3, 21)
        self.maskSizeInput.setSingleStep(2)
        self.maskSizeInput.setValue(3)
        self.maskSizeInput.valueChanged.connect(self.update_mask_inputs)
        self.maskSizeInput.valueChanged.connect(self.update_sigma_inputs)

        self.sigmaLabel = QLabel('高斯標準差(Sigma):')
        self.sigmaInput = QLineEdit('自動(可手動輸入調整)')
        self.sigmaLabel.hide()  # 初始隱藏
        self.sigmaInput.hide()  # 初始隱藏

        # 隱藏的標籤和輸入欄位，僅當 Mask Type 為 Other 時顯示
        self.averagingLabel = QLabel('自定義濾波器(Other)的歸一化因子 [ 輸入:9 == 1/9 ]:')
        self.averagingInput = QLineEdit('1')
        self.averagingLabel.hide()  # 初始隱藏
        self.averagingInput.hide()  # 初始隱藏

        self.applyButton = QPushButton('套用遮罩')
        self.applyButton.clicked.connect(self.apply_mask)

        self.imageLabel = QLabel()
        self.resultLabel = QLabel()

        # 水平佈局來顯示原始圖像和處理後圖像
        imageLayout = QHBoxLayout()
        imageLayout.addWidget(QLabel('原始影像:'))
        imageLayout.addWidget(self.imageLabel)
        imageLayout.addWidget(QLabel('處理後影像:'))
        imageLayout.addWidget(self.resultLabel)

        imageLayout.setAlignment(Qt.AlignCenter)

        topLayout = QHBoxLayout()
        topLayout.addWidget(self.maskTypeLabel)
        topLayout.addWidget(self.maskTypeCombo)
        topLayout.addWidget(self.maskSizeLabel)
        topLayout.addWidget(self.maskSizeInput)
        topLayout.addWidget(self.sigmaLabel)
        topLayout.addWidget(self.sigmaInput)

        averageLayout = QHBoxLayout()
        averageLayout.addWidget(self.averagingLabel)
        averageLayout.addWidget(self.averagingInput)

        layout = QVBoxLayout()
        layout.addWidget(self.loadButton)
        layout.addLayout(topLayout)
        layout.addLayout(averageLayout)
        self.maskLayout = QGridLayout()
        layout.addLayout(self.maskLayout)
        layout.addWidget(self.applyButton)
        layout.addLayout(imageLayout)  # 使用水平佈局來顯示圖像

        self.setLayout(layout)

        # 創建滾動區域，將主布局設置為滾動區域的內容
        scrollArea = QScrollArea()  # 創建滾動區域
        scrollArea.setWidgetResizable(True)  # 使內容自動調整
        container = QWidget()
        container.setLayout(layout)

        # 將滾動區域的內容設置為 QWidget
        scrollArea.setWidget(container)

        # 設置滾動區域為主界面
        self.setLayout(QVBoxLayout())  # 清空主界面
        self.layout().addWidget(scrollArea)  # 將滾動區域添加到主界面

    def update_sigma_inputs(self):
        mask_type = self.maskTypeCombo.currentText()
        if mask_type == 'Gaussian':
            self.sigmaLabel.show()
            self.sigmaInput.show()
        else:
            self.sigmaLabel.hide()
            self.sigmaInput.hide()

    def update_mask_inputs(self):
        mask_type = self.maskTypeCombo.currentText()
        self.clear_mask_layout()  # 清理 UI
        if mask_type == 'Other':
            self.averagingLabel.show()  # 顯示標籤
            self.averagingInput.show()  # 顯示輸入欄位
            self.create_mask_inputs()
        else:
            self.averagingLabel.hide()  # 隱藏標籤
            self.averagingInput.hide()  # 隱藏輸入欄位

    def load_image(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp);;All Files (*)")
        if fileName:
            self.image = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
            self.show_image(self.image, self.imageLabel)

    def show_image(self, image, label):
        height, width = image.shape
        bytesPerLine = width
        qimage = QImage(image.data, width, height, bytesPerLine, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        label.setPixmap(pixmap.scaled(256, 256))

    def clear_mask_layout(self):
        for i in reversed(range(self.maskLayout.count())):
            widget = self.maskLayout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)
        self.maskInputs = []

    def create_mask_inputs(self):
        size = self.maskSizeInput.value()
        for i in range(size):
            row = []
            for j in range(size):
                lineEdit = QLineEdit('0')
                self.maskLayout.addWidget(lineEdit, i, j)
                row.append(lineEdit)
            self.maskInputs.append(row)

    def apply_mask(self):

        def manual_gaussian_kernel(size, sigma=None):
            if sigma is None:
                sigma = size / 6  # 自動推導 sigma

            kernel = np.zeros((size, size), np.float32)
            center = size // 2
            sum_val = 0  # 用於歸一化

            for i in range(size):
                for j in range(size):
                    x, y = i - center, j - center
                    kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
                    sum_val += kernel[i, j]

            kernel /= sum_val  # 正規化，使其總和為 1
            return kernel

        def manual_filter2D(image, kernel):
            image_height, image_width = image.shape
            kernel_size = kernel.shape[0]
            pad = kernel_size // 2

            padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='reflect')
            filtered_image = np.zeros_like(image)

            for i in range(image_height):
                for j in range(image_width):
                    roi = padded_image[i:i + kernel_size, j:j + kernel_size]
                    filtered_value = np.sum(roi * kernel)
                    filtered_image[i, j] = filtered_value

            filtered_image = np.clip(filtered_image, 0, 255)
            return filtered_image.astype(np.uint8)

        if self.image is None:
            QMessageBox.warning(self, 'Warning', 'Please load an image first!')
            return

        mask_size = self.maskSizeInput.value()
        mask_type = self.maskTypeCombo.currentText()

        if mask_type == 'Box':
            mask = np.ones((mask_size, mask_size), np.float32) / (mask_size * mask_size)
        elif mask_type == 'Gaussian':
            sigma_text = self.sigmaInput.text()
            if sigma_text == 'Auto' or sigma_text == '':
                sigma = None
            else:
                try:
                    sigma = float(sigma_text)
                except ValueError:
                    QMessageBox.warning(self, 'Warning', 'Please enter a valid sigma value!')
                    return

            mask = manual_gaussian_kernel(mask_size, sigma)
        elif mask_type == 'Other':
            mask = self.get_custom_mask(mask_size)
            if mask is None:
                return

        start_time = time.time()

        filtered_image = manual_filter2D(self.image, mask)

        end_time = time.time()

        self.show_image(filtered_image, self.resultLabel)
        QMessageBox.information(self, 'Computation Time', f'Processing time: {end_time - start_time:.4f} seconds')

    def get_custom_mask(self, size):
        mask = np.zeros((size, size), np.float32)
        for i in range(size):
            for j in range(size):
                try:
                    value = float(self.maskInputs[i][j].text())
                    mask[i, j] = value
                except ValueError:
                    QMessageBox.warning(self, 'Warning', 'Please enter valid numbers!')
                    return None
        try:
            averaging_value = float(self.averagingInput.text())
            return mask / averaging_value
        except ValueError:
            QMessageBox.warning(self, 'Warning', 'Please enter a valid averaging value!')
            return None

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

    # 手動載入灰階影像
    def load_grayscale_image(self, file_path):
        from PIL import Image
        img = Image.open(file_path).convert('L')  # 轉換為灰階
        return np.array(img)


class LocalEnhancementApp(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('局部增強方法')
        self.setGeometry(100, 100, 1200, 800)

        # 設置字體
        self.setFont(QFont('微軟正黑體', 12, QFont.Bold))

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

        self.imageLabel = QLabel()
        self.localResultLabel = QLabel()  # 局部增強結果顯示
        self.histEqualizationResultLabel = QLabel()  # 直方圖均衡化結果顯示

        imageLayout = QHBoxLayout()
        imageLayout.addWidget(QLabel('原始影像:'))
        imageLayout.addWidget(self.imageLabel)
        imageLayout.addWidget(QLabel('局部增強結果:'))
        imageLayout.addWidget(self.localResultLabel)
        imageLayout.addWidget(QLabel('均衡化結果:'))
        imageLayout.addWidget(self.histEqualizationResultLabel)
        imageLayout.setAlignment(Qt.AlignCenter)

        SXYLayout = QHBoxLayout()
        SXYLayout.addWidget(self.neighborhoodSizeLabel)
        SXYLayout.addWidget(self.neighborhoodSizeSpinBox)

        layout = QVBoxLayout()
        layout.addWidget(self.loadButton)
        layout.addLayout(SXYLayout)
        layout.addWidget(self.applyLocalButton)
        layout.addWidget(self.applyHistButton)
        layout.addLayout(imageLayout)

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

        result_image = cv2.equalizeHist(self.image)
        self.show_image(result_image, self.histEqualizationResultLabel)

    def local_enhancement(self, image, sxy):
        image_float = image.astype(np.float32)
        result_image = np.zeros_like(image_float)
        padded_image = cv2.copyMakeBorder(image_float, sxy//2, sxy//2, sxy//2, sxy//2, cv2.BORDER_REFLECT)

        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                local_region = padded_image[y:y + sxy, x:x + sxy]
                local_mean = np.mean(local_region)
                local_std = np.std(local_region)
                k = 1.0
                result_image[y, x] = k * (image_float[y, x] - local_mean) / (local_std + 1e-5) + 128

        result_image = np.clip(result_image, 0, 255).astype(np.uint8)
        return result_image

# 主視窗使用分頁選項
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Digital Image Processing HW3')
        self.setGeometry(100, 100, 1200, 900)

        # 創建分頁
        self.tabs = QTabWidget()

        # 第一個分頁：Spatial Filtering
        self.tab1 = SpatialFilteringApp()
        self.tabs.addTab(self.tab1, "Part 2 : Spatial Filtering")

        # 第二個分頁：Edge Detection
        self.tab2 = EdgeDetectionApp()
        self.tabs.addTab(self.tab2, "Part 3 : Edge Detection")

        # 第三個分頁：Local Enhancement
        self.tab3 = LocalEnhancementApp()
        self.tabs.addTab(self.tab3, "Part 4 : Local Enhancement")

        self.setCentralWidget(self.tabs)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())