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
        self.setGeometry(100, 100, 1200, 700)

        # 設置字體
        self.setFont(QFont('微軟正黑體', 12, QFont.Bold))

        # Load and Mask UI
        self.loadButton = QPushButton('Load Image')
        self.loadButton.clicked.connect(self.load_image)

        self.maskTypeLabel = QLabel('Mask Type:')
        self.maskTypeCombo = QComboBox()
        self.maskTypeCombo.addItems(['Box', 'Gaussian', 'Other'])
        self.maskTypeCombo.currentTextChanged.connect(self.update_mask_inputs)

        self.maskSizeLabel = QLabel('Mask Size:')
        self.maskSizeInput = QSpinBox()
        self.maskSizeInput.setRange(3, 21)
        self.maskSizeInput.setSingleStep(2)
        self.maskSizeInput.setValue(3)
        self.maskSizeInput.valueChanged.connect(self.update_mask_inputs)

        self.sigmaLabel = QLabel('Gaussian Sigma:')
        self.sigmaInput = QLineEdit('Auto')
        self.sigmaInput.setPlaceholderText("Enter sigma value")

        # 隱藏的標籤和輸入欄位，僅當 Mask Type 為 Other 時顯示
        self.averagingLabel = QLabel('Normalization factor of "Other" filter [ enter:9 == 1/9 ]:')
        self.averagingInput = QLineEdit('1')
        self.averagingLabel.hide()  # 初始隱藏
        self.averagingInput.hide()  # 初始隱藏

        self.applyButton = QPushButton('Apply Mask')
        self.applyButton.clicked.connect(self.apply_mask)

        self.imageLabel = QLabel()
        self.resultLabel = QLabel()

        # 水平佈局來顯示原始圖像和處理後圖像
        imageLayout = QHBoxLayout()
        imageLayout.addWidget(QLabel('Original Image:'))
        imageLayout.addWidget(self.imageLabel)
        imageLayout.addWidget(QLabel('Processed Image:'))
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
        self.setWindowTitle('Edge Detection Operations')
        self.setGeometry(100, 100, 1200, 700)

        # 設置字體
        self.setFont(QFont('微軟正黑體', 12, QFont.Bold))

        # Load Image
        self.loadButton = QPushButton('Load Image')
        self.loadButton.clicked.connect(self.load_image)

        # Edge Detection Method Combo Box
        self.edgeMethodLabel = QLabel('Edge Detection Method:')
        self.edgeMethodCombo = QComboBox()
        self.edgeMethodCombo.addItems(['Sobel', 'Laplacian', 'Canny'])
        self.edgeMethodCombo.currentTextChanged.connect(self.update_edge_method)

        self.applyButton = QPushButton('Apply Edge Detection')
        self.applyButton.clicked.connect(self.apply_edge_detection)

        self.imageLabel = QLabel()
        self.resultLabel = QLabel()

        # 水平佈局來顯示原始圖像和處理後圖像
        imageLayout = QHBoxLayout()
        imageLayout.addWidget(QLabel('Original Image:'))
        imageLayout.addWidget(self.imageLabel)
        imageLayout.addWidget(QLabel('Processed Image:'))
        imageLayout.addWidget(self.resultLabel)

        imageLayout.setAlignment(Qt.AlignCenter)

        # Layout
        topLayout = QHBoxLayout()
        topLayout.addWidget(self.edgeMethodLabel)
        topLayout.addWidget(self.edgeMethodCombo)

        layout = QVBoxLayout()
        layout.addWidget(self.loadButton)
        layout.addLayout(topLayout)
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

    def update_edge_method(self):
        pass  # 根據需求更新方法（如參數設置）

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

    def apply_edge_detection(self):
        if self.image is None:
            QMessageBox.warning(self, 'Warning', 'Please load an image first!')
            return

        method = self.edgeMethodCombo.currentText()
        if method == 'Sobel':
            processed_image = cv2.Sobel(self.image, cv2.CV_64F, 1, 0, ksize=5)
        elif method == 'Laplacian':
            processed_image = cv2.Laplacian(self.image, cv2.CV_64F)
        elif method == 'Canny':
            processed_image = cv2.Canny(self.image, 100, 200)

        processed_image = np.uint8(np.abs(processed_image))
        self.show_image(processed_image, self.resultLabel)

class LocalEnhancementApp(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Local Enhancement Method')
        self.setGeometry(100, 100, 1200, 700)

        # 設置字體
        self.setFont(QFont('微軟正黑體', 12, QFont.Bold))

        # Buttons and controls
        self.loadButton = QPushButton('Load Image')
        self.loadButton.clicked.connect(self.load_image)

        self.neighborhoodSizeLabel = QLabel('Neighborhood Size (Sxy):')
        self.neighborhoodSizeSpinBox = QSpinBox()
        self.neighborhoodSizeSpinBox.setRange(1, 20)
        self.neighborhoodSizeSpinBox.setValue(3)

        self.applyLocalButton = QPushButton('Apply Local Enhancement')
        self.applyLocalButton.clicked.connect(self.apply_local_enhancement)

        self.applyHistButton = QPushButton('Apply Histogram Equalization')
        self.applyHistButton.clicked.connect(self.apply_histogram_equalization)

        self.imageLabel = QLabel()
        self.resultLabel = QLabel()

        layout = QVBoxLayout()
        layout.addWidget(self.loadButton)
        layout.addWidget(self.neighborhoodSizeLabel)
        layout.addWidget(self.neighborhoodSizeSpinBox)
        layout.addWidget(self.applyLocalButton)
        layout.addWidget(self.applyHistButton)
        layout.addWidget(QLabel('Original Image:'))
        layout.addWidget(self.imageLabel)
        layout.addWidget(QLabel('Processed Image:'))
        layout.addWidget(self.resultLabel)

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
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio))

    def apply_local_enhancement(self):
        if self.image is None:
            return

        neighborhood_size = self.neighborhoodSizeSpinBox.value()
        result_image = self.local_enhancement(self.image, neighborhood_size)
        self.show_image(result_image, self.resultLabel)

    def apply_histogram_equalization(self):
        if self.image is None:
            return

        result_image = cv2.equalizeHist(self.image)
        self.show_image(result_image, self.resultLabel)

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

# 主視窗使用分頁選項 (Main Window Tab Setup)
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Processing Application')
        self.setGeometry(100, 100, 1200, 800)

        # 創建分頁
        self.tabs = QTabWidget()

        # 第一個分頁：Spatial Filtering
        self.tab1 = SpatialFilteringApp()
        self.tabs.addTab(self.tab1, "Part 2: Spatial Filtering")

        # 第二個分頁：Edge Detection
        self.tab2 = EdgeDetectionApp()
        self.tabs.addTab(self.tab2, "Part 3: Edge Detection")

        # 第三個分頁：Local Enhancement
        self.tab3 = LocalEnhancementApp()
        self.tabs.addTab(self.tab3, "Part 4: Local Enhancement")

        self.setCentralWidget(self.tabs)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())