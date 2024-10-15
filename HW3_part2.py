import sys
import cv2
import numpy as np
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QGridLayout, QWidget, QPushButton,
                             QLineEdit, QFileDialog, QMessageBox, QSpinBox, QComboBox, QHBoxLayout, QScrollArea)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt

class SpatialFilteringApp(QMainWindow):
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
        # 置中對齊
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

        self.setCentralWidget(scrollArea)

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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SpatialFilteringApp()
    window.show()
    sys.exit(app.exec_())
