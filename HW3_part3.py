import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QFileDialog, QSlider, QWidget, \
    QComboBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage


class EdgeDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Marr-Hildreth Edge Detection')
        self.setGeometry(100, 100, 800, 600)

        # Buttons and controls
        self.loadButton = QPushButton('Load Image')
        self.loadButton.clicked.connect(self.load_image)

        self.filterSelect = QComboBox()
        self.filterSelect.addItems(['Marr-Hildreth', 'Sobel'])

        self.sigmaSlider = QSlider(Qt.Horizontal)
        self.sigmaSlider.setRange(1, 10)
        self.sigmaSlider.setValue(2)
        self.sigmaSlider.valueChanged.connect(self.update_sigma)

        self.thresholdSlider = QSlider(Qt.Horizontal)
        self.thresholdSlider.setRange(1, 100)
        self.thresholdSlider.setValue(10)
        self.thresholdSlider.valueChanged.connect(self.update_threshold)

        self.applyButton = QPushButton('Apply Filter')
        self.applyButton.clicked.connect(self.apply_filter)

        self.imageLabel = QLabel()
        self.resultLabel = QLabel()

        layout = QVBoxLayout()
        layout.addWidget(self.loadButton)
        layout.addWidget(QLabel('Select Filter:'))
        layout.addWidget(self.filterSelect)
        layout.addWidget(QLabel('Gaussian Sigma:'))
        layout.addWidget(self.sigmaSlider)
        layout.addWidget(QLabel('Zero-crossing Threshold:'))
        layout.addWidget(self.thresholdSlider)
        layout.addWidget(self.applyButton)
        layout.addWidget(QLabel('Original Image:'))
        layout.addWidget(self.imageLabel)
        layout.addWidget(QLabel('Processed Image:'))
        layout.addWidget(self.resultLabel)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_image(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp);;All Files (*)",
                                                  options=options)
        if fileName:
            self.image = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
            self.show_image(self.image, self.imageLabel)

    def show_image(self, image, label):
        height, width = image.shape
        bytesPerLine = width
        qimage = QImage(image.data, width, height, bytesPerLine, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        label.setPixmap(pixmap.scaled(256, 256, Qt.KeepAspectRatio))

    def update_sigma(self):
        self.sigma = self.sigmaSlider.value()

    def update_threshold(self):
        self.threshold = self.thresholdSlider.value()

    def apply_filter(self):
        if self.image is None:
            return

        selected_filter = self.filterSelect.currentText()
        if selected_filter == 'Marr-Hildreth':
            result_image = self.marr_hildreth(self.image, self.sigmaSlider.value(), self.thresholdSlider.value())
        elif selected_filter == 'Sobel':
            result_image = self.sobel_operator(self.image)

        self.show_image(result_image, self.resultLabel)

    def marr_hildreth(self, image, sigma, threshold):
        # Step 1: Smooth the image with a Gaussian filter
        smoothed_image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)

        # Step 2: Apply Laplacian to the smoothed image
        laplacian = cv2.Laplacian(smoothed_image, cv2.CV_64F)

        # Step 3: Zero-crossing detection
        zero_crossing = np.zeros_like(laplacian)
        laplacian = laplacian.astype(np.float32)
        for y in range(1, laplacian.shape[0] - 1):
            for x in range(1, laplacian.shape[1] - 1):
                region = laplacian[y - 1:y + 2, x - 1:x + 2]
                if np.min(region) < 0 and np.max(region) > 0 and (np.max(region) - np.min(region)) > threshold:
                    zero_crossing[y, x] = 255

        return zero_crossing.astype(np.uint8)

    def sobel_operator(self, image):
        # Apply Sobel operator
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        return np.uint8(np.clip(sobel_magnitude, 0, 255))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = EdgeDetectionApp()
    window.show()
    sys.exit(app.exec_())
