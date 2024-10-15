import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QFileDialog, QSlider, QWidget, QComboBox, QSpinBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage

class LocalEnhancementApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Local Enhancement Method')
        self.setGeometry(100, 100, 800, 600)

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

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

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
        # Convert the image to float for processing
        image_float = image.astype(np.float32)
        result_image = np.zeros_like(image_float)

        # Padding the image to handle borders
        padded_image = cv2.copyMakeBorder(image_float, sxy//2, sxy//2, sxy//2, sxy//2, cv2.BORDER_REFLECT)

        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                # Define the local neighborhood region
                local_region = padded_image[y:y + sxy, x:x + sxy]

                # Calculate local mean and standard deviation
                local_mean = np.mean(local_region)
                local_std = np.std(local_region)

                # Apply enhancement formula (Example 3.10 concept)
                k = 1.0  # Enhancement constant, can be adjusted as needed
                result_image[y, x] = k * (image_float[y, x] - local_mean) / (local_std + 1e-5) + 128  # Centered around 128

        # Clip the values to the valid range [0, 255] and convert to uint8
        result_image = np.clip(result_image, 0, 255).astype(np.uint8)
        return result_image

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LocalEnhancementApp()
    window.show()
    sys.exit(app.exec_())
