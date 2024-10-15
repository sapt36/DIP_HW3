import sys
import cv2
import numpy as np
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QGridLayout, QWidget, QPushButton, QLineEdit, QFileDialog, QMessageBox, QSpinBox, QComboBox, QHBoxLayout)
from PyQt5.QtGui import QImage, QPixmap

class SpatialFilteringApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Spatial Filtering Operations')
        self.setGeometry(100, 100, 800, 600)

        # Load and Mask UI
        self.loadButton = QPushButton('Load Image')
        self.loadButton.clicked.connect(self.load_image)

        self.maskTypeLabel = QLabel('Mask Type:')
        self.maskTypeCombo = QComboBox()
        self.maskTypeCombo.addItems(['Box', 'Gaussian', 'Smoothing', 'Other'])
        self.maskTypeCombo.currentTextChanged.connect(self.update_mask_inputs)

        self.maskSizeLabel = QLabel('Mask Size:')
        self.maskSizeInput = QSpinBox()
        self.maskSizeInput.setRange(3, 9)
        self.maskSizeInput.setSingleStep(2)
        self.maskSizeInput.setValue(3)
        self.maskSizeInput.valueChanged.connect(self.update_mask_inputs)

        self.averagingLabel = QLabel('Averaging Value:')
        self.averagingInput = QLineEdit('1')

        self.applyButton = QPushButton('Apply Mask')
        self.applyButton.clicked.connect(self.apply_mask)

        self.imageLabel = QLabel()
        self.resultLabel = QLabel()

        # Layouts
        topLayout = QHBoxLayout()
        topLayout.addWidget(self.maskTypeLabel)
        topLayout.addWidget(self.maskTypeCombo)
        topLayout.addWidget(self.maskSizeLabel)
        topLayout.addWidget(self.maskSizeInput)

        layout = QVBoxLayout()
        layout.addWidget(self.loadButton)
        layout.addLayout(topLayout)
        layout.addWidget(self.averagingLabel)
        layout.addWidget(self.averagingInput)
        self.maskLayout = QGridLayout()
        layout.addLayout(self.maskLayout)
        layout.addWidget(self.applyButton)
        layout.addWidget(QLabel('Original Image:'))
        layout.addWidget(self.imageLabel)
        layout.addWidget(QLabel('Processed Image:'))
        layout.addWidget(self.resultLabel)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.update_mask_inputs()

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

    def update_mask_inputs(self):
        mask_type = self.maskTypeCombo.currentText()
        self.clear_mask_layout()  # Use function to clear UI components
        if mask_type == 'Other':
            self.averagingLabel.show()
            self.averagingInput.show()
            self.create_mask_inputs()
        else:
            self.averagingLabel.hide()
            self.averagingInput.hide()

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
        if self.image is None:
            QMessageBox.warning(self, 'Warning', 'Please load an image first!')
            return

        mask_size = self.maskSizeInput.value()
        mask_type = self.maskTypeCombo.currentText()

        if mask_type in ['Box', 'Smoothing']:
            mask = np.ones((mask_size, mask_size), np.float32) / (mask_size * mask_size)
        elif mask_type == 'Gaussian':
            mask = cv2.getGaussianKernel(mask_size, 0)
            mask = mask * mask.T
        elif mask_type == 'Other':
            mask = self.get_custom_mask(mask_size)
            if mask is None:
                return

        start_time = time.time()
        filtered_image = cv2.filter2D(self.image, -1, mask)
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
