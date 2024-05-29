from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)

        self.nx_label = QLabel("Grid Size (nx):")
        self.nx_input = QLineEdit(str(parent.nx))

        self.ny_label = QLabel("Grid Size (ny):")
        self.ny_input = QLineEdit(str(parent.ny))

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.nx_label)
        self.layout.addWidget(self.nx_input)
        self.layout.addWidget(self.ny_label)
        self.layout.addWidget(self.ny_input)

        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_settings)
        self.layout.addWidget(self.save_button)

        self.setLayout(self.layout)

    def save_settings(self):
        try:
            self.parent().nx = int(self.nx_input.text())
            self.parent().ny = int(self.ny_input.text())
            self.accept()
        except ValueError:
            print("Invalid input. Please enter numerical values.")

