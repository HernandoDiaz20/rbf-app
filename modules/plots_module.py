from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel

class PlotsModule(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        title = QLabel("📈 GRÁFICAS")
        layout.addWidget(title)