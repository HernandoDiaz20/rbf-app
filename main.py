import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QFrame, QStackedWidget,
                             QLabel, QSizePolicy)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

# Importar módulos
try:
    from modules.data_loader import DataLoaderModule
    from modules.config_module import ConfigModule
    from modules.train_module import TrainingModule
    from modules.simulation_module import SimulationModule
    from modules.plots_module import PlotsModule
    MODULES_LOADED = True
except ImportError as e:
    print(f"Error importando módulos: {e}")
    print("Creando módulos básicos...")
    MODULES_LOADED = False

# Crear módulos básicos si falla la importación
class BasicModule(QWidget):
    def __init__(self, main_window, title, icon):
        super().__init__()
        self.main_window = main_window
        layout = QVBoxLayout(self)
        
        # Título del módulo
        title_label = QLabel(f"{icon} {title}")
        title_label.setObjectName("title_label")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Contenido placeholder
        content_label = QLabel(f"Módulo {title} - En desarrollo")
        content_label.setAlignment(Qt.AlignCenter)
        content_label.setStyleSheet("color: #6c757d; font-size: 16px; margin: 50px;")
        layout.addWidget(content_label)

if not MODULES_LOADED:
    class DataLoaderModule(BasicModule):
        def __init__(self, main_window):
            super().__init__(main_window, "CARGA DE DATOS", "📊")
    
    class ConfigModule(BasicModule):
        def __init__(self, main_window):
            super().__init__(main_window, "CONFIGURACIÓN RBF", "⚙️")
    
    class TrainingModule(BasicModule):
        def __init__(self, main_window):
            super().__init__(main_window, "ENTRENAMIENTO", "🧠")
    
    class SimulationModule(BasicModule):
        def __init__(self, main_window):
            super().__init__(main_window, "SIMULACIÓN", "🔮")
    
    class PlotsModule(BasicModule):
        def __init__(self, main_window):
            super().__init__(main_window, "GRÁFICAS", "📈")

class RBFMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Red Neuronal RBF - Entrenamiento y Simulación")
        self.setMinimumSize(1200, 800)
        
        # Variables globales para compartir datos
        self.dataset = None
        self.preprocessed_data = None
        self.rbf_config = None
        self.training_results = None
        self.simulation_results = None
        
        self.setup_ui()
        self.load_styles()
        
        # Mostrar ventana maximizada
        self.showMaximized()
    
    def setup_ui(self):
        """Configurar la interfaz principal"""
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Crear barra de navegación
        self.create_navigation_bar(main_layout)
        
        # Crear área de contenido
        self.create_content_area(main_layout)
    
    def create_navigation_bar(self, parent_layout):
        """Crear barra de navegación superior"""
        nav_frame = QFrame()
        nav_frame.setObjectName("nav_frame")
        nav_frame.setFixedHeight(80)
        nav_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        # Layout horizontal para los botones
        nav_layout = QHBoxLayout(nav_frame)
        nav_layout.setContentsMargins(20, 10, 20, 10)
        nav_layout.setSpacing(5)
        
        # Botones de navegación
        self.nav_buttons = {}
        modules = [
            ("Carga de Datos", "📊"),
            ("Configuración RBF", "⚙️"),
            ("Entrenamiento", "🧠"),
            ("Simulación", "🔮"),
            ("Gráficas", "📈")
        ]
        
        for module_name, icon in modules:
            btn = QPushButton(f"{icon} {module_name}")
            btn.setObjectName("nav_button")
            btn.setCheckable(True)
            btn.setFixedHeight(50)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.clicked.connect(lambda checked, name=module_name: self.switch_module(name))
            
            # Fuente más grande para los botones
            font = QFont()
            font.setPointSize(11)
            font.setBold(True)
            btn.setFont(font)
            
            nav_layout.addWidget(btn)
            self.nav_buttons[module_name] = btn
        
        # Marcar primer botón como activo
        self.nav_buttons["Carga de Datos"].setChecked(True)
        
        parent_layout.addWidget(nav_frame)
    
    def create_content_area(self, parent_layout):
        """Crear área de contenido para los módulos"""
        content_frame = QFrame()
        content_frame.setObjectName("content_frame")
        
        content_layout = QVBoxLayout(content_frame)
        content_layout.setContentsMargins(20, 20, 20, 20)
        content_layout.setSpacing(15)
        
        # Widget apilado para módulos
        self.stacked_widget = QStackedWidget()
        
        # Inicializar módulos
        self.modules = {
            "Carga de Datos": DataLoaderModule(self),
            "Configuración RBF": ConfigModule(self),
            "Entrenamiento": TrainingModule(self),
            "Simulación": SimulationModule(self),
            "Gráficas": PlotsModule(self)
        }
        
        # Agregar módulos al stacked widget
        for name, module in self.modules.items():
            self.stacked_widget.addWidget(module)
        
        content_layout.addWidget(self.stacked_widget)
        parent_layout.addWidget(content_frame)
    
    def switch_module(self, module_name):
        """Cambiar entre módulos"""
        # Desmarcar todos los botones
        for btn in self.nav_buttons.values():
            btn.setChecked(False)
        
        # Marcar botón actual
        self.nav_buttons[module_name].setChecked(True)
        
        # Cambiar módulo
        module_index = list(self.modules.keys()).index(module_name)
        self.stacked_widget.setCurrentIndex(module_index)
        
        # Actualizar módulo si es necesario
        current_module = self.modules[module_name]
        if hasattr(current_module, 'on_activate'):
            current_module.on_activate()
    
    def load_styles(self):
        """Cargar estilos desde archivo QSS con forzado para alertas"""
        try:
            # Intentar cargar desde archivo
            style_path = os.path.join('styles', 'styles.qss')
            if os.path.exists(style_path):
                with open(style_path, 'r', encoding='utf-8') as f:
                    stylesheet = f.read()
            else:
                # Si no existe el archivo, usar estilos embebidos
                stylesheet = self.get_embedded_styles()
            
            # Aplicar estilos SOLO a la ventana principal (no a QApplication)
            self.setStyleSheet(stylesheet)
            
        except Exception as e:
            print(f"Error cargando estilos: {e}")
            # Estilos de emergencia
            emergency_styles = self.get_emergency_styles()
            self.setStyleSheet(emergency_styles)
    
    def get_embedded_styles(self):
        """Estilos embebidos como fallback"""
        return """
        /* ===== ESTILOS EMBEBIDOS RBF APP ===== */
        
        QMainWindow {
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                        stop: 0 #f8f9fa, stop: 1 #e9ecef);
        }
        
        QWidget {
            background: transparent;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        
        /* Barra de Navegación */
        QFrame#nav_frame {
            background: #ffffff;
            border-bottom: 2px solid #dee2e6;
            border-radius: 0px;
            padding: 5px;
        }
        
        QPushButton#nav_button {
            background: #ffffff;
            border: 2px solid #e9ecef;
            border-radius: 10px;
            color: #495057;
            font-weight: 600;
            font-size: 12px;
            padding: 15px 10px;
            margin: 2px;
        }
        
        QPushButton#nav_button:hover {
            background: #f8f9fa;
            border: 2px solid #007bff;
            color: #007bff;
        }
        
        QPushButton#nav_button:pressed {
            background: #007bff;
            color: #ffffff;
        }
        
        QPushButton#nav_button:checked {
            background: #007bff;
            border: 2px solid #0056b3;
            color: #ffffff;
        }
        
        /* Frames de Contenido */
        QFrame#content_frame {
            background: #ffffff;
            border: 1px solid #dee2e6;
            border-radius: 12px;
            margin: 15px;
            padding: 5px;
        }
        
        QLabel#title_label {
            color: #212529;
            font-size: 24px;
            font-weight: 700;
            background: transparent;
            padding: 15px;
            border-bottom: 2px solid #e9ecef;
        }
        
        /* ===== ESTILOS NUCLEARES PARA QMessageBox ===== */
        QMessageBox {
            background-color: white !important;
            color: #333333 !important;
            border: 2px solid #007bff !important;
            border-radius: 10px !important;
        }
        
        QMessageBox QLabel {
            background-color: white !important;
            color: #333333 !important;
            font-size: 14px !important;
            font-weight: normal !important;
            padding: 15px !important;
        }
        
        QMessageBox QPushButton {
            background-color: #007bff !important;
            color: white !important;
            border: none !important;
            border-radius: 5px !important;
            padding: 8px 15px !important;
            font-weight: bold !important;
            min-width: 80px !important;
            min-height: 35px !important;
            margin: 5px !important;
        }
        
        QMessageBox QPushButton:hover {
            background-color: #0056b3 !important;
        }
        
        QMessageBox QPushButton:pressed {
            background-color: #004085 !important;
        }
        
        /* Botón por defecto */
        QMessageBox QPushButton[default="true"] {
            background-color: #28a745 !important;
        }
        
        QMessageBox QPushButton[default="true"]:hover {
            background-color: #218838 !important;
        }
        
        /* Reset total para asegurar visibilidad */
        * {
            color: #333333 !important;
            background-color: white !important;
        }
        
        QLabel {
            color: #333333 !important;
            background: transparent !important;
        }
        """
    
    def get_emergency_styles(self):
        """Estilos de emergencia mínimos"""
        return """
        /* Estilos de emergencia para alertas */
        QMessageBox {
            background-color: white;
            color: black;
        }
        
        QMessageBox QLabel {
            background-color: white;
            color: black;
            font-size: 14px;
        }
        
        QMessageBox QPushButton {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            min-width: 80px;
        }
        
        QMessageBox QPushButton:hover {
            background-color: #0056b3;
        }
        
        /* Reset básico */
        * {
            color: black !important;
            background-color: white !important;
        }
        """

def main():
    app = QApplication(sys.argv)
    
    # Configurar aplicación
    app.setApplicationName("RBF Neural Network")
    app.setApplicationVersion("1.0")
    
    # Forzar estilos fusionados para mejor compatibilidad
    app.setStyle("Fusion")
    
    # Crear y mostrar ventana principal
    window = RBFMainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()