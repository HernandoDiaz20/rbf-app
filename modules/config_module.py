import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QGroupBox, QSpinBox, QDoubleSpinBox,
                             QComboBox, QTextEdit, QFrame, QMessageBox,
                             QTableWidget, QTableWidgetItem, QHeaderView,
                             QScrollArea, QProgressBar)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
import random

class ConfigModule(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.rbf_config = None
        self.centers = None
        self.setup_ui()
    
    def setup_ui(self):
        # Layout principal con scroll
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Área de scroll para contenido principal
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Widget contenedor para el área de scroll
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(15)
        
        # Título
        title = QLabel("⚙️ CONFIGURACIÓN RBF")
        title.setObjectName("title_label")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel#title_label {
                color: #2c3e50;
                font-size: 24px;
                font-weight: bold;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                          stop:0 #3498db, stop:1 #2c3e50);
                color: white;
                border-radius: 10px;
                padding: 15px;
                margin: 5px;
            }
        """)
        scroll_layout.addWidget(title)
        
        # Crear secciones
        self.create_parameters_section(scroll_layout)
        self.create_centers_section(scroll_layout)
        self.create_actions_section(scroll_layout)
        self.create_info_section(scroll_layout)
        
        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area)
        
        # Inicializar configuración por defecto
        self.initialize_default_config()
    
    def create_parameters_section(self, parent_layout):
        """Crear sección de parámetros configurables"""
        params_group = QGroupBox("🔧 Parámetros de la Red RBF")
        params_layout = QVBoxLayout(params_group)
        
        # Número de centros radiales
        centers_frame = QFrame()
        centers_layout = QHBoxLayout(centers_frame)
        centers_layout.addWidget(QLabel("Número de Centros Radiales (Neuronas Ocultas):"))
        
        self.centers_spin = QSpinBox()
        self.centers_spin.setRange(1, 50)  # Rango de 1 a 50 centros
        self.centers_spin.setValue(2)      # Valor por defecto: 2
        self.centers_spin.setSuffix(" centros")
        self.centers_spin.valueChanged.connect(self.on_parameters_changed)
        
        centers_layout.addWidget(self.centers_spin)
        centers_layout.addStretch()
        params_layout.addWidget(centers_frame)
        
        # Error de aproximación óptimo
        error_frame = QFrame()
        error_layout = QHBoxLayout(error_frame)
        error_layout.addWidget(QLabel("Error de Aproximación Óptimo:"))
        
        self.error_spin = QDoubleSpinBox()
        self.error_spin.setRange(0.0, 1.0)  # Rango de 0.0 a 1.0
        self.error_spin.setValue(0.1)       # Valor por defecto: 0.1
        self.error_spin.setDecimals(3)      # Tres decimales para mayor precisión
        self.error_spin.setSingleStep(0.01)
        self.error_spin.valueChanged.connect(self.on_parameters_changed)
        
        error_layout.addWidget(self.error_spin)
        
        # Etiqueta informativa del rango
        range_label = QLabel("(Rango: 0 - 1.0)")
        range_label.setStyleSheet("color: #7f8c8d; font-size: 11px;")
        error_layout.addWidget(range_label)
        
        error_layout.addStretch()
        params_layout.addWidget(error_frame)
        
        # Información de la configuración actual
        self.config_info = QLabel("Configuración actual: 2 centros, FA(d)=d²*ln(d), Error=0.1")
        self.config_info.setStyleSheet("color: #2c3e50; font-weight: bold; padding: 5px;")
        params_layout.addWidget(self.config_info)
        
        parent_layout.addWidget(params_group)
    
    def create_centers_section(self, parent_layout):
        """Crear sección para visualización de centros radiales"""
        centers_group = QGroupBox("🎯 Centros Radiales Inicializados")
        centers_layout = QVBoxLayout(centers_group)
        
        # Información de centros
        self.centers_info = QLabel("Centros radiales: No inicializados")
        self.centers_info.setStyleSheet("color: #7f8c8d; font-weight: bold;")
        centers_layout.addWidget(self.centers_info)
        
        # Información del método
        method_info = QLabel("Método: Aleatorio dentro del rango real de las entradas")
        method_info.setStyleSheet("color: #3498db; font-weight: bold; font-size: 12px;")
        centers_layout.addWidget(method_info)
        
        # Tabla para mostrar centros
        self.centers_table = QTableWidget()
        self.centers_table.setAlternatingRowColors(True)
        self.centers_table.setMinimumHeight(200)
        centers_layout.addWidget(self.centers_table)
        
        parent_layout.addWidget(centers_group)
    
    def create_actions_section(self, parent_layout):
        """Crear sección de acciones"""
        actions_group = QGroupBox("🚀 Acciones de Configuración")
        actions_layout = QVBoxLayout(actions_group)
        
        # Botones de acción
        buttons_frame = QFrame()
        buttons_layout = QHBoxLayout(buttons_frame)
        
        btn_initialize = QPushButton("🎲 Inicializar Centros Radiales")
        btn_initialize.setMinimumHeight(40)
        btn_initialize.clicked.connect(self.initialize_centers)
        
        btn_save = QPushButton("💾 Guardar Configuración")
        btn_save.setMinimumHeight(40)
        btn_save.clicked.connect(self.save_configuration)
        
        btn_reset = QPushButton("🔄 Restablecer Valores")
        btn_reset.setMinimumHeight(40)
        btn_reset.clicked.connect(self.reset_configuration)
        
        buttons_layout.addWidget(btn_initialize)
        buttons_layout.addWidget(btn_save)
        buttons_layout.addWidget(btn_reset)
        
        actions_layout.addWidget(buttons_frame)
        
        # Estado de la configuración
        self.status_label = QLabel("⏳ Esperando inicialización de centros...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("padding: 10px; font-weight: bold;")
        actions_layout.addWidget(self.status_label)
        
        parent_layout.addWidget(actions_group)
    
    def create_info_section(self, parent_layout):
        """Crear sección de información"""
        info_group = QGroupBox("📚 Información de la Configuración RBF")
        info_layout = QVBoxLayout(info_group)
        
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setHtml("""
        <h3>Configuración de la Red RBF</h3>
        
        <h4>Número de Centros Radiales (Neuronas Ocultas):</h4>
        <ul>
            <li>Define la cantidad de neuronas en la capa oculta</li>
            <li>Cada centro radial representa un prototipo en el espacio de entrada</li>
            <li>Valor típico: 2 o más según la complejidad del problema</li>
            <li><b>Más centros = Mayor capacidad de aproximación</b></li>
        </ul>
        
        <h4>Función de Activación Radial (FIJA):</h4>
        <ul>
            <li><b>FA(d) = d² * ln(d)</b>: Función de base radial logarítmica</li>
            <li>Esta función está predefinida y no puede ser modificada</li>
            <li>Para d &lt; 1, ln(d) es negativo → activaciones negativas</li>
            <li>Para d &gt; 1, ln(d) es positivo → activaciones positivas</li>
        </ul>
        
        <h4>Error de Aproximación Óptimo:</h4>
        <ul>
            <li>Valor objetivo para el error de entrenamiento</li>
            <li>Valor por defecto: 0.1</li>
            <li>Rango permitido: 0.0 a 1.0</li>
            <li>Controla la precisión deseada de la red</li>
            <li><b>Valores más pequeños = Mayor precisión requerida</b></li>
        </ul>
        
        <h4>Inicialización de Centros Radiales (AUTOMÁTICA):</h4>
        <ul>
            <li><b>Método:</b> Aleatorio dentro del rango real de las entradas</li>
            <li>Los centros se inicializan aleatoriamente dentro del rango [min(X), max(X)]</li>
            <li>Cada centro tiene la misma dimensionalidad que las ENTRADAS</li>
            <li>La cantidad de valores por centro = número de ENTRADAS del dataset</li>
            <li><b>¡Se mantienen los valores originales del dataset!</b></li>
        </ul>
        
        <p><b>Nota:</b> Solo se realiza una iteración por defecto en el entrenamiento.</p>
        """)
        
        info_layout.addWidget(info_text)
        parent_layout.addWidget(info_group)
    
    def initialize_default_config(self):
        """Inicializar configuración por defecto"""
        self.rbf_config = {
            'num_centers': 2,
            'activation_function': 'd2_lnd',
            'activation_name': 'FA(d) = d² * ln(d)',
            'target_error': 0.1,
            'center_method': 'random',
            'centers': None,
            'iterations': 1
        }
        self.update_config_info()
    
    def on_parameters_changed(self):
        """Manejar cambios en los parámetros"""
        self.update_config_info()
    
    def update_config_info(self):
        """Actualizar información de configuración"""
        num_centers = self.centers_spin.value()
        error = self.error_spin.value()
        
        info_text = f"Configuración actual: {num_centers} centros, FA(d)=d²*ln(d), Error={error:.3f}"
        self.config_info.setText(info_text)
    
    def get_num_inputs_from_data(self):
        """Obtener el número real de entradas desde los datos preprocesados"""
        if self.main_window.preprocessed_data is None:
            return 0
        
        # Obtener el número de entradas desde input_columns del data_loader
        input_columns = self.main_window.preprocessed_data.get('input_columns', [])
        num_inputs = len(input_columns)
        
        print(f"=== DEBUG CONFIG - INFORMACIÓN DE ENTRADAS ===")
        print(f"Número de entradas desde input_columns: {num_inputs}")
        print(f"Columnas de entrada: {input_columns}")
        
        # VERIFICACIÓN CRÍTICA: También verificar la forma de X_train
        X_train = self.main_window.preprocessed_data.get('X_train')
        if X_train is not None:
            print(f"Forma de X_train: {X_train.shape}")
            print(f"Número de columnas en X_train: {X_train.shape[1]}")
            print(f"Rango REAL de X_train: [{np.min(X_train):.6f}, {np.max(X_train):.6f}]")
            
            # Si hay discrepancia, mostrar advertencia
            if X_train.shape[1] != num_inputs:
                print(f"¡ADVERTENCIA! Discrepancia detectada:")
                print(f"  - input_columns tiene {num_inputs} entradas")
                print(f"  - X_train tiene {X_train.shape[1]} columnas")
        
        return num_inputs
    
    def initialize_centers(self):
        """Inicializar centros radiales aleatoriamente dentro del rango real de CADA ENTRADA - SEGÚN EJEMPLO"""
        if self.main_window.preprocessed_data is None:
            QMessageBox.warning(self, "Advertencia", 
                            "Primero cargue y preprocese los datos en el módulo 'Carga de Datos'")
            return
        
        try:
            # Obtener número real de ENTRADAS desde input_columns
            num_inputs = self.get_num_inputs_from_data()
            
            if num_inputs == 0:
                QMessageBox.warning(self, "Advertencia", 
                                "No se pudieron determinar las entradas de los datos")
                return
            
            num_centers = self.centers_spin.value()
            
            # Obtener datos de entrenamiento (X_train ya contiene solo entradas)
            X_train = self.main_window.preprocessed_data['X_train']
            
            print(f"=== DEBUG CONFIG - INICIALIZACIÓN CENTROS ===")
            print(f"Numero de centros: {num_centers}")
            print(f"Numero de entradas desde input_columns: {num_inputs}")
            print(f"Forma de X_train: {X_train.shape}")
            print(f"Rango REAL de X_train: [{np.min(X_train):.6f}, {np.max(X_train):.6f}]")
            
            # VERIFICACIÓN CRÍTICA: Si hay discrepancia, usar el número REAL de X_train
            if X_train.shape[1] != num_inputs:
                print(f"¡CORRECCIÓN AUTOMÁTICA! Usando número real de columnas de X_train")
                print(f"Anterior: {num_inputs} entradas")
                num_inputs = X_train.shape[1]
                print(f"Actual: {num_inputs} entradas")
            
            # CONVERSIÓN ROBUSTA DE DATOS - Asegurar que sean numéricos
            if hasattr(X_train, 'dtype') and X_train.dtype == object:
                try:
                    X_train = X_train.astype(np.float64)
                except:
                    import pandas as pd
                    X_train = pd.DataFrame(X_train).apply(pd.to_numeric, errors='coerce').values
            
            # Verificar que los datos son numéricos
            if not np.issubdtype(X_train.dtype, np.number):
                QMessageBox.warning(self, "Advertencia", 
                                "Los datos de entrenamiento deben ser numéricos")
                return
            
            # Verificar la forma de los datos
            if len(X_train.shape) != 2:
                QMessageBox.warning(self, "Advertencia", 
                                "Los datos de entrenamiento no tienen la forma correcta")
                return
            
            # ✅ CORRECCIÓN SEGÚN EJEMPLO: Obtener valores mínimos y máximos de CADA ENTRADA individualmente
            min_vals = np.min(X_train, axis=0)  # Mínimo por columna (por entrada)
            max_vals = np.max(X_train, axis=0)  # Máximo por columna (por entrada)
            
            print(f"=== RANGOS POR ENTRADA ===")
            for i in range(num_inputs):
                print(f"Entrada {i+1}: [{min_vals[i]:.6f}, {max_vals[i]:.6f}]")
            
            print(f"Forma final usada para centros: ({num_centers}, {num_inputs})")
            
            # ✅ CORRECCIÓN DEFINITIVA SEGÚN EJEMPLO: Inicializar centros aleatoriamente dentro del rango [min(X_i), max(X_i)] para CADA entrada
            self.centers = np.zeros((num_centers, num_inputs))
            
            for j in range(num_inputs):  # Para cada entrada/columna
                min_val = min_vals[j]
                max_val = max_vals[j]
                
                # Generar valores aleatorios para esta entrada específica dentro de su rango real
                random_values = np.random.uniform(
                    low=min_val, 
                    high=max_val, 
                    size=num_centers
                )
                
                self.centers[:, j] = random_values
                
                print(f"Entrada {j+1}: Centros entre {np.min(random_values):.6f} y {np.max(random_values):.6f} (Datos: [{min_val:.6f}, {max_val:.6f}])")
            
            # Aplicar más precisión a los valores (10 decimales)
            self.centers = np.round(self.centers, 10)
            
            # Actualizar interfaz
            self.display_centers_table()
            self.update_centers_info(len(self.centers), num_inputs)
            
            self.status_label.setText("✅ Centros radiales inicializados correctamente")
            self.status_label.setStyleSheet("color: #27ae60; font-weight: bold; padding: 10px;")
            
            print(f"✅ Centros inicializados exitosamente: {self.centers.shape}")
            print(f"✅ Cada centro tiene {num_inputs} valores (entradas)")
            print(f"✅ Rango de centros: [{np.min(self.centers):.6f}, {np.max(self.centers):.6f}]")
            
            # VERIFICACIÓN FINAL
            print(f"=== VERIFICACIÓN FINAL DE RANGOS ===")
            all_within_range = True
            for j in range(num_inputs):
                center_min = np.min(self.centers[:, j])
                center_max = np.max(self.centers[:, j])
                data_min = min_vals[j]
                data_max = max_vals[j]
                
                if center_min >= data_min and center_max <= data_max:
                    print(f"   ✅ Entrada {j+1}: Rango CORRECTO - Centros [{center_min:.6f}, {center_max:.6f}] en Datos [{data_min:.6f}, {data_max:.6f}]")
                else:
                    print(f"   ❌ Entrada {j+1}: Rango INCORRECTO - Centros [{center_min:.6f}, {center_max:.6f}] debería estar en [{data_min:.6f}, {data_max:.6f}]")
                    all_within_range = False
            
            if all_within_range:
                print("🎉 TODOS los centros están dentro de los rangos correctos de los datos")
            else:
                print("⚠️ ALGUNOS centros están fuera de los rangos esperados")
            
        except Exception as e:
            error_msg = f"❌ Error al inicializar centros: {str(e)}"
            self.status_label.setText(error_msg)
            self.status_label.setStyleSheet("color: #c0392b; font-weight: bold; padding: 10px;")
            QMessageBox.critical(self, "Error", f"No se pudieron inicializar los centros:\n{str(e)}")
            print(f"Error detallado: {e}")
            import traceback
            traceback.print_exc()
    
    def display_centers_table(self):
        """Mostrar centros radiales en la tabla - SIMPLIFICADO"""
        if self.centers is None:
            return
        
        num_centers, num_inputs = self.centers.shape
        
        # Configurar tabla con EXACTAMENTE num_inputs columnas
        self.centers_table.setRowCount(num_centers)
        self.centers_table.setColumnCount(num_inputs + 1)  # +1 para la columna "Centro #"
        
        # Encabezados simples como antes
        headers = ["Centro #"]
        headers.extend([f"Entrada {i+1}" for i in range(num_inputs)])
        
        self.centers_table.setHorizontalHeaderLabels(headers)
        
        # Llenar datos con precisión (10 decimales)
        for i in range(num_centers):
            # Número de centro
            center_item = QTableWidgetItem(f"R{i+1}")
            center_item.setTextAlignment(Qt.AlignCenter)
            self.centers_table.setItem(i, 0, center_item)
            
            # Valores del centro con 10 decimales de precisión
            for j in range(num_inputs):
                value = self.centers[i, j]
                item = QTableWidgetItem(f"{value:.10f}")
                item.setTextAlignment(Qt.AlignCenter)
                self.centers_table.setItem(i, j + 1, item)
        
        # Ajustar tamaño de columnas
        self.centers_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        
        print(f"✅ Tabla de centros configurada: {num_centers} centros x {num_inputs} entradas")
    
    def update_centers_info(self, num_centers, num_inputs):
        """Actualizar información de centros"""
        range_info = ""
        if self.centers is not None:
            center_min = np.min(self.centers)
            center_max = np.max(self.centers)
            range_info = f" - Rango: [{center_min:.4f}, {center_max:.4f}]"
        
        info_text = f"Centros radiales: {num_centers} centros, {num_inputs} entradas cada uno{range_info}"
        self.centers_info.setText(info_text)
        print(f"✅ Información actualizada: {info_text}")
    
    def save_configuration(self):
        """Guardar configuración RBF"""
        if self.centers is None:
            QMessageBox.warning(self, "Advertencia", 
                              "Primero inicialice los centros radiales")
            return
        
        try:
            # Crear configuración completa
            num_inputs = self.centers.shape[1] if self.centers is not None else 0
            
            self.rbf_config = {
                'num_centers': self.centers_spin.value(),
                'num_inputs': num_inputs,
                'activation_function': 'd2_lnd',
                'activation_name': 'FA(d) = d² * ln(d)',
                'target_error': self.error_spin.value(),
                'center_method': 'random',
                'centers': self.centers.copy(),
                'iterations': 1
            }
            
            # Guardar en la ventana principal
            self.main_window.rbf_config = self.rbf_config
            
            # Mostrar confirmación
            config_summary = f"""
            ✅ Configuración RBF Guardada:
            
            • Centros radiales: {self.rbf_config['num_centers']}
            • Entradas por centro: {num_inputs}
            • Función de activación: {self.rbf_config['activation_name']}
            • Error objetivo: {self.rbf_config['target_error']:.3f}
            • Método de inicialización: Aleatorio dentro del rango real de las entradas
            • Iteraciones: 1
            • Precisión: 10 decimales
            
            Los centros han sido inicializados aleatoriamente dentro del rango
            real de las {num_inputs} entradas del dataset.
            
            Rango actual de centros: [{np.min(self.centers):.4f}, {np.max(self.centers):.4f}]
            """
            
            QMessageBox.information(self, "Configuración Guardada", config_summary)
            
            self.status_label.setText("💾 Configuración guardada - Lista para entrenamiento")
            self.status_label.setStyleSheet("color: #2980b9; font-weight: bold; padding: 10px;")
            
            print(f"✅ Configuración guardada: {num_inputs} entradas")
            
        except Exception as e:
            error_msg = f"❌ Error al guardar configuración: {str(e)}"
            self.status_label.setText(error_msg)
            self.status_label.setStyleSheet("color: #c0392b; font-weight: bold; padding: 10px;")
            QMessageBox.critical(self, "Error", f"No se pudo guardar la configuración:\n{str(e)}")
    
    def reset_configuration(self):
        """Restablecer configuración a valores por defecto"""
        reply = QMessageBox.question(self, 'Confirmar Reset', 
                                   '¿Está seguro de que desea restablecer la configuración a los valores por defecto?',
                                   QMessageBox.Yes | QMessageBox.No, 
                                   QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.centers_spin.setValue(2)
            self.error_spin.setValue(0.1)
            
            self.centers = None
            self.centers_table.setRowCount(0)
            self.centers_table.setColumnCount(0)
            
            self.centers_info.setText("Centros radiales: No inicializados")
            self.status_label.setText("⏳ Esperando inicialización de centros...")
            self.status_label.setStyleSheet("color: #7f8c8d; font-weight: bold; padding: 10px;")
            
            self.initialize_default_config()
    
    def debug_data_info(self):
        """Función temporal para debug de datos"""
        if self.main_window.preprocessed_data is not None:
            X_train = self.main_window.preprocessed_data['X_train']
            input_columns = self.main_window.preprocessed_data.get('input_columns', [])
            output_columns = self.main_window.preprocessed_data.get('output_columns', [])
            
            print(f"=== DEBUG CONFIG - INFORMACIÓN COMPLETA ===")
            print(f"Forma de X_train: {X_train.shape}")
            print(f"Numero de entradas desde input_columns: {len(input_columns)}")
            print(f"Columnas de entrada: {input_columns}")
            print(f"Columnas de salida: {output_columns}")
            
            # Mostrar rangos reales de los datos
            if X_train is not None and len(X_train) > 0:
                print("=== RANGOS REALES DE LOS DATOS ===")
                for i in range(min(5, X_train.shape[1])):  # Mostrar primeras 5 columnas
                    col_min = np.min(X_train[:, i])
                    col_max = np.max(X_train[:, i])
                    print(f"Entrada {i+1}: [{col_min:.6f}, {col_max:.6f}]")
    
    def on_activate(self):
        """Método llamado cuando el módulo se activa"""
        # Debug temporal
        self.debug_data_info()
        
        # Verificar si hay datos preprocesados disponibles
        if self.main_window.preprocessed_data is not None:
            num_inputs = self.get_num_inputs_from_data()
            self.status_label.setText(f"✅ Datos disponibles - {num_inputs} entradas - Puede inicializar centros")
            self.status_label.setStyleSheet("color: #27ae60; font-weight: bold; padding: 10px;")
        else:
            self.status_label.setText("⚠️ Cargue datos primero en el módulo 'Carga de Datos'")
            self.status_label.setStyleSheet("color: #f39c12; font-weight: bold; padding: 10px;")