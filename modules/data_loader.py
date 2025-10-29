import os
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QGroupBox, QListWidget, QTreeWidget,
                             QTreeWidgetItem, QTextEdit, QTabWidget, QFrame,
                             QProgressBar, QRadioButton, QButtonGroup, QFileDialog,
                             QMessageBox, QSplitter, QComboBox, QLineEdit,
                             QTableWidget, QTableWidgetItem, QHeaderView,
                             QCheckBox, QApplication, QScrollArea, QGridLayout,
                             QSizePolicy, QSpinBox)
from PyQt5.QtCore import Qt, QTimer, QMimeData, QUrl
from PyQt5.QtGui import QFont, QDragEnterEvent, QDropEvent
import requests
from io import StringIO
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

class DataLoaderModule(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.df = None
        self.preprocessed_data = None
        self.input_columns = []
        self.output_columns = []
        self.test_size = 0.3  # 30% para prueba por defecto
        
        self.setup_ui()
        self.setAcceptDrops(True)
    
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
        title = QLabel("📊 CARGA DE DATOS")
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
        
        # Crear secciones en un grid para mejor distribución
        grid_layout = QGridLayout()
        grid_layout.setSpacing(15)
        
        # Sección de carga (fila 0, columna 0)
        self.create_loading_section(grid_layout, 0, 0)
        
        # Sección de información (fila 0, columna 1)
        self.create_info_section(grid_layout, 0, 1)
        
        # Sección de detección automática (fila 1, columna 0)
        self.create_auto_detection_section(grid_layout, 1, 0)
        
        # Sección de preprocesamiento (fila 1, columna 1)
        self.create_preprocessing_section(grid_layout, 1, 1)
        
        # Sección de vista previa (fila 2, columnas 0-1)
        self.create_preview_section(grid_layout, 2, 0, 1, 2)
        
        scroll_layout.addLayout(grid_layout)
        
        # Sección de log (fuera del grid)
        self.create_log_section(scroll_layout)
        
        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area)
    
    def create_loading_section(self, parent_layout, row, col):
        """Crear sección de carga de datos"""
        loading_group = QGroupBox("📁 Carga de Dataset")
        loading_layout = QVBoxLayout(loading_group)
        
        # Métodos de carga local
        local_group = QGroupBox("Carga Local")
        local_layout = QVBoxLayout(local_group)
        
        btn_frame = QFrame()
        btn_layout = QHBoxLayout(btn_frame)
        
        btn_csv = QPushButton("📄 Cargar CSV")
        btn_csv.setMinimumHeight(35)
        btn_csv.clicked.connect(lambda: self.load_file('csv'))
        
        btn_json = QPushButton("📊 Cargar JSON")
        btn_json.setMinimumHeight(35)
        btn_json.clicked.connect(lambda: self.load_file('json'))
        
        btn_drag = QPushButton("⬆️ Arrastrar Archivo")
        btn_drag.setMinimumHeight(35)
        btn_drag.clicked.connect(self.enable_drag_drop_help)
        
        btn_layout.addWidget(btn_csv)
        btn_layout.addWidget(btn_json)
        btn_layout.addWidget(btn_drag)
        local_layout.addWidget(btn_frame)
        
        # Carga desde URL
        url_group = QGroupBox("Carga desde URL")
        url_layout = QVBoxLayout(url_group)
        
        self.url_entry = QLineEdit()
        self.url_entry.setPlaceholderText("https://ejemplo.com/dataset.csv")
        self.url_entry.setMinimumHeight(30)
        
        btn_url_frame = QFrame()
        btn_url_layout = QHBoxLayout(btn_url_frame)
        btn_url = QPushButton("🌐 Descargar desde URL")
        btn_url.setMinimumHeight(35)
        btn_url.clicked.connect(self.load_from_url)
        btn_url_layout.addWidget(btn_url)
        
        url_layout.addWidget(QLabel("URL del dataset:"))
        url_layout.addWidget(self.url_entry)
        url_layout.addWidget(btn_url_frame)
        
        # Botón para resetear
        reset_frame = QFrame()
        reset_layout = QHBoxLayout(reset_frame)
        btn_reset = QPushButton("🗑️ Limpiar Todo")
        btn_reset.setMinimumHeight(35)
        btn_reset.clicked.connect(self.reset_all)
        reset_layout.addWidget(btn_reset)
        
        # Agregar al layout principal
        loading_layout.addWidget(local_group)
        loading_layout.addWidget(url_group)
        loading_layout.addWidget(reset_frame)
        loading_layout.addStretch()
        
        parent_layout.addWidget(loading_group, row, col)
    
    def create_info_section(self, parent_layout, row, col):
        """Crear sección de información del dataset"""
        info_group = QGroupBox("📈 Información del Dataset")
        info_group.setMinimumHeight(200)
        info_layout = QVBoxLayout(info_group)
        
        self.info_label = QLabel("Dataset: No cargado\n\nEntradas: 0\nSalidas: 0\nPatrones: 0")
        self.info_label.setWordWrap(True)
        self.info_label.setAlignment(Qt.AlignLeft)
        
        # Estadísticas adicionales
        self.detailed_stats = QLabel("")
        self.detailed_stats.setWordWrap(True)
        
        info_layout.addWidget(self.info_label)
        info_layout.addWidget(self.detailed_stats)
        info_layout.addStretch()
        
        parent_layout.addWidget(info_group, row, col)
    
    def create_auto_detection_section(self, parent_layout, row, col):
        """Crear sección de detección automática"""
        auto_group = QGroupBox("🔍 Detección Automática")
        auto_group.setMinimumHeight(300)
        auto_layout = QVBoxLayout(auto_group)
        
        # Información de detección
        self.detection_info = QLabel("El programa detecta automáticamente:\n• Última columna como SALIDA\n• Demás columnas como ENTRADAS")
        self.detection_info.setWordWrap(True)
        auto_layout.addWidget(self.detection_info)
        
        # Columnas detectadas automáticamente
        detected_frame = QFrame()
        detected_layout = QVBoxLayout(detected_frame)
        
        self.auto_inputs_label = QLabel("🔹 Entradas detectadas: Ninguna")
        self.auto_output_label = QLabel("🔸 Salida detectada: Ninguna")
        
        detected_layout.addWidget(self.auto_inputs_label)
        detected_layout.addWidget(self.auto_output_label)
        auto_layout.addWidget(detected_frame)
        
        parent_layout.addWidget(auto_group, row, col)
    
    def create_preprocessing_section(self, parent_layout, row, col):
        """Crear sección de preprocesamiento automático"""
        preprocess_group = QGroupBox("⚙️ Preprocesamiento Automático")
        preprocess_group.setMinimumHeight(300)
        preprocess_layout = QVBoxLayout(preprocess_group)
        
        # Información de preprocesamiento
        info_frame = QFrame()
        info_layout = QVBoxLayout(info_frame)
        
        preprocess_info = QLabel(
            "El preprocesamiento se aplica automáticamente:\n"
            "• Valores faltantes: Rellenar con mediana\n"
            "• Codificación: Automática para variables categóricas\n"
            "• NO se aplica escalado (se mantienen valores originales)"
        )
        preprocess_info.setWordWrap(True)
        info_layout.addWidget(preprocess_info)
        preprocess_layout.addWidget(info_frame)
        
        # Configuración de partición
        partition_frame = QFrame()
        partition_layout = QHBoxLayout(partition_frame)
        
        partition_layout.addWidget(QLabel("Partición:"))
        
        # Spinbox para porcentaje de entrenamiento
        self.train_spin = QSpinBox()
        self.train_spin.setRange(50, 90)  # Entre 50% y 90%
        self.train_spin.setValue(70)      # 70% por defecto
        self.train_spin.setSuffix("% Entrenamiento")
        self.train_spin.valueChanged.connect(self.update_partition_percentages)
        
        # Label para porcentaje de prueba (calculado automáticamente)
        self.test_label = QLabel("30% Prueba")
        
        partition_layout.addWidget(self.train_spin)
        partition_layout.addWidget(self.test_label)
        partition_layout.addStretch()
        
        preprocess_layout.addWidget(partition_frame)
        
        # Información de partición
        self.split_info_label = QLabel("📊 Entrenamiento: 0 (70%) | Prueba: 0 (30%)")
        self.split_info_label.setAlignment(Qt.AlignCenter)
        preprocess_layout.addWidget(self.split_info_label)
        
        # Botón de aplicar preprocesamiento automático
        btn_preprocess = QPushButton("🚀 Aplicar Preprocesamiento Automático")
        btn_preprocess.setMinimumHeight(45)
        btn_preprocess.clicked.connect(self.apply_auto_preprocessing)
        preprocess_layout.addWidget(btn_preprocess)
        
        # Estado del preprocesamiento
        self.preprocess_status = QLabel("⏳ Esperando datos...")
        self.preprocess_status.setAlignment(Qt.AlignCenter)
        preprocess_layout.addWidget(self.preprocess_status)
        
        preprocess_layout.addStretch()
        parent_layout.addWidget(preprocess_group, row, col)
    
    def create_preview_section(self, parent_layout, row, col, row_span, col_span):
        """Crear sección de vista previa"""
        preview_group = QGroupBox("👁️ Vista Previa de Datos")
        preview_group.setMinimumHeight(400)
        preview_layout = QVBoxLayout(preview_group)
        
        # Notebook para pestañas
        self.notebook = QTabWidget()
        self.notebook.setMinimumHeight(350)
        
        # Pestaña de datos originales
        data_tab = QWidget()
        data_layout = QVBoxLayout(data_tab)
        
        data_scroll = QScrollArea()
        data_scroll.setWidgetResizable(True)
        
        self.data_table = QTableWidget()
        self.data_table.setAlternatingRowColors(True)
        data_scroll.setWidget(self.data_table)
        data_layout.addWidget(data_scroll)
        
        # Pestaña de estadísticas
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)
        
        stats_scroll = QScrollArea()
        stats_scroll.setWidgetResizable(True)
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setStyleSheet("font-family: 'Consolas', 'Monaco', monospace; font-size: 10pt;")
        stats_scroll.setWidget(self.stats_text)
        stats_layout.addWidget(stats_scroll)
        
        # Pestaña de preprocesamiento
        preprocess_tab = QWidget()
        preprocess_layout = QVBoxLayout(preprocess_tab)
        
        preprocess_scroll = QScrollArea()
        preprocess_scroll.setWidgetResizable(True)
        
        self.preprocess_text = QTextEdit()
        self.preprocess_text.setReadOnly(True)
        self.preprocess_text.setStyleSheet("font-family: 'Consolas', 'Monaco', monospace; font-size: 10pt;")
        preprocess_scroll.setWidget(self.preprocess_text)
        preprocess_layout.addWidget(preprocess_scroll)
        
        # Agregar pestañas
        self.notebook.addTab(data_tab, "📋 Datos Originales")
        self.notebook.addTab(stats_tab, "📊 Estadísticas")
        self.notebook.addTab(preprocess_tab, "🔧 Preprocesamiento")
        
        preview_layout.addWidget(self.notebook)
        parent_layout.addWidget(preview_group, row, col, row_span, col_span)
    
    def create_log_section(self, parent_layout):
        """Crear sección de log de operaciones"""
        log_group = QGroupBox("📝 Log de Operaciones")
        log_layout = QVBoxLayout(log_group)
        
        log_scroll = QScrollArea()
        log_scroll.setWidgetResizable(True)
        log_scroll.setMinimumHeight(120)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("font-family: 'Consolas', 'Monaco', monospace; font-size: 9pt;")
        log_scroll.setWidget(self.log_text)
        log_layout.addWidget(log_scroll)
        
        parent_layout.addWidget(log_group)
    
    def auto_detect_columns(self):
        """Detección automática de columnas de entrada y salida"""
        if self.df is None:
            return
        
        # La última columna es la salida, las demás son entradas
        all_columns = self.df.columns.tolist()
        if len(all_columns) > 0:
            self.output_columns = [all_columns[-1]]  # Última columna como salida
            self.input_columns = all_columns[:-1]    # Todas las demás como entradas
        
        # VERIFICACIÓN EXTRA: Asegurar que no hay solapamiento
        for col in self.output_columns:
            if col in self.input_columns:
                self.input_columns.remove(col)
        
        print(f"=== DEBUG DETECCIÓN COLUMNAS ===")
        print(f"Columnas totales en dataset: {len(all_columns)}")
        print(f"Entradas detectadas: {len(self.input_columns)} - {self.input_columns}")
        print(f"Salida detectada: {self.output_columns}")
        
        self.update_auto_detection_labels()
        self.log_message(f"🔍 Detección automática: {len(self.input_columns)} entradas, 1 salida")
    
    def update_auto_detection_labels(self):
        """Actualizar etiquetas de detección automática"""
        inputs_text = f"🔹 Entradas detectadas: {', '.join(self.input_columns) if self.input_columns else 'Ninguna'}"
        outputs_text = f"🔸 Salida detectada: {', '.join(self.output_columns) if self.output_columns else 'Ninguna'}"
        
        self.auto_inputs_label.setText(inputs_text)
        self.auto_output_label.setText(outputs_text)
    
    def reset_all(self):
        """Resetear todo el formulario a su estado inicial"""
        reply = QMessageBox.question(self, 'Confirmar Reset', 
                                   '¿Está seguro de que desea limpiar todos los datos?',
                                   QMessageBox.Yes | QMessageBox.No, 
                                   QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.df = None
            self.preprocessed_data = None
            self.input_columns = []
            self.output_columns = []
            self.test_size = 0.3
            self.train_spin.setValue(70)
            self.test_label.setText("30% Prueba")
            
            # Limpiar interfaz
            self.info_label.setText("Dataset: No cargado\n\nEntradas: 0\nSalidas: 0\nPatrones: 0")
            self.detailed_stats.setText("")
            self.auto_inputs_label.setText("🔹 Entradas detectadas: Ninguna")
            self.auto_output_label.setText("🔸 Salida detectada: Ninguna")
            self.data_table.setRowCount(0)
            self.data_table.setColumnCount(0)
            self.stats_text.clear()
            self.preprocess_text.clear()
            self.split_info_label.setText("📊 Entrenamiento: 0 (70%) | Prueba: 0 (30%)")
            self.preprocess_status.setText("⏳ Esperando datos...")
            self.url_entry.clear()
            
            # Limpiar datos en la aplicación principal
            self.main_window.preprocessed_data = None
            self.main_window.dataset = None
            
            self.log_message("🧹 Todos los datos han sido limpiados")
    
    def update_partition_percentages(self):
        """Actualizar porcentajes de partición cuando cambia el valor del spinbox"""
        train_percent = self.train_spin.value()
        test_percent = 100 - train_percent
        self.test_size = test_percent / 100.0
        
        self.test_label.setText(f"{test_percent}% Prueba")
        
        # Actualizar información de partición si hay datos cargados
        if self.df is not None:
            train_size = int(len(self.df) * (self.test_size))
            test_size = len(self.df) - train_size
            self.update_split_info(train_size, test_size)
    
    def enable_drag_drop_help(self):
        """Mostrar ayuda para drag and drop"""
        QMessageBox.information(self, "Arrastrar y Soltar", 
                              "Puede arrastrar archivos CSV o JSON directamente sobre cualquier parte de la ventana.")
        self.log_message("🖱️ Modo arrastrar y soltar activado - Arrastre archivos sobre la ventana")
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Manejar evento de arrastrar sobre el widget"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        """Manejar evento de soltar archivo"""
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.load_dropped_file(file_path)
    
    def load_dropped_file(self, file_path):
        """Cargar archivo arrastrado"""
        try:
            if file_path.endswith('.csv'):
                self.df = pd.read_csv(file_path)
                self.log_message(f"✅ CSV cargado: {os.path.basename(file_path)}")
            elif file_path.endswith('.json'):
                self.df = pd.read_json(file_path)
                self.log_message(f"✅ JSON cargado: {os.path.basename(file_path)}")
            else:
                QMessageBox.warning(self, "Formato no soportado", 
                                  "Solo se admiten archivos CSV y JSON")
                return
            
            # Detección automática después de cargar
            self.auto_detect_columns()
            self.update_interface()
            
        except Exception as e:
            self.log_message(f"❌ Error al cargar archivo: {str(e)}")
            QMessageBox.critical(self, "Error", f"No se pudo cargar el archivo:\n{str(e)}")
    
    def load_file(self, file_type):
        """Cargar archivo local"""
        try:
            if file_type == 'csv':
                file_path, _ = QFileDialog.getOpenFileName(self, "Seleccionar CSV", "", "CSV Files (*.csv)")
                if file_path:
                    self.df = pd.read_csv(file_path)
                    self.log_message(f"✅ CSV cargado: {os.path.basename(file_path)}")
            elif file_type == 'json':
                file_path, _ = QFileDialog.getOpenFileName(self, "Seleccionar JSON", "", "JSON Files (*.json)")
                if file_path:
                    self.df = pd.read_json(file_path)
                    self.log_message(f"✅ JSON cargado: {os.path.basename(file_path)}")
            
            if self.df is not None:
                # Detección automática después de cargar
                self.auto_detect_columns()
                self.update_interface()
                
        except Exception as e:
            self.log_message(f"❌ Error al cargar archivo: {str(e)}")
            QMessageBox.critical(self, "Error", f"No se pudo cargar el archivo:\n{str(e)}")
    
    def load_from_url(self):
        """Cargar dataset desde URL"""
        url = self.url_entry.text().strip()
        if not url or url == "https://ejemplo.com/dataset.csv":
            QMessageBox.warning(self, "Advertencia", "Por favor ingrese una URL válida")
            return
        
        try:
            self.log_message(f"🌐 Intentando cargar desde: {url}")
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            if url.endswith('.csv'):
                self.df = pd.read_csv(StringIO(response.text))
            elif url.endswith('.json'):
                self.df = pd.read_json(StringIO(response.text))
            else:
                # Intentar detectar automáticamente
                try:
                    self.df = pd.read_csv(StringIO(response.text))
                except:
                    self.df = pd.read_json(StringIO(response.text))
            
            self.log_message("✅ Dataset cargado exitosamente desde URL")
            
            # Detección automática después de cargar
            self.auto_detect_columns()
            self.update_interface()
            
        except Exception as e:
            self.log_message(f"❌ Error al cargar desde URL: {str(e)}")
            QMessageBox.critical(self, "Error", f"No se pudo cargar desde la URL:\n{str(e)}")

    
    def apply_auto_preprocessing(self):
        """Aplicar preprocesamiento automático completo - SIN ESCALADO"""
        if self.df is None:
            QMessageBox.warning(self, "Advertencia", "Primero cargue un dataset")
            return
        
        if not self.input_columns or not self.output_columns:
            QMessageBox.warning(self, "Advertencia", "No se pudieron detectar automáticamente las columnas de entrada y salida")
            return
        
        try:
            self.preprocess_status.setText("🔄 Procesando...")
            QApplication.processEvents()  # Actualizar la interfaz
            
            # Crear copia del dataframe para preprocesamiento
            df_processed = self.df.copy()
            preprocess_log = "=== PREPROCESAMIENTO AUTOMÁTICO ===\n\n"
            
            # VERIFICACIÓN CRÍTICA: Mostrar exactamente lo que se va a procesar
            preprocess_log += f"ENTRADAS A PROCESAR: {len(self.input_columns)} columnas\n"
            preprocess_log += f"  • {self.input_columns}\n"
            preprocess_log += f"SALIDA A PROCESAR: {self.output_columns}\n\n"
            
            print(f"=== DEBUG PREPROCESAMIENTO ===")
            print(f"DataFrame original: {self.df.shape}")
            print(f"Entradas a procesar: {len(self.input_columns)} - {self.input_columns}")
            print(f"Salida a procesar: {self.output_columns}")
            
            # 1. Manejar valores faltantes (automático) - SOLO EN ENTRADAS
            preprocess_log += "1. MANEJO DE VALORES FALTANTES (solo entradas):\n"
            
            # Identificar tipos de columnas - SOLO ENTRADAS
            numeric_cols = []
            categorical_cols = []
            
            for col in self.input_columns:
                if pd.api.types.is_numeric_dtype(df_processed[col]):
                    numeric_cols.append(col)
                else:
                    categorical_cols.append(col)
            
            # Para numéricas: rellenar con mediana - SOLO ENTRADAS
            if len(numeric_cols) > 0:
                imputer_num = SimpleImputer(strategy='median')
                df_processed[numeric_cols] = imputer_num.fit_transform(df_processed[numeric_cols])
                preprocess_log += f"   • Numéricas: rellenadas con mediana ({len(numeric_cols)} columnas)\n"
            
            # Para categóricas: rellenar con moda - SOLO ENTRADAS
            if len(categorical_cols) > 0:
                imputer_cat = SimpleImputer(strategy='most_frequent')
                # Convertir a string para evitar problemas con tipos mixtos
                for col in categorical_cols:
                    df_processed[col] = df_processed[col].astype(str)
                df_processed[categorical_cols] = imputer_cat.fit_transform(df_processed[categorical_cols])
                preprocess_log += f"   • Categóricas: rellenadas con moda ({len(categorical_cols)} columnas)\n"
            
            # 2. ✅ CORRECCIÓN: NO APLICAR ESCALADO - Mantener valores originales
            preprocess_log += "\n2. ESCALADO: NO APLICADO (se mantienen valores originales)\n"
            preprocess_log += "   • Los centros radiales se inicializarán con el rango original de los datos\n"
            
            # 3. Codificación automática de variables categóricas - SOLO ENTRADAS
            preprocess_log += "\n3. CODIFICACIÓN DE VARIABLES (solo entradas):\n"
            new_input_columns = []
            
            for col in self.input_columns:
                if col in categorical_cols:
                    try:
                        # One-hot encoding para entradas categóricas
                        dummies = pd.get_dummies(df_processed[col], prefix=col)
                        df_processed = pd.concat([df_processed, dummies], axis=1)
                        df_processed = df_processed.drop(col, axis=1)
                        
                        # Agregar nuevas columnas one-hot
                        new_input_columns.extend(dummies.columns.tolist())
                        preprocess_log += f"   • '{col}' → {len(dummies.columns)} variables one-hot\n"
                    except Exception as e:
                        # Fallback: label encoding si one-hot falla
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                        new_input_columns.append(col)
                        preprocess_log += f"   • '{col}' → label encoding ({len(le.classes_)} categorías)\n"
                else:
                    # Mantener columnas numéricas
                    new_input_columns.append(col)
            
            # Actualizar lista de columnas de entrada
            self.input_columns = new_input_columns
            
            # 4. Codificación de salida si es categórica
            output_col = self.output_columns[0]
            if not pd.api.types.is_numeric_dtype(df_processed[output_col]):
                try:
                    unique_values = df_processed[output_col].unique()
                    mapping = {val: i for i, val in enumerate(unique_values)}
                    df_processed[output_col] = df_processed[output_col].map(mapping)
                    preprocess_log += f"   • Salida '{output_col}' codificada: {mapping}\n"
                except:
                    # Fallback: label encoding
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    df_processed[output_col] = le.fit_transform(df_processed[output_col].astype(str))
                    preprocess_log += f"   • Salida '{output_col}' → label encoding\n"
            
            # 5. Partición automática con porcentajes configurados por el usuario
            train_percent = self.train_spin.value()
            test_percent = 100 - train_percent
            preprocess_log += f"\n4. PARTICIÓN DEL DATASET ({train_percent}%/{test_percent}%):\n"
            
            # CORRECIÓN DEFINITIVA: X debe contener SOLO las columnas de ENTRADA
            # Y debe EXCLUIR completamente la columna de salida
            X = df_processed[self.input_columns].values
            y = df_processed[self.output_columns].values.ravel()
            
            print(f"DEBUG - Forma de X (entradas): {X.shape}")
            print(f"DEBUG - Forma de y (salida): {y.shape}")
            print(f"DEBUG - Entradas finales: {len(self.input_columns)}")
            print(f"DEBUG - Salida final: {self.output_columns}")
            print(f"DEBUG - Rango de X: [{np.min(X):.6f}, {np.max(X):.6f}]")
            print(f"DEBUG - Rango de y: [{np.min(y):.6f}, {np.max(y):.6f}]")
            
            # Verificar que hay suficientes datos para la estratificación
            unique_classes = len(np.unique(y))
            if unique_classes > 1 and len(y) >= 2 * unique_classes:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=self.test_size, random_state=42, stratify=y
                )
                preprocess_log += "   • Partición con estratificación\n"
            else:
                # Sin estratificación si no hay suficientes clases
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=self.test_size, random_state=42
                )
                preprocess_log += "   • Partición sin estratificación (datos insuficientes)\n"
            
            preprocess_log += f"   • Entrenamiento: {len(X_train)} patrones ({train_percent}%)\n"
            preprocess_log += f"   • Prueba: {len(X_test)} patrones ({test_percent}%)\n"
            preprocess_log += f"   • Total ENTRADAS: {len(self.input_columns)}\n"
            preprocess_log += f"   • Total SALIDAS: 1\n"
            preprocess_log += f"   • Rango final de X_train: [{np.min(X_train):.6f}, {np.max(X_train):.6f}]\n"
            
            # 6. Guardar estadísticas básicas
            preprocess_log += "\n5. ESTADÍSTICAS FINALES:\n"
            preprocess_log += f"   • Forma X_train: {X_train.shape}\n"
            preprocess_log += f"   • Forma X_test: {X_test.shape}\n"
            preprocess_log += f"   • Forma y_train: {y_train.shape}\n"
            preprocess_log += f"   • Forma y_test: {y_test.shape}\n"
            
            # Asegurar que todos los datos sean float64
            X_train = X_train.astype(np.float64)
            X_test = X_test.astype(np.float64)
            y_train = y_train.astype(np.float64)
            y_test = y_test.astype(np.float64)

            # Guardar en preprocessed_data
            self.preprocessed_data = {
                'X_train': X_train, 'X_test': X_test,
                'y_train': y_train, 'y_test': y_test,
                'input_columns': self.input_columns,
                'output_columns': self.output_columns,
                'df_processed': df_processed,
                'preprocess_log': preprocess_log,
                'partition_percentages': {'train': train_percent, 'test': test_percent},
                'scaler_type': 'Ninguno'  # ✅ Sin escalado
            }
            
            # Actualizar aplicación principal
            self.main_window.preprocessed_data = self.preprocessed_data
            
            # Actualizar interfaz
            self.update_split_info(len(X_train), len(X_test))
            self.preprocess_text.setText(preprocess_log)
            
            self.preprocess_status.setText("✅ Preprocesamiento completado")
            
            self.log_message("✅ Preprocesamiento automático aplicado exitosamente")
            
            # CORRECCIÓN DEL MENSAJE DE CONFIRMACIÓN
            QMessageBox.information(self, "Preprocesamiento Completado", 
                                f"Procesamiento automático finalizado:\n"
                                f"• Entrenamiento: {len(X_train)} patrones ({train_percent}%)\n"
                                f"• Prueba: {len(X_test)} patrones ({test_percent}%)\n"
                                f"• ENTRADAS: {len(self.input_columns)}\n"
                                f"• SALIDA: 1\n"
                                f"• Ver detalles en pestaña 'Preprocesamiento'")
            
        except Exception as e:
            self.preprocess_status.setText("❌ Error en preprocesamiento")
            error_msg = f"❌ Error en preprocesamiento automático: {str(e)}"
            self.log_message(error_msg)
            QMessageBox.critical(self, "Error", f"Error en preprocesamiento automático:\n{str(e)}")
    
    def update_interface(self):
        """Actualizar toda la interfaz con los datos cargados"""
        if self.df is not None:
            # Actualizar información básica
            num_patterns = len(self.df)
            num_features = len(self.df.columns)
            num_inputs = len(self.input_columns)
            num_outputs = len(self.output_columns)
            
            info_text = f"Dataset: Cargado\n\nEntradas: {num_inputs}\nSalidas: {num_outputs}\nPatrones: {num_patterns}\nCaracterísticas: {num_features}"
            self.info_label.setText(info_text)
            
            # Actualizar estadísticas detalladas
            self.update_detailed_stats()
            
            # Actualizar detección automática
            self.update_auto_detection_labels()
            
            # Actualizar tabla de datos
            self.update_data_table()
            
            # Actualizar estadísticas
            self.update_statistics()
            
            # Actualizar información de partición
            train_percent = self.train_spin.value()
            train_size = int(num_patterns * (train_percent / 100.0))
            test_size = num_patterns - train_size
            self.update_split_info(train_size, test_size)
            
            # Resetear estado de preprocesamiento
            self.preprocess_status.setText("✅ Datos listos para preprocesamiento")
    
    def update_detailed_stats(self):
        """Actualizar estadísticas detalladas"""
        if self.df is not None:
            stats_text = ""
            
            # Tipos de datos
            numeric_count = len(self.df.select_dtypes(include=[np.number]).columns)
            categorical_count = len(self.df.select_dtypes(include=['object']).columns)
            
            stats_text += f"Numéricas: {numeric_count}\n"
            stats_text += f"Categóricas: {categorical_count}\n"
            
            # Valores faltantes
            total_missing = self.df.isnull().sum().sum()
            if total_missing > 0:
                stats_text += f"Valores faltantes: {total_missing}"
            else:
                stats_text += "Valores faltantes: 0"
            
            self.detailed_stats.setText(stats_text)
    
    def update_data_table(self):
        """Actualizar tabla de datos"""
        if self.df is not None:
            # Mostrar máximo 100 filas para mejor rendimiento
            display_rows = min(100, len(self.df))
            
            self.data_table.setRowCount(display_rows)
            self.data_table.setColumnCount(len(self.df.columns))
            self.data_table.setHorizontalHeaderLabels(self.df.columns)
            
            for i in range(display_rows):
                for j, value in enumerate(self.df.iloc[i]):
                    item = QTableWidgetItem(str(value))
                    self.data_table.setItem(i, j, item)
            
            # Ajustar tamaño de columnas
            self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            
            if len(self.df) > 100:
                self.log_message(f"📋 Mostrando 100 de {len(self.df)} filas")
    
    def update_statistics(self):
        """Actualizar estadísticas del dataset"""
        if self.df is not None:
            stats_text = "=== ESTADÍSTICAS DEL DATASET ===\n\n"
            
            # Información general
            stats_text += f"Forma: {self.df.shape}\n"
            stats_text += f"Memoria: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n"
            
            # Tipos de datos
            stats_text += "TIPOS DE DATOS:\n"
            for col in self.df.columns:
                dtype = self.df[col].dtype
                unique_count = self.df[col].nunique()
                stats_text += f"  {col}: {dtype} (únicos: {unique_count})\n"
            
            stats_text += "\nVALORES FALTANTES:\n"
            missing = self.df.isnull().sum()
            for col in self.df.columns:
                if missing[col] > 0:
                    stats_text += f"  {col}: {missing[col]}\n"
            
            # Estadísticas numéricas
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                stats_text += "\nESTADÍSTICAS NUMÉRICAS:\n"
                stats = self.df[numeric_cols].describe()
                stats_text += stats.to_string()
            
            self.stats_text.setText(stats_text)
    
    def update_split_info(self, train_size, test_size):
        """Actualizar información de partición"""
        train_percent = self.train_spin.value()
        test_percent = 100 - train_percent
        
        info_text = f"📊 Entrenamiento: {train_size} ({train_percent}%) | Prueba: {test_size} ({test_percent}%)"
        self.split_info_label.setText(info_text)
    
    def log_message(self, message):
        """Agregar mensaje al log"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        # Auto-scroll al final
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
    
    def on_activate(self):
        """Método llamado cuando el módulo se activa"""
        self.log_message("🔍 Módulo de carga de datos activado")