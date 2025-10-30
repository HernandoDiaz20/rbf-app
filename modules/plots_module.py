import numpy as np
import pandas as pd
import os
import json
import pickle
from datetime import datetime
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QGroupBox, QTextEdit, QFrame, 
                             QMessageBox, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QScrollArea, QProgressBar,
                             QSplitter, QTabWidget, QApplication, QFileDialog,
                             QLineEdit, QInputDialog, QDialog, QFormLayout,
                             QDialogButtonBox, QGridLayout, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QDoubleValidator, QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
matplotlib.use('Qt5Agg')

class PlotsModule(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.figures = []
        self.current_plots = {}
        self.loaded_images = {}  # Para almacenar imágenes cargadas
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
        title = QLabel("📈 GRÁFICAS RBF")
        title.setObjectName("title_label")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel#title_label {
                color: #2c3e50;
                font-size: 24px;
                font-weight: bold;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                          stop:0 #e74c3c, stop:1 #c0392b);
                color: white;
                border-radius: 10px;
                padding: 15px;
                margin: 5px;
            }
        """)
        scroll_layout.addWidget(title)
        
        # Crear secciones
        self.create_requirements_section(scroll_layout)
        self.create_actions_section(scroll_layout)
        self.create_plots_section(scroll_layout)
        self.create_loaded_images_section(scroll_layout)  # Nueva sección para imágenes cargadas
        self.create_info_section(scroll_layout)
        
        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area)
    
    def create_requirements_section(self, parent_layout):
        """Crear sección de requisitos"""
        requirements_group = QGroupBox("📋 Requisitos para Gráficas")
        requirements_layout = QVBoxLayout(requirements_group)
        
        # Información de requisitos
        self.requirements_label = QLabel(
            "Requisitos para generar gráficas:\n"
            "• Simulación ejecutada: ❌\n" 
            "• Resultados disponibles: ❌\n"
        )
        self.requirements_label.setStyleSheet("color: #7f8c8d; font-size: 12px; padding: 10px;")
        requirements_layout.addWidget(self.requirements_label)
        
        parent_layout.addWidget(requirements_group)
    
    def create_actions_section(self, parent_layout):
        """Crear sección de acciones"""
        actions_group = QGroupBox("🚀 Acciones de Gráficas")
        actions_layout = QVBoxLayout(actions_group)
        
        # Primera fila de botones - MÁS ANCHOS
        row1_frame = QFrame()
        row1_layout = QHBoxLayout(row1_frame)
        
        btn_generate = QPushButton("🎨 GENERAR GRÁFICAS")
        btn_generate.setMinimumHeight(50)
        btn_generate.setMinimumWidth(200)
        btn_generate.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn_generate.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                          stop:0 #27ae60, stop:1 #2ecc71);
                color: white;
                font-weight: bold;
                font-size: 14px;
                border-radius: 8px;
                padding: 12px 20px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                          stop:0 #219653, stop:1 #27ae60);
            }
            QPushButton:disabled {
                background: #bdc3c7;
                color: #7f8c8d;
            }
        """)
        btn_generate.clicked.connect(self.generate_plots)
        self.btn_generate = btn_generate
        
        btn_save = QPushButton("💾 GUARDAR GRÁFICAS")
        btn_save.setMinimumHeight(50)
        btn_save.setMinimumWidth(200)
        btn_save.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn_save.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                          stop:0 #3498db, stop:1 #2980b9);
                color: white;
                font-weight: bold;
                font-size: 14px;
                border-radius: 8px;
                padding: 12px 20px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                          stop:0 #2980b9, stop:1 #2471a3);
            }
            QPushButton:disabled {
                background: #bdc3c7;
                color: #7f8c8d;
            }
        """)
        btn_save.clicked.connect(self.save_plots)
        self.btn_save = btn_save
        
        row1_layout.addWidget(btn_generate)
        row1_layout.addWidget(btn_save)
        
        # Segunda fila de botones - MÁS ANCHOS
        row2_frame = QFrame()
        row2_layout = QHBoxLayout(row2_frame)
        
        btn_load = QPushButton("📂 CARGAR GRÁFICAS")
        btn_load.setMinimumHeight(50)
        btn_load.setMinimumWidth(200)
        btn_load.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn_load.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                          stop:0 #9b59b6, stop:1 #8e44ad);
                color: white;
                font-weight: bold;
                font-size: 14px;
                border-radius: 8px;
                padding: 12px 20px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                          stop:0 #8e44ad, stop:1 #7d3c98);
            }
        """)
        btn_load.clicked.connect(self.load_plots)
        
        btn_clear = QPushButton("🗑️ LIMPIAR GRÁFICAS")
        btn_clear.setMinimumHeight(50)
        btn_clear.setMinimumWidth(200)
        btn_clear.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn_clear.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                          stop:0 #e74c3c, stop:1 #c0392b);
                color: white;
                font-weight: bold;
                font-size: 14px;
                border-radius: 8px;
                padding: 12px 20px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                          stop:0 #c0392b, stop:1 #a93226);
            }
        """)
        btn_clear.clicked.connect(self.clear_plots)
        
        row2_layout.addWidget(btn_load)
        row2_layout.addWidget(btn_clear)
        
        # Agregar filas al layout principal
        actions_layout.addWidget(row1_frame)
        actions_layout.addWidget(row2_frame)
        
        # Estado de las gráficas
        self.plots_status = QLabel("⏳ Esperando generación de gráficas...")
        self.plots_status.setAlignment(Qt.AlignCenter)
        self.plots_status.setStyleSheet("""
            padding: 15px; 
            font-weight: bold; 
            font-size: 12px;
            background: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        """)
        actions_layout.addWidget(self.plots_status)
        
        parent_layout.addWidget(actions_group)
    
    def create_plots_section(self, parent_layout):
        """Crear sección de gráficas - CONTENEDOR MÁS GRANDE"""
        plots_group = QGroupBox("📊 Visualización de Gráficas")
        plots_layout = QVBoxLayout(plots_group)
        
        # Notebook para pestañas de gráficas - CONTENEDOR MÁS GRANDE
        self.plots_notebook = QTabWidget()
        self.plots_notebook.setMinimumHeight(650)  # Aumentado de 500 a 650
        self.plots_notebook.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Crear pestañas para cada gráfica
        self.plot_tabs = {}
        plot_types = [
            ("yd_vs_yr", "📈 YD vs YR (Entrenamiento y Prueba)"),
            ("error_vs_optimal", "📊 Error General vs Error Óptimo"),
            ("scatter_predictions", "🎯 Dispersión YD vs YR")
        ]
        
        for plot_id, plot_name in plot_types:
            tab = QWidget()
            layout = QVBoxLayout(tab)
            
            # Canvas para la gráfica - TAMAÑO ORIGINAL (no cambiar)
            figure = plt.Figure(figsize=(10, 6), dpi=100)  # Mantener tamaño original
            canvas = FigureCanvas(figure)
            canvas.setMinimumHeight(450)  # Mantener tamaño original
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            
            # Toolbar de navegación
            toolbar = NavigationToolbar(canvas, self)
            
            layout.addWidget(toolbar)
            layout.addWidget(canvas)
            
            self.plot_tabs[plot_id] = {
                'tab': tab,
                'figure': figure,
                'canvas': canvas,
                'toolbar': toolbar
            }
            
            self.plots_notebook.addTab(tab, plot_name)
        
        plots_layout.addWidget(self.plots_notebook)
        parent_layout.addWidget(plots_group)
    
    def create_loaded_images_section(self, parent_layout):
        """Nueva sección para mostrar imágenes cargadas - CONTENEDOR MÁS GRANDE"""
        self.loaded_images_group = QGroupBox("📁 Gráficas Cargadas")
        self.loaded_images_group.setVisible(False)
        loaded_layout = QVBoxLayout(self.loaded_images_group)
        
        # Notebook para imágenes cargadas - CONTENEDOR MÁS GRANDE
        self.loaded_images_notebook = QTabWidget()
        self.loaded_images_notebook.setMinimumHeight(550)  # Aumentado de 400 a 550
        self.loaded_images_notebook.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        loaded_layout.addWidget(self.loaded_images_notebook)
        parent_layout.addWidget(self.loaded_images_group)
    
    def create_info_section(self, parent_layout):
        """Crear sección de información"""
        info_group = QGroupBox("📋 Información de las Gráficas")
        info_layout = QVBoxLayout(info_group)
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setStyleSheet("""
            font-family: 'Consolas', 'Monaco', monospace; 
            font-size: 10pt;
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 10px;
        """)
        self.info_text.setMinimumHeight(150)
        info_layout.addWidget(self.info_text)
        
        parent_layout.addWidget(info_group)
    
    def check_simulation_results(self):
        """Verificar si hay resultados de simulación disponibles"""
        has_simulation = False
        
        if hasattr(self.main_window, 'simulation_results'):
            has_simulation = self.main_window.simulation_results is not None
        
        if not has_simulation and hasattr(self.main_window, 'training_results'):
            has_simulation = self.main_window.training_results is not None
        
        if not has_simulation and hasattr(self.main_window, 'simulation_module'):
            if hasattr(self.main_window.simulation_module, 'loaded_model'):
                has_simulation = self.main_window.simulation_module.loaded_model is not None
        
        return has_simulation
    
    def get_simulation_results(self):
        """Obtener resultados de simulación"""
        if hasattr(self.main_window, 'simulation_results') and self.main_window.simulation_results is not None:
            return self.main_window.simulation_results
        elif hasattr(self.main_window, 'training_results') and self.main_window.training_results is not None:
            return self.create_simulation_from_training()
        elif hasattr(self.main_window, 'simulation_module'):
            if hasattr(self.main_window.simulation_module, 'simulation_results'):
                return self.main_window.simulation_module.simulation_results
            elif hasattr(self.main_window.simulation_module, 'loaded_model'):
                return self.create_simulation_from_loaded_model()
        
        return None
    
    def create_simulation_from_training(self):
        """Crear estructura de simulación a partir de training_results"""
        if self.main_window.training_results is None or self.main_window.preprocessed_data is None:
            return None
        
        try:
            training_results = self.main_window.training_results
            preprocessed_data = self.main_window.preprocessed_data
            
            simulation_results = {
                'train': {
                    'y_true': preprocessed_data['y_train'],
                    'y_pred': training_results['y_pred'],
                    'errors': training_results['errors'],
                    'convergence': training_results['convergence']
                },
                'test': {
                    'y_true': preprocessed_data['y_test'] if 'y_test' in preprocessed_data else preprocessed_data['y_train'],
                    'y_pred': training_results['y_pred'][:len(preprocessed_data.get('y_test', preprocessed_data['y_train']))],
                    'errors': training_results['errors']
                },
                'custom': [],
                'model_info': {
                    'num_centers': training_results['num_centers'],
                    'target_error': training_results['target_error'],
                    'training_date': training_results['training_date'],
                    'input_columns': preprocessed_data.get('input_columns', []),
                    'dataset_info': f"{preprocessed_data['X_train'].shape[1]} entradas, 1 salida"
                }
            }
            
            return simulation_results
            
        except Exception as e:
            print(f"Error creando simulación desde entrenamiento: {e}")
            return None
    
    def create_simulation_from_loaded_model(self):
        """Crear estructura de simulación a partir de modelo cargado"""
        if not hasattr(self.main_window.simulation_module, 'loaded_model'):
            return None
        
        try:
            loaded_model = self.main_window.simulation_module.loaded_model
            training_results = loaded_model['training_results']
            preprocessed_info = loaded_model['preprocessed_data_info']
            
            simulation_results = {
                'train': {
                    'y_true': training_results['y_pred'],
                    'y_pred': training_results['y_pred'],
                    'errors': training_results['errors'],
                    'convergence': training_results['convergence']
                },
                'test': {
                    'y_true': training_results['y_pred'],
                    'y_pred': training_results['y_pred'],
                    'errors': training_results['errors']
                },
                'custom': [],
                'model_info': {
                    'num_centers': training_results['num_centers'],
                    'target_error': training_results['target_error'],
                    'training_date': training_results['training_date'],
                    'input_columns': preprocessed_info.get('input_columns', []),
                    'dataset_info': f"{preprocessed_info['X_train_shape'][1]} entradas, 1 salida"
                }
            }
            
            return simulation_results
            
        except Exception as e:
            print(f"Error creando simulación desde modelo cargado: {e}")
            return None
    
    def update_requirements_status(self):
        """Actualizar el estado de los requisitos"""
        has_simulation = self.check_simulation_results()
        
        simulation_status = "✅" if has_simulation else "❌"
        results_status = "✅" if has_simulation else "❌"
        
        requirements_text = (
            f"Requisitos para generar gráficas:\n"
            f"• Simulación ejecutada: {simulation_status}\n" 
            f"• Resultados disponibles: {results_status}\n"
        )
        
        self.requirements_label.setText(requirements_text)
        
        if has_simulation:
            self.plots_status.setText("✅ LISTO - Puede generar gráficas")
            self.plots_status.setStyleSheet("""
                color: #27ae60; 
                font-weight: bold; 
                padding: 15px;
                font-size: 12px;
                background: #d5f4e6;
                border-radius: 8px;
                border: 1px solid #27ae60;
            """)
            self.btn_generate.setEnabled(True)
            self.btn_save.setEnabled(False)
        else:
            self.plots_status.setText("⚠️ EJECUTE una simulación primero en el módulo 'Simulación'")
            self.plots_status.setStyleSheet("""
                color: #f39c12; 
                font-weight: bold; 
                padding: 15px;
                font-size: 12px;
                background: #fef5e7;
                border-radius: 8px;
                border: 1px solid #f39c12;
            """)
            self.btn_generate.setEnabled(False)
            self.btn_save.setEnabled(False)
    
    def verify_plots_requirements(self):
        """Verificar requisitos para generar gráficas"""
        if not self.check_simulation_results():
            QMessageBox.warning(self, "Simulación Requerida", 
                              "Primero ejecute una simulación en el módulo 'Simulación'\n\n"
                              "O cargue un modelo entrenado en el módulo 'Simulación'")
            return False
        return True
    
    def generate_plots(self):
        """Generar las tres gráficas requeridas"""
        if not self.verify_plots_requirements():
            return
        
        try:
            self.plots_status.setText("🔄 GENERANDO GRÁFICAS...")
            QApplication.processEvents()
            
            simulation_results = self.get_simulation_results()
            if simulation_results is None:
                QMessageBox.warning(self, "Error", "No se pudieron obtener los resultados de simulación")
                return
            
            train_results = simulation_results['train']
            test_results = simulation_results['test']
            model_info = simulation_results['model_info']
            
            self.info_text.clear()
            self.info_text.append("=== GENERANDO GRÁFICAS ===\n")
            self.info_text.append(f"• Dataset: {model_info['dataset_info']}\n")
            self.info_text.append(f"• Centros radiales: {model_info['num_centers']}\n")
            self.info_text.append(f"• Error objetivo: {model_info['target_error']}\n")
            self.info_text.append(f"• Convergencia: {'✅ CONVERGE' if train_results['convergence'] else '❌ NO CONVERGE'}\n")
            
            # Generar las tres gráficas requeridas
            self.generate_yd_vs_yr_plot(train_results, test_results, model_info)
            self.generate_error_vs_optimal_plot(train_results, test_results, model_info)
            self.generate_scatter_predictions_plot(train_results, test_results, model_info)
            
            # Habilitar botón de guardar
            self.btn_save.setEnabled(True)
            self.current_plots = simulation_results
            
            self.plots_status.setText("✅ GRÁFICAS GENERADAS EXITOSAMENTE")
            self.plots_status.setStyleSheet("""
                color: #27ae60; 
                font-weight: bold; 
                padding: 15px;
                font-size: 12px;
                background: #d5f4e6;
                border-radius: 8px;
                border: 1px solid #27ae60;
            """)
            
            self.info_text.append("\n✅ Todas las gráficas han sido generadas exitosamente")
            self.info_text.append("💾 Use el botón 'GUARDAR GRÁFICAS' para exportarlas como PNG")
            
        except Exception as e:
            self.plots_status.setText("❌ ERROR GENERANDO GRÁFICAS")
            self.plots_status.setStyleSheet("""
                color: #c0392b; 
                font-weight: bold; 
                padding: 15px;
                font-size: 12px;
                background: #fadbd8;
                border-radius: 8px;
                border: 1px solid #c0392b;
            """)
            QMessageBox.critical(self, "Error", f"Error al generar gráficas:\n{str(e)}")
    
    def generate_yd_vs_yr_plot(self, train_results, test_results, model_info):
        """Gráfica 1: YD vs YR (Entrenamiento y Prueba)"""
        plot_data = self.plot_tabs['yd_vs_yr']
        figure = plot_data['figure']
        canvas = plot_data['canvas']
        
        figure.clear()
        
        # Crear subplots
        ax1 = figure.add_subplot(211)
        ax2 = figure.add_subplot(212)
        
        # Datos de entrenamiento
        y_true_train = train_results['y_true']
        y_pred_train = train_results['y_pred']
        patterns_train = np.arange(1, len(y_true_train) + 1)
        
        # Datos de prueba
        y_true_test = test_results['y_true']
        y_pred_test = test_results['y_pred']
        patterns_test = np.arange(1, len(y_true_test) + 1)
        
        # Gráfica de entrenamiento
        ax1.plot(patterns_train, y_true_train, 'b-', linewidth=2, label='YD (Deseado)', alpha=0.7)
        ax1.plot(patterns_train, y_pred_train, 'r--', linewidth=2, label='YR (Red)', alpha=0.7)
        ax1.fill_between(patterns_train, y_true_train, y_pred_train, alpha=0.3, color='gray')
        ax1.set_title('COMPARACIÓN YD vs YR - ENTRENAMIENTO', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Patrón #')
        ax1.set_ylabel('Valor de Salida')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfica de prueba
        ax2.plot(patterns_test, y_true_test, 'b-', linewidth=2, label='YD (Deseado)', alpha=0.7)
        ax2.plot(patterns_test, y_pred_test, 'r--', linewidth=2, label='YR (Red)', alpha=0.7)
        ax2.fill_between(patterns_test, y_true_test, y_pred_test, alpha=0.3, color='gray')
        ax2.set_title('COMPARACIÓN YD vs YR - PRUEBA', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Patrón #')
        ax2.set_ylabel('Valor de Salida')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        figure.tight_layout(pad=3.0)
        canvas.draw()
        
        self.info_text.append("✅ Gráfica 1: YD vs YR generada")
    
    def generate_error_vs_optimal_plot(self, train_results, test_results, model_info):
        """Gráfica 2: Error General vs Error Óptimo"""
        plot_data = self.plot_tabs['error_vs_optimal']
        figure = plot_data['figure']
        canvas = plot_data['canvas']
        
        figure.clear()
        ax = figure.add_subplot(111)
        
        errors = {
            'Entrenamiento': train_results['errors']['eg'],
            'Prueba': test_results['errors']['eg']
        }
        
        target_error = model_info['target_error']
        
        categories = list(errors.keys())
        values = list(errors.values())
        colors = ['green' if err <= target_error else 'red' for err in values]
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        
        ax.axhline(y=target_error, color='red', linestyle='--', linewidth=2, 
                  label=f'Error Óptimo = {target_error}')
        
        ax.set_title('ERROR GENERAL (EG) vs ERROR ÓPTIMO', fontsize=14, fontweight='bold')
        ax.set_ylabel('Error General (EG)')
        ax.set_xlabel('Conjunto de Datos')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        max_error = max(values) if values else target_error * 2
        ax.set_ylim(0, max_error * 1.2)
        
        figure.tight_layout()
        canvas.draw()
        
        self.info_text.append("✅ Gráfica 2: Error vs Óptimo generada")
    
    def generate_scatter_predictions_plot(self, train_results, test_results, model_info):
        """Gráfica 3: Curva de dispersión YD vs YR"""
        plot_data = self.plot_tabs['scatter_predictions']
        figure = plot_data['figure']
        canvas = plot_data['canvas']
        
        figure.clear()
        ax = figure.add_subplot(111)
        
        y_true_train = train_results['y_true']
        y_pred_train = train_results['y_pred']
        y_true_test = test_results['y_true']
        y_pred_test = test_results['y_pred']
        
        all_y_true = np.concatenate([y_true_train, y_true_test])
        all_y_pred = np.concatenate([y_pred_train, y_pred_test])
        min_val = min(np.min(all_y_true), np.min(all_y_pred))
        max_val = max(np.max(all_y_true), np.max(all_y_pred))
        
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, 
                label='YD = YR (Predicción Perfecta)')
        
        ax.scatter(y_true_train, y_pred_train, alpha=0.6, color='blue', 
                  s=50, label='Entrenamiento', edgecolors='black', linewidth=0.5)
        
        ax.scatter(y_true_test, y_pred_test, alpha=0.6, color='red', 
                  s=50, label='Prueba', edgecolors='black', linewidth=0.5)
        
        ax.set_title('DISPERSIÓN: YD (Deseado) vs YR (Red)', fontsize=14, fontweight='bold')
        ax.set_xlabel('YD - Salida Deseada')
        ax.set_ylabel('YR - Salida de la Red')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        from sklearn.metrics import r2_score
        r2_train = r2_score(y_true_train, y_pred_train)
        r2_test = r2_score(y_true_test, y_pred_test)
        
        info_text = f'R² Entrenamiento: {r2_train:.4f}\nR² Prueba: {r2_test:.4f}'
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        figure.tight_layout()
        canvas.draw()
        
        self.info_text.append("✅ Gráfica 3: Dispersión YD vs YR generada")
        self.info_text.append(f"   • R² Entrenamiento: {r2_train:.4f}")
        self.info_text.append(f"   • R² Prueba: {r2_test:.4f}")
    
    def save_plots(self):
        """Guardar todas las gráficas como archivos PNG"""
        if not self.current_plots:
            QMessageBox.warning(self, "Advertencia", "Primero genere las gráficas")
            return
        
        try:
            results_dir = "resultados"
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plots_dir = os.path.join(results_dir, f"graficas_{timestamp}")
            os.makedirs(plots_dir)
            
            plot_files = {
                'yd_vs_yr': 'grafico_yr_vs_yd.png',
                'error_vs_optimal': 'grafico_error.png', 
                'scatter_predictions': 'grafico_dispersion.png'
            }
            
            saved_files = []
            
            for plot_id, filename in plot_files.items():
                if plot_id in self.plot_tabs:
                    plot_data = self.plot_tabs[plot_id]
                    filepath = os.path.join(plots_dir, filename)
                    plot_data['figure'].savefig(filepath, dpi=300, bbox_inches='tight', 
                                              facecolor='white', edgecolor='none')
                    saved_files.append(f"• {filename}")
            
            combined_path = os.path.join(plots_dir, 'graficas_combinadas.png')
            self.save_combined_plot(combined_path)
            saved_files.append("• graficas_combinadas.png")
            
            info_path = os.path.join(plots_dir, 'info_graficas.json')
            self.save_plots_info(info_path)
            saved_files.append("• info_graficas.json")
            
            QMessageBox.information(self, "Gráficas Guardadas", 
                                f"✅ Gráficas guardadas en:\n{plots_dir}\n\n"
                                f"Archivos creados:\n" + "\n".join(saved_files))
            
            self.plots_status.setText(f"💾 GRÁFICAS GUARDADAS EN: {plots_dir}")
            self.info_text.append(f"\n💾 Gráficas exportadas a: {plots_dir}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudieron guardar las gráficas:\n{str(e)}")
    
    def save_combined_plot(self, filepath):
        """Guardar gráfica combinada con las tres visualizaciones"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('RESUMEN GRÁFICO - RED NEURONAL RBF', fontsize=16, fontweight='bold')
        
        simulation_results = self.current_plots
        train_results = simulation_results['train']
        test_results = simulation_results['test']
        model_info = simulation_results['model_info']
        
        y_true_test = test_results['y_true']
        y_pred_test = test_results['y_pred']
        patterns_test = np.arange(1, len(y_true_test) + 1)
        
        axes[0, 0].plot(patterns_test, y_true_test, 'b-', linewidth=2, label='YD (Deseado)', alpha=0.7)
        axes[0, 0].plot(patterns_test, y_pred_test, 'r--', linewidth=2, label='YR (Red)', alpha=0.7)
        axes[0, 0].set_title('YD vs YR - Conjunto de Prueba')
        axes[0, 0].set_xlabel('Patrón #')
        axes[0, 0].set_ylabel('Valor de Salida')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        errors = {
            'Entrenamiento': train_results['errors']['eg'],
            'Prueba': test_results['errors']['eg']
        }
        target_error = model_info['target_error']
        
        categories = list(errors.keys())
        values = list(errors.values())
        colors = ['green' if err <= target_error else 'red' for err in values]
        
        bars = axes[0, 1].bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        axes[0, 1].axhline(y=target_error, color='red', linestyle='--', linewidth=2, 
                          label=f'Error Óptimo = {target_error}')
        axes[0, 1].set_title('Error General vs Error Óptimo')
        axes[0, 1].set_ylabel('Error General (EG)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        y_true_train = train_results['y_true']
        y_pred_train = train_results['y_pred']
        
        min_val = min(np.min(y_true_test), np.min(y_pred_test))
        max_val = max(np.max(y_true_test), np.max(y_pred_test))
        
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)
        axes[1, 0].scatter(y_true_test, y_pred_test, alpha=0.6, color='red', s=50)
        axes[1, 0].set_title('Dispersión: YD vs YR')
        axes[1, 0].set_xlabel('YD - Salida Deseada')
        axes[1, 0].set_ylabel('YR - Salida de la Red')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_aspect('equal', adjustable='box')
        
        axes[1, 1].axis('off')
        model_text = (
            f"INFORMACIÓN DEL MODELO\n\n"
            f"• Centros radiales: {model_info['num_centers']}\n"
            f"• Error objetivo: {target_error}\n"
            f"• EG Entrenamiento: {errors['Entrenamiento']:.4f}\n"
            f"• EG Prueba: {errors['Prueba']:.4f}\n"
            f"• Convergencia: {'SI' if train_results['convergence'] else 'NO'}\n"
            f"• Dataset: {model_info['dataset_info']}\n"
            f"• Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        axes[1, 1].text(0.1, 0.9, model_text, transform=axes[1, 1].transAxes, fontsize=11,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
    
    def save_plots_info(self, filepath):
        """Guardar información de las gráficas en JSON"""
        if not self.current_plots:
            return
        
        simulation_results = self.current_plots
        train_results = simulation_results['train']
        test_results = simulation_results['test']
        model_info = simulation_results['model_info']
        
        plots_info = {
            'fecha_generacion': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'modelo': {
                'num_centros': model_info['num_centers'],
                'error_objetivo': float(model_info['target_error']),
                'dataset': model_info['dataset_info']
            },
            'metricas': {
                'eg_entrenamiento': float(train_results['errors']['eg']),
                'eg_prueba': float(test_results['errors']['eg']),
                'convergencia': "CONVERGE" if train_results['convergence'] else "NO_CONVERGE"
            },
            'graficas_generadas': [
                'grafico_yr_vs_yd.png',
                'grafico_error.png', 
                'grafico_dispersion.png',
                'graficas_combinadas.png'
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(plots_info, f, indent=4, ensure_ascii=False)
    
    def load_plots(self):
        """Cargar y mostrar gráficas desde archivos PNG - MEJOR VISUALIZACIÓN"""
        try:
            file_paths, _ = QFileDialog.getOpenFileNames(
                self, "Seleccionar archivos de gráficas PNG", "", 
                "Archivos PNG (*.png);;Todos los archivos (*)"
            )
            
            if file_paths:
                self.loaded_images_group.setVisible(True)
                self.loaded_images_notebook.clear()
                self.loaded_images = {}
                
                for file_path in file_paths:
                    filename = os.path.basename(file_path)
                    
                    # Crear pestaña para esta imagen
                    tab = QWidget()
                    layout = QVBoxLayout(tab)
                    
                    # Crear QLabel para mostrar la imagen - MEJOR VISUALIZACIÓN
                    image_label = QLabel()
                    image_label.setAlignment(Qt.AlignCenter)
                    image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                    image_label.setStyleSheet("background: white; border: 1px solid #ccc;")
                    
                    # Cargar y escalar la imagen - MEJOR ESCALADO
                    pixmap = QPixmap(file_path)
                    if not pixmap.isNull():
                        # Escalar la imagen para aprovechar el espacio del contenedor más grande
                        scaled_pixmap = pixmap.scaled(900, 700, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        image_label.setPixmap(scaled_pixmap)
                        
                        # Agregar scroll area para imágenes grandes
                        scroll_area = QScrollArea()
                        scroll_area.setWidgetResizable(True)
                        scroll_area.setWidget(image_label)
                        scroll_area.setMinimumHeight(500)  # Asegurar altura mínima
                        
                        # Agregar a la pestaña
                        layout.addWidget(scroll_area)
                        
                        # Agregar información del archivo
                        file_info = QLabel(f"📄 Archivo: {filename}\n📍 Ruta: {file_path}")
                        file_info.setStyleSheet("color: #7f8c8d; font-size: 10px; padding: 10px; background: #f8f9fa; border-radius: 5px;")
                        layout.addWidget(file_info)
                        
                        # Agregar pestaña al notebook
                        self.loaded_images_notebook.addTab(tab, filename)
                        self.loaded_images[filename] = file_path
                    
                self.info_text.append(f"📂 Gráficas cargadas: {len(file_paths)} archivos")
                self.info_text.append("📍 Las gráficas cargadas se muestran en la sección 'Gráficas Cargadas'")
                
                QMessageBox.information(self, "Gráficas Cargadas", 
                                    f"✅ Se cargaron {len(file_paths)} gráficas exitosamente.\n\n"
                                    f"Las gráficas están disponibles en la sección 'Gráficas Cargadas'.\n"
                                    f"Use las barras de scroll si es necesario para ver las gráficas completas.")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al cargar gráficas:\n{str(e)}")
    
    def clear_plots(self):
        """Limpiar todas las gráficas"""
        reply = QMessageBox.question(self, 'Confirmar Limpieza', 
                                   '¿Está seguro de que desea limpiar todas las gráficas?',
                                   QMessageBox.Yes | QMessageBox.No, 
                                   QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            # Limpiar gráficas generadas
            for plot_id, plot_data in self.plot_tabs.items():
                plot_data['figure'].clear()
                plot_data['canvas'].draw()
            
            # Limpiar gráficas cargadas
            self.loaded_images_notebook.clear()
            self.loaded_images_group.setVisible(False)
            self.loaded_images.clear()
            
            self.current_plots.clear()
            self.info_text.clear()
            
            self.btn_save.setEnabled(False)
            
            self.plots_status.setText("🧹 GRÁFICAS LIMPIADAS - Listas para nueva generación")
            self.plots_status.setStyleSheet("""
                color: #7f8c8d; 
                font-weight: bold; 
                padding: 15px;
                font-size: 12px;
                background: #f8f9fa;
                border-radius: 8px;
                border: 1px solid #dee2e6;
            """)
            
            QMessageBox.information(self, "Gráficas Limpiadas", 
                                "Todas las gráficas han sido limpiadas.")
    
    def on_activate(self):
        """Método llamado cuando el módulo se activa"""
        self.info_text.clear()
        self.info_text.append("=== MÓDULO DE GRÁFICAS RBF ===\n")
        self.info_text.append("Este módulo genera visualizaciones automáticas:\n")
        self.info_text.append("1. 📈 YD vs YR: Comparación entrenamiento y prueba\n")
        self.info_text.append("2. 📊 Error vs Óptimo: EG vs error objetivo (0.1)\n")
        self.info_text.append("3. 🎯 Dispersión: YD vs YR (diagonal perfecta)\n\n")
        self.info_text.append("NUEVAS CARACTERÍSTICAS:\n")
        self.info_text.append("• 📁 Carga directa de gráficas PNG\n")
        self.info_text.append("• 🖼️ Visualización inmediata sin simulación\n")
        self.info_text.append("• 📊 Área de gráficas más grande\n")
        self.info_text.append("• 💾 Exportación en alta calidad (300 DPI)\n\n")
        self.info_text.append("INSTRUCCIONES:\n")
        self.info_text.append("1. Genere gráficas desde resultados de simulación\n")
        self.info_text.append("2. O cargue gráficas existentes con 'CARGAR GRÁFICAS'\n")
        self.info_text.append("3. Las gráficas cargadas se muestran en sección separada\n")
        
        # Actualizar estado de requisitos
        self.update_requirements_status()