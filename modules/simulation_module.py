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
                             QDialogButtonBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QDoubleValidator

class SimulationModule(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.simulation_results = None
        self.loaded_model = None
        self.custom_test_patterns = []
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
        title = QLabel("🔮 SIMULACIÓN RBF")
        title.setObjectName("title_label")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel#title_label {
                color: #2c3e50;
                font-size: 24px;
                font-weight: bold;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                          stop:0 #9b59b6, stop:1 #8e44ad);
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
        self.create_results_section(scroll_layout)
        self.create_details_section(scroll_layout)
        
        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area)
    
    def create_requirements_section(self, parent_layout):
        """Crear sección de requisitos"""
        requirements_group = QGroupBox("📋 Requisitos para Simulación")
        requirements_layout = QVBoxLayout(requirements_group)
        
        # Información de requisitos
        self.requirements_label = QLabel(
            "Requisitos para la simulación:\n"
            "• Dataset cargado y preprocesado: ❌\n" 
            "• Pesos RBF cargados: ❌\n"
            "• Centros radiales disponibles: ❌"
        )
        self.requirements_label.setStyleSheet("color: #7f8c8d; font-size: 12px; padding: 10px;")
        requirements_layout.addWidget(self.requirements_label)
        
        # Información del modelo cargado
        self.model_info_label = QLabel("Modelo: No cargado")
        self.model_info_label.setStyleSheet("color: #e74c3c; font-weight: bold; padding: 5px;")
        requirements_layout.addWidget(self.model_info_label)
        
        parent_layout.addWidget(requirements_group)
    
    def create_actions_section(self, parent_layout):
        """Crear sección de acciones"""
        actions_group = QGroupBox("🚀 Acciones de Simulación")
        actions_layout = QVBoxLayout(actions_group)
        
        # Primera fila de botones
        row1_frame = QFrame()
        row1_layout = QHBoxLayout(row1_frame)
        
        btn_load_model = QPushButton("📂 Cargar Modelo Entrenado")
        btn_load_model.setMinimumHeight(45)
        btn_load_model.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                          stop:0 #3498db, stop:1 #2980b9);
                color: white;
                font-weight: bold;
                font-size: 14px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                          stop:0 #2980b9, stop:1 #2471a3);
            }
        """)
        btn_load_model.clicked.connect(self.load_trained_model)
        
        btn_simulate = QPushButton("🎯 Ejecutar Simulación")
        btn_simulate.setMinimumHeight(45)
        btn_simulate.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                          stop:0 #27ae60, stop:1 #2ecc71);
                color: white;
                font-weight: bold;
                font-size: 14px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                          stop:0 #219653, stop:1 #27ae60);
            }
        """)
        btn_simulate.clicked.connect(self.run_simulation)
        
        row1_layout.addWidget(btn_load_model)
        row1_layout.addWidget(btn_simulate)
        
        # Segunda fila de botones
        row2_frame = QFrame()
        row2_layout = QHBoxLayout(row2_frame)
        
        btn_add_pattern = QPushButton("➕ Agregar Patrón Prueba")
        btn_add_pattern.setMinimumHeight(45)
        btn_add_pattern.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                          stop:0 #f39c12, stop:1 #e67e22);
                color: white;
                font-weight: bold;
                font-size: 14px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                          stop:0 #e67e22, stop:1 #d35400);
            }
        """)
        btn_add_pattern.clicked.connect(self.add_custom_test_pattern)
        
        btn_save_results = QPushButton("💾 Guardar Simulación")
        btn_save_results.setMinimumHeight(45)
        btn_save_results.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                          stop:0 #9b59b6, stop:1 #8e44ad);
                color: white;
                font-weight: bold;
                font-size: 14px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                          stop:0 #8e44ad, stop:1 #7d3c98);
            }
        """)
        btn_save_results.clicked.connect(self.save_simulation_results)
        
        row2_layout.addWidget(btn_add_pattern)
        row2_layout.addWidget(btn_save_results)
        
        # Tercera fila de botones
        row3_frame = QFrame()
        row3_layout = QHBoxLayout(row3_frame)
        
        btn_clear = QPushButton("🗑️ Limpiar Simulación")
        btn_clear.setMinimumHeight(45)
        btn_clear.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                          stop:0 #e74c3c, stop:1 #c0392b);
                color: white;
                font-weight: bold;
                font-size: 14px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                          stop:0 #c0392b, stop:1 #a93226);
            }
        """)
        btn_clear.clicked.connect(self.clear_simulation)
        
        row3_layout.addWidget(btn_clear)
        row3_layout.addStretch()
        
        # Agregar filas al layout principal
        actions_layout.addWidget(row1_frame)
        actions_layout.addWidget(row2_frame)
        actions_layout.addWidget(row3_frame)
        
        # Estado de la simulación
        self.simulation_status = QLabel("⏳ Esperando configuración de simulación...")
        self.simulation_status.setAlignment(Qt.AlignCenter)
        self.simulation_status.setStyleSheet("padding: 10px; font-weight: bold;")
        actions_layout.addWidget(self.simulation_status)
        
        parent_layout.addWidget(actions_group)
    
    def create_results_section(self, parent_layout):
        """Crear sección de resultados"""
        results_group = QGroupBox("📊 Resultados de la Simulación")
        results_layout = QVBoxLayout(results_group)
        
        # Notebook para pestañas de resultados
        self.results_notebook = QTabWidget()
        
        # Pestaña de métricas comparativas
        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(metrics_tab)
        
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setStyleSheet("font-family: 'Consolas', 'Monaco', monospace; font-size: 10pt;")
        self.metrics_text.setMinimumHeight(200)
        metrics_layout.addWidget(self.metrics_text)
        
        # Pestaña de resultados detallados
        details_tab = QWidget()
        details_layout = QVBoxLayout(details_tab)
        
        self.results_table = QTableWidget()
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setMinimumHeight(300)
        details_layout.addWidget(self.results_table)
        
        # Pestaña de patrones personalizados
        custom_tab = QWidget()
        custom_layout = QVBoxLayout(custom_tab)
        
        self.custom_patterns_table = QTableWidget()
        self.custom_patterns_table.setAlternatingRowColors(True)
        self.custom_patterns_table.setMinimumHeight(200)
        custom_layout.addWidget(self.custom_patterns_table)
        
        # Agregar pestañas
        self.results_notebook.addTab(metrics_tab, "📈 Métricas Comparativas")
        self.results_notebook.addTab(details_tab, "🔍 Resultados Detallados")
        self.results_notebook.addTab(custom_tab, "➕ Patrones Personalizados")
        
        results_layout.addWidget(self.results_notebook)
        parent_layout.addWidget(results_group)
    
    def create_details_section(self, parent_layout):
        """Crear sección de detalles de la simulación"""
        details_group = QGroupBox("📋 Información de la Simulación")
        details_layout = QVBoxLayout(details_group)
        
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setStyleSheet("font-family: 'Consolas', 'Monaco', monospace; font-size: 9pt;")
        self.details_text.setMinimumHeight(200)
        details_layout.addWidget(self.details_text)
        
        parent_layout.addWidget(details_group)
    
    def update_requirements_status(self):
        """Actualizar el estado de los requisitos"""
        has_data = self.main_window.preprocessed_data is not None
        has_weights = (self.main_window.training_results is not None or 
                      self.loaded_model is not None)
        has_centers = has_weights  # Si hay pesos, hay centros
        
        data_status = "✅" if has_data else "❌"
        weights_status = "✅" if has_weights else "❌"
        centers_status = "✅" if has_centers else "❌"
        
        requirements_text = (
            f"Requisitos para la simulación:\n"
            f"• Dataset cargado y preprocesado: {data_status}\n" 
            f"• Pesos RBF cargados: {weights_status}\n"
            f"• Centros radiales disponibles: {centers_status}"
        )
        
        self.requirements_label.setText(requirements_text)
        
        # Actualizar estado de simulación
        if has_data and has_weights:
            self.simulation_status.setText("✅ Listo para simular - Todos los requisitos cumplidos")
            self.simulation_status.setStyleSheet("color: #27ae60; font-weight: bold; padding: 10px;")
        else:
            self.simulation_status.setText("⚠️ Complete los requisitos primero")
            self.simulation_status.setStyleSheet("color: #f39c12; font-weight: bold; padding: 10px;")
    
    def load_trained_model(self):
        """Cargar modelo entrenado desde archivo"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Seleccionar modelo entrenado", "", 
                "Archivos de modelo (*.pkl);;Todos los archivos (*)"
            )
            
            if file_path:
                with open(file_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.loaded_model = model_data
                
                # Extraer información del modelo
                training_results = model_data['training_results']
                rbf_config = model_data['rbf_config']
                preprocessed_info = model_data['preprocessed_data_info']
                
                # Actualizar información del modelo
                model_info = (
                    f"Modelo cargado: {os.path.basename(file_path)}\n"
                    f"• Centros radiales: {training_results['num_centers']}\n"
                    f"• Fecha entrenamiento: {training_results['training_date']}\n"
                    f"• Error de entrenamiento: {training_results['errors']['eg']:.6f}\n"
                    f"• Convergencia: {'✅ CONVERGE' if training_results['convergence'] else '❌ NO CONVERGE'}"
                )
                
                self.model_info_label.setText(model_info)
                self.model_info_label.setStyleSheet("color: #27ae60; font-weight: bold; padding: 5px;")
                
                # Actualizar requisitos
                self.update_requirements_status()
                
                self.details_text.append(f"✅ Modelo cargado exitosamente: {os.path.basename(file_path)}")
                QMessageBox.information(self, "Modelo Cargado", "Modelo entrenado cargado exitosamente.")
                
        except Exception as e:
            error_msg = f"❌ Error al cargar modelo: {str(e)}"
            self.details_text.append(error_msg)
            QMessageBox.critical(self, "Error", f"No se pudo cargar el modelo:\n{str(e)}")
    
    def get_current_model(self):
        """Obtener el modelo actual (desde entrenamiento o cargado)"""
        if self.main_window.training_results is not None:
            return {
                'training_results': self.main_window.training_results,
                'rbf_config': self.main_window.rbf_config,
                'preprocessed_data_info': {
                    'input_columns': self.main_window.preprocessed_data.get('input_columns', []),
                    'output_columns': self.main_window.preprocessed_data.get('output_columns', []),
                    'X_train_shape': self.main_window.preprocessed_data['X_train'].shape
                }
            }
        elif self.loaded_model is not None:
            return self.loaded_model
        else:
            return None
    
    def radial_basis_function(self, d):
        """Función de base radial - misma que en entrenamiento"""
        # Usar la misma función que en el entrenamiento para consistencia
        d_safe = np.where(d < 1e-10, 1e-10, d)
        sigma = np.mean(d_safe)
        if sigma < 1e-10:
            sigma = 1.0
        activations = np.exp(-d_safe**2 / (2 * sigma**2))
        
        if np.any(np.isnan(activations)) or np.any(np.isinf(activations)):
            c = 1.0
            activations = np.sqrt(d_safe**2 + c**2)
        
        return activations
    
    def calculate_euclidean_distance(self, X, centers):
        """Calcular distancias euclidianas - misma que en entrenamiento"""
        n_patterns = X.shape[0]
        n_centers = centers.shape[0]
        distances = np.zeros((n_patterns, n_centers))
        
        for j in range(n_centers):
            diff = X - centers[j]
            distances[:, j] = np.sqrt(np.sum(diff**2, axis=1))
        
        return distances
    
    def build_interpolation_matrix(self, activations):
        """Construir matriz de interpolación - misma que en entrenamiento"""
        n_patterns = activations.shape[0]
        ones_column = np.ones((n_patterns, 1))
        A = np.hstack([ones_column, activations])
        return A
    
    def calculate_predictions(self, A, weights):
        """Calcular predicciones: ŷ = A * W"""
        return np.dot(A, weights)
    
    def calculate_errors(self, y_true, y_pred):
        """Calcular métricas de error"""
        absolute_errors = np.abs(y_true - y_pred)
        eg = np.mean(absolute_errors)
        mae = np.mean(absolute_errors)
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        
        return {
            'eg': eg,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'absolute_errors': absolute_errors
        }
    
    def run_simulation(self):
        """Ejecutar simulación completa"""
        if not self.verify_simulation_requirements():
            return
        
        try:
            self.simulation_status.setText("🔄 Ejecutando simulación...")
            self.details_text.clear()
            self.metrics_text.clear()
            
            # Obtener modelo actual
            model = self.get_current_model()
            training_results = model['training_results']
            rbf_config = model['rbf_config']
            preprocessed_info = model['preprocessed_data_info']
            
            # Obtener datos
            X_train = self.main_window.preprocessed_data['X_train']
            X_test = self.main_window.preprocessed_data['X_test']
            y_train = self.main_window.preprocessed_data['y_train']
            y_test = self.main_window.preprocessed_data['y_test']
            
            centers = training_results['centers']
            weights = training_results['weights']
            target_error = training_results['target_error']
            
            self.details_text.append("=== INFORMACIÓN DE LA SIMULACIÓN ===\n")
            self.details_text.append(f"• Dataset: {len(preprocessed_info['input_columns'])} entradas, 1 salida\n")
            self.details_text.append(f"• Centros radiales: {training_results['num_centers']}\n")
            self.details_text.append(f"• Patrones entrenamiento: {X_train.shape[0]}\n")
            self.details_text.append(f"• Patrones prueba: {X_test.shape[0]}\n")
            self.details_text.append(f"• Error objetivo: {target_error}\n")
            self.details_text.append(f"• Fecha modelo: {training_results['training_date']}\n")
            
            # SIMULACIÓN EN CONJUNTO DE ENTRENAMIENTO
            self.details_text.append("\n=== SIMULACIÓN ENTRENAMIENTO ===\n")
            
            # Calcular activaciones para entrenamiento
            distances_train = self.calculate_euclidean_distance(X_train, centers)
            activations_train = self.radial_basis_function(distances_train)
            A_train = self.build_interpolation_matrix(activations_train)
            
            # Calcular predicciones para entrenamiento
            y_pred_train = self.calculate_predictions(A_train, weights)
            errors_train = self.calculate_errors(y_train, y_pred_train)
            
            self.details_text.append(f"• Error General (EG): {errors_train['eg']:.6f}\n")
            self.details_text.append(f"• Convergencia: {'✅ CONVERGE' if errors_train['eg'] <= target_error else '❌ NO CONVERGE'}\n")
            
            # SIMULACIÓN EN CONJUNTO DE PRUEBA
            self.details_text.append("\n=== SIMULACIÓN PRUEBA ===\n")
            
            # Calcular activaciones para prueba
            distances_test = self.calculate_euclidean_distance(X_test, centers)
            activations_test = self.radial_basis_function(distances_test)
            A_test = self.build_interpolation_matrix(activations_test)
            
            # Calcular predicciones para prueba
            y_pred_test = self.calculate_predictions(A_test, weights)
            errors_test = self.calculate_errors(y_test, y_pred_test)
            
            self.details_text.append(f"• Error General (EG): {errors_test['eg']:.6f}\n")
            
            # SIMULACIÓN EN PATRONES PERSONALIZADOS
            custom_predictions = []
            if self.custom_test_patterns:
                self.details_text.append("\n=== SIMULACIÓN PATRONES PERSONALIZADOS ===\n")
                X_custom = np.array(self.custom_test_patterns)
                distances_custom = self.calculate_euclidean_distance(X_custom, centers)
                activations_custom = self.radial_basis_function(distances_custom)
                A_custom = self.build_interpolation_matrix(activations_custom)
                y_pred_custom = self.calculate_predictions(A_custom, weights)
                
                for i, (pattern, prediction) in enumerate(zip(self.custom_test_patterns, y_pred_custom)):
                    custom_predictions.append({
                        'patron': f"Personalizado_{i+1}",
                        'entradas': pattern,
                        'prediccion': prediction
                    })
                    self.details_text.append(f"• Patrón {i+1}: {pattern} → {prediction:.6f}\n")
            
            # Guardar resultados
            self.simulation_results = {
                'train': {
                    'y_true': y_train,
                    'y_pred': y_pred_train,
                    'errors': errors_train,
                    'convergence': errors_train['eg'] <= target_error
                },
                'test': {
                    'y_true': y_test,
                    'y_pred': y_pred_test,
                    'errors': errors_test
                },
                'custom': custom_predictions,
                'model_info': {
                    'num_centers': training_results['num_centers'],
                    'target_error': target_error,
                    'training_date': training_results['training_date'],
                    'input_columns': preprocessed_info['input_columns'],
                    'dataset_info': f"{X_train.shape[1]} entradas, 1 salida"
                }
            }
            
            # Actualizar interfaz
            self.update_results_display()
            self.simulation_status.setText("✅ Simulación completada exitosamente")
            self.simulation_status.setStyleSheet("color: #27ae60; font-weight: bold; padding: 10px;")
            
            # Mostrar resumen
            self.show_simulation_summary(errors_train, errors_test)
            
        except Exception as e:
            self.simulation_status.setText("❌ Error en la simulación")
            self.simulation_status.setStyleSheet("color: #c0392b; font-weight: bold; padding: 10px;")
            QMessageBox.critical(self, "Error", f"Error durante la simulación:\n{str(e)}")
    
    def verify_simulation_requirements(self):
        """Verificar requisitos para simulación"""
        if self.main_window.preprocessed_data is None:
            QMessageBox.warning(self, "Datos Requeridos", 
                              "Primero cargue un dataset en el módulo 'Carga de Datos'")
            return False
        
        if self.get_current_model() is None:
            QMessageBox.warning(self, "Modelo Requerido", 
                              "Primero cargue un modelo entrenado o realice un entrenamiento")
            return False
        
        return True
    
    def update_results_display(self):
        """Actualizar visualización de resultados"""
        if self.simulation_results is None:
            return
        
        # Actualizar métricas comparativas
        self.update_metrics_display()
        
        # Actualizar tabla de resultados detallados
        self.update_results_table()
        
        # Actualizar tabla de patrones personalizados
        self.update_custom_patterns_table()
    
    def update_metrics_display(self):
        """Actualizar display de métricas comparativas"""
        train_errors = self.simulation_results['train']['errors']
        test_errors = self.simulation_results['test']['errors']
        model_info = self.simulation_results['model_info']
        
        convergence_status = "✅ CONVERGE" if self.simulation_results['train']['convergence'] else "❌ NO CONVERGE"
        
        metrics_text = f"""=== MÉTRICAS COMPARATIVAS DE SIMULACIÓN ===

INFORMACIÓN DEL MODELO:
• Centros radiales: {model_info['num_centers']}
• Dataset: {model_info['dataset_info']}
• Error objetivo: {model_info['target_error']}
• Fecha modelo: {model_info['training_date']}
• Convergencia: {convergence_status}

MÉTRICAS - CONJUNTO DE ENTRENAMIENTO:
• Error General (EG): {train_errors['eg']:.8f}
• Error Absoluto Medio (MAE): {train_errors['mae']:.8f}
• Raíz del Error Cuadrático Medio (RMSE): {train_errors['rmse']:.8f}

MÉTRICAS - CONJUNTO DE PRUEBA:
• Error General (EG): {test_errors['eg']:.8f}
• Error Absoluto Medio (MAE): {test_errors['mae']:.8f}
• Raíz del Error Cuadrático Medio (RMSE): {test_errors['rmse']:.8f}

VERIFICACIÓN DE CONVERGENCIA:
• EG entrenamiento ({train_errors['eg']:.6f}) {'≤' if train_errors['eg'] <= model_info['target_error'] else '>'} Error objetivo ({model_info['target_error']})
• Estado: {convergence_status}
"""
        
        # Agregar información de patrones personalizados si existen
        if self.simulation_results['custom']:
            metrics_text += "\nPATRONES PERSONALIZADOS:\n"
            for custom in self.simulation_results['custom']:
                metrics_text += f"• {custom['patron']}: {custom['prediccion']:.6f}\n"
        
        self.metrics_text.setText(metrics_text)
    
    def update_results_table(self):
        """Actualizar tabla de resultados detallados"""
        # Mostrar resultados de prueba (más representativos para simulación)
        y_true = self.simulation_results['test']['y_true']
        y_pred = self.simulation_results['test']['y_pred']
        absolute_errors = self.simulation_results['test']['errors']['absolute_errors']
        
        # Configurar tabla
        self.results_table.setRowCount(len(y_true))
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels([
            "Patrón #", "Entradas", "YD (Deseado)", "YR (Red)", "Error |YD-YR|"
        ])
        
        # Llenar datos
        for i in range(len(y_true)):
            # Número de patrón
            pattern_item = QTableWidgetItem(f"P{i+1}")
            pattern_item.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(i, 0, pattern_item)
            
            # Información de entradas (resumida)
            input_columns = self.simulation_results['model_info']['input_columns']
            inputs_info = f"{len(input_columns)} entradas" if len(input_columns) > 3 else ", ".join(input_columns)
            inputs_item = QTableWidgetItem(inputs_info)
            self.results_table.setItem(i, 1, inputs_item)
            
            # YD (Deseado)
            yd_item = QTableWidgetItem(f"{y_true[i]:.6f}")
            yd_item.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(i, 2, yd_item)
            
            # YR (Red)
            yr_item = QTableWidgetItem(f"{y_pred[i]:.6f}")
            yr_item.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(i, 3, yr_item)
            
            # Error
            error_item = QTableWidgetItem(f"{absolute_errors[i]:.6f}")
            error_item.setTextAlignment(Qt.AlignCenter)
            # Colorear según magnitud del error
            if absolute_errors[i] > 0.5:
                error_item.setBackground(Qt.red)
            elif absolute_errors[i] > 0.1:
                error_item.setBackground(Qt.yellow)
            else:
                error_item.setBackground(Qt.green)
            self.results_table.setItem(i, 4, error_item)
        
        # Ajustar tamaño de columnas
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    
    def update_custom_patterns_table(self):
        """Actualizar tabla de patrones personalizados"""
        if not self.simulation_results['custom']:
            self.custom_patterns_table.setRowCount(0)
            self.custom_patterns_table.setColumnCount(0)
            return
        
        custom_data = self.simulation_results['custom']
        
        # Configurar tabla
        self.custom_patterns_table.setRowCount(len(custom_data))
        self.custom_patterns_table.setColumnCount(3)
        self.custom_patterns_table.setHorizontalHeaderLabels([
            "Patrón", "Entradas", "YR (Predicción)"
        ])
        
        # Llenar datos
        for i, custom in enumerate(custom_data):
            # Nombre del patrón
            pattern_item = QTableWidgetItem(custom['patron'])
            pattern_item.setTextAlignment(Qt.AlignCenter)
            self.custom_patterns_table.setItem(i, 0, pattern_item)
            
            # Entradas
            inputs_text = ", ".join([f"{x:.4f}" for x in custom['entradas']])
            inputs_item = QTableWidgetItem(inputs_text)
            self.custom_patterns_table.setItem(i, 1, inputs_item)
            
            # Predicción
            prediction_item = QTableWidgetItem(f"{custom['prediccion']:.6f}")
            prediction_item.setTextAlignment(Qt.AlignCenter)
            self.custom_patterns_table.setItem(i, 2, prediction_item)
        
        # Ajustar tamaño de columnas
        self.custom_patterns_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    
    def add_custom_test_pattern(self):
        """Agregar patrón de prueba personalizado"""
        if self.main_window.preprocessed_data is None:
            QMessageBox.warning(self, "Datos Requeridos", "Primero cargue un dataset")
            return
        
        try:
            num_inputs = self.main_window.preprocessed_data['X_train'].shape[1]
            input_columns = self.main_window.preprocessed_data.get('input_columns', [f"Entrada_{i+1}" for i in range(num_inputs)])
            
            # Crear diálogo para ingresar valores
            dialog = QDialog(self)
            dialog.setWindowTitle("Agregar Patrón de Prueba Personalizado")
            dialog.setMinimumWidth(400)
            
            layout = QFormLayout(dialog)
            
            # Crear campos de entrada para cada variable
            input_fields = []
            for i in range(num_inputs):
                field = QLineEdit()
                field.setValidator(QDoubleValidator())
                field.setPlaceholderText(f"Valor para {input_columns[i]}")
                layout.addRow(f"{input_columns[i]}:", field)
                input_fields.append(field)
            
            # Botones
            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            layout.addRow(button_box)
            
            if dialog.exec_() == QDialog.Accepted:
                # Recoger valores
                pattern = []
                for field in input_fields:
                    value = field.text().strip()
                    if value:
                        pattern.append(float(value))
                    else:
                        pattern.append(0.0)  # Valor por defecto
                
                self.custom_test_patterns.append(pattern)
                self.details_text.append(f"➕ Patrón personalizado agregado: {pattern}")
                
                # Si hay simulación previa, actualizar
                if self.simulation_results is not None:
                    self.details_text.append("⚠️ Ejecute la simulación nuevamente para incluir este patrón")
                
                QMessageBox.information(self, "Patrón Agregado", 
                                      f"Patrón personalizado agregado exitosamente:\n{pattern}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al agregar patrón:\n{str(e)}")
    
    def show_simulation_summary(self, train_errors, test_errors):
        """Mostrar resumen de la simulación"""
        model_info = self.simulation_results['model_info']
        convergence_status = "✅ CONVERGE" if self.simulation_results['train']['convergence'] else "❌ NO CONVERGE"
        
        summary = f"""
{convergence_status}

🎯 RESUMEN DE SIMULACIÓN:

📊 MÉTRICAS PRINCIPALES:
• Entrenamiento - EG: {train_errors['eg']:.6f}
• Prueba - EG: {test_errors['eg']:.6f}
• Error objetivo: {model_info['target_error']}

🔧 CONFIGURACIÓN:
• Centros radiales: {model_info['num_centers']}
• Dataset: {model_info['dataset_info']}
• Patrones personalizados: {len(self.simulation_results['custom'])}

La simulación se completó exitosamente. Puede guardar los resultados o agregar más patrones de prueba.
"""
        
        QMessageBox.information(self, "Simulación Completada", summary)
    
    def save_simulation_results(self):
        """Guardar resultados de la simulación"""
        if self.simulation_results is None:
            QMessageBox.warning(self, "Advertencia", "Primero ejecute una simulación")
            return
        
        try:
            # Crear carpeta de resultados si no existe
            results_dir = "resultados"
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            
            # Crear subcarpeta con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            simulation_dir = os.path.join(results_dir, f"simulacion_{timestamp}")
            os.makedirs(simulation_dir)
            
            model_info = self.simulation_results['model_info']
            train_errors = self.simulation_results['train']['errors']
            test_errors = self.simulation_results['test']['errors']
            
            # 1. Guardar métricas en JSON - VERSIÓN CORREGIDA
            metrics_path = os.path.join(simulation_dir, "metricas_simulacion.json")
            
            # CORRECCIÓN: Convertir booleanos a strings
            convergence_str = "CONVERGE" if self.simulation_results['train']['convergence'] else "NO_CONVERGE"
            
            metrics_data = {
                'fecha_simulacion': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'modelo': {
                    'num_centros': int(model_info['num_centers']),
                    'error_objetivo': float(model_info['target_error']),
                    'fecha_entrenamiento': model_info['training_date'],
                    'entradas': len(model_info['input_columns']),
                    'columnas_entrada': model_info['input_columns'],
                    'columna_salida': model_info.get('output_columns', ['Y'])
                },
                'metricas_entrenamiento': {
                    'eg': float(train_errors['eg']),
                    'mae': float(train_errors['mae']),
                    'rmse': float(train_errors['rmse']),
                    'convergencia': convergence_str  # ✅ CORREGIDO: string en lugar de bool
                },
                'metricas_prueba': {
                    'eg': float(test_errors['eg']),
                    'mae': float(test_errors['mae']),
                    'rmse': float(test_errors['rmse'])
                },
                'patrones_personalizados': len(self.simulation_results['custom']),
                'resumen_convergencia': f"EG entrenamiento ({train_errors['eg']:.6f}) {'≤' if train_errors['eg'] <= model_info['target_error'] else '>'} Error objetivo ({model_info['target_error']})"
            }
            
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, indent=4, ensure_ascii=False)
            
            # 2. Guardar resultados detallados en CSV
            results_path = os.path.join(simulation_dir, "resultados_detallados.csv")
            results_data = []
            
            # Datos de prueba
            y_true_test = self.simulation_results['test']['y_true']
            y_pred_test = self.simulation_results['test']['y_pred']
            errors_test = self.simulation_results['test']['errors']['absolute_errors']
            
            for i in range(len(y_true_test)):
                results_data.append({
                    'conjunto': 'prueba',
                    'patron': f"P{i+1}",
                    'yd_deseado': float(y_true_test[i]),
                    'yr_red': float(y_pred_test[i]),
                    'error_absoluto': float(errors_test[i]),
                    'clasificacion_error': 'BAJO' if errors_test[i] <= 0.1 else 'MEDIO' if errors_test[i] <= 0.5 else 'ALTO'
                })
            
            # Datos personalizados
            for custom in self.simulation_results['custom']:
                results_data.append({
                    'conjunto': 'personalizado',
                    'patron': custom['patron'],
                    'entradas': str(custom['entradas']),  # Convertir lista a string
                    'yd_deseado': '',
                    'yr_red': float(custom['prediccion']),
                    'error_absoluto': '',
                    'clasificacion_error': 'N/A'
                })
            
            results_df = pd.DataFrame(results_data)
            results_df.to_csv(results_path, index=False, encoding='utf-8')
            
            # 3. Guardar resumen en texto
            summary_path = os.path.join(simulation_dir, "resumen_simulacion.txt")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("=== RESUMEN DE SIMULACIÓN RBF ===\n\n")
                f.write(self.metrics_text.toPlainText())
                f.write("\n\n=== DETALLES ===\n\n")
                f.write(self.details_text.toPlainText())
            
            # 4. Guardar información de patrones personalizados (opcional)
            if self.simulation_results['custom']:
                custom_path = os.path.join(simulation_dir, "patrones_personalizados.json")
                custom_data = []
                
                for custom in self.simulation_results['custom']:
                    custom_data.append({
                        'patron': custom['patron'],
                        'entradas': [float(x) for x in custom['entradas']],
                        'prediccion': float(custom['prediccion']),
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                
                with open(custom_path, 'w', encoding='utf-8') as f:
                    json.dump(custom_data, f, indent=4, ensure_ascii=False)
            
            # Mensaje de confirmación mejorado
            archivos_creados = [
                "• metricas_simulacion.json (métricas principales)",
                "• resultados_detallados.csv (resultados completos)",
                "• resumen_simulacion.txt (resumen ejecutivo)"
            ]
            
            if self.simulation_results['custom']:
                archivos_creados.append("• patrones_personalizados.json (patrones custom)")
            
            QMessageBox.information(self, "Simulación Guardada", 
                                f"✅ Resultados de simulación guardados en:\n{simulation_dir}\n\n"
                                f"Archivos creados:\n" + "\n".join(archivos_creados))
            
            self.simulation_status.setText(f"💾 Simulación guardada en {simulation_dir}")
            self.details_text.append(f"💾 Resultados guardados en: {simulation_dir}")
            
        except Exception as e:
            error_msg = f"❌ Error al guardar simulación: {str(e)}"
            self.details_text.append(error_msg)
            QMessageBox.critical(self, "Error", f"No se pudieron guardar los resultados:\n{str(e)}")
    
    def clear_simulation(self):
        """Limpiar resultados de la simulación"""
        reply = QMessageBox.question(self, 'Confirmar Limpieza', 
                                   '¿Está seguro de que desea limpiar todos los resultados de la simulación?',
                                   QMessageBox.Yes | QMessageBox.No, 
                                   QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.simulation_results = None
            self.custom_test_patterns = []
            
            # Limpiar interfaces
            self.metrics_text.clear()
            self.results_table.setRowCount(0)
            self.results_table.setColumnCount(0)
            self.custom_patterns_table.setRowCount(0)
            self.custom_patterns_table.setColumnCount(0)
            self.details_text.clear()
            
            self.simulation_status.setText("🧹 Simulación limpiada - Lista para nueva simulación")
            self.simulation_status.setStyleSheet("color: #7f8c8d; font-weight: bold; padding: 10px;")
            
            QMessageBox.information(self, "Simulación Limpiada", 
                                "Todos los resultados de la simulación han sido limpiados.")
    
    def on_activate(self):
        """Método llamado cuando el módulo se activa"""
        self.details_text.clear()
        self.details_text.append("=== MÓDULO DE SIMULACIÓN RBF ===\n")
        self.details_text.append("Este módulo permite:\n")
        self.details_text.append("• 📂 Cargar modelos entrenados previamente\n")
        self.details_text.append("• 🎯 Ejecutar simulaciones en datos de prueba\n")
        self.details_text.append("• ➕ Agregar patrones de prueba personalizados\n")
        self.details_text.append("• 📊 Calcular métricas comparativas (EG, MAE, RMSE)\n")
        self.details_text.append("• 🔍 Verificar convergencia del modelo\n")
        self.details_text.append("• 💾 Guardar resultados de simulación\n\n")
        self.details_text.append("REQUISITOS:\n")
        self.details_text.append("• Dataset cargado y preprocesado\n")
        self.details_text.append("• Modelo RBF entrenado (desde entrenamiento o archivo)\n")
        
        # Actualizar estado de requisitos
        self.update_requirements_status()