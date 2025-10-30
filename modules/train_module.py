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
                             QSplitter, QTabWidget, QApplication, QFileDialog)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class TrainingModule(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.training_results = None
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
        title = QLabel("🧠 ENTRENAMIENTO RBF")
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
        self.create_training_section(scroll_layout)
        self.create_results_section(scroll_layout)
        self.create_details_section(scroll_layout)
        
        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area)
    
    def create_training_section(self, parent_layout):
        """Crear sección de entrenamiento"""
        training_group = QGroupBox("🚀 Entrenamiento de la Red RBF")
        training_layout = QVBoxLayout(training_group)
        
        # Información de requisitos
        requirements_frame = QFrame()
        requirements_layout = QVBoxLayout(requirements_frame)
        
        requirements_label = QLabel(
            "Requisitos para el entrenamiento:\n"
            "• Datos cargados y preprocesados ✓\n" 
            "• Configuración RBF guardada ✓\n"
            "• Centros radiales inicializados ✓"
        )
        requirements_label.setStyleSheet("color: #7f8c8d; font-size: 12px; padding: 10px;")
        requirements_layout.addWidget(requirements_label)
        
        training_layout.addWidget(requirements_frame)
        
        # Botones de entrenamiento
        buttons_frame = QFrame()
        buttons_layout = QHBoxLayout(buttons_frame)
        
        btn_train = QPushButton("🎯 Iniciar Entrenamiento RBF")
        btn_train.setMinimumHeight(50)
        btn_train.setStyleSheet("""
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
        btn_train.clicked.connect(self.start_training)
        
        btn_save = QPushButton("💾 Guardar Entrenamiento")
        btn_save.setMinimumHeight(50)
        btn_save.setStyleSheet("""
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
        btn_save.clicked.connect(self.save_training)
        
        btn_reset = QPushButton("🔄 Limpiar Entrenamiento")
        btn_reset.setMinimumHeight(50)
        btn_reset.setStyleSheet("""
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
        btn_reset.clicked.connect(self.reset_training)
        
        buttons_layout.addWidget(btn_train)
        buttons_layout.addWidget(btn_save)
        buttons_layout.addWidget(btn_reset)
        
        training_layout.addWidget(buttons_frame)
        
        # Barra de progreso
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        training_layout.addWidget(self.progress_bar)
        
        # Estado del entrenamiento
        self.training_status = QLabel("⏳ Esperando inicio del entrenamiento...")
        self.training_status.setAlignment(Qt.AlignCenter)
        self.training_status.setStyleSheet("padding: 10px; font-weight: bold;")
        training_layout.addWidget(self.training_status)
        
        parent_layout.addWidget(training_group)
    
    def create_results_section(self, parent_layout):
        """Crear sección de resultados"""
        results_group = QGroupBox("📊 Resultados del Entrenamiento")
        results_layout = QVBoxLayout(results_group)
        
        # Notebook para pestañas de resultados
        self.results_notebook = QTabWidget()
        
        # Pestaña de pesos
        weights_tab = QWidget()
        weights_layout = QVBoxLayout(weights_tab)
        
        self.weights_table = QTableWidget()
        self.weights_table.setAlternatingRowColors(True)
        self.weights_table.setMinimumHeight(200)
        weights_layout.addWidget(self.weights_table)
        
        # Pestaña de matriz de activaciones
        activations_tab = QWidget()
        activations_layout = QVBoxLayout(activations_tab)
        
        self.activations_table = QTableWidget()
        self.activations_table.setAlternatingRowColors(True)
        self.activations_table.setMinimumHeight(200)
        activations_layout.addWidget(self.activations_table)
        
        # Pestaña de centros finales
        centers_tab = QWidget()
        centers_layout = QVBoxLayout(centers_tab)
        
        self.centers_table = QTableWidget()
        self.centers_table.setAlternatingRowColors(True)
        self.centers_table.setMinimumHeight(200)
        centers_layout.addWidget(self.centers_table)
        
        # Pestaña de errores
        errors_tab = QWidget()
        errors_layout = QVBoxLayout(errors_tab)
        
        self.errors_text = QTextEdit()
        self.errors_text.setReadOnly(True)
        self.errors_text.setStyleSheet("font-family: 'Consolas', 'Monaco', monospace; font-size: 10pt;")
        errors_layout.addWidget(self.errors_text)
        
        # Agregar pestañas
        self.results_notebook.addTab(weights_tab, "⚖️ Pesos Finales")
        self.results_notebook.addTab(centers_tab, "🎯 Centros Finales")
        self.results_notebook.addTab(activations_tab, "🔢 Activaciones")
        self.results_notebook.addTab(errors_tab, "📈 Errores y Métricas")
        
        results_layout.addWidget(self.results_notebook)
        parent_layout.addWidget(results_group)
    
    def create_details_section(self, parent_layout):
        """Crear sección de detalles del entrenamiento"""
        details_group = QGroupBox("📋 Detalles del Proceso de Entrenamiento")
        details_layout = QVBoxLayout(details_group)
        
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setStyleSheet("font-family: 'Consolas', 'Monaco', monospace; font-size: 9pt;")
        self.details_text.setMinimumHeight(300)
        details_layout.addWidget(self.details_text)
        
        parent_layout.addWidget(details_group)
    
    def radial_basis_function(self, d):
        """Función de base radial MEJORADA y ESTABLE"""
        # Evitar problemas con ln(0) y valores negativos
        d_safe = np.where(d < 1e-10, 1e-10, d)
        
        # CORRECCIÓN: Mejorar estabilidad numérica
        # Para d < 1, ln(d) es negativo, lo que causa problemas
        # Usamos una versión estabilizada
        
        # Opción A: Gaussiana (recomendada para estabilidad)
        sigma = np.mean(d_safe)  # Sigma adaptativo basado en distancias promedio
        if sigma < 1e-10:
            sigma = 1.0  # Valor por defecto si las distancias son muy pequeñas
        
        activations = np.exp(-d_safe**2 / (2 * sigma**2))
        
        # Verificar que no hay problemas numéricos
        if np.any(np.isnan(activations)) or np.any(np.isinf(activations)):
            # Fallback: función multiquadrática
            c = 1.0
            activations = np.sqrt(d_safe**2 + c**2)
        
        return activations
    
    def calculate_euclidean_distance(self, X, centers):
        """Calcular distancias euclidianas - OPTIMIZADA"""
        n_patterns = X.shape[0]
        n_centers = centers.shape[0]
        
        # Usar broadcasting para cálculo vectorizado (más eficiente y preciso)
        distances = np.zeros((n_patterns, n_centers))
        
        for j in range(n_centers):
            # Calcular distancia entre todos los patrones y el centro j
            diff = X - centers[j]
            distances[:, j] = np.sqrt(np.sum(diff**2, axis=1))
        
        return distances
    
    def build_interpolation_matrix(self, activations):
        """Construir matriz de interpolación A agregando columna de 1's (umbral)"""
        n_patterns = activations.shape[0]
        
        # Agregar columna de 1's para el umbral (W0) - EXACTO como en el ejemplo
        ones_column = np.ones((n_patterns, 1))
        A = np.hstack([ones_column, activations])
        
        return A
    
    def solve_weights(self, A, y):
        """Resolver pesos - VERSIÓN MEJORADA con múltiples métodos de respaldo"""
        try:
            # DEBUG: Información de la matriz A
            self.details_text.append(f"\n🔍 DEBUG MATRIZ A:")
            self.details_text.append(f"   - Forma: {A.shape}")
            self.details_text.append(f"   - Rango: {np.linalg.matrix_rank(A)}")
            self.details_text.append(f"   - Condición: {np.linalg.cond(A):.2e}")
            
            # Método 1: np.linalg.lstsq (MÁS ROBUSTO)
            try:
                weights, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
                self.details_text.append(f"   - LSTSQ - Rank: {rank}, W₀: {weights[0]:.6f}")
                
                if rank < A.shape[1]:
                    self.details_text.append("   ⚠️ Matriz de rango incompleto")
                
                return weights
                
            except np.linalg.LinAlgError:
                self.details_text.append("   ❌ LSTSQ falló, usando método alternativo")
            
            # Método 2: SVD directo
            try:
                U, s, Vt = np.linalg.svd(A, full_matrices=False)
                # Filtrar valores singulares muy pequeños
                s_inv = np.zeros_like(s)
                tolerance = max(A.shape) * np.finfo(float).eps * np.max(s)
                s_inv[s > tolerance] = 1 / s[s > tolerance]
                
                weights = np.dot(Vt.T, s_inv * np.dot(U.T, y))
                self.details_text.append(f"   - SVD - W₀: {weights[0]:.6f}")
                return weights
                
            except np.linalg.LinAlgError:
                self.details_text.append("   ❌ SVD falló, usando pseudoinversa")
            
            # Método 3: Pseudoinversa con regularización (último recurso)
            lambda_reg = 1e-8  # Regularización muy pequeña
            ATA = np.dot(A.T, A) + lambda_reg * np.eye(A.shape[1])
            weights = np.dot(np.linalg.pinv(ATA), np.dot(A.T, y))
            self.details_text.append(f"   - Pseudoinversa regularizada - W₀: {weights[0]:.6f}")
            
            return weights
            
        except Exception as e:
            raise Exception(f"Error en cálculo de pesos: {str(e)}")
    
    def debug_training_process(self, A, y_train, activations, distances):
        """Función para debug detallado del proceso de entrenamiento"""
        debug_info = "\n=== DEBUG DETALLADO ===\n"
        
        # 1. Información de la matriz A
        debug_info += f"1. MATRIZ A:\n"
        debug_info += f"   - Forma: {A.shape}\n"
        debug_info += f"   - Rango: {np.linalg.matrix_rank(A)} (de {min(A.shape)})\n"
        debug_info += f"   - Condición: {np.linalg.cond(A):.2e}\n"
        debug_info += f"   - Valores únicos en columna umbral: {np.unique(A[:, 0])}\n"
        
        # 2. Información de activaciones
        debug_info += f"\n2. ACTIVACIONES:\n"
        debug_info += f"   - Mínimo: {np.min(activations):.6e}\n"
        debug_info += f"   - Máximo: {np.max(activations):.6e}\n"
        debug_info += f"   - Media: {np.mean(activations):.6e}\n"
        debug_info += f"   - ¿Tiene NaN?: {np.any(np.isnan(activations))}\n"
        debug_info += f"   - ¿Tiene Inf?: {np.any(np.isinf(activations))}\n"
        
        # 3. Información de distancias
        debug_info += f"\n3. DISTANCIAS:\n"
        debug_info += f"   - Mínimo: {np.min(distances):.6e}\n"
        debug_info += f"   - Máximo: {np.max(distances):.6e}\n"
        debug_info += f"   - Media: {np.mean(distances):.6e}\n"
        
        # 4. Información de salidas
        debug_info += f"\n4. SALIDAS (y_train):\n"
        debug_info += f"   - Mínimo: {np.min(y_train):.6f}\n"
        debug_info += f"   - Máximo: {np.max(y_train):.6f}\n"
        debug_info += f"   - Media: {np.mean(y_train):.6f}\n"
        
        # 5. Calcular pesos con diferentes métodos para comparar
        try:
            # Método 1: Pseudoinversa (actual)
            ATA = np.dot(A.T, A)
            ATA_inv = np.linalg.pinv(ATA)
            weights_pinv = np.dot(np.dot(ATA_inv, A.T), y_train)
            
            # Método 2: SVD directo
            U, s, Vt = np.linalg.svd(A, full_matrices=False)
            weights_svd = np.dot(Vt.T, np.dot(U.T, y_train) / s)
            
            # Método 3: np.linalg.lstsq (más robusto)
            weights_lstsq, residuals, rank, s_values = np.linalg.lstsq(A, y_train, rcond=None)
            
            debug_info += f"\n5. COMPARACIÓN DE MÉTODOS DE CÁLCULO:\n"
            debug_info += f"   - Pseudoinversa - W₀: {weights_pinv[0]:.10f}\n"
            debug_info += f"   - SVD directo - W₀: {weights_svd[0]:.10f}\n"
            debug_info += f"   - LSTSQ - W₀: {weights_lstsq[0]:.10f}\n"
            debug_info += f"   - Rank detectado por LSTSQ: {rank}\n"
            
            # Usar el método que dé el W₀ más razonable
            if abs(weights_lstsq[0]) > 1e-10:  # Si LSTSQ da un W₀ no-cero
                selected_weights = weights_lstsq
                debug_info += f"   - SELECCIONADO: LSTSQ (W₀ no-cero)\n"
            elif abs(weights_svd[0]) > 1e-10:  # Si SVD da un W₀ no-cero
                selected_weights = weights_svd
                debug_info += f"   - SELECCIONADO: SVD (W₀ no-cero)\n"
            else:
                selected_weights = weights_lstsq  # Por defecto LSTSQ
                debug_info += f"   - SELECCIONADO: LSTSQ (por defecto)\n"
            
            return selected_weights
            
        except Exception as e:
            debug_info += f"\n❌ Error en cálculo comparativo: {str(e)}\n"
            return None
        
        finally:
            # Mostrar debug info en la interfaz
            self.details_text.append(debug_info)
    
    def calculate_predictions(self, A, weights):
        """Calcular predicciones: ŷ = A * W"""
        return np.dot(A, weights)
    
    def calculate_errors(self, y_true, y_pred):
        """Calcular diferentes métricas de error"""
        # Error absoluto
        absolute_errors = np.abs(y_true - y_pred)
        
        # Error General (EG) - promedio de errores absolutos
        eg = np.mean(absolute_errors)
        
        # Error Absoluto Medio (MAE)
        mae = np.mean(absolute_errors)
        
        # Error Cuadrático Medio (MSE)
        mse = np.mean((y_true - y_pred) ** 2)
        
        # Raíz del Error Cuadrático Medio (RMSE)
        rmse = np.sqrt(mse)
        
        return {
            'eg': eg,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'absolute_errors': absolute_errors
        }
    
    def start_training(self):
        """Iniciar proceso de entrenamiento - VERSIÓN MEJORADA"""
        # Verificar requisitos
        if not self.verify_requirements():
            return
        
        try:
            self.training_status.setText("🔄 Iniciando entrenamiento...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.details_text.clear()
            
            # Obtener datos
            X_train = self.main_window.preprocessed_data['X_train']
            y_train = self.main_window.preprocessed_data['y_train']
            centers = self.main_window.rbf_config['centers']
            num_centers = self.main_window.rbf_config['num_centers']
            target_error = self.main_window.rbf_config['target_error']
            
            self.details_text.append("=== INFORMACIÓN GENERAL ===\n")
            self.details_text.append(f"• Número de centros radiales: {num_centers}\n")
            self.details_text.append(f"• Iteración: 1 (entrenamiento único)\n")
            self.details_text.append(f"• Error objetivo: {target_error}\n")
            self.details_text.append(f"• Patrones de entrenamiento: {X_train.shape[0]}\n")
            self.details_text.append(f"• Entradas por patrón: {X_train.shape[1]}\n")
            self.details_text.append(f"• Función de activación: Gaussiana mejorada\n")
            self.details_text.append(f"• Rango de centros: [{np.min(centers):.4f}, {np.max(centers):.4f}]\n")
            self.details_text.append(f"• Rango de datos X_train: [{np.min(X_train):.4f}, {np.max(X_train):.4f}]\n")
            self.details_text.append(f"• Rango de salidas y_train: [{np.min(y_train):.4f}, {np.max(y_train):.4f}]\n")
            
            self.progress_bar.setValue(20)
            QApplication.processEvents()
            
            # Paso 1: Calcular distancias euclidianas
            self.details_text.append("\n=== CÁLCULO DE DISTANCIAS EUCLIDIANAS ===\n")
            distances = self.calculate_euclidean_distance(X_train, centers)
            self.details_text.append(f"Matriz de distancias: {distances.shape} (patrones x centros)\n")
            self.details_text.append(f"Rango distancias: [{np.min(distances):.6f}, {np.max(distances):.6f}]\n")
            
            # Mostrar algunas distancias de ejemplo
            if distances.shape[0] >= 2 and distances.shape[1] >= 2:
                self.details_text.append(f"Ejemplo distancias:\n")
                self.details_text.append(f"  D11 (Patrón1-Centro1): {distances[0,0]:.4f}\n")
                self.details_text.append(f"  D21 (Patrón1-Centro2): {distances[0,1]:.4f}\n")
                self.details_text.append(f"  D12 (Patrón2-Centro1): {distances[1,0]:.4f}\n")
                self.details_text.append(f"  D22 (Patrón2-Centro2): {distances[1,1]:.4f}\n")
            
            self.progress_bar.setValue(40)
            QApplication.processEvents()
            
            # Paso 2: Aplicar función de activación radial MEJORADA
            self.details_text.append("\n=== APLICACIÓN FUNCIÓN DE ACTIVACIÓN GAUSSIANA MEJORADA ===\n")
            activations = self.radial_basis_function(distances)
            self.details_text.append(f"Matriz de activaciones Φ: {activations.shape}\n")
            self.details_text.append(f"Rango activaciones: [{np.min(activations):.6f}, {np.max(activations):.6f}]\n")
            
            # Mostrar algunas activaciones de ejemplo
            if activations.shape[0] >= 2 and activations.shape[1] >= 2:
                self.details_text.append(f"Ejemplo activaciones:\n")
                self.details_text.append(f"  FA(D11): {activations[0,0]:.4f}\n")
                self.details_text.append(f"  FA(D21): {activations[0,1]:.4f}\n")
                self.details_text.append(f"  FA(D12): {activations[1,0]:.4f}\n")
                self.details_text.append(f"  FA(D22): {activations[1,1]:.4f}\n")
            
            self.progress_bar.setValue(60)
            QApplication.processEvents()
            
            # Paso 3: Construir matriz de interpolación
            self.details_text.append("\n=== CONSTRUCCIÓN MATRIZ DE INTERPOLACIÓN ===\n")
            A = self.build_interpolation_matrix(activations)
            self.details_text.append(f"Matriz A: {A.shape} (con columna de unos para umbral)\n")
            self.details_text.append(f"Rango de A: {np.linalg.matrix_rank(A)} (mínimo requerido: {min(A.shape)})\n")
            
            self.progress_bar.setValue(80)
            QApplication.processEvents()
            
            # Paso 4: Calcular pesos con debug detallado
            self.details_text.append("\n=== CÁLCULO DE PESOS CON DEBUG ===\n")
            
            # Llamar a la función de debug
            weights = self.debug_training_process(A, y_train, activations, distances)
            
            if weights is None:
                # Si el debug falla, usar el método mejorado
                self.details_text.append("⚠️ Usando método de pesos mejorado\n")
                weights = self.solve_weights(A, y_train)
            
            self.details_text.append(f"Vector de pesos W calculado: {len(weights)} pesos\n")
            self.details_text.append(f"  W₀ (Umbral): {weights[0]:.10f}\n")
            for i in range(1, len(weights)):
                self.details_text.append(f"  W{i}: {weights[i]:.10f}\n")
            
            # Paso 5: Calcular predicciones y errores
            y_pred = self.calculate_predictions(A, weights)
            errors = self.calculate_errors(y_train, y_pred)
            
            self.progress_bar.setValue(100)
            QApplication.processEvents()
            
            # Guardar resultados
            self.training_results = {
                'weights': weights,
                'activations': activations,
                'distances': distances,
                'interpolation_matrix': A,
                'y_pred': y_pred,
                'errors': errors,
                'centers': centers,
                'num_centers': num_centers,
                'iteration': 1,
                'target_error': target_error,
                'convergence': errors['eg'] <= target_error,
                'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Actualizar aplicación principal
            self.main_window.training_results = self.training_results
            
            # Actualizar interfaz
            self.update_results_display()
            self.training_status.setText("✅ Entrenamiento completado exitosamente")
            self.training_status.setStyleSheet("color: #27ae60; font-weight: bold; padding: 10px;")
            
            # Mostrar resumen
            self.show_training_summary()
            
        except Exception as e:
            self.training_status.setText("❌ Error en el entrenamiento")
            self.training_status.setStyleSheet("color: #c0392b; font-weight: bold; padding: 10px;")
            QMessageBox.critical(self, "Error", f"Error durante el entrenamiento:\n{str(e)}")
            print(f"Error detallado: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.progress_bar.setVisible(False)
    
    def verify_requirements(self):
        """Verificar que se cumplen todos los requisitos para el entrenamiento"""
        if self.main_window.preprocessed_data is None:
            QMessageBox.warning(self, "Datos Requeridos", 
                              "Primero cargue y preprocese los datos en el módulo 'Carga de Datos'")
            return False
        
        if self.main_window.rbf_config is None:
            QMessageBox.warning(self, "Configuración Requerida", 
                              "Primero configure y guarde la red RBF en el módulo 'Configuración RBF'")
            return False
        
        if self.main_window.rbf_config.get('centers') is None:
            QMessageBox.warning(self, "Centros Requeridos", 
                              "Primero inicialice los centros radiales en el módulo 'Configuración RBF'")
            return False
        
        return True
    
    def update_results_display(self):
        """Actualizar la visualización de resultados"""
        if self.training_results is None:
            return
        
        # Actualizar tabla de pesos
        self.update_weights_table()
        
        # Actualizar tabla de centros finales
        self.update_centers_table()
        
        # Actualizar tabla de activaciones
        self.update_activations_table()
        
        # Actualizar información de errores
        self.update_errors_info()
    
    def update_weights_table(self):
        """Actualizar tabla de pesos finales"""
        weights = self.training_results['weights']
        
        # Configurar tabla
        self.weights_table.setRowCount(len(weights))
        self.weights_table.setColumnCount(2)
        self.weights_table.setHorizontalHeaderLabels(["Tipo", "Valor"])
        
        # Llenar datos
        for i, weight in enumerate(weights):
            # Tipo de peso
            if i == 0:
                type_item = QTableWidgetItem("W₀ (Umbral)")
            else:
                type_item = QTableWidgetItem(f"W{i} (Centro {i})")
            
            # Valor del peso
            value_item = QTableWidgetItem(f"{weight:.10f}")
            
            self.weights_table.setItem(i, 0, type_item)
            self.weights_table.setItem(i, 1, value_item)
        
        # Ajustar tamaño de columnas
        self.weights_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    
    def update_centers_table(self):
        """Actualizar tabla de centros radiales finales"""
        centers = self.training_results['centers']
        num_centers, num_inputs = centers.shape
        
        # Configurar tabla
        self.centers_table.setRowCount(num_centers)
        self.centers_table.setColumnCount(num_inputs + 1)
        
        # Encabezados
        headers = ["Centro #"]
        headers.extend([f"Entrada {i+1}" for i in range(num_inputs)])
        self.centers_table.setHorizontalHeaderLabels(headers)
        
        # Llenar datos
        for i in range(num_centers):
            # Número de centro
            center_item = QTableWidgetItem(f"R{i+1}")
            center_item.setTextAlignment(Qt.AlignCenter)
            self.centers_table.setItem(i, 0, center_item)
            
            # Valores del centro
            for j in range(num_inputs):
                value = centers[i, j]
                item = QTableWidgetItem(f"{value:.10f}")
                item.setTextAlignment(Qt.AlignCenter)
                self.centers_table.setItem(i, j + 1, item)
        
        # Ajustar tamaño de columnas
        self.centers_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    
    def update_activations_table(self):
        """Actualizar tabla de activaciones (muestra solo primeras filas)"""
        activations = self.training_results['activations']
        n_centers = activations.shape[1]
        
        # Mostrar máximo 20 patrones para mejor rendimiento
        display_rows = min(20, activations.shape[0])
        
        # Configurar tabla
        self.activations_table.setRowCount(display_rows)
        self.activations_table.setColumnCount(n_centers + 1)
        
        # Encabezados
        headers = ["Patrón #"]
        headers.extend([f"Centro {i+1}" for i in range(n_centers)])
        self.activations_table.setHorizontalHeaderLabels(headers)
        
        # Llenar datos
        for i in range(display_rows):
            # Número de patrón
            pattern_item = QTableWidgetItem(f"P{i+1}")
            pattern_item.setTextAlignment(Qt.AlignCenter)
            self.activations_table.setItem(i, 0, pattern_item)
            
            # Valores de activación
            for j in range(n_centers):
                value = activations[i, j]
                item = QTableWidgetItem(f"{value:.6f}")
                item.setTextAlignment(Qt.AlignCenter)
                self.activations_table.setItem(i, j + 1, item)
        
        # Ajustar tamaño de columnas
        self.activations_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        
        if activations.shape[0] > 20:
            self.details_text.append(f"Nota: Mostrando 20 de {activations.shape[0]} patrones en la tabla de activaciones\n")
    
    def update_errors_info(self):
        """Actualizar información de errores y métricas"""
        errors = self.training_results['errors']
        y_train = self.main_window.preprocessed_data['y_train']
        y_pred = self.training_results['y_pred']
        num_centers = self.training_results['num_centers']
        iteration = self.training_results['iteration']
        target_error = self.training_results['target_error']
        convergence = self.training_results['convergence']
        
        convergence_status = "✅ CONVERGE" if convergence else "❌ NO CONVERGE"
        
        error_text = f"""=== RESULTADOS FINALES DEL ENTRENAMIENTO ===

INFORMACIÓN GENERAL:
• Número de centros radiales: {num_centers}
• Iteración: {iteration}
• Error objetivo: {target_error}
• Estado: {convergence_status}
• Función de activación: Gaussiana mejorada
• Fecha: {self.training_results['training_date']}

MÉTRICAS DE ERROR (REQUERIDAS):
• Error General (EG): {errors['eg']:.10f}
• Error Absoluto Medio (MAE): {errors['mae']:.10f}
• Raíz del Error Cuadrático Medio (RMSE): {errors['rmse']:.10f}

ESTADÍSTICAS DE LAS SALIDAS:
• Salidas reales (y_train): 
  - Mínimo: {np.min(y_train):.6f}
  - Máximo: {np.max(y_train):.6f}
  - Media: {np.mean(y_train):.6f}
  - Desviación estándar: {np.std(y_train):.6f}

• Salidas predichas (ŷ): 
  - Mínimo: {np.min(y_pred):.6f}
  - Máximo: {np.max(y_pred):.6f}
  - Media: {np.mean(y_pred):.6f}
  - Desviación estándar: {np.std(y_pred):.6f}

PESOS FINALES CALCULADOS:
"""
        weights = self.training_results['weights']
        for i, w in enumerate(weights):
            if i == 0:
                error_text += f"• W₀ (Umbral): {w:.10f}\n"
            else:
                error_text += f"• W{i}: {w:.10f}\n"
        
        error_text += f"\nCONVERGENCIA: "
        if convergence:
            error_text += f"EG ({errors['eg']:.6f}) <= Error objetivo ({target_error})"
        else:
            error_text += f"EG ({errors['eg']:.6f}) > Error objetivo ({target_error})"
        
        self.errors_text.setText(error_text)
    
    def show_training_summary(self):
        """Mostrar resumen del entrenamiento"""
        errors = self.training_results['errors']
        weights = self.training_results['weights']
        num_centers = self.training_results['num_centers']
        iteration = self.training_results['iteration']
        convergence = self.training_results['convergence']
        
        convergence_msg = "✅ LA RED CONVERGE" if convergence else "⚠️ LA RED NO CONVERGE"
        
        summary = f"""
{convergence_msg}

🎯 INFORMACIÓN DE ENTRENAMIENTO:
• Centros radiales: {num_centers}
• Iteración: {iteration}
• Patrones de entrenamiento: {len(self.training_results['y_pred'])}
• Entradas por patrón: {self.main_window.preprocessed_data['X_train'].shape[1]}
• Función de activación: Gaussiana mejorada
• Fecha: {self.training_results['training_date']}

📊 RESULTADOS FINALES (REQUERIDAS):
• Error General (EG): {errors['eg']:.8f}
• Error Absoluto Medio (MAE): {errors['mae']:.8f}
• Raíz del Error Cuadrático Medio (RMSE): {errors['rmse']:.8f}

⚖️ PESOS CALCULADOS: {len(weights)}
  - W₀ (Umbral): {weights[0]:.6f}
  - W₁ a W{len(weights)-1}: {len(weights)-1} pesos para centros radiales

La red RBF está lista para realizar simulaciones.
"""
        
        QMessageBox.information(self, "Entrenamiento Completado", summary)
    
    def save_training(self):
        """Guardar resultados del entrenamiento en archivos - VERSIÓN CORREGIDA"""
        if self.training_results is None:
            QMessageBox.warning(self, "Advertencia", "Primero realice un entrenamiento")
            return
        
        try:
            # Crear carpeta de resultados si no existe
            results_dir = "resultados"
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            
            # Crear subcarpeta con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            training_dir = os.path.join(results_dir, f"entrenamiento_{timestamp}")
            os.makedirs(training_dir)
            
            # 1. Guardar modelo completo
            model_path = os.path.join(training_dir, "modelo_rbf.pkl")
            model_data = {
                'training_results': self.training_results,
                'rbf_config': self.main_window.rbf_config,
                'preprocessed_data_info': {
                    'input_columns': self.main_window.preprocessed_data.get('input_columns', []),
                    'output_columns': self.main_window.preprocessed_data.get('output_columns', []),
                    'X_train_shape': self.main_window.preprocessed_data['X_train'].shape,
                    'scaler_type': self.main_window.preprocessed_data.get('scaler_type', 'Ninguno')
                }
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            # 2. Guardar métricas en JSON - VERSIÓN CORREGIDA
            metrics_path = os.path.join(training_dir, "metricas.json")
            
            # CORRECCIÓN: Convertir booleanos a strings para JSON
            convergence_str = "CONVERGE" if self.training_results['convergence'] else "NO_CONVERGE"
            
            metrics_data = {
                'fecha_entrenamiento': self.training_results['training_date'],
                'convergencia': convergence_str,  # ✅ CORREGIDO: string en lugar de bool
                'error_objetivo': float(self.training_results['target_error']),  # ✅ Asegurar float
                'metricas': {
                    'eg': float(self.training_results['errors']['eg']),
                    'mae': float(self.training_results['errors']['mae']),
                    'mse': float(self.training_results['errors']['mse']),
                    'rmse': float(self.training_results['errors']['rmse'])
                },
                'configuracion': {
                    'num_centros': int(self.training_results['num_centers']),  # ✅ Asegurar int
                    'iteraciones': int(self.training_results['iteration']),    # ✅ Asegurar int
                    'funcion_activacion': 'Gaussiana mejorada',
                    'centros_radiales': int(self.training_results['num_centers']),
                    'entradas_por_centro': self.training_results['centers'].shape[1] if self.training_results['centers'] is not None else 0
                },
                'pesos': {
                    'w0_umbral': float(self.training_results['weights'][0]),  # ✅ Umbral
                    'pesos_centros': [float(w) for w in self.training_results['weights'][1:]]  # ✅ Pesos de centros
                }
            }
            
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, indent=4, ensure_ascii=False)
            
            # 3. Guardar detalles del entrenamiento en texto
            details_path = os.path.join(training_dir, "detalles_entrenamiento.txt")
            with open(details_path, 'w', encoding='utf-8') as f:
                f.write("=== DETALLES DEL ENTRENAMIENTO RBF ===\n\n")
                f.write(self.details_text.toPlainText())
                f.write("\n\n=== RESULTADOS FINALES ===\n\n")
                f.write(self.errors_text.toPlainText())
            
            # 4. Guardar pesos en CSV - VERSIÓN MEJORADA
            weights_path = os.path.join(training_dir, "pesos_rbf.csv")
            weights_data = []
            
            # Agregar umbral (W0)
            weights_data.append({
                'tipo': 'W₀ (Umbral)',
                'valor': float(self.training_results['weights'][0]),
                'centro_asociado': 'Ninguno (Umbral)'
            })
            
            # Agregar pesos de centros radiales
            for i in range(1, len(self.training_results['weights'])):
                weights_data.append({
                    'tipo': f'W{i}',
                    'valor': float(self.training_results['weights'][i]),
                    'centro_asociado': f'Centro Radial {i}'
                })
            
            weights_df = pd.DataFrame(weights_data)
            weights_df.to_csv(weights_path, index=False, encoding='utf-8')
            
            # 5. Guardar centros radiales en CSV
            centers_path = os.path.join(training_dir, "centros_radiales.csv")
            if self.training_results['centers'] is not None:
                centers_data = []
                num_centers, num_inputs = self.training_results['centers'].shape
                
                for i in range(num_centers):
                    center_info = {'centro': f'R{i+1}'}
                    for j in range(num_inputs):
                        center_info[f'entrada_{j+1}'] = float(self.training_results['centers'][i, j])
                    centers_data.append(center_info)
                
                centers_df = pd.DataFrame(centers_data)
                centers_df.to_csv(centers_path, index=False, encoding='utf-8')
            
            # 6. Guardar predicciones vs reales
            predictions_path = os.path.join(training_dir, "predicciones_entrenamiento.csv")
            y_train = self.main_window.preprocessed_data['y_train']
            y_pred = self.training_results['y_pred']
            
            predictions_data = {
                'salida_real': [float(y) for y in y_train],
                'salida_predicha': [float(y) for y in y_pred],
                'error_absoluto': [float(abs(y_true - y_pred)) for y_true, y_pred in zip(y_train, y_pred)]
            }
            
            predictions_df = pd.DataFrame(predictions_data)
            predictions_df.to_csv(predictions_path, index=False, encoding='utf-8')
            
            # Mensaje de confirmación mejorado
            archivos_creados = [
                "• modelo_rbf.pkl (modelo completo serializado)",
                "• metricas.json (métricas de evaluación)",
                "• detalles_entrenamiento.txt (proceso detallado)",
                "• pesos_rbf.csv (pesos de la red)",
                "• centros_radiales.csv (centros radiales)",
                "• predicciones_entrenamiento.csv (comparación real vs predicho)"
            ]
            
            QMessageBox.information(self, "Entrenamiento Guardado", 
                                f"✅ Resultados del entrenamiento guardados en:\n{training_dir}\n\n"
                                f"Archivos creados:\n" + "\n".join(archivos_creados))
            
            self.training_status.setText(f"💾 Entrenamiento guardado en {training_dir}")
            
        except Exception as e:
            error_msg = f"❌ Error al guardar entrenamiento: {str(e)}"
            QMessageBox.critical(self, "Error", f"No se pudieron guardar los resultados:\n{str(e)}")
    
    def reset_training(self):
        """Limpiar resultados del entrenamiento"""
        if self.training_results is None:
            QMessageBox.information(self, "Información", "No hay entrenamiento para limpiar")
            return
        
        reply = QMessageBox.question(self, 'Confirmar Limpieza', 
                                   '¿Está seguro de que desea limpiar todos los resultados del entrenamiento?',
                                   QMessageBox.Yes | QMessageBox.No, 
                                   QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.training_results = None
            self.main_window.training_results = None
            
            # Limpiar interfaces
            self.weights_table.setRowCount(0)
            self.weights_table.setColumnCount(0)
            self.centers_table.setRowCount(0)
            self.centers_table.setColumnCount(0)
            self.activations_table.setRowCount(0)
            self.activations_table.setColumnCount(0)
            self.errors_text.clear()
            self.details_text.clear()
            
            self.training_status.setText("🧹 Entrenamiento limpiado - Listo para nuevo entrenamiento")
            self.training_status.setStyleSheet("color: #7f8c8d; font-weight: bold; padding: 10px;")
            
            QMessageBox.information(self, "Entrenamiento Limpiado", 
                                "Todos los resultados del entrenamiento han sido limpiados.\n"
                                "Puede realizar un nuevo entrenamiento cuando lo desee.")
    
    def on_activate(self):
        """Método llamado cuando el módulo se activa"""
        self.details_text.clear()
        self.details_text.append("=== MÓDULO DE ENTRENAMIENTO RBF ===\n")
        self.details_text.append("Este módulo entrena la red neuronal RBF usando:\n")
        self.details_text.append("1. Cálculo de distancias euclidianas entre patrones y centros\n")
        self.details_text.append("2. Aplicación de función de base radial: Gaussiana mejorada\n")
        self.details_text.append("3. Construcción de matriz de interpolación\n")
        self.details_text.append("4. Cálculo de pesos por mínimos cuadrados\n")
        self.details_text.append("5. Cálculo de métricas de error (EG, MAE, RMSE)\n")
        self.details_text.append("6. Verificación de convergencia vs error objetivo\n\n")
        self.details_text.append("NUEVAS FUNCIONALIDADES:\n")
        self.details_text.append("• 💾 Guardar entrenamiento: Exporta resultados a archivos\n")
        self.details_text.append("• 🔄 Limpiar entrenamiento: Prepara para nuevo entrenamiento\n")
        self.details_text.append("• 🔍 Debug detallado: Diagnóstico completo del proceso\n")
        
        # Verificar estado actual
        if (self.main_window.preprocessed_data is not None and 
            self.main_window.rbf_config is not None):
            self.training_status.setText("✅ Listo para entrenar - Todos los requisitos cumplidos")
            self.training_status.setStyleSheet("color: #27ae60; font-weight: bold; padding: 10px;")
        else:
            self.training_status.setText("⚠️ Complete los módulos anteriores primero")
            self.training_status.setStyleSheet("color: #f39c12; font-weight: bold; padding: 10px;")