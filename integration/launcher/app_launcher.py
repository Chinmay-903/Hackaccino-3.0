import sys
import os
import sqlite3
import shutil

import time
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QGridLayout, 
    QLineEdit, QSizePolicy, QHBoxLayout, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QPixmap, QDragEnterEvent, QDropEvent


MODERN_STYLE = """
    QWidget {
        background-color: #1E1E2E;
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }
    QPushButton {
        background-color: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 12px 24px;
        color: white;
        font-size: 14px;
        min-width: 140px;
        font-weight: 500;
    }
    QPushButton:hover {
        background-color: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    QPushButton:pressed {
        background-color: rgba(255, 255, 255, 0.15);
    }
    QLineEdit {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 12px 24px;
        color: white;
        font-size: 16px;
        font-weight: 500;
    }
    QLineEdit:focus {
        border: 2px solid #2196F3;
        background-color: rgba(33, 150, 243, 0.1);
    }
"""

class ModernSoftwareWidget(QWidget):
    def __init__(self, name, path, logo_path, parent=None):
        super().__init__(parent)
        self.name = name
        self.path = path
        self.logo_path = logo_path
        self.selected = False
        
        self.initUI()
    
    def initUI(self):
        self.setMinimumSize(200, 180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 15)
        layout.setSpacing(10)
        
        logo_container = QWidget()
        logo_container.setStyleSheet("""
            QWidget {
                background-color: rgba(255, 255, 255, 0.03);
                border-radius: 20px;
                padding: 10px;
            }
        """)
        logo_layout = QVBoxLayout()
        logo_layout.setContentsMargins(10, 10, 10, 10)
        
        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignCenter)
        self.update_logo()
        logo_layout.addWidget(self.logo_label)
        logo_container.setLayout(logo_layout)
        layout.addWidget(logo_container)
        
        self.name_label = QLabel(self.name)
        self.name_label.setAlignment(Qt.AlignCenter)
        self.name_label.setMaximumHeight(28)
        self.name_label.setWordWrap(False)
        self.name_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: 600;
                color: #2196F3;
                padding: 4px;
                margin: 0px;
                background-color: rgba(33, 150, 243, 0.05);
                border-radius: 8px;
            }
        """)
        
        metrics = self.name_label.fontMetrics()
        elided_text = metrics.elidedText(self.name, Qt.ElideRight, 180)
        self.name_label.setText(elided_text)
        
        layout.addWidget(self.name_label)
        layout.addStretch(0)
        
        self.setLayout(layout)
        self.update_style()
    
    def update_logo(self):
        if self.logo_path and os.path.exists(self.logo_path):
            pixmap = QPixmap(self.logo_path)
        else:
            app_name = os.path.splitext(os.path.basename(self.path))[0]
            logos_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logos")
            logo_path = os.path.join(logos_dir, f"{app_name}.png")
            
            if os.path.exists(logo_path):
                pixmap = QPixmap(logo_path)
            else:
                default_logo = os.path.join(logos_dir, "default_logo.png")
                if os.path.exists(default_logo):
                    pixmap = QPixmap(default_logo)
                else:
                    pixmap = QPixmap(140, 140)
                    pixmap.fill(Qt.transparent)
        
        pixmap = pixmap.scaled(140, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.logo_label.setPixmap(pixmap)
    
    def set_selected(self, selected):
        self.selected = selected
        self.update_style()
    
    def update_style(self):
        style = """
            QWidget {
                background-color: %s;
                border-radius: 24px;
                border: 2px solid %s;
            }
        """ % (
            "rgba(33, 150, 243, 0.12)" if self.selected else "rgba(255, 255, 255, 0.02)",
            "#2196F3" if self.selected else "rgba(255, 255, 255, 0.05)"
        )
        self.setStyleSheet(style)

class ModernSoftwareLauncher(QWidget):
    def __init__(self, validation_callback=None):
        super().__init__()
        self.widgets = []
        self.selected_index = 0
        self.max_cols = 4
        self.validation_callback = validation_callback
        self.validation_timer = QTimer()
        self.initUI()
        self.load_software()
        self.setFocusPolicy(Qt.StrongFocus)
        self.setup_validation_timer()
    
    def setup_validation_timer(self):
        self.validation_timer.timeout.connect(self.validate_user)
        self.validation_timer.start(10000)  # 10 seconds
    
    def validate_user(self):
        if self.validation_callback and not self.validation_callback():
            self.close()
    
    def initUI(self):
        self.setWindowTitle("Modern App Drawer")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet(MODERN_STYLE)
        
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)
        
        header_layout = QHBoxLayout()
        header_layout.setSpacing(15)
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search applications...")
        self.search_input.textChanged.connect(self.filter_apps)
        self.search_input.setFixedHeight(48)
        header_layout.addWidget(self.search_input)
        
        self.add_button = QPushButton("Add App")
        self.add_button.clicked.connect(self.browse_for_app)
        self.add_button.setFixedHeight(48)
        self.add_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(33, 150, 243, 0.1);
                border: 1px solid rgba(33, 150, 243, 0.2);
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: rgba(33, 150, 243, 0.15);
                border: 1px solid rgba(33, 150, 243, 0.3);
            }
        """)
        header_layout.addWidget(self.add_button)
        
        self.logo_button = QPushButton("Set Logo")
        self.logo_button.clicked.connect(self.select_app_logo)
        self.logo_button.setFixedHeight(48)
        self.logo_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(76, 175, 80, 0.1);
                border: 1px solid rgba(76, 175, 80, 0.2);
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: rgba(76, 175, 80, 0.15);
                border: 1px solid rgba(76, 175, 80, 0.3);
            }
        """)
        header_layout.addWidget(self.logo_button)
        
        self.remove_button = QPushButton("Remove App")
        self.remove_button.clicked.connect(self.remove_selected_app)
        self.remove_button.setFixedHeight(48)
        self.remove_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(244, 67, 54, 0.1);
                border: 1px solid rgba(244, 67, 54, 0.2);
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: rgba(244, 67, 54, 0.15);
                border: 1px solid rgba(244, 67, 54, 0.3);
            }
        """)
        header_layout.addWidget(self.remove_button)
        
        main_layout.addLayout(header_layout)
        
        self.app_grid = QGridLayout()
        self.app_grid.setSpacing(20)
        self.app_grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        main_layout.addLayout(self.app_grid)
        
        main_layout.addStretch()
        
        self.setLayout(main_layout)
        self.setAcceptDrops(True)
    
    def select_app_logo(self):
        if not self.widgets or self.selected_index < 0:
            return
            
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp)")
        
        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                self.set_app_logo(selected_files[0])
    
    def set_app_logo(self, logo_path):
        if not self.widgets or self.selected_index < 0:
            return
            
        selected_widget = self.widgets[self.selected_index]
        
        logos_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logos")
        os.makedirs(logos_dir, exist_ok=True)
        
        app_name = os.path.splitext(os.path.basename(selected_widget.path))[0]
        new_logo_path = os.path.join(logos_dir, f"{app_name}.png")
        
        try:
            shutil.copy2(logo_path, new_logo_path)
            
            conn = sqlite3.connect("launcher.db")
            cursor = conn.cursor()
            cursor.execute("UPDATE software SET logo_path=? WHERE path=?", 
                         (new_logo_path, selected_widget.path))
            conn.commit()
            conn.close()
            
            self.load_software()
            self.select_app(self.widgets[self.selected_index])
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not set logo: {str(e)}")
    
    def filter_apps(self, text):
        conn = sqlite3.connect("launcher.db")
        cursor = conn.cursor()
        cursor.execute("SELECT name, path, logo_path FROM software WHERE name LIKE ?", 
                      (f"%{text}%",))
        programs = cursor.fetchall()
        conn.close()
        self.load_software(programs)
    
    def load_software(self, programs=None):
        for i in reversed(range(self.app_grid.count())):
            widget = self.app_grid.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        self.widgets.clear()
        
        if not programs:
            conn = sqlite3.connect("launcher.db")
            cursor = conn.cursor()
            cursor.execute("SELECT name, path, logo_path FROM software")
            programs = cursor.fetchall()
            conn.close()
        
        row, col = 0, 0
        for program in programs:
            name, path, logo_path = program
            widget = ModernSoftwareWidget(name, path, logo_path)
            widget.mousePressEvent = lambda e, w=widget: self.select_app(w)
            widget.mouseDoubleClickEvent = lambda e, w=widget: self.launch_app(w.path)
            
            self.app_grid.addWidget(widget, row, col)
            self.widgets.append(widget)
            
            col += 1
            if col >= self.max_cols:
                col = 0
                row += 1
        
        if self.widgets:
            self.select_app(self.widgets[0])
    
    def select_app(self, widget):
        for w in self.widgets:
            w.set_selected(False)
        widget.set_selected(True)
        self.selected_index = self.widgets.index(widget)
    
    def launch_app(self, path):
        if os.path.exists(path):
            try:
                os.startfile(path)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not launch application: {str(e)}")
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()
    
    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            path = url.toLocalFile()
            if os.path.exists(path):
                self.add_app(path)
        event.accept()
    
    def add_app(self, path):
        if not os.path.exists(path):
            QMessageBox.warning(self, "Error", "The specified application does not exist")
            return
            
        name = os.path.splitext(os.path.basename(path))[0]
        
        logos_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logos")
        logo_name = f"{name}.png"
        logo_path = os.path.join(logos_dir, logo_name)
        
        if not os.path.exists(logo_path):
            logo_path = None
        
        conn = sqlite3.connect("launcher.db")
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS software (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                path TEXT NOT NULL,
                logo_path TEXT
            )
        """)
        
        cursor.execute("SELECT id FROM software WHERE path = ?", (path,))
        if cursor.fetchone():
            QMessageBox.information(self, "Info", "This application is already in the launcher")
        else:
            cursor.execute("INSERT INTO software (name, path, logo_path) VALUES (?, ?, ?)",
                         (name, path, logo_path))
            conn.commit()
            self.load_software()
        
        conn.close()

    def browse_for_app(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Applications (*.exe);;All Files (*.*)")
        
        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                self.add_app(selected_files[0])

    def remove_selected_app(self):
        if not self.widgets or self.selected_index < 0:
            return
            
        selected_widget = self.widgets[self.selected_index]
        reply = QMessageBox.question(
            self,
            'Remove Application',
            f'Are you sure you want to remove "{selected_widget.name}"?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            conn = sqlite3.connect("launcher.db")
            cursor = conn.cursor()
            cursor.execute("DELETE FROM software WHERE path = ?", (selected_widget.path,))
            conn.commit()
            conn.close()
            self.load_software()

    def keyPressEvent(self, event):
        if not self.widgets:
            return

        current_row = self.selected_index // self.max_cols
        current_col = self.selected_index % self.max_cols
        total_rows = (len(self.widgets) - 1) // self.max_cols + 1
        
        new_index = self.selected_index

        if event.key() == Qt.Key_Left:
            new_index = max(0, self.selected_index - 1)
        elif event.key() == Qt.Key_Right:
            new_index = min(len(self.widgets) - 1, self.selected_index + 1)
        elif event.key() == Qt.Key_Up:
            if current_row > 0:
                new_index = self.selected_index - self.max_cols
        elif event.key() == Qt.Key_Down:
            if current_row < total_rows - 1:
                new_index = min(len(self.widgets) - 1, self.selected_index + self.max_cols)
        elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if 0 <= self.selected_index < len(self.widgets):
                self.launch_app(self.widgets[self.selected_index].path)
            return
        elif event.key() == Qt.Key_Escape:
            self.close()
            return

        if new_index != self.selected_index and 0 <= new_index < len(self.widgets):
            self.select_app(self.widgets[new_index])

    def closeEvent(self, event):
        self.validation_timer.stop()
        event.accept()

def init_db():
    conn = sqlite3.connect("launcher.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS software (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            path TEXT NOT NULL,
            logo_path TEXT
        )
    """)
    conn.commit()
    conn.close()

def run_launcher(validation_callback=None):
    init_db()
    
    logos_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logos")
    os.makedirs(logos_dir, exist_ok=True)
    
    app = QApplication(sys.argv)
    launcher = ModernSoftwareLauncher(validation_callback)
    launcher.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run_launcher()