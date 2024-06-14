import sys
import cv2
import face_recognition
import numpy as np
import sqlite3
import pickle
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QLineEdit, QFileDialog, QDialog, QInputDialog, QScrollArea, QHBoxLayout, QMessageBox
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from datetime import datetime
import os

# Veritabanı bağlantısı
def create_connection():
    return sqlite3.connect('student_faces.db')

# Veritabanı kurulum ve güncelleme
def setup_database():
    with create_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS faces (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT,
                            encoding BLOB,
                            image BLOB,
                            access_allowed INTEGER DEFAULT 1)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS recognition_logs (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT,
                            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()

# Yüze veritabanına ekle
def add_face(name, encoding, image):
    encoding_blob = pickle.dumps(encoding)
    with create_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO faces (name, encoding, image) VALUES (?, ?, ?)", (name, encoding_blob, image))
        conn.commit()

# Veritabanındaki tüm yüzleri al
def get_faces():
    with create_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM faces")
        faces = cursor.fetchall()
    return [{"id": face[0], "name": face[1], "encoding": pickle.loads(face[2]), "image": face[3], "access_allowed": face[4]} for face in faces]

# Tanıma etkinliğini kaydet
def log_recognition(name):
    with create_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO recognition_logs (name) VALUES (?)", (name,))
        conn.commit()

# Yüz kodlama
def encode_face(image):
    face_encodings = face_recognition.face_encodings(image)
    if face_encodings:
        return face_encodings[0]
    return None

# Çerçeve içindeki yüzleri tanı
def recognize_faces(frame, known_faces):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces([face["encoding"] for face in known_faces], face_encoding)
        name = "Yeni Yuz"
        access_allowed = False
        face_distances = face_recognition.face_distance([face["encoding"] for face in known_faces], face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_faces[best_match_index]["name"]
                access_allowed = known_faces[best_match_index]["access_allowed"]
                log_recognition(name)
        face_names.append((name, access_allowed))

    return face_locations, face_names

class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        self.clicked.emit()

# Ana pencere sınıfı
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.known_faces = []
        self.load_known_faces()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Kamera açılamadı.")
            return
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
        self.image_displayed = False

    def initUI(self):
        self.setWindowTitle('Yüz Tanıma ve Turnike Geçiş Sistemi')
        self.camera_label = ClickableLabel(self)
        self.camera_label.clicked.connect(self.clear_turnstile_image)
        self.status_label = QLabel(self)
        
        self.recognize_button = QPushButton('Yüzünü Tanıt', self)
        self.recognize_button.clicked.connect(self.start_recognition)
        self.recognize_button.setStyleSheet('background-color: red; font-size: 16px; font-weight: bold')

        self.upload_face_button = QPushButton('Yüz Yükle', self)
        self.upload_face_button.clicked.connect(self.upload_face)
        self.upload_face_button.setStyleSheet('background-color: lightgreen')

        self.view_faces_button = QPushButton('Kayıtlı Öğrencileri Gör', self)
        self.view_faces_button.clicked.connect(self.view_faces)
        self.view_faces_button.setStyleSheet('background-color: lightyellow')
        
        self.view_logs_button = QPushButton('Giriş Kayıtlarını Gör', self)
        self.view_logs_button.clicked.connect(self.view_logs)
        self.view_logs_button.setStyleSheet('background-color: lightcoral')

        layout = QVBoxLayout()
        layout.addWidget(self.recognize_button)
        layout.addWidget(self.camera_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.upload_face_button)
        layout.addWidget(self.view_faces_button)
        layout.addWidget(self.view_logs_button)

        self.setLayout(layout)

    def upload_face(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Yüz Resmi Seç", "",
                                                  "All Files (*);;JPEG (*.jpg;*.jpeg);;PNG (*.png)", options=options)
        if fileName:
            image = cv2.imread(fileName)
            if image is None:
                print("Resim yüklenemedi.")
                return
            encoding = encode_face(image)
            if encoding is not None:
                name, ok = QInputDialog.getText(self, 'Yüz Kaydet', 'Öğrenci İsmi:')
                if ok:
                    with open(fileName, 'rb') as f:
                        image_blob = f.read()
                    add_face(name, encoding, image_blob)
                    self.load_known_faces()

    def load_known_faces(self):
        self.known_faces = get_faces()

    def view_faces(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Kayıtlı Öğrenciler")
        scroll = QScrollArea(dialog)
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)

        faces = get_faces()
        for face in faces:
            if face["image"]:
                image = QImage.fromData(face["image"])
                pixmap = QPixmap.fromImage(image)
                label = QLabel()
                label.setPixmap(pixmap.scaled(100, 100, Qt.KeepAspectRatio))
                layout.addWidget(QLabel(f"Adı: {face['name']}"))
                layout.addWidget(label)
                if face["access_allowed"] == 1:
                    access_label = QLabel("Geçiş İzinli")
                    access_label.setStyleSheet('background-color: green')
                    layout.addWidget(access_label)
                else:
                    access_label = QLabel("Geçiş İzni Yok")
                    access_label.setStyleSheet('background-color: red')
                    layout.addWidget(access_label)

                delete_button = QPushButton("Sil", container)
                delete_button.clicked.connect(lambda _, name=face['name']: self.delete_and_notify(name))
                layout.addWidget(delete_button)

                toggle_access_button = QPushButton("Geçiş İzni Değiştir", container)
                toggle_access_button.clicked.connect(lambda _, face_id=face['id']: self.toggle_access_and_notify(face_id))
                layout.addWidget(toggle_access_button)

        scroll.setWidget(container)
        layout_wrapper = QVBoxLayout(dialog)
        layout_wrapper.addWidget(scroll)
        dialog.setLayout(layout_wrapper)
        dialog.exec_()

    def delete_and_notify(self, name):
        with create_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM faces WHERE name = ?", (name,))
            conn.commit()
        self.load_known_faces()
        QMessageBox.information(self, "Başarılı", f"Öğrenci {name} silindi.")

    def toggle_access_and_notify(self, face_id):
        with create_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT access_allowed FROM faces WHERE id = ?", (face_id,))
            access_allowed = cursor.fetchone()[0]
            new_access_allowed = 0 if access_allowed == 1 else 1
            cursor.execute("UPDATE faces SET access_allowed = ? WHERE id = ?", (new_access_allowed, face_id))
            conn.commit()
        self.load_known_faces()
        QMessageBox.information(self, "Başarılı", "Geçiş izni güncellendi.")

    def view_logs(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Giriş Kayıtları")
        layout = QVBoxLayout(dialog)
        with create_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM recognition_logs ORDER BY timestamp DESC")
            logs = cursor.fetchall()
        for log in logs:
            layout.addWidget(QLabel(f"Adı: {log[1]}, Zaman: {log[2]}"))
        dialog.setLayout(layout)
        dialog.exec_()

    def start_recognition(self):
        self.status_label.setText("Yüz tanıma işlemi başladı, lütfen bekleyin...")
        self.status_label.setStyleSheet('background-color: yellow')
        self.image_displayed = False
        self.recognize_face()

    def recognize_face(self):
        if self.image_displayed:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.status_label.setText("Kamera açılamadı.")
            return

        frame = cv2.resize(frame, (640, 480))  # Performans için çerçeveyi yeniden boyutlandır
        face_locations, face_names = recognize_faces(frame, self.known_faces)

        if face_names:
            for (top, right, bottom, left), (name, access_allowed) in zip(face_locations, face_names):
                color = (0, 255, 0) if access_allowed else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                if access_allowed:
                    self.status_label.setText(f"Geçiş izni verildi: {name}\nGeçebilirsiniz.")
                    self.status_label.setStyleSheet('background-color: green')
                    self.show_turnstile_image("open")
                    return
                else:
                    self.status_label.setText(f"Geçiş izni yok: {name}\nGeçemezsiniz.")
                    self.status_label.setStyleSheet('background-color: red')
                    self.show_turnstile_image("closed")
                    return
        else:
            self.status_label.setText("Yüz tanınmadı veya geçiş izni yok.\nGeçemezsiniz.")
            self.status_label.setStyleSheet('background-color: red')
            self.show_turnstile_image("closed")

    def update_frame(self):
        if self.image_displayed:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.resize(frame, (640, 480))  # Performans için çerçeveyi yeniden boyutlandır
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(qt_image))

        face_locations = face_recognition.face_locations(rgb_image)
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_image = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], QImage.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(qt_image))

    def show_turnstile_image(self, status):
        self.image_displayed = True
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if status == "open":
            turnstile_image_path = os.path.join(current_dir, "resimler/acikturnike.webp")
            turnstile_image = QImage(turnstile_image_path)
            if turnstile_image.isNull():
                print(f"Açık turnike resmi yüklenemedi: {turnstile_image_path}")
            self.status_label.setText(self.status_label.text() + "\nGeçebilirsiniz.")
        else:
            turnstile_image_path = os.path.join(current_dir, "resimler/kapaliturnike.webp")
            turnstile_image = QImage(turnstile_image_path)
            if turnstile_image.isNull():
                print(f"Kapalı turnike resmi yüklenemedi: {turnstile_image_path}")
            self.status_label.setText(self.status_label.text() + "\nGeçemezsiniz.")

        pixmap = QPixmap.fromImage(turnstile_image)
        if not pixmap.isNull():
            self.camera_label.setPixmap(pixmap.scaled(self.camera_label.width(), self.camera_label.height(), Qt.KeepAspectRatio))

    def clear_turnstile_image(self):
        self.camera_label.clear()
        self.status_label.clear()
        self.image_displayed = False

if __name__ == '__main__':
    setup_database()
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
