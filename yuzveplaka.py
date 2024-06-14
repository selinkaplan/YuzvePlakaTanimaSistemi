import sys
import cv2
import face_recognition
import numpy as np
import sqlite3
import re
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QLineEdit, QFileDialog, QDialog, QInputDialog, QScrollArea, QHBoxLayout, QMessageBox
from PyQt5.QtCore import QTimer, QDateTime
from PyQt5.QtGui import QImage, QPixmap
from datetime import datetime
import pytesseract
import pickle

detected_plates = {}

# Veritabanı bağlantısı
def create_connection():
    return sqlite3.connect('recognition.db')

# Veritabanı kurulumu ve güncelleme
def setup_database():
    with create_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS faces (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT,
                            encoding BLOB,
                            image BLOB,
                            marked INTEGER DEFAULT 0)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS plates (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            plate_number TEXT,
                            marked INTEGER DEFAULT 0)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS recognition_logs (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            type TEXT,
                            identifier TEXT,
                            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()

# Veritabanına yüz ekleme
def add_face(name, encoding, image):
    encoding_blob = pickle.dumps(encoding)
    with create_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO faces (name, encoding, image) VALUES (?, ?, ?)", (name, encoding_blob, image))
        conn.commit()

# Veritabanındaki tüm yüzleri alma
def get_faces():
    with create_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM faces")
        faces = cursor.fetchall()
    return [{"id": face[0], "name": face[1], "encoding": pickle.loads(face[2]), "image": face[3], "marked": face[4]} for face in faces]

# Veritabanında bir yüzü işaretleme
def mark_face(name):
    with create_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE faces SET marked = 1 WHERE name = ?", (name,))
        conn.commit()

# Veritabanında bir yüzün işaretini kaldırma
def unmark_face(name):
    with create_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE faces SET marked = 0 WHERE name = ?", (name,))
        conn.commit()

# Veritabanından bir yüzü silme
def delete_face(name):
    with create_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM faces WHERE name = ?", (name,))
        conn.commit()

# Veritabanındaki tüm yüzleri işaretleme
def mark_all_faces():
    with create_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE faces SET marked = 1")
        conn.commit()

# Veritabanına plaka ekleme
def add_plate(plate_number):
    with create_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO plates (plate_number) VALUES (?)", (plate_number,))
        conn.commit()

# Veritabanındaki tüm plakaları alma
def get_plates():
    with create_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM plates")
        plates = cursor.fetchall()
    return [{"id": plate[0], "plate_number": plate[1], "marked": plate[2]} for plate in plates]

# Veritabanında bir plakayı işaretleme
def mark_plate(plate_number):
    with create_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE plates SET marked = 1 WHERE plate_number = ?", (plate_number,))
        conn.commit()

# Veritabanında bir plakanın işaretini kaldırma
def unmark_plate(plate_number):
    with create_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE plates SET marked = 0 WHERE plate_number = ?", (plate_number,))
        conn.commit()

# Tanıma olayını kaydetme
def log_recognition(rec_type, identifier):
    with create_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO recognition_logs (type, identifier) VALUES (?, ?)", (rec_type, identifier))
        conn.commit()

# Yüzün yeni olup olmadığını kontrol etme
def is_new_face(face_encoding, known_faces):
    for known_face in known_faces:
        matches = face_recognition.compare_faces([known_face["encoding"]], face_encoding)
        if True in matches:
            return False
    return True

# Yüz kodlama
def encode_face(image):
    face_encodings = face_recognition.face_encodings(image)
    if face_encodings:
        return face_encodings[0]
    return None

# Bir karedeki yüzleri tanıma
def recognize_faces(frame, known_faces):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces([face["encoding"] for face in known_faces], face_encoding)
        name = "Yeni Yuz"
        face_distances = face_recognition.face_distance([face["encoding"] for face in known_faces], face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_faces[best_match_index]["name"]
                if name == "Isimsiz":
                    name = "Kayitli Ama Isimsiz"
                # Bu yüz işaretlenmiş mi kontrol et
                if known_faces[best_match_index]["marked"]:
                    log_recognition('face', name)
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(f"Marked face {name} recognized at {timestamp}!")
        face_names.append(name)

    return face_locations, face_names

# Bir karedeki plakaları tanıma
def recognize_plate(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    recognized_plates = []
    plate_locations = []

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            plate_img = gray[y:y + h, x:x + w]
            plate_img = cv2.adaptiveThreshold(plate_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            plate_text = extract_plate_text(plate_img)
            if plate_text and is_turkish_license_plate(plate_text) and plate_text not in recognized_plates:
                recognized_plates.append(plate_text)
                plate_locations.append((x, y, w, h))
                log_recognition('plate', plate_text)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, plate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return plate_locations, recognized_plates

def is_turkish_license_plate(plate_text):
    pattern = r'^\d{2}\s[A-Z]{1,3}\s\d{2,4}$'
    return re.match(pattern, plate_text) is not None

def extract_plate_text(plate_image):
    config = '--oem 3 --psm 7'
    plate_text = pytesseract.image_to_string(plate_image, config=config)
    return plate_text.strip()

# Ana pencere sınıfı
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        try:
            self.initUI()
            self.known_faces = []
            self.known_plates = []
            self.detected_plates = {}
            self.detected_faces = set()  # Önceden tespit edilen yüzleri takip etmek için
            self.load_known_faces()
            self.load_known_plates()
            self.timer = QTimer(self)
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Çözünürlüğü ayarla
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Çözünürlüğü ayarla
            if not self.cap.isOpened():
                print("Kamera açilamadi.")
                return
            self.auto_save_active = False
        except Exception as e:
            print(f"Initialization error: {e}")

    def initUI(self):
        self.setWindowTitle('Yüz ve Plaka Tanıma Sistemi')
        self.camera_label = QLabel(self)

        self.select_face_recognition_button = QPushButton('Yüz Tanıma', self)
        self.select_face_recognition_button.clicked.connect(self.show_face_recognition_buttons)
        self.select_face_recognition_button.setStyleSheet('background-color: lightblue; font-size: 18px; border-radius: 10px; padding: 20px')

        self.select_plate_recognition_button = QPushButton('Plaka Tanıma', self)
        self.select_plate_recognition_button.clicked.connect(self.show_plate_recognition_buttons)
        self.select_plate_recognition_button.setStyleSheet('background-color: lightgreen; font-size: 18px; border-radius: 10px; padding: 20px')

        self.upload_face_button = QPushButton('Yüz Yükle', self)
        self.upload_face_button.clicked.connect(self.upload_face)
        self.upload_face_button.setStyleSheet('background-color: lightblue; border-radius: 10px; padding: 10px')
        self.upload_face_button.hide()

        self.upload_plate_button = QPushButton('Plaka Yükle', self)
        self.upload_plate_button.clicked.connect(self.upload_plate)
        self.upload_plate_button.setStyleSheet('background-color: lightgreen; border-radius: 10px; padding: 10px')
        self.upload_plate_button.hide()

        self.view_faces_button = QPushButton('Kayıtlı Yüzleri Gör', self)
        self.view_faces_button.clicked.connect(self.view_faces)
        self.view_faces_button.setStyleSheet('background-color: lightcoral; border-radius: 10px; padding: 10px')
        self.view_faces_button.hide()

        self.view_plates_button = QPushButton('Kayıtlı Plakaları Gör', self)
        self.view_plates_button.clicked.connect(self.view_plates)
        self.view_plates_button.setStyleSheet('background-color: lightgoldenrodyellow; border-radius: 10px; padding: 10px')
        self.view_plates_button.hide()

        self.mark_all_faces_button = QPushButton('Tüm Yüzleri İşaretle', self)
        self.mark_all_faces_button.clicked.connect(mark_all_faces)
        self.mark_all_faces_button.setStyleSheet('background-color: lightgray; border-radius: 10px; padding: 10px')
        self.mark_all_faces_button.hide()

        self.auto_save_button = QPushButton('Oto Kaydet', self)
        self.auto_save_button.clicked.connect(self.toggle_auto_save)
        self.auto_save_button.setStyleSheet('background-color: lightpink; border-radius: 10px; padding: 10px')
        self.auto_save_button.hide()

        self.name_input = QLineEdit(self)
        self.name_input.hide()

        self.plate_input = QLineEdit(self)
        self.plate_input.hide()

        # Layout setup
        layout = QVBoxLayout()
        button_layout = QVBoxLayout()

        button_layout.addWidget(self.select_face_recognition_button)
        button_layout.addWidget(self.select_plate_recognition_button)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.camera_label)
        main_layout.addLayout(button_layout)
        layout.addLayout(main_layout)

        self.setLayout(layout)

        button_layout.addWidget(self.upload_face_button)
        button_layout.addWidget(self.name_input)
        button_layout.addWidget(self.upload_plate_button)
        button_layout.addWidget(self.plate_input)
        button_layout.addWidget(self.view_faces_button)
        button_layout.addWidget(self.view_plates_button)
        button_layout.addWidget(self.mark_all_faces_button)
        button_layout.addWidget(self.auto_save_button)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.camera_label)
        main_layout.addLayout(button_layout)
        layout.addLayout(main_layout)

        self.setLayout(layout)

    def show_face_recognition_buttons(self):
        self.upload_face_button.show()
        self.view_faces_button.show()
        self.mark_all_faces_button.show()
        self.upload_plate_button.hide()
        self.view_plates_button.hide()
        self.auto_save_button.hide()
        self.name_input.show()
        self.plate_input.hide()

        self.timer.stop()
        self.timer.timeout.connect(self.update_frame_face_recognition)
        self.timer.start(30)

    def show_plate_recognition_buttons(self):
        self.upload_plate_button.show()
        self.view_plates_button.show()
        self.auto_save_button.show()
        self.upload_face_button.hide()
        self.view_faces_button.hide()
        self.mark_all_faces_button.hide()
        self.name_input.hide()
        self.plate_input.show()

        self.timer.stop()
        self.timer.timeout.connect(self.update_frame_plate_recognition)
        self.timer.start(30)

    def upload_face(self):
        try:
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
                    name, ok = QInputDialog.getText(self, 'Yüz Kaydet', 'Yüz İsmi:')
                    if ok:
                        with open(fileName, 'rb') as f:
                            image_blob = f.read()
                        add_face(name, encoding, image_blob)
                        self.load_known_faces()
        except Exception as e:
            print(f"Error in uploading face: {e}")

    def upload_plate(self):
        try:
            plate_number, ok = QInputDialog.getText(self, "Plaka Girişi", "Plaka Numarası:")
            if ok and plate_number:
                plate_number = plate_number.upper()
                if is_turkish_license_plate(plate_number):
                    add_plate(plate_number)
                    self.load_known_plates()
                    self.plate_input.clear()
                else:
                    QMessageBox.warning(self, "Hatalı Format", "Lütfen geçerli bir Türk plakası girin.")
        except Exception as e:
            print(f"Error in uploading plate: {e}")

    def load_known_faces(self):
        try:
            self.known_faces = get_faces()
        except Exception as e:
            print(f"Error in loading known faces: {e}")

    def load_known_plates(self):
        try:
            self.known_plates = get_plates()
        except Exception as e:
            print(f"Error in loading known plates: {e}")

    def view_faces(self):
        try:
            dialog = QDialog(self)
            dialog.setWindowTitle("Kayıtlı Yüzler")
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
                    label.setPixmap(pixmap.scaled(100, 100, aspectRatioMode=1))
                    layout.addWidget(QLabel(f"Name: {face['name']}"))
                    layout.addWidget(label)
                    if face["marked"]:
                        mark_label = QLabel("Isaretli")
                        mark_label.setStyleSheet('background-color: yellow')
                        layout.addWidget(mark_label)
                    mark_button = QPushButton("Isaretle", container)
                    mark_button.clicked.connect(lambda _, name=face['name']: self.mark_and_notify(name))
                    mark_button.setStyleSheet('background-color: lightblue')
                    layout.addWidget(mark_button)
                    unmark_button = QPushButton("Isaret Kaldir", container)
                    unmark_button.clicked.connect(lambda _, name=face['name']: self.unmark_and_notify(name))
                    unmark_button.setStyleSheet('background-color: lightgreen')
                    layout.addWidget(unmark_button)
                    delete_button = QPushButton("Sil", container)
                    delete_button.clicked.connect(lambda _, name=face['name']: self.delete_and_notify(name))
                    delete_button.setStyleSheet('background-color: lightcoral')
                    layout.addWidget(delete_button)
                    name_button = QPushButton("Isimle", container)
                    name_button.clicked.connect(lambda _, face_id=face['id']: self.rename_face(face_id))
                    name_button.setStyleSheet('background-color: lightgoldenrodyellow')
                    layout.addWidget(name_button)

            scroll.setWidget(container)
            layout_wrapper = QVBoxLayout(dialog)
            layout_wrapper.addWidget(scroll)
            dialog.setLayout(layout_wrapper)
            dialog.exec_()
        except Exception as e:
            print(f"Error in viewing faces: {e}")

    def mark_and_notify(self, name):
        try:
            mark_face(name)
            self.load_known_faces()
            QMessageBox.information(self, "Başarılı", f"Yüz {name} işaretlendi.")
        except Exception as e:
            print(f"Error in marking face: {e}")

    def unmark_and_notify(self, name):
        try:
            unmark_face(name)
            self.load_known_faces()
            QMessageBox.information(self, "Başarılı", f"Yüz {name} için işaret kaldırıldı.")
        except Exception as e:
            print(f"Error in unmarking face: {e}")

    def delete_and_notify(self, name):
        try:
            delete_face(name)
            self.load_known_faces()
            QMessageBox.information(self, "Başarılı", f"Yüz {name} silindi.")
        except Exception as e:
            print(f"Error in deleting face: {e}")

    def rename_face(self, face_id):
        try:
            new_name, ok = QInputDialog.getText(self, 'Yüzü Yeniden İsimle', 'Yeni ismi girin:')
            if ok and new_name:
                with create_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("UPDATE faces SET name = ? WHERE id = ?", (new_name, face_id))
                    conn.commit()
                self.load_known_faces()
                QMessageBox.information(self, "Başarılı", f"Yüz {face_id} ismi {new_name} olarak değiştirildi.")
        except Exception as e:
            print(f"Error in renaming face: {e}")

    def view_plates(self):
        try:
            dialog = QDialog(self)
            dialog.setWindowTitle("Kayıtlı Plakalar")
            layout = QVBoxLayout(dialog)
            plates = get_plates()
            for plate in plates:
                plate_label = QLabel(f"Plaka Numarası: {plate['plate_number']}")
                layout.addWidget(plate_label)
                if plate["marked"]:
                    mark_label = QLabel("Isaretli")
                    mark_label.setStyleSheet('background-color: yellow')
                    layout.addWidget(mark_label)
                mark_button = QPushButton("Isaretle", self)
                mark_button.clicked.connect(lambda _, plate_number=plate['plate_number']: self.mark_and_notify_plate(plate_number))
                mark_button.setStyleSheet('background-color: lightblue')
                layout.addWidget(mark_button)
                unmark_button = QPushButton("Isaret Kaldir", self)
                unmark_button.clicked.connect(lambda _, plate_number=plate['plate_number']: self.unmark_and_notify_plate(plate_number))
                unmark_button.setStyleSheet('background-color: lightgreen')
                layout.addWidget(unmark_button)
                delete_button = QPushButton("Sil", self)
                delete_button.clicked.connect(lambda _, plate_number=plate['plate_number']: self.delete_and_notify_plate(plate_number))
                delete_button.setStyleSheet('background-color: lightcoral')
                layout.addWidget(delete_button)

            dialog.setLayout(layout)
            dialog.exec_()
        except Exception as e:
            print(f"Error in viewing plates: {e}")

    def mark_and_notify_plate(self, plate_number):
        try:
            mark_plate(plate_number)
            self.load_known_plates()
            QMessageBox.information(self, "Başarılı", f"Plaka {plate_number} işaretlendi.")
        except Exception as e:
            print(f"Error in marking plate: {e}")

    def unmark_and_notify_plate(self, plate_number):
        try:
            unmark_plate(plate_number)
            self.load_known_plates()
            QMessageBox.information(self, "Başarılı", f"Plaka {plate_number} için işaret kaldırıldı.")
        except Exception as e:
            print(f"Error in unmarking plate: {e}")

    def delete_and_notify_plate(self, plate_number):
        try:
            with create_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM plates WHERE plate_number = ?", (plate_number,))
                conn.commit()
            self.load_known_plates()
            QMessageBox.information(self, "Başarılı", f"Plaka {plate_number} silindi.")
        except Exception as e:
            print(f"Error in deleting plate: {e}")

    def toggle_auto_save(self):
        self.auto_save_active = not self.auto_save_active
        if self.auto_save_active:
            self.auto_save_button.setStyleSheet('background-color: red')
        else:
            self.auto_save_button.setStyleSheet('background-color: lightpink')

    def update_frame_face_recognition(self):
        try:
            ret, frame = self.cap.read()
            if not ret:
                return

            frame = cv2.resize(frame, (640, 480))  # Daha iyi performans için kare boyutunu değiştir

            face_locations, face_names = self.process_face_recognition(frame)

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                color = (0, 255, 0) if name == "Yeni Yuz" else (0, 255, 255) if name == "Kayitli Ama Isimsiz" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(qt_image))
        except Exception as e:
            print(f"Error in updating frame: {e}")

    def update_frame_plate_recognition(self):
        try:
            ret, frame = self.cap.read()
            if not ret:
                return

            frame = cv2.resize(frame, (640, 480))  # Daha iyi performans için kare boyutunu değiştir

            plate_locations, plate_numbers = self.process_plate_recognition(frame)

            for (x, y, w, h), plate_number in zip(plate_locations, plate_numbers):
                color = (0, 255, 0)  # Varsayılan renk
                # Plakanın işaretli olup olmadığını kontrol et
                if plate_number in [plate["plate_number"] for plate in self.known_plates if plate["marked"]]:
                    color = (0, 0, 255)  # İşaretli plakalar için kırmızı renk
                    # Plaka numarasına tarih ve saat ekle
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    plate_number = f"{plate_number} - {timestamp}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, plate_number, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            if self.auto_save_active:
                for plate_number in plate_numbers:
                    if plate_number not in self.detected_plates:
                        add_plate(plate_number)
                        self.detected_plates[plate_number] = {'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'last_seen': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(qt_image))
        except Exception as e:
            print(f"Error in updating frame: {e}")

    def process_face_recognition(self, frame):
        try:
            face_locations, face_names = recognize_faces(frame, self.known_faces)
            return face_locations, face_names
        except Exception as e:
            print(f"Error in face recognition processing: {e}")
            return [], []

    def process_plate_recognition(self, frame):
        try:
            plate_locations, plate_numbers = recognize_plate(frame)
            return plate_locations, plate_numbers
        except Exception as e:
            print(f"Error in plate recognition processing: {e}")
            return [], []

if __name__ == '__main__':
    try:
        setup_database()
        app = QApplication(sys.argv)
        mainWindow = MainWindow()
        mainWindow.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Error in main: {e}")
