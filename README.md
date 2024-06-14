# Yüz ve Plaka Tanıma Sistemi

Bu proje, yüz ve plaka tanıma işlemlerini gerçekleştiren bir sistemdir. Sistemde yüz tanıma ve plaka tanıma olmak üzere iki ana modül bulunmaktadır. Yüz tanıma modülü, kaydedilmiş yüzleri tanıyabilir ve bu yüzleri veritabanına kaydedebilir. Plaka tanıma modülü ise araç plakalarını tanıyabilir ve bu plakaları veritabanına kaydedebilir.

## Gereksinimler

- Python 3.x
- OpenCV
- face_recognition
- numpy
- sqlite3
- PyQt5
- pytesseract
- pickle

## Kurulum

1. Gereksinimleri yükleyin:
    ```bash
    pip install opencv-python face_recognition numpy pyqt5 pytesseract pickle-mixin
    ```

2. Veritabanını oluşturmak için `setup_database` fonksiyonunu çalıştırın.

## Kullanım

### Yüz ve Plaka Tanıma Sistemi

1. `main.py` dosyasını çalıştırın:
    ```bash
    python main.py
    ```

2. Ana pencerede yüz tanıma veya plaka tanıma modüllerinden birini seçin.
3. Yüz tanıma modülünde, yüzleri yükleyebilir ve veritabanına kaydedebilirsiniz.
4. Plaka tanıma modülünde, plakaları yükleyebilir ve veritabanına kaydedebilirsiniz.

### Ekran Görüntüleri

#### Ana Pencere

Ana pencerede, yüz tanıma, plaka tanıma ve yüz yükleme gibi işlemler için seçenekler bulunmaktadır. Ayrıca, kayıtlı yüzleri görüntüleyebilir ve tüm yüzleri işaretleyebilirsiniz.

![Ana Pencere](screenshots/Ekran%20Resmi%202024-06-04%2016.31.17.png)

#### Yüz Tanıma

Bu ekran görüntüsünde, sistemin yüzleri tanıma ve etiketleme işlevi gösterilmektedir. "Yeni Yuz" etiketiyle belirtilen yüzler sistemde yeni tanınmış ve henüz kaydedilmemiş yüzlerdir.

![Yüz Tanıma](screenshots/Ekran%20Resmi%202024-06-04%2016.33.07.png)

#### Kayıtlı Yüzler

Kayıtlı yüzler ekranında, veritabanına kaydedilmiş yüzleri görüntüleyebilir, işaretleyebilir, işaretleri kaldırabilir ve silebilirsiniz. Ayrıca, yüzlerin isimlerini de güncelleyebilirsiniz.

![Kayıtlı Yüzler 1](screenshots/Ekran%20Resmi%202024-06-04%2016.35.12.png)
![Kayıtlı Yüzler 2](screenshots/Ekran%20Resmi%202024-06-04%2016.36.01.png)

#### Yüz Tanıma ve Kod İncelemesi

Bu ekran görüntüsü, yüz tanıma işlevi sırasında tanınan yüzlerin ve terminaldeki kod çıktısının bir örneğini göstermektedir.

![Yüz Tanıma ve Kod İncelemesi](screenshots/Ekran%20Resmi%202024-06-04%2016.38.53.png)

#### Veritabanı Görüntüleme

Veritabanı tarayıcı kullanarak yüzlerin ve plakaların veritabanında nasıl saklandığını görebilirsiniz. Aşağıdaki görüntülerde, yüzlerin ve plakaların veritabanı kayıtları gösterilmektedir.

![Veritabanı Görüntüleme 1](screenshots/Ekran%20Resmi%202024-06-04%2016.47.45.png)
![Veritabanı Görüntüleme 2](screenshots/Ekran%20Resmi%202024-06-04%2016.48.23.png)

#### Giriş Kayıtları

Giriş kayıtları ekranında, tanınan yüzlerin giriş zamanlarını görebilirsiniz. Bu ekran, hangi yüzlerin hangi zamanlarda tanındığını listeler.

![Giriş Kayıtları](screenshots/Ekran%20Resmi%202024-06-04%2018.58.05.png)

## Fonksiyonlar

### Veritabanı Fonksiyonları

- `create_connection()`: Veritabanı bağlantısı oluşturur.
- `setup_database()`: Veritabanı kurulum ve güncellemeleri gerçekleştirir.
- `add_face(name, encoding, image)`: Veritabanına yeni bir yüz ekler.
- `get_faces()`: Veritabanındaki tüm yüzleri getirir.
- `log_recognition(rec_type, identifier)`: Tanıma olayını veritabanına kaydeder.
- `is_new_face(face_encoding, known_faces)`: Yeni bir yüz olup olmadığını kontrol eder.
- `encode_face(image)`: Bir yüzü kodlar.
- `recognize_faces(frame, known_faces)`: Bir karedeki yüzleri tanır.
- `recognize_plate(frame)`: Bir karedeki plakaları tanır.
- `add_plate(plate_number)`: Veritabanına yeni bir plaka ekler.
- `get_plates()`: Veritabanındaki tüm plakaları getirir.

### UI Fonksiyonları

- `initUI()`: Kullanıcı arayüzünü başlatır.
- `show_face_recognition_buttons()`: Yüz tanıma butonlarını gösterir.
- `show_plate_recognition_buttons()`: Plaka tanıma butonlarını gösterir.
- `upload_face()`: Yüz yükleme işlemini gerçekleştirir.
- `upload_plate()`: Plaka yükleme işlemini gerçekleştirir.
- `load_known_faces()`: Kayıtlı yüzleri yükler.
- `load_known_plates()`: Kayıtlı plakaları yükler.
- `view_faces()`: Kayıtlı yüzleri görüntüler.
- `view_plates()`: Kayıtlı plakaları görüntüler.
- `mark_and_notify(name)`: Bir yüzü işaretler ve kullanıcıyı bilgilendirir.
- `unmark_and_notify(name)`: Bir yüzün işaretini kaldırır ve kullanıcıyı bilgilendirir.
- `delete_and_notify(name)`: Bir yüzü siler ve kullanıcıyı bilgilendirir.
- `mark_and_notify_plate(plate_number)`: Bir plakayı işaretler ve kullanıcıyı bilgilendirir.
- `unmark_and_notify_plate(plate_number)`: Bir plakanın işaretini kaldırır ve kullanıcıyı bilgilendirir.
- `delete_and_notify_plate(plate_number)`: Bir plakayı siler ve kullanıcıyı bilgilendirir.
- `toggle_auto_save()`: Otomatik kaydetme işlemini açar/kapatır.
- `update_frame_face_recognition()`: Yüz tanıma çerçevesini günceller.
- `update_frame_plate_recognition()`: Plaka tanıma çerçevesini günceller.

## Katkıda Bulunma

Eğer bu projeye katkıda bulunmak isterseniz, lütfen aşağıdaki adımları takip edin:

1. Bu depoyu forklayın.
2. Yeni bir dal (branch) oluşturun: `git checkout -b yeni-ozellik`
3. Değişikliklerinizi yapın ve commit edin: `git commit -am 'Yeni özellik ekle'`
4. Dalınıza push yapın: `git push origin yeni-ozellik`
5. Bir pull isteği (pull request) oluşturun.

