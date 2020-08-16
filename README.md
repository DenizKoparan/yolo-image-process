# yolo-image-process
Projenin amacı araç tespiti ile araç sayılarını ve bekleme sürelerini hesaplamaktır.
Bu veriler kullanılarak araçların trafikte bekleme sürelerini minimize etmek planlanmıştır.
Kullanılan görüntülerdeki araçların tespiti için YOLO algoritması kullanılmıştır.
Başarılı bir araç tespiti için hazır verisetleri kullanılmamıştır. Kendi eğittiğimiz verisetleri
ve cfg dosyaları kullanılmıştır. Araçların konumu ve bekleme sürelerinin hesaplanılması
için çizgi methodu uygulanmıştır.

Örnek Çalıştırma Komutu:
python imagesA.py --images inputs/inputA/ --outputs outputs/