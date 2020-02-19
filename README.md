About
Tenserflow kütüphanesi ile el yazıları için tahminleme yapan ve bunu öğrenerek yapan yapak zeka uygulaması.

Install
Python için ANACONDA kurulumu gerekiyor.
-https://www.anaconda.com/download/
-64-bit Anaconda indirip kurun. Ayrıca Python kurulumu yapmanıza gerek yok. Python Anaconda'yla birlikte gelecek.
-Kurulum esnasında "add python to your PATH" seçeneğini işaretleyin.

İlk olarak PYCHARM'ı kurmak gerekiyor.
- https://www.jetbrains.com/pycharm/download/
- Ben Pycharm Community Edition kullanıyorum. Siz isterseniz farklı bir IDE'de kullanabilirsiniz.
- Pycharm kurulumu yaptıysanız interpreter belirtin.
- File/Default Settings/Project Interpreter
- Sağdaki küçük butona tıklayıp "Add Local" seçin.
- Açılan pencereden "System Interpreter" seçeneğini seçip anaconda içerisinde python.exe yolunu verin. Bu yol şurası:
- C:\Users\KullanıcıAdı\Anaconda3\python.exe

Tenserflow Kurulumu
1. Tensorflow CPU (GPU kurulumunu internetten bulabilirsiniz.)
- Komut penceresinde aşağıdaki pip install'u çalıştırın:
C:\> pip install --upgrade tensorflow
- CPU için kurulum bu kadar. Komut penceresinde "python" yazıp aşağıdaki kod ile test edebilirsiniz:
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> print(sess.run(hello))
- Not: import tensorflow as tf yazdıktan sonra FutureWarning alabilirsiniz. Bu bir hata değil sadece numpy ile ilgili bir uyarı. Kodlarınız sorunsuz çalışacaktır.
- Sonuç olarak "Hello, Tensorflow!" göreceksiniz.