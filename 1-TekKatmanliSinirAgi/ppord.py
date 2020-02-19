import tensorflow as tf
# Tensorflow kütüphanesini tf olarak çektik.

from tensorflow.examples.tutorials.mnist import input_data
# Google tarafından sunulan datasete ulaştık.

mnist = input_data.read_data_sets("data/MNIST/", one_hot=True)
# Datasetimizi ilgili klasöre ekledik ve mnist değişkenine atadık.

x = tf.placeholder(tf.float32, [None, 784])
# Yer tutucu oluşturduk ve datasetteki her veriyi tek tek taşır. Eğitim aşamasında doldur boşalt yapılacaktır. (Tf kütüphanesine ait.)
# 784(28x28) gelen datasetlerin(fotoğraf) boyutu.
# None olması ise gelen verinin sınırlandırılmasıdır. None olduğu için sınırı olmayacak.

y_true = tf.placeholder(tf.float32, [None, 10])
# 10 sayısı sınıf sayımızdır. Verisetimizde 0-9 rakamları olduğu için toplam 10 sınıf olmaktadır.

w = tf.Variable(tf.zeros([784, 10]))
#w ağırlık
# Variable olarak verdiğimizde bunun eğitilebilir değerler olduğunu belirtmiş oluyoruz.

b = tf.Variable(tf.zeros([10]))
#b, yapay sinir ağlarındaki baestir ve 10 elemanlı bir vektör olarak tanımladık.

logits = tf.matmul(x, w) + b
#tf.matmul ile matrislerde çarpma işlemini gerçekleştiriyoruz ve baes ile topluyoruz.

y = tf.nn.softmax(logits)
# y tahmin değerleri
# tf kütüphanesinde aktivasyon formüllerine ulaşarak softmax ile değerleri 0-1 aralığına sıkıştırıyoruz.

xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
# Kaybı 0'a çekerek hatayı azaltmaya çalışıyoruz.

loss = tf.reduce_mean(xent)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
#tahminin doğrumu yanlışmı olduğunun sonucunu mantıksal olarak döndürür.

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimize = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
# Hedefe giderken atılacak adım sayısı 0.5 olarak verildi.

sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 128

#Eğitimin gerçekleştiği fonksiyon.
def training_step(iterations):
    for i in range(iterations):
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        # x batch resimler, y_batch resim etiketleri, mesela 10. verinin etiketi 3 ise o veri 3'tür.
        feed_dict_train = {x: x_batch, y_true: y_batch}
        sess.run(optimize, feed_dict=feed_dict_train)

#Testin gerçekleştiği fonksiyon
def test_accuarcy():
    feed_dict_test = {x: mnist.test.images, y_true: mnist.test.labels}
    acc = sess.run(accuracy, feed_dict=feed_dict_test)
    print('Testing accuracy:', acc)

training_step(2000)
test_accuarcy()