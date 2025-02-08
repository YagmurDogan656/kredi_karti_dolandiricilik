# Kredi Kartı Dolandırıcılığı Tespiti

Bu proje, kredi kartı dolandırıcılığını tespit etmek için makine öğrenimi tekniklerini kullanmaktadır. Proje, kredi kartı işlemlerinin sınıflandırılması amacıyla **DistilBERT** modelini kullanarak verileri analiz etmektedir.

## İçindekiler

- [Proje Hakkında](#proje-hakkında)
- [Kullanılan Kütüphaneler](#kullanılan-kütüphaneler)
- [Veri Seti](#veri-seti)
- [Model Eğitimi](#model-eğitimi)
- [Sonuçlar](#sonuçlar)
- [Kurulum](#kurulum)

## Proje Hakkında

Bu proje, kredi kartı işlemlerinin dolandırıcılık olup olmadığını belirlemek için bir makine öğrenimi modeli geliştirmeyi amaçlamaktadır. Model, verileri analiz ederek dolandırıcılık işlemlerini tespit etmeye çalışmaktadır.

## Kullanılan Kütüphaneler

- `pandas`: Veri analizi için.
- `datasets`: Hugging Face veri seti yönetimi için.
- `transformers`: Model ve tokenizer'ı yüklemek için.
- `sklearn`: Veri setini bölmek ve doğruluk hesaplamak için.
- `numpy`: Sayısal işlemler için.

## Veri Seti

Proje, `creditcard5.csv` adlı bir CSV dosyasını kullanmaktadır. Bu dosya, kredi kartı işlemlerini ve dolandırıcılık sınıfını içermektedir. Veri seti, eğitim ve değerlendirme olarak ikiye ayrılmaktadır.

## Model Eğitimi

Model, **DistilBERT** tabanlı bir sınıflandırıcıdır. DistilBERT, BERT modelinin daha hafif ve hızlı bir versiyonudur ve doğal dil işleme görevlerinde yüksek performans sunar. Eğitim sürecinde aşağıdaki adımlar izlenmektedir:

1. **Veri Yükleme**: `creditcard5.csv` dosyası yüklenir.
2. **Veri Ön İşleme**: Hedef değişken ayarlanır ve veri seti eğitim ve değerlendirme olarak bölünür.
3. **Tokenizasyon**: Veri seti, modelin anlayabileceği bir formata dönüştürülür.
4. **Model Eğitimi**: Model, eğitim veri seti üzerinde eğitilir ve değerlendirme veri seti ile test edilir.

### Kullanılan Model

Proje, Hugging Face'in `transformers` kütüphanesinden **DistilBERT** modelini kullanmaktadır. Bu model, metin sınıflandırma görevlerinde etkili bir şekilde kullanılmakta ve dolandırıcılık tespiti gibi uygulamalarda yüksek doğruluk sağlamaktadır.

## Sonuçlar

Modelin eğitimi tamamlandıktan sonra, doğruluk ve diğer metrikler hesaplanır. Bu sonuçlar, modelin performansını değerlendirmek için kullanılır.

## Kurulum

Projenin çalışabilmesi için aşağıdaki kütüphanelerin yüklenmesi gerekmektedir:
pip install pandas datasets transformers scikit-learn numpy
