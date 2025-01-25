# Heart Disease Prediction

## Proje Amacı

Bu proje, bireylerin kalp hastalığına yakalanıp yakalanmadığını tahmin etmek üzere bir makine öğrenimi modeli geliştirmek için tasarlanmıştır. Proje, çeşitli algoritmaların başarımlarını kıyaslayarak en iyi sonucu veren modeli belirlemeyi amaçlamaktadır.

---

## Veriseti

- **Kaynak**: [Kaggle Üzerinden Erişim](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease)
- **Boyut**: 319,795 satır, 38 özellik
- **Açıklama**: Verisetinde bireylerin yaşam tarzları, fiziksel ve zihinsel sağlık durumları ve hastalık geçmişlerine dair bilgiler yer almaktadır.

### Özellikler

| Özellik Adı        | Açıklama                                     |
|--------------------|---------------------------------------------|
| **BMI**            | Beden kitle indeksi.                       |
| **PhysicalHealth** | Fiziksel sağlık sorunları yaşanılan gün sayısı. |
| **MentalHealth**   | Zihinsel sağlık sorunları yaşanılan gün sayısı. |
| **SleepTime**      | Ortalama uyku süresi (saat).               |
| **SkinCancer**     | Cilt kanseri geçmişi (1: Evet, 0: Hayır).   |

Ek olarak, verisetinde cinsiyet, yaş aralığı, sigara kullanımı gibi çeşitli kategorik değişkenler bulunmaktadır.

---

## Kullanılan Algoritmalar

Projede aşağıdaki algoritmalar kullanılmıştır:

1. **K-Nearest Neighbors (KNN)**
2. **Random Forest**
3. **Gaussian Naive Bayes**
4. **XGBoost**

---

## Sonuçlar

### Model Performans Karşılaştırması

| Model             | Doğruluk | Precision | Recall | F1 Score |
|-------------------|----------|-----------|--------|----------|
| **XGBoost**       | 0.884    | 0.893     | 0.872  | 0.883    |
| **KNN**           | 0.853    | -         | -      | -        |
| **Random Forest** | 0.722    | -         | -      | -        |
| **GaussianNB**    | 0.698    | -         | -      | -        |

**XGBoost** algoritması en iyi performansı göstermiştir. Hyperparametre optimizasyonu ile aşağıdaki değerler elde edilmiştir:

- **learning_rate**: 0.2
- **max_depth**: 7
- **n_estimators**: 300

### Performans Değerlendirmesi

| Metric       | Değer   |
|--------------|---------|
| **Accuracy** | 0.8838  |
| **Precision**| 0.8934  |
| **Recall**   | 0.8719  |
| **F1 Score** | 0.8825  |

---

## Proje Dosyaları

- **Kodlar ve Analizler**: [Kaggle Notebook](https://www.kaggle.com/code/ahmetyasirduman/Heart_Diaese_Prediction/notebook)
- **GitHub**: Kodların GitHub'da paylaşılması için yakında güncellemeler yapılacaktır.

---

## LinkedIn Paylaşımı

Projeyi LinkedIn profilimde incelemek için: [LinkedIn Profilim](https://www.linkedin.com/in/ahmet-yasir-duman-03b689256)

---

## Gelecek Adımlar

- Daha fazla veri için modele transfer öğrenme uygulanması.
- Diğer algoritmaların hyperparametre optimizasyonu.
- Daha detaylı görsel ve çıktı analizleri.
