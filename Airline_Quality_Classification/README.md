# Airline Quality Classification

## Proje Hakkında
Bu proje, uçak yolculuğu deneyimlerinin değerlendirilmesini amaçlayan bir sınıflandırma çalışmasıdır. XGBoost algoritması kullanılarak yolcuların memnuniyet durumu tahmin edilmiştir. Proje, Kaggle üzerinde yer alan [Airline Quality Ratings Dataset](https://www.kaggle.com/datasets/mikhail1681/airline-quality-ratings) ile gerçekleştirilmiştir.

---

## Kullanılan Teknolojiler
- **Programlama Dili:** Python
- **Kütüphaneler:**
  - pandas
  - numpy
  - seaborn
  - matplotlib
  - scikit-learn
  - xgboost

---

## Veri Seti
**Veri Seti Kaynağı:** [Kaggle](https://www.kaggle.com/datasets/mikhail1681/airline-quality-ratings)

**Değişkenler:**
- Yolcu bilgileri (Cinsiyet, Yaş vb.)
- Yolculuk özellikleri (Seyahat tipi, Uçuş mesafesi vb.)
- Hizmet değerlendirmeleri (Yemek kalitesi, Temizlik vb.)
- Memnuniyet durumu (Sınıflandırma hedefi: Memnun veya Memnun Değil)

---

## Çalışma Adımları
1. **Veri Ön İşleme:**
   - Eksik verilerin analizi ve doldurulması.
   - Değişkenlerin ölçeklenmesi.
2. **Modelleme:**
   - XGBoost algoritması ile sınıflandırma modeli oluşturuldu.
   - Model parametreleri GridSearchCV ile optimize edildi.
3. **Değerlendirme:**
   - Confusion Matrix, ROC-AUC gibi metriklerle modelin başarımı değerlendirildi.
   
---

## Sonuçlar
- Model, yolcuların memnuniyet durumunu yüksek doğruluk oranıyla tahmin edebilmiştir.
- Veri görselleştirme yöntemleriyle modelin performansı ve önemli değişkenler analiz edilmiştir.

---

## Dosya Yapısı
```
Airline-Quality-Classification/
├── airline-quality-xgboost.ipynb   # Jupyter Notebook dosyası
├── README.md                       # Açıklama dosyası

```

---

## Nasıl Çalıştırılır?
1. Gerekli Python kütüphanelerini yükleyin:
   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn xgboost
   ```
2. Projeyi çalıştırmak için Jupyter Notebook dosyasını açın:
   ```bash
   jupyter notebook airline-quality-xgboost.ipynb
   ```

---

## İlgili Linkler
- **Proje Kodu (Kaggle):** [Ahmet Yasir Duman](https://www.kaggle.com/ahmetyasirduman/code)
- **Veri Seti:** [Airline Quality Ratings Dataset](https://www.kaggle.com/datasets/mikhail1681/airline-quality-ratings)
- **LinkedIn Profilim:** [Ahmet Yasir Duman](https://www.linkedin.com/in/ahmet-yasir-duman-03b689256)

---

## İletişim
Herhangi bir soru veya öneriniz için:
- **E-posta:** [ahmetyasirduman@gmail.com](mailto:ahmetyasirduman@gmail.com)
- **LinkedIn:** [Ahmet Yasir Duman](https://www.linkedin.com/in/ahmet-yasir-duman-03b689256)
