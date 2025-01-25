(# KMeans Kümeleme Projesi

## Proje Özeti
Bu proje, **KMeans Kümeleme Algoritması**nın perakende verisi üzerinde nasıl uygulandığını ve müşterileri satın alma davranışlarına göre gruplandırmayı gösterir. Bu analiz için kullanılan veri seti, Kaggle'dan temin edilmiştir.

### Veri Seti Detayları
- **Kaynak**: [Online Retail II Veri Seti](https://www.kaggle.com/datasets/kabilan45/online-retail-ii-dataset)
- **Açıklama**: Bu veri seti, Birleşik Krallık merkezli bir çevrimiçi perakendecinin işlem verilerini içermektedir; faturalar, stok kodları, açıklamalar, miktarlar ve müşteri kimlikleri gibi bilgiler yer almaktadır.

## Yöntem
1. **Veri Hazırlığı**
   - Eksik değerlerin işlenmesi ve aykırı değerlerin filtrelenmesi gibi veri temizleme ve ön işleme adımları.
   - Recency, Frequency ve Monetary değeri (RFM) gibi anlamlı metrikler oluşturmak için özellik çıkarımı.

2. **KMeans Algoritması**
   - Müşterilerin RFM puanlarına dayalı olarak farklı gruplara kümelemek için kullanıldı.
   - **Dirsek Yöntemi** kullanılarak en uygun küme sayısı belirlendi.

3. **Görselleştirme**
   - Sonuçları etkili bir şekilde yorumlamak için kümeler çizildi.

## Anahtar İpuçları
- Müşteriler, yüksek değerli ve düşük değerli müşteri segmentlerini belirlemek amacıyla gruplandırıldı.
- Sonuçlar, pazarlama stratejileri ve müşteri sadakati için uygulanabilir içgörüler sağladı.

## Kod ve Uygulama
Proje kodu Kaggle profilimde mevcuttur:
- [Kaggle'daki Proje Not Defteri](https://www.kaggle.com/ahmetyasirduman/code)

## Araçlar ve Teknolojiler
- **Programlama Dili**: Python
- **Kullanılan Kütüphaneler**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## Nasıl Çalıştırılır
1. Not defterini klonlayın veya veri setini Kaggle'dan indirin.
2. Tüm bağımlılıkların yüklendiğinden emin olun.
3. Sonuçları ve görselleştirmeleri görmek için not defterini çalıştırın.

## Benimle İletişime Geçin
Çalışmalarımla ilgili geri bildirimde bulunabilir veya takip edebilirsiniz:
- **LinkedIn**: [Ahmet Yasir Duman](https://www.linkedin.com/in/ahmet-yasir-duman-03b689256)
- **Kaggle Profilim**: [Ahmet Yasir Duman](https://www.kaggle.com/ahmetyasirduman)

---
**Not**: Detaylı bir açıklama için Kaggle profilimde yüklenen not defterine göz atabilirsiniz.)
