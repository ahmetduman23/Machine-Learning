## Proje 1: Denetimsiz Makine Öğrenmesi - Kalp Hastalığı Risk Analizi

Bu projede, KMeans ve Gaussian Mixture Model (GMM) kullanılarak kalp hastalığı risk faktörleri temelinde bireyler gruplandırılmıştır. Veri setinde, kalp hastalığı ve cilt kanseri gibi hastalıkların varlığını gösteren kategorik veriler ve diğer sağlık göstergeleri yer almaktadır.

Adımlar:
  1.Veri İşleme:

  *  Categorical veriler etiketlenmiş (Label Encoding) ve One-Hot     Encoding ile dönüştürülmüştür.

  *  Eksik veriler temizlenmiştir.
  
  *  Veriler ölçeklendirilmiştir (Standard Scaler).


  2.Modelleme:

  *   KMeans ve GMM algoritmaları kullanılarak 2 kümeye ayırma işlemi 
  gerçekleştirilmiştir.
  
  * Her iki algoritma için Silhouette Skoru ve Adjusted Rand Index (ARI) hesaplanmıştır.


  3.Sonuçlar:
  * Karışıklık matrisleri görselleştirilerek kümelerin doğruluğu incelenmiştir.

Daha fazla detaya ve kodlara aşağıdaki Kaggle bağlantısından ulaşabilirsiniz:(https://www.kaggle.com/code/ahmetyasirduman/unsupervised-learning)


## Proje 2: Denetimli Makine Öğrenmesi - Kalp Hastalığı Tahmin Modeli

Bu projede, Random Forest, XGBoost, KNN, Gaussian Naive Bayes gibi farklı makine öğrenmesi algoritmaları kullanılarak kalp hastalığı tahmini yapılmıştır. Proje kapsamında hiperparametre optimizasyonu ve çapraz doğrulama yapılmıştır.

Adımlar:
  1.Veri İşleme:

  * Kategorik veriler One-Hot Encoding ile dönüştürülmüş, eksik veriler temizlenmiştir.

  * Veriler ölçeklendirilmiş ve dengesiz veri seti SMOTE ile dengelenmiştir.

  2.Modelleme:

  * Random Forest, XGBoost, KNN ve Naive Bayes algoritmaları eğitilmiş ve performansları karşılaştırılmıştır.

  * GridSearchCV kullanılarak en iyi hiperparametreler belirlenmiştir.

  3.Sonuçlar:

  * Çeşitli metrikler (Accuracy, Precision, Recall, F1 Score)   kullanılarak sonuçlar değerlendirilmiştir.

  * Hata oranları ve karışıklık matrisi görselleştirilmiştir.

Daha fazla detaya ve kodlara aşağıdaki Kaggle bağlantısından ulaşabilirsiniz:(https://www.kaggle.com/code/ahmetyasirduman/4-models-88-accuracy)
