import numpy as np
import re
import string

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances


def my_preprocessing(line):
    line = line.lower()
    line = re.sub(r"[{}]".format(string.punctuation), " ", line)
    return line


def normalize(array, euclidean=False, cosine_distance=True):
    """
    Для перевода в Евклидово расстояние или косинусного подобия (пока не пригодилось)

    """
    X_normalized = preprocessing.normalize(array, norm='l2') #test_array = np.random.rand(3, 3)  # 3x3 - размерность масива
    if euclidean:
        euclidean_dist = euclidean_distances(X_normalized)
        return np.square(euclidean_dist)

    if cosine_distance:
        return 2 - 2*cosine_similarity(X_normalized)

# Текст для обучения (возможно он неподходит)
training_text = """
Объектом исследования являются алгоритм поиска текстовой информации по ключевым словам в социальных сетях.
Целью работы является получение алгоритма поиска текстовой информации по ключевым словам в социальных сетях. 
 Для этого необходимо решить задачу получения алгоритмов для поиска ключевых слов и извлечения признаков из текста. 
В процессе работы был выполнен анализ предметной области и обзор существующих алгоритмов для поиска ключевых слов 
и извлечения признаков из текста.  На основе анализа существующих решений был выбран алгоритм для реализации.
В результате работы получен алгоритма поиска текстовой информации по ключевым словам в социальных сетях.  
""".split("\n")[1:-1]

def kmeans_predict(*, text1, text2):
    tfidf_vectorizer = TfidfVectorizer(preprocessor=my_preprocessing)
    tfidf = tfidf_vectorizer.fit_transform(training_text)
    kmeans = KMeans(n_clusters=2).fit(tfidf)
    lines_for_predicting = [text1, text2]
    rezult = kmeans.predict(tfidf_vectorizer.transform(lines_for_predicting))
    if rezult[0] == rezult[1]:
        return 1   # Означает что тексты в одно кластере
    else:
        return 0


def compare_texts(*, tf_idf_list, tf_idf_etalon):
    array = np.array([[tf_idf_etalon]])
    index = 0
    for tf_idf in tf_idf_list:
        tmp = np.array([[tf_idf]])
        array = np.concatenate((array, tmp))
        index += 1
    kmeans = KMeans(n_clusters=2, random_state=0).fit(array)
    rezult = []
    for elem in array[1:]:
        predict = kmeans.predict([[tf_idf_etalon], elem])
        if predict[0] == predict[1]:    # 1 - элементы в одном кластере
            rezult.append(1)
        else:
            rezult.append(0)
    print(" len(rezult) " + str(len(rezult)))
    return rezult


if __name__ == "__main__":
    print(compare_texts(tf_idf_list=[1, 2, 3, 4, 5], tf_idf_etalon=2))
