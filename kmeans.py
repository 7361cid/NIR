import numpy as np

from sklearn.cluster import KMeans


def compare_texts(*, tf_idf_list, tf_idf_etalon_vector):
    array = np.array([tf_idf_etalon_vector])
    index = 0
    for tf_idf in tf_idf_list:
        tmp = np.array([tf_idf])
        array = np.concatenate((array, tmp))
        index += 1
    print("array " + str(array))
    kmeans = KMeans(n_clusters=2, random_state=0).fit(array)
    rezult = []
    for elem in array[1:]:
        predict = kmeans.predict([tf_idf_etalon_vector, elem])
        if predict[0] == predict[1]:    # 1 - элементы в одном кластере
            rezult.append(1)
        else:
            rezult.append(0)
    print(" len(rezult) " + str(len(rezult)))
    return rezult


if __name__ == "__main__":
    print(compare_texts(tf_idf_list=[[1, 2, 3], [2, 5, 2]], tf_idf_etalon_vector=[1, 1, 1]))
