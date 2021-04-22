import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_cosine_sim(*strs):
    vectors = [t for t in get_vectors(*strs)]
    return cosine_similarity(vectors)


def get_vectors(*strs):
    text = [t for t in strs]
    vectorizer = CountVectorizer(text)
    vectorizer.fit(text)
    return vectorizer.transform(text).toarray()


def show_info_about_compare_vectors(array):
    for i in range(len(array)):
        for j in range(len(array)):
            procent_shodstva = re.findall("\d+", array[i][j])  # Иногда vectorizer помещает не число и его нужно вытащить регулярным выражением
            print("Текст " + str(i+1) + " схож с текстом " + str(j+1) + " на " + str(procent_shodstva) + " %")


if __name__ == "__main__":
    show_info_about_compare_vectors(get_cosine_sim("Мой текст 1", "Мой текст еще", "Просто текст"))
