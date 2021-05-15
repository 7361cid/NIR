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


def show_info_about_compare_vectors(*, texts_list):
    """Первый текст должен быть эталлоном для вывода статистики"""
    array = get_cosine_sim(*texts_list)

    #for i in range(len(texts_list)):
      #  print("text №" + str(i+1) + " " + texts_list[i])
    rezult = []
    for j in range(len(array)):
        procent_shodstva = float(".".join(re.findall(r"\d+", str(array[0][j]))))   #
        # Иногда vectorizer помещает не число и его нужно вытащить регулярным выражением
        procent_shodstva = round(procent_shodstva * 100)
        if j != 0:
            print("Текст 1 схож с текстом " + str(j + 1) + " на " + str(procent_shodstva) + " %")
            rezult.append(procent_shodstva)
    return rezult  # возвращает список косинусных расстояний эталонного текста относительно найденных



if __name__ == "__main__":
    show_info_about_compare_vectors(texts_list=("Пример текста", "Еще больше текста", "Не похож на предыдущие"))
