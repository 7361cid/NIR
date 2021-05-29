import text_search
import word_vector
import my_vk_parser as vk
import openpyxl
import kmeans
from sklearn.datasets import fetch_20newsgroups


def basic_test(*, key_info, count, domain, sheet_name, book_name, excel_index,
               lemotize=False, tf_idf_avg=False, Ngram=0):
    """
    Функция тестирования
    Алгоритм
        1 - выполняет поиск постов в соцсети по ключевым словам
        2 - на основе одного из постов расширяет набор ключевых слов
        3 - повторный поиск по ключевым словам
        4 - для найденных постов считается TF-IDF
        5 - для постов с самыми высоким значениями рассчитывается косинсное расстояние
    Параметры
    key_info - информация для поиска
    count - количество постов
    domain - домен паблика
    sheet - лист для записи в excel
    excel_index - индекс для записи   в таблицу excel
    индекс нужно менять между вызовами тестов для того чтобы один тест не переписал другой
    lemotize - использовать лематизацию или нет
    Ngram - размер N-граммы для разделения
    tf_idf_avg - если True то tf-idf усредняется
    """
    wb = openpyxl.load_workbook(filename=book_name)  # файл должен существовать с листом "test"
    sheet = wb[sheet_name]
    sheet['A1'] = "косинусное расстояние"
    sheet['B1'] = "tf-idf"
    sheet['C1'] = "kmeans"
    sheet['D1'] = "ari"
    sheet['E1'] = "косинусное расстояние ari"
    sheet['F1'] = "Параметры теста"


    # поиск постов с ключевыми словами
    texts_list = vk.get_text_from_vk(query=key_info, count=count, domain=domain,
                                     lemotize=lemotize, Ngram=Ngram)
    # выбор текста пользователем(берем 1ый не пустой текст)
    user_text = vk.user_chouse_emulator(texts_list=texts_list, random=False)
    # выкинуть из текста символы пунтуации, чтобы не было слов в стиле "Я,"
    text_search.delete_bad_symbols(key_text)
    text_search.delete_bad_symbols(user_text)
    key_list = key_info.split()  # Разбиение по пробелам
    user_text_words_list = user_text.split()
    new_key_list = []
    new_key_list.extend(key_list)
    for key in key_list:  # поиск новых ключевых слов
        new_keys = text_search.frequancy_analis(words_list=user_text_words_list, key_word=key, max_new_words=2)
        if new_keys:
            if type(new_keys) == tuple or type(new_keys) == list:
                new_key_list.extend(list(new_keys))
            else:
                new_key_list.append(new_keys)

    new_key_list = list(set(new_key_list))  # удаление одинаковых ключей
    print(new_key_list)

    new_key_info = " ".join(new_key_list)
    # поиск постов с расширенным набором ключевых слов
    new_texts_list = vk.get_text_from_vk(query=new_key_info, count=count, domain=domain,
                                         lemotize=lemotize, Ngram=Ngram)
    new_texts_list = list(set(new_texts_list))  # удаление одинаковых текстов
    new_texts_list.insert(0, user_text)  # вставка эталонного текста в новый набор
    kolvo_postov = len(new_texts_list)
    rezult = text_search.search(key_info=key_info, texts_list=new_texts_list,
                                tf_idf_avg=tf_idf_avg)  # расчет TF-IDF
    find_user_text_rezult = find_user_text(rezult=rezult, user_text=user_text_words_list)
    tf_idf_etalon = find_user_text_rezult["TF-IDF"]
    tf_idf_etalon_vector = find_user_text_rezult["TF-IDF_Vector"]
    ari_etalon = text_search.ari(text=user_text)
    rezult.sort(key=lambda x: x["TF-IDF"])  # Сортируем найденные тексты по TF-IDF
    top_five_tf_idf = find_top_5(rezult=rezult, tf_idf_etalon=tf_idf_etalon)
    top_five_tf_idf_texts_list = [" ".join(r["text_words_list"]) for r in top_five_tf_idf]
    top_five_tf_idf_value = [r["TF-IDF"] for r in top_five_tf_idf]
    top_five_tf_idf_vectors_list = [r["TF-IDF_Vector"] for r in top_five_tf_idf]
    texts_list_from_top_five_tf_idf = [user_text]
    texts_list_from_top_five_tf_idf.extend(top_five_tf_idf_texts_list)
    # рассчет косинусного расстояния для текстов отобранных по tf-idf
    cos_sim_list = word_vector.show_info_about_compare_vectors(texts_list=texts_list_from_top_five_tf_idf) # рассчет косинусного расстояния
    kmeans_rezult_list = kmeans.compare_texts(tf_idf_etalon_vector=tf_idf_etalon_vector,
                                              tf_idf_list=top_five_tf_idf_vectors_list)
    print(top_five_tf_idf_value)
    print(cos_sim_list)

    top_five_ari = find_top_5(rezult=rezult, ari_etalon=ari_etalon)
    top_five_ari_value = [r["ARI"] for r in top_five_ari]
    top_five_ari_texts_list = [" ".join(r["text_words_list"]) for r in top_five_ari]
    texts_list_from_top_five_ari = [user_text]
    texts_list_from_top_five_ari.extend(top_five_ari_texts_list)
    # рассчет косинусного расстояния для текстов отобранных по ari
    cos_sim_list_ari = word_vector.show_info_about_compare_vectors(texts_list=texts_list_from_top_five_ari)

    cos_sim_excel_index = tfidf_excel_index = kmeans_excel_index = ari_excel_index = \
        cos_sim_ari_excel_index = excel_index
    info = " domain " + str(domain) + " count " + str(count) + " keyinfo " + str(key_info) + " lemotize " \
           + str(lemotize) + " tf_idf_avg " + str(tf_idf_avg) + " Ngramm " + str(Ngram)
    sheet['F' + str(excel_index)] = info
    sheet['G' + str(excel_index)] = "итоговое количество постов " + str(kolvo_postov)
    for cos_sim in cos_sim_list:
        sheet['A' + str(cos_sim_excel_index)] = str(cos_sim)
        cos_sim_excel_index += 1

    for tfidf_value in top_five_tf_idf_value:
        sheet['B' + str(tfidf_excel_index)] = str(tfidf_value).replace('.', ',')  # разделитель в excel это ,
        tfidf_excel_index += 1

    for kmeans_value in kmeans_rezult_list:
        sheet['C' + str(kmeans_excel_index)] = str(kmeans_value).replace('.', ',')  # разделитель в excel это ,
        kmeans_excel_index += 1

    for ari in top_five_ari_value:
        sheet['D' + str(ari_excel_index)] = str(ari).replace('.', ',')  # разделитель в excel это ,
        ari_excel_index += 1

    for ari in cos_sim_list_ari:
        sheet['E' + str(cos_sim_ari_excel_index)] = str(ari).replace('.', ',')  # разделитель в excel это ,
        cos_sim_ari_excel_index += 1

    wb.save(book_name)


def find_user_text(*, rezult, user_text):
    """
    Чтобы найти TF-IDF эталонного текста, он должен быть найден среди результа повторного поиска
    """
    similar_to_user_text = rezult[0]
    procent_shodstva = 0
    for r in rezult:
        tmp_procent_shodstva = word_list_compare(user_text, r["text_words_list"])
        if tmp_procent_shodstva > procent_shodstva:
            similar_to_user_text = r
            procent_shodstva = tmp_procent_shodstva

    return similar_to_user_text


def word_list_compare(list1, list2):
    similar_words_count = 0
    for word in list2:
        if word in list1:
            similar_words_count += 1
    return similar_words_count/len(list2)


def test_newsgroups(*, categories, count, book_name, sheet_name, excel_index, tf_idf_avg, key_text_length):
    """
       Функция тестирования
       Алгоритм
           1 - выполняет поиск текстов (выбор по категории)
           2 - на основе одного из текстов создается набор ключевых слов
           3 - для текстов считается TF-IDF
           4 - для текстов с самыми высоким значениями TF-IDF рассчитывается косинусное расстояние
       Параметры
       count - количество постов
       categories - список категорий текста
       sheet - лист для записи в excel
       excel_index - индекс для записи   в таблицу excel
       индекс нужно менять между вызовами тестов для того чтобы один тест не переписал другой
       tf_idf_avg - если True то tf-idf усредняется
       """
    wb = openpyxl.load_workbook(filename=book_name)  # файл должен существовать с листом "test"
    sheet = wb[sheet_name]
    sheet['A1'] = "косинусное расстояние"
    sheet['B1'] = "tf-idf"
    sheet['C1'] = "kmeans"
    sheet['D1'] = "ari"
    sheet['E1'] = "косинусное расстояние ari"
    sheet['F1'] = "Параметры теста"

    # поиск постов с по категориям
    texts_list = get_text_from_newsgroups(categories=categories, count=count)
    # выбор текста пользователем(берем 1ый не пустой текст)
    user_text = vk.user_chouse_emulator(texts_list=texts_list, random=False)
    # в начале новостных текстов идет email поэтому лучше брать не первые слова
    key_text = " ".join(user_text.split()[50:50+key_text_length])
    text_search.delete_bad_symbols(key_text)
    text_search.delete_bad_symbols(user_text)
    user_text_words_list = user_text.split()
    new_texts_list = list(set(texts_list))  # удаление одинаковых текстов
    rezult = text_search.search(key_info=key_text, texts_list=new_texts_list,
                                tf_idf_avg=tf_idf_avg, ENG=True)  # расчет TF-IDF
    find_user_text_rezult = find_user_text(rezult=rezult, user_text=user_text_words_list)
    tf_idf_etalon = find_user_text_rezult["TF-IDF"]
    tf_idf_etalon_vector = find_user_text_rezult["TF-IDF_Vector"]
    ari_etalon = text_search.ari(text=user_text)
    rezult.sort(key=lambda x: x["TF-IDF"])  # Сортируем найденные тексты по TF-IDF

    top_five_tf_idf = find_top_5(rezult=rezult, tf_idf_etalon=tf_idf_etalon)
    top_five_tf_idf_texts_list = [" ".join(r["text_words_list"]) for r in top_five_tf_idf]
    top_five_tf_idf_value = [r["TF-IDF"] for r in top_five_tf_idf]
    top_five_tf_idf_vectors_list = [r["TF-IDF_Vector"] for r in top_five_tf_idf]
    texts_list_from_top_five_tf_idf = [user_text]
    texts_list_from_top_five_tf_idf.extend(top_five_tf_idf_texts_list)
    # рассчет косинусного расстояния для текстов отобранных по tf-idf
    cos_sim_list = word_vector.show_info_about_compare_vectors(
        texts_list=texts_list_from_top_five_tf_idf)  # рассчет косинусного расстояния
    kmeans_rezult_list = kmeans.compare_texts(tf_idf_etalon_vector=tf_idf_etalon_vector,
                                              tf_idf_list=top_five_tf_idf_vectors_list)
    print(top_five_tf_idf_value)
    print(cos_sim_list)

    top_five_ari = find_top_5(rezult=rezult, ari_etalon=ari_etalon)
    top_five_ari_value = [r["ARI"] for r in top_five_ari]
    top_five_ari_texts_list = [" ".join(r["text_words_list"]) for r in top_five_ari]
    texts_list_from_top_five_ari = [user_text]
    texts_list_from_top_five_ari.extend(top_five_ari_texts_list)
    # рассчет косинусного расстояния для текстов отобранных по ari
    cos_sim_list_ari = word_vector.show_info_about_compare_vectors(
        texts_list=texts_list_from_top_five_ari)  # рассчет косинусного расстояния

    cos_sim_excel_index = tfidf_excel_index = kmeans_excel_index = ari_excel_index = \
        cos_sim_ari_excel_index = excel_index

    info = " categories " + str(categories) + " count " + str(count) + " keytext " + str(key_text) + \
           " tf_idf_avg " + str(tf_idf_avg)
    sheet['G' + str(excel_index)] = info

    for cos_sim in cos_sim_list:
        sheet['A' + str(cos_sim_excel_index)] = str(cos_sim)
        cos_sim_excel_index += 1

    for tfidf_value in top_five_tf_idf_value:
        sheet['B' + str(tfidf_excel_index)] = str(tfidf_value).replace('.', ',')  # разделитель в excel это ,
        tfidf_excel_index += 1

    for kmeans_value in kmeans_rezult_list:
        sheet['C' + str(kmeans_excel_index)] = str(kmeans_value).replace('.', ',')  # разделитель в excel это ,
        kmeans_excel_index += 1

    for ari in top_five_ari_value:
        sheet['D' + str(ari_excel_index)] = str(ari).replace('.', ',')  # разделитель в excel это ,
        ari_excel_index += 1

    for ari in cos_sim_list_ari:
        sheet['E' + str(cos_sim_ari_excel_index)] = str(ari).replace('.', ',')  # разделитель в excel это ,
        cos_sim_ari_excel_index += 1

    wb.save(book_name)


def get_text_from_newsgroups(*, categories, count):
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
    return newsgroups_train.data[:count]


def find_top_5(rezult, tf_idf_etalon="", ari_etalon=""):
    """
    Данная функция выполняет поиск 5 значенией максимально близких к эталону
    """
    rezult = unique_rezult(rezult=rezult)
    index = 0
    if tf_idf_etalon:
        for i in range(len(rezult)):
            if rezult[i]["TF-IDF"] == tf_idf_etalon:
                index = i
                break
    if ari_etalon:
        for i in range(len(rezult)):
            if rezult[i]["ARI"] == tf_idf_etalon:
                index = i
                break
    print("len(rezult) " + str(len(rezult)) + " index " + str(index))
    if index == len(rezult) - 1:
        top_rezult = [rezult[index - 5], rezult[index - 4], rezult[index - 3], rezult[index - 2], rezult[index - 1]]
    elif index == len(rezult) - 2:
        top_rezult = [rezult[index - 4], rezult[index - 3], rezult[index - 2], rezult[index - 1], rezult[index + 1]]
    elif index == 0:
        top_rezult = [rezult[index+1], rezult[index+2], rezult[index+3], rezult[index+4], rezult[index+5]]
    elif index == 1:
        top_rezult = [rezult[index - 1], rezult[index + 1], rezult[index + 2], rezult[index + 3], rezult[index + 4]]
    else:
        top_rezult = [rezult[index - 3], rezult[index - 2], rezult[index - 1], rezult[index + 1], rezult[index + 2]]
    return top_rezult


def unique_rezult(*, rezult):
    rezult_unique = []
    rezult_texts = []
    for r in rezult:
        if r["text_words_list"] not in rezult_texts:
            rezult_unique.append(r)
        rezult_texts.append(r["text_words_list"])
    return rezult_unique


if __name__ == "__main__":

    # взял тектст из одного поста, для проверки N-грамм лучше фраза подлинее, так как фраза длиннее уменьшим count
    key_text = "Такая программа прорабатывает каждую группу мышц"
    """
    basic_test(key_info=key_text, count=3, domain='sports_books', book_name="NIR2021.xlsx",
               sheet_name="test8", excel_index=10, lemotize=False, tf_idf_avg=False)

    basic_test(key_info=key_text, count=3, domain='sports_books', book_name="NIR2021.xlsx",
               sheet_name="test8", excel_index=20, lemotize=True, tf_idf_avg=False)

    basic_test(key_info=key_text, count=3, domain='sports_books', book_name="NIR2021.xlsx",
               sheet_name="test8", excel_index=30, lemotize=False, tf_idf_avg=True)

    basic_test(key_info=key_text, count=3, domain='sports_books', book_name="NIR2021.xlsx",
               sheet_name="test8", excel_index=40, lemotize=True, tf_idf_avg=True)

    basic_test(key_info=key_text, count=3, domain='sports_books',book_name="NIR2021.xlsx",
               sheet_name="test8", excel_index=50, lemotize=False, tf_idf_avg=False, Ngram=2)

    basic_test(key_info=key_text, count=3, domain='sports_books', book_name="NIR2021.xlsx",
               sheet_name="test8", excel_index=60, lemotize=False, tf_idf_avg=True,  Ngram=2)

    basic_test(key_info=key_text, count=3, domain='sports_books', book_name="NIR2021.xlsx",
               sheet_name="test8", excel_index=70, lemotize=False, tf_idf_avg=False, Ngram=3)

    basic_test(key_info=key_text, count=3, domain='sports_books', book_name="NIR2021.xlsx",
               sheet_name="test8", excel_index=80, lemotize=False, tf_idf_avg=True, Ngram=3)
    """
    categories = ['sci.space',  'talk.politics.guns', 'comp.windows.x', 'rec.sport.hockey']
    test_newsgroups(categories=categories, count=10, book_name="NIR2021.xlsx",
                    sheet_name="test9", excel_index=10, key_text_length=10, tf_idf_avg=False)

    test_newsgroups(categories=categories, count=10, book_name="NIR2021.xlsx",
                    sheet_name="test9", excel_index=20, key_text_length=10, tf_idf_avg=True)


