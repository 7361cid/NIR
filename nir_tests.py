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
    sheet['D1'] = "Параметры теста"


    # поиск постов с ключевыми словами
    texts_list = vk.get_text_from_vk(query=key_info, count=count, domain=domain,
                                     lemotize=lemotize, Ngram=Ngram)
    # выбор текста пользователем(берем 1ый не пустой текст)
    user_text = vk.user_chouse_emulator(texts_list=texts_list, random=False)
    print("Эталон для TF-IDF 1 " + user_text[:20])
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
    tf_idf_etalon = find_user_text_rezult[0]
    print("Эталон для TF-IDF 2 " + " ".join(find_user_text_rezult[1]["text_words_list"][:20]))
    rezult.sort(key=lambda x: x["TF-IDF"])  # Сортируем найденные тексты по TF-IDF
    rezult_texts_list = [" ".join(r["text_words_list"]) for r in rezult]
    rezult_tfidf_list = [r["TF-IDF"] for r in rezult]
    texts_list = [user_text]
    texts_list.extend(rezult_texts_list[-5:])  # Берем последние 5 текстов
    cos_sim_list = word_vector.show_info_about_compare_vectors(texts_list=texts_list) # рассчет косинусного расстояния
    kmeans_rezult_list = kmeans.compare_texts(tf_idf_etalon=tf_idf_etalon, tf_idf_list=rezult_tfidf_list[-5:])
    print(rezult_tfidf_list[-5:])
    print(cos_sim_list)
    # среднее косинусное расстояние для первого текста и остальных
    cos_sim_excel_index = tfidf_excel_index = kmeans_excel_index = excel_index
    info = " domain " + str(domain) + " count " + str(count) + " keyinfo " + str(key_info) + " lemotize " \
           + str(lemotize) + " tf_idf_avg " + str(tf_idf_avg) + " Ngramm " + str(Ngram)
    sheet['D' + str(excel_index)] = info
    sheet['E' + str(excel_index)] = "итоговое количество постов " + str(kolvo_postov)
    for cos_sim in cos_sim_list:
        sheet['A' + str(cos_sim_excel_index)] = str(cos_sim)
        cos_sim_excel_index += 1

    for tfidf_value in rezult_tfidf_list[-5:]:
        sheet['B' + str(tfidf_excel_index)] = str(tfidf_value).replace('.', ',')  # разделитель в excel это ,
        tfidf_excel_index += 1

    for kmeans_value in kmeans_rezult_list:
        sheet['C' + str(kmeans_excel_index)] = str(kmeans_value).replace('.', ',')  # разделитель в excel это ,
        kmeans_excel_index += 1

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

    return similar_to_user_text["TF-IDF"], similar_to_user_text


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
    sheet['D1'] = "Параметры теста"

    # поиск постов с по категориям
    texts_list = get_text_from_newsgroups(categories=categories, count=count)
    # выбор текста пользователем(берем 1ый не пустой текст)
    user_text = vk.user_chouse_emulator(texts_list=texts_list, random=False)
    print("Эталон для TF-IDF 1 " + user_text[:20])
    # выкинуть из текста символы пунтуации, чтобы не было слов в стиле "Я,"
    key_text = " ".join(user_text.split()[key_text_length])  # ключевые слова берем из выбранного текста
    text_search.delete_bad_symbols(key_text)
    text_search.delete_bad_symbols(user_text)
    key_list = key_text.split()  # Разбиение по пробелам
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

    rezult = text_search.search(key_info=key_text, texts_list=texts_list,
                                tf_idf_avg=tf_idf_avg)  # расчет TF-IDF
    find_user_text_rezult = find_user_text(rezult=rezult, user_text=user_text_words_list)
    tf_idf_etalon = find_user_text_rezult[0]
    print("Эталон для TF-IDF 2 " + " ".join(find_user_text_rezult[1]["text_words_list"][:20]))
    rezult.sort(key=lambda x: x["TF-IDF"])  # Сортируем найденные тексты по TF-IDF
    rezult_texts_list = [" ".join(r["text_words_list"]) for r in rezult]
    rezult_tfidf_list = [r["TF-IDF"] for r in rezult]
    texts_list = [user_text]
    texts_list.extend(rezult_texts_list[-5:])  # Берем последние 5 текстов
    cos_sim_list = word_vector.show_info_about_compare_vectors(texts_list=texts_list)  # рассчет косинусного расстояния
    kmeans_rezult_list = kmeans.compare_texts(tf_idf_etalon=tf_idf_etalon, tf_idf_list=rezult_tfidf_list[-5:])
    print(rezult_tfidf_list[-5:])
    print(cos_sim_list)
    # среднее косинусное расстояние для первого текста и остальных
    cos_sim_excel_index = tfidf_excel_index = kmeans_excel_index = excel_index
    info = " categories " + str(categories) + " count " + str(count) + " keytext " + str(key_text) + \
           " tf_idf_avg " + str(tf_idf_avg)
    sheet['D' + str(excel_index)] = info
    for cos_sim in cos_sim_list:
        sheet['A' + str(cos_sim_excel_index)] = str(cos_sim)
        cos_sim_excel_index += 1

    for tfidf_value in rezult_tfidf_list[-5:]:
        sheet['B' + str(tfidf_excel_index)] = str(tfidf_value).replace('.', ',')  # разделитель в excel это ,
        tfidf_excel_index += 1

    for kmeans_value in kmeans_rezult_list:
        sheet['C' + str(kmeans_excel_index)] = str(kmeans_value).replace('.', ',')  # разделитель в excel это ,
        kmeans_excel_index += 1

    wb.save(book_name)

def get_text_from_newsgroups(*, categories, count):
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
    return newsgroups_train.data[:count]


if __name__ == "__main__":

    # взял тектст из одного поста, для проверки N-грамм лучше фраза подлинее, так как фраза длиннее уменьшим count
    key_text = "Такая программа прорабатывает каждую группу мышц"
    basic_test(key_info=key_text, count=2, domain='sports_books', book_name="NIR2021.xlsx",
               sheet_name="test6", excel_index=10, lemotize=False, tf_idf_avg=False)

    basic_test(key_info=key_text, count=2, domain='sports_books', book_name="NIR2021.xlsx",
               sheet_name="test6", excel_index=20, lemotize=True, tf_idf_avg=False)

    basic_test(key_info=key_text, count=2, domain='sports_books', book_name="NIR2021.xlsx",
               sheet_name="test6", excel_index=30, lemotize=False, tf_idf_avg=True)

    basic_test(key_info=key_text, count=2, domain='sports_books', book_name="NIR2021.xlsx",
               sheet_name="test6", excel_index=40, lemotize=True, tf_idf_avg=True)

    basic_test(key_info=key_text, count=2, domain='sports_books',book_name="NIR2021.xlsx",
               sheet_name="test6", excel_index=50, lemotize=False, tf_idf_avg=False, Ngram=2)

    basic_test(key_info=key_text, count=2, domain='sports_books', book_name="NIR2021.xlsx",
               sheet_name="test6", excel_index=60, lemotize=False, tf_idf_avg=True,  Ngram=2)

    basic_test(key_info=key_text, count=2, domain='sports_books', book_name="NIR2021.xlsx",
               sheet_name="test6", excel_index=70, lemotize=False, tf_idf_avg=False, Ngram=3)

    basic_test(key_info=key_text, count=2, domain='sports_books', book_name="NIR2021.xlsx",
               sheet_name="test6", excel_index=80, lemotize=False, tf_idf_avg=True, Ngram=3)
    

    test_newsgroups(categories=['comp.windows.x'], count=10, book_name="NIR2021.xlsx",
                    sheet_name="test7", excel_index=10, key_text_length=5, tf_idf_avg=False)

    test_newsgroups(categories=['comp.windows.x'], count=10, book_name="NIR2021.xlsx",
                    sheet_name="test7", excel_index=20, key_text_length=5, tf_idf_avg=True)


