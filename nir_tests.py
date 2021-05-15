import text_search
import word_vector
import my_vk_parser as vk
import openpyxl


def basic_test(*, key_info, count, domain, sheet, cos_sim_excel_index, tfidf_excel_index, lemotize=False):
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
    cos_sim_excel_index - индекс для записи косинусного расстояния  в таблицу excel
    tfidf_excel_index -  индекс для записи tf-idf  в таблицу excel
    индексы нужно менять между вызовами тестов для того чтобы один тест не переписал другой
    lemotize - использовать лематизацию или нет
    """
    # поиск постов с ключевыми словами
    texts_list = vk.get_text_from_vk(query=key_info, count=count, domain=domain, lemotize=lemotize)
    # выбор текста пользователем(берем 1ый текст)
    user_text = vk.user_chouse_emulator(texts_list=texts_list, random=False)
    for bad_symbol in [',', '.', '!', '?']:  # выкинуть из текста символы пунтуации, чтобы не было слов в стиле "Я,"
        key_info.replace(bad_symbol, '')
        user_text.replace(bad_symbol, '')

    key_list = key_info.split()  # Разбиение по пробелам
    words_list = user_text.split()
    new_key_list = []
    new_key_list.extend(key_list)
    for key in key_list:  # поиск новых ключевых слов
        new_key_list.extend(text_search.frequancy_analis(words_list=words_list, key_word=key, max_new_words=2))
        print(new_key_list)

    new_key_info = " ".join(new_key_list)
    # поиск постов с расширенным набором ключевых слов
    new_texts_list = vk.get_text_from_vk(query=new_key_info, count=count, domain=domain, lemotize=lemotize)
    rezult = text_search.search(key_info=key_info, texts_list=new_texts_list)  # расчет TF-IDF
    rezult.sort(key=lambda x: x["TF-IDF"])  # Сортируем найденные тексты по TF-IDF
    rezult_texts_list = [" ".join(r["text_words_list"]) for r in rezult]
    rezult_tfidf_list = [r["TF-IDF"] for r in rezult]
    texts_list = [user_text]
    texts_list.extend(rezult_texts_list[-5:])  # Берем последние 5 текстов
    cos_sim_list = word_vector.show_info_about_compare_vectors(texts_list=texts_list) # рассчет косинусного расстояния
    print(rezult_tfidf_list[-5:])
    print(cos_sim_list)
    # среднее косинусное расстояние для первого текста и остальных
    info = " domain" + str(domain) + " count " + str(count) + " keyinfo " + str(key_info) + " lemotize " + str(lemotize)
    sheet['C' + str(cos_sim_excel_index)] = info

    for cos_sim in cos_sim_list:
        sheet['A' + str(cos_sim_excel_index)] = str(cos_sim)
        cos_sim_excel_index += 1

    for tfidf_value in rezult_tfidf_list[-5:]:
        sheet['B' + str(tfidf_excel_index)] = str(tfidf_value).replace('.', ',')  # разделитель в excel это ,
        tfidf_excel_index += 1


if __name__ == "__main__":
    wb = openpyxl.load_workbook(filename='NIR2021.xlsx')  # файл должен существовать с листом "test"
    sheet = wb["test"]
    sheet['A1'] = "косинусное расстояние"
    sheet['B1'] = "tf-idf"
    sheet['C1'] = "Параметры теста"
    # взял текст из одного поста, чтобы точно было совпадение
    basic_test(key_info="каждого раунда", count=10, domain='sports_books',
               sheet=sheet, cos_sim_excel_index=2, tfidf_excel_index=2)

    basic_test(key_info="каждого раунда", count=10, domain='sports_books',
               sheet=sheet, cos_sim_excel_index=20, tfidf_excel_index=20, lemotize=True)
    wb.save("NIR2021.xlsx")
