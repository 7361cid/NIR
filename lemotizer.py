from pymorphy2 import MorphAnalyzer

def lemmotize(word):
    morph = MorphAnalyzer()
    return morph.normal_forms(word)[0]

if __name__ == "__main__":
    print(lemmotize("Книги"), end=" ")
    print(lemmotize("Бегает"), end=" ")
    print(lemmotize("Бумажная"), end=" ")
