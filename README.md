Program składa się z 3 części:
- część pierwsza służy stworzeniu pliku z cechami i zapisania ich w processed_files
- część druga to analiza i podzielenie danych na treningowe i testowe 
- część druga to uczenie modeli.

Na część pierwszą składają się klasy:
- data_reader  ---> ta klasa czyta jakie pliki z sesjami użytkownika powinny być użyte, czyli kreauje ścieżki do plików  i przekazuje je  do data processera
- data_reader_[zbiór danych] --> kazdy ze zbiorów danych [Balabit, Chaoshen, Singapur, DFL] ma swoja własną klasę
- data_processer ---> ta klasa służy przeprocesowaniu danych i zapisaniu ich do pliku 
- data_processer_[zbiór danych] --> processing dla każdego zbioru jest inny 

Jak działa processing:
W każdym ze zbiorów sprawdzane jest kiedy puszczony został przycisk myszki. Aż do momentu puszczenia myszki zapisywane są wszystkie akcje, które nastąpiły. Mając te dane można skupić określona ilość akcji, aby stworzyć wydarzenia.
Wydarzenia podzielone zostały na 3 typy:
- MM (Mouse Move) - ruszanie myszką
- PC (Process and Click) ruszanie z przycisnieciem na koncu ruchu
- DD (Drag and Realease) trzymanie przycisku i puszczenie na koncu.

Wzorowałem się na artykule 'Intrusion Detection Using Mouse Dynamics', którzy robili ekstrakcje danych do BALABITU. Zacytowałem, gdzie użyłem części ich kodu. 

Wydarzenia mogą być filtrowoane po czasie oraz po ilości akcji (chodzi o to, że za krótkie wydarzenia mało nam dają, więć trzeba je usunąć). W pliku util.consts znajdują się stałe filtrujące te wydarzenia. 

Każdy z tych zbiorów miał swoją specyfikę, więc processing był inny dla każdego z nich.

Na część drugą skłądają sie klasy
- nonlegality_analyser używany dla każdego zbioru, gdzie uczymy użytkownika każdy vs każdy (legalność sesji jest dodana po stworzeniu wektora danych lub w ogole nie musi być wykonana) 0--> klasa używana może być w UNSUPERVISED  i SUPERVISED learningu
- legality_analyser używany tylko dla BALABITU, gdy chcemy przeprocesować pliki testowe (legalność sesji jest sprawdzana podczas kreaowania wektora danych) --> tylko SUPERVISED learning

Metody tych klas mogą wyplotować nam wykresy :
- plotActionHistograms() historgram podzielonych wydarzeń
- plotTypeOfActions(stacked) wykres słupokowy wydarzeń

oraz  to tutaj zachodzi split danych na testowe i treningowe.

Część trzecia to modele. Jak narazie są w nich:
- Random Forest,
- SVM,
- One class SVM,
- Isoltion forest.

Głowna funkcja - evaluate().


Jak włączyć ---> 
1) otwórz jupyter notebooka o nazwie main.ipynb
2) wyekstrachowane dane powinny znajdować się w pliki processed_files, dzięki temu zbiory danych są niepotrzebne
3) przeklikaj cały arkusz (dane nie powinny się tworzyć)


Inne pliki:
cutter.py - usuwa sesje testowe użytkowników, które nie znajdują się w pliku public_labels.csv (BALABIT)
ChaoShenCSVCreator.py - dekoduje pliki binarne ze zbioru ChaoShen, a nastenie zapisuje je do folderu i scala
debbuging -> prosze nie zwracać uwagi
utils/consts - stałe wykorzystywane w uczeniu
utils/helper_functions - kilka funkcji statystycznych
utils/plotting - nieuzywane


Środowisko w którym robiono testy:
- python wersja  3.9.13
- Windows 10 Education N 22H2


