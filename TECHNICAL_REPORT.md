# C5-alpha-0.1 — dodatkowa metoda zespołu

Status dokumentu: opis istniejącego wariantu źródłowego i plan przeniesienia.
Ta gałąź zawiera szkielet; nie odtwarza jeszcze wyników. Główne zgłoszenie
zespołu korzysta z metody kolegi, zgodnie z informacją użytkownika.

## Podejście

Źródłowe C5 łączy ensemble pięciu zamrożonych TCN B+CE, pełny N14 z modelem
liniowym oraz dekoder Viterbiego i reguły C3/C5. Sieci używają 37 cech
obserwacyjnych. N14 dostarcza dowody kontaktu, przerw i wznowień.
C5-alpha-0.1 stosuje ochronę plateau D i mieszanie z meta-modelem przy
alpha = 0,10. Koszty zdarzeń i lotu pozostają 6/14, minimalne D: 40 klatek.
To odrębny system od sześciomodelowego TCN kolegi z cechami przy 5 fps.

## Wejście i zasoby

Historyczny eksporter używał metadanych i surowego trackingu tactical ATD.
Nie potrzebował konkursowego GT ani filmu do wykonania inferencji.
Przed przeniesieniem należy sprawdzić kompletny zestaw modułów, checkpointów,
normalizacji, modelu Linear, meta-modelu i konfiguracji względem manifestu.
Wagi oraz pliki konfiguracyjne nie zostały jeszcze przeniesione do tej gałęzi.

## Dane treningowe i warianty porównawcze

Według raportu źródłowego TCN pochodzą z istniejących foldów B+CE na IDSSE,
a głowa Linear N14 jest zamrożonym modelem wytrenowanym na syntetyce.
Przed publikacją trzeba uzupełnić dokładne źródło, rewizję, licencję,
pochodzenie etykiet, liczbę przykładów i podziały dla wszystkich wag.
Nie deklarujemy kompletnego audytu treningu na podstawie samego szkieletu.

Porównywano N9/N14, C2, C3-Rule, C3-Meta i C4. Ich źródła i krótkie
podsumowania można zachować jako badania, oddzielając je od finalnego C5.
Wyników porównań na różnych wejściach nie łączymy w jeden ranking.

## Reguły posiadania

N14 dostarcza stan H/A/D/X i fizyczne dowody. Dekoder łączy je z TCN;
reguły C3 wyrównują granice i chronią krótkie zdarzenia. Ochrona plateau C5
blokuje kandydata przy fizycznie potwierdzonym LIVE przez cały odcinek.
Jakiekolwiek D lub surowe X zachowuje plateau. Dokładną semantykę luźnej piłki,
pojedynków i wznowień należy opisać po przeglądzie zamrożonego kodu.

## Wyniki i ograniczenia

Źródłowa finalna konfiguracja uzyskała średnie spellF1
0,66763161 native / 0,59011657 mixed na pięciu znanych meczach IDSSE.
Wynik zagnieżdżonego wyboru etapu: 0,66702306 / 0,59250772.
Pozostają regresje dwóch par native i wzrost pominiętych klatek D.
Panelu 3567 dla C5 nie wykonano. LOMO jest warunkowe na historycznym cache
TCN; znane mecze były wcześniej używane w rozwoju. Nie jest to niezależna Q
ani wynik konkursowy, ani porównanie z modelem kolegi na wspólnym wejściu.

Nie wyprowadzamy widoczności piłki podczas D z ogólnych statystyk widoczności.
Nazwy i jednostki cech, w tym zmiany prędkości, trzeba zachować zgodnie z kodem.

## Reprodukcja

Historyczny raport C5 opisuje P4 PASS: dwa identyczne pełne eksporty
i dwie oficjalne walidacje. Nie jest to odbiór nowego układu tej gałęzi.
Po przeniesieniu wymagane są czyste środowisko, sprawdzone hashe zasobów,
działające polecenie oraz kontrola dwóch wyników u uprawnionego operatora.
Szczegóły: [plan reprodukcji](docs/REPRODUKCJA.md).

CSV C5 pozostaje prywatny. Dodatkowa metoda w repo nie zmienia pliku
zgłoszonego formularzem ani metody, która ma go odtwarzać.
