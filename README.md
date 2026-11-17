# More Than a Hack 2027 — Phase 1

Projekt startowy do zadania rozpoznawania posiadania piłki.
Na tym etapie korzystamy **wyłącznie z publicznego kodu, danych syntetycznych i publicznego zbioru IDSSE**.

Wprowadzenie dla nowej osoby: [co już mamy, jakie testy wykonaliśmy i propozycja repozytorium](PODSUMOWANIE-DLA-ZESPOLU.md).

## Repozytorium i współpraca

Docelowe repo: `sebesio/more-than-hack-2027`. Wysyłanie zmian odbywa się lokalnym
Gitem; dodatkowe konto we wtyczce GitHub nie jest potrzebne. Nazwę i e-mail autora
każdy ustawia lokalnie w swoim klonie (`git config --local user.name ...` oraz
`git config --local user.email ...`), używając własnego zweryfikowanego adresu GitHub
lub adresu noreply. Ustawienia dostępu i autora nie są plikami repozytorium.

Jedno zadanie = osobna gałąź; przy równoległej pracy agentów także osobny worktree.
Nowy worktree wymaga własnego środowiska: uruchom w nim `./setup.ps1`.
Zmiany łączymy do `main` przez pull request po przeglądzie drugiej osoby i testach.
Na GitHub Free w prywatnym repo jest to umowa zespołu, bez wymuszania ochroną gałęzi.
Codex korzysta z `AGENTS.md`, a `CLAUDE.md` importuje te same zasady dla Claude Code.

[GitHub Actions](.github/workflows/check.yml) uruchamia `check`, `demo` i `improve`
na Windows i Linuxie z Pythonem 3.12: po pushu do `main`, dla PR do `main` lub ręcznie.
CI instaluje przypięte zależności, sprawdza integralność `official/` i porównuje
metody na wspólnych danych syntetycznych. Nie pobiera danych konkursowych,
nie wysyła artefaktów i nie przechowuje cache wyników. Pełne lokalne sprawdzenie:

```powershell
.\run.ps1 check
.\run.ps1 demo
.\run.ps1 improve
```

Przed pierwszym wysłaniem przejrzyj `git diff --cached --stat` i
`git diff --cached`. Do historii trafiają wyłącznie sprawdzony kod i dokumentacja;
dane oraz wyniki pozostają lokalnie, także gdy są syntetyczne. Samo dodanie
`origin` nie wysyła plików. `git push` wykonujemy jako osobny, uzgodniony krok,
po sprawdzeniu konta, docelowego repozytorium i zawartości commita.

## Teraz: krok 1 — brakująca piłka i przerwy

Dodaliśmy pierwszą własną metodę w `solution/possession.py`. Oficjalny baseline
pozostaje punktem odniesienia. Uruchom nowe porównanie:

```powershell
.\run.ps1 improve
.\run.ps1 check
```

Co zmieniliśmy, po kolei:

1. **Pamięć posiadania.** Gdy piłka znika, zachowujemy ostatnią potwierdzoną drużynę
   przez maksymalnie 2 sekundy. Dzięki temu krótkie zasłonięcie nie tworzy przerwy.
2. **Jawna niepewność.** Jeśli nadal nie ma przekonujących obserwacji, zwracamy `X`.
   Brak wszystkich zawodników również daje `X`; brak danych nie jest zerową prędkością.
3. **Dodatkowe oznaki przerwy.** `D` wymaga około 1,5 sekundy małej aktywności
   obserwowanych zawodników oraz brakującej piłki albo widocznej, nieruchomej piłki
   daleko od zawodników. Sama nieobecność piłki nie wystarcza.
4. **Potwierdzenie kontroli i wznowienia.** Piłka blisko zawodnika musi utrzymać
   wskazanie drużyny przez około 0,2 sekundy. Jeden krótki kontakt nie zmienia od razu posiadania.
5. **Porównanie na tym samym wejściu.** Cztery oficjalne baseline'y i nasza metoda
   dostają identyczny nowy zestaw scenariuszy. Każdy CSV przechodzi tę samą walidację
   i ocenę oficjalnym scorerem. Nasz algorytm dostaje wyłącznie obserwacje i drużyny;
   etykiety oraz nazwy scenariuszy trafiają dopiero do oceny po zapisaniu predykcji.

Progi są wstępnymi założeniami inżynierskimi. Nie dopasowywaliśmy ich do meczu
konkursowego ani jego publicznego GT. Po pierwszym porównaniu nie zmienialiśmy
progów ani zestawu przypadków, aby podnieść wynik.

Nowe pliki wejściowe powstają w `data/synthetic-step1/`, wyniki w
`outputs/synthetic-step1/`. Starsze `demo` nadal pokazuje poprzedni, prosty przykład.
Nie porównuj liczbowo ocen z tych dwóch różnych zestawów danych.

Wyniki kroku 1 na danych syntetycznych (27 października 2026):

| Metoda | macroF1: klatki | spellF1: całe posiadania | F1 klasy D: przerwy |
|---|---:|---:|---:|
| Oficjalny raw | 0,6497 | 0,3730 | 0,2941 |
| Oficjalny smooth25 | 0,6517 | 0,6329 | 0,2609 |
| Oficjalny smooth50 | 0,6234 | **0,8400** | 0,1538 |
| Oficjalny smooth125 | 0,5666 | 0,6667 | 0,0000 |
| Nasz step1_memory_dead | 0,8029 | 0,7636 | 0,7000 |
| Nasz step1_filled (`X` zastąpione etykietą, 6 listopada) | 0,8250 | 0,7636 | 0,7000 |
| Nasz segment_decoder (6 listopada) | **0,8812** | 0,8364 | **0,8319** |

**Dekoder wyrównuje smooth50 na tej syntetyce i wyraźnie lepiej rozpoznaje przerwy.**
Step 1 działa tylko na bieżących i wcześniejszych klatkach. Oficjalne wygładzanie i dekoder
wykorzystują także późniejsze. Dekoder nadal daje fałszywe `D` w `ambiguous_live_missing`
(0% poprawnych klatek; step1 37%), a przy ukrytej zmianie drużyny stawia granicę w środku luki.

## Dekoder całych odcinków

[solution/decoder.py](solution/decoder.py) wybiera najlepszy ciąg odcinków H/A/D dla całej połowy.
Działa na dowodach liczonych dla drużyn, nie slotów: odległość do piłki i aktywność zawodników.
Zmiana odcinka ma koszt, a posiadanie minimalną długość. Wewnątrz bloku nie zwraca `X`.
Progi są założeniami; jeszcze ich nie stroiliśmy.

`fill_abstentions` zapisuje etykietę, którą scorer i tak by przyjął za `X`. SpellF1 się nie zmienia.
MacroF1 może jednak spaść, jeśli `X` leży głównie na przerwach: na IDSSE wypełnienie step1 obniżyło
go z 0,50 do 0,41. Wypełnianie ma sens dopiero przy dobrym wykrywaniu przerw.

## IDSSE: mecze z etykietami do treningu i testów

[IDSSE](https://doi.org/10.1038/s41597-025-04505-y) to 7 meczów Bundesligi 2022/23 (CC BY 4.0) z etykietami
`BallPossession` i `BallStatus` w każdej klatce. Pliki trafiają do ignorowanego `datasets/idsse/`:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\fetch_idsse.py
.\.venv\Scripts\python.exe -X utf8 scripts\research_idsse.py
```

Wyniki z 6 listopada 2026, średnio na 7 meczach, z ukrytą piłką i bez zamrożenia piłki w przerwach:

| Metoda | spellF1: luki niezależne | macroF1 | spellF1: luki głównie w przerwach | macroF1 |
|---|---:|---:|---:|---:|
| Oficjalny smooth50 | 0,208 | 0,454 | 0,310 | 0,728 |
| step1_memory_dead | 0,262 | 0,393 | 0,334 | 0,507 |
| segment_decoder | **0,274** | **0,678** | **0,392** | **0,803** |

Dekoder wygrywa ze smooth50 w każdym meczu obu wariantów. Przewaga nad step1 w lukach niezależnych (0,012)
mieści się w szumie. Rozkład luk jest założeniem, a mecze to inne rozgrywki i kamery: **to nie jest wynik konkursowy.**

Trzy ustalenia przed treningiem na tych danych:

1. **Inna definicja posiadania.** Zmiana następuje przy kontakcie: średnio 4,4 posiadania na minutę, mediana
   ok. 3,7 s, a 26–39% bezpośrednich przekazań wraca w ciągu 3 s. Konkurs wyklucza samo dotknięcie.
   `merge_brief_spells` tylko przybliża tę różnicę; progu scalania jeszcze nie wybraliśmy.
2. **Przeciek przez piłkę.** W przerwie pozycja piłki stoi w 100% klatek (w grze ok. 1%). Trzeba ją zastąpić
   przed treningiem, inaczej model przeczyta status wprost.
3. **Dane są czystsze od ATD.** Zawsze ok. 22 zawodników, piłka w każdej klatce, stałe tożsamości.
   Trening wymaga zniekształceń: anonimowych, porwanych torów, duplikatów i luk w piłce.

### IDSSE w formacie ATD konkursu

`scripts/degrade_idsse.py` zamienia jeden mecz IDSSE w plik o tej samej strukturze co ATD konkursu.
Etykiety trafiają do osobnego `frame,label` CSV, nigdy do wejścia.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.degrade_idsse --match J03WMX --seed 0 --evaluate
```

Wynik trafia do `datasets/idsse/degraded/<mecz>/seed<n>/`: `tactical/<mecz>_tactical_atd.xml` i `.txt`,
`<mecz>_gt.csv` oraz `manifest.json`. Opcja `--evaluate` dodaje oficjalny baseline, scorer i raport jakości.

- **Struktura:** 199 319 klatek, `frame_id` od 1, `x,y,s` dla 46 anonimowych slotów (po 23 na drużynę),
  piłka `x,y,s`, współrzędne 0–1 od lewego górnego rogu, prędkość w m/s, `NaN` dla braków.
  Przed meczem i w przerwie jest tyle samo pustych klatek co w konkursie. Wczytuje go `kloppy` 3.18.0,
  oficjalny `baseline.py` i `solution/tracking.py`.
- **Usunięte:** wysokość piłki, `D`, `A`, `M`, sędziowie, identyfikatory osób, `BallStatus`
  i `BallPossession`. Usunięto też zamrożoną piłkę w przerwach: zostaje tylko część końca przerwy,
  z piłką na miejscu wznowienia.
- **Kalibracja na publicznych liczbach:** piłki brak w 40,7% ocenianych klatek, średnio 20,5 obiektu.
  Na J03WMX (ziarno 0) wyszło 40,67% i 20,49.
- **Założenia, nie pomiary:** braki piłki są częstsze w tłoku, w powietrzu i przy szybkich podaniach.
  Piłkę w powietrzu widzi jedna kamera, która rzutuje ją na murawę. Zawodnicy znikają częściej w tłoku
  i po dalszej stronie boiska. Dochodzą zamiany tożsamości przy zbliżeniach, duplikaty i rzadkie fałszywe
  piłki. Wagi są w `Config`.
- **Etykiety:** zostają surowe etykiety IDSSE (posiadanie zmienia się już przy kontakcie). Opcja
  `--merge-seconds N` scala krótsze posiadania, ale progu jeszcze nie wybraliśmy.

Oficjalny baseline na J03WMX (ziarno 0) daje spellF1: smooth50 0,240, smooth25 0,219,
smooth125 0,189, raw 0,016. MacroF1 wynosi 0,54–0,61. Kolejność jest jak w opublikowanych wynikach
całego meczu konkursowego (smooth50 0,243), ale macroF1 jest niższe. Parametrów na tych liczbach nie
stroiliśmy. **To nie jest wynik konkursowy.**

Czego brakuje względem konkursu: wideo, homografii, Smart Tagging i danych w klatkach poza grą (w konkursie
piłka jest tam widoczna w ok. 48% klatek). Rzeczywisty rozkład błędów Metrica nie jest znany.

Co udało się sprawdzić:

| Sztuczna sytuacja | Zachowanie naszej metody |
|---|---|
| Krótkie zniknięcie piłki podczas ruchu zawodników | Utrzymuje drużynę; 0 fałszywych `D` w tym przypadku |
| Długie zniknięcie podczas ruchu | Najpierw pamięć, potem `X`; 0 fałszywych `D` |
| Stojący zawodnicy i odległa nieruchoma piłka | Rozpoznaje przerwę z opóźnieniem potrzebnym do jej potwierdzenia |
| Stojący zawodnicy, piłka widoczna blisko gracza | Zachowuje posiadanie |
| Zniknięcie całego trackingu | Zwraca `X` zamiast udawać, że zawodnicy stoją |
| Zawodnicy stoją i piłki nie widać, ale gra trwa | Nadal popełnia fałszywe `D` — jawnie uwzględniony trudny przypadek |
| Przeciwnik przejmuje piłkę podczas jej niewidoczności | Nie odkrywa zmiany bez nowej obserwacji |

`X` nie usuwa klatek z oceny: obniża macroF1, a przy spellF1 scorer uzupełnia je
ostatnią etykietą w ocenianym odcinku. Dlatego niewidoczne przejęcie nadal kosztuje wynik.
Również rzeczywista przerwa z piłką stojącą blisko zawodnika może wyglądać jak kontrola.
Te heurystyki nie zastępują informacji o decyzji sędziego.

Wyniki do obejrzenia po uruchomieniu:

- [step1_memory_dead.csv](outputs/synthetic-step1/submissions/step1_memory_dead.csv) — nasza predykcja;
- [scoreboard.csv](outputs/synthetic-step1/evaluation/scoreboard.csv) — porównanie wszystkich metod;
- [comparison.json](outputs/synthetic-step1/comparison.json) — liczby błędów, fałszywych `D` i odpowiedzi `X` według sytuacji;
- [run_manifest.json](outputs/synthetic-step1/run_manifest.json) — parametry, wersje i hashe dla odtwarzalności.

To nadal test inżynierski na wymyślonych scenariuszach, nie wynik konkursowy.

### Sieć TCN + dekoder Viterbiego na zniekształconym IDSSE

`scripts/train_tcn.py` trenuje dwukierunkową sieć splotową (styl MS-TCN, ok. 0,3 mln parametrów, CPU)
na cechach co 5 klatek: piłka i jej braki, 3 najbliższe sloty każdej drużyny posortowane po odległości,
ruch drużyn. Wyjście H/A/D wygładza dekoder Viterbiego z jedną karą za zmianę etykiety.

- **Podział:** J03WN1 jest zamkniętym meczem testowym (losowanie `random.Random(2027)`); nie wolno go używać
  do treningu ani strojenia. Na pozostałych 6 meczach działa leave-one-match-out: w każdym foldzie 4 mecze
  treningowe, 1 walidacyjny (wybór epoki i kary) i 1 testowy.
- **Ziarna:** trening losuje ziarno 0–4 dla każdego wycinka. Walidacja i test zawsze na ziarnach 0 i 1.
- **Augmentacja:** zamiana drużyn (H↔A) i odbicia boiska.
- **Wyniki:** `outputs/tcn/<przebieg>/WYNIKI.md` (tabela), `wyniki.csv` (każde ziarno i metoda),
  `epochs.csv` (każda epoka), `log.txt`, `config.json` (ustawienia, commit, wersje) i modele foldów.
  Każdy fold porównuje TCN z Viterbim, TCN bez Viterbiego i baseline smooth50 na tych samych danych.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-nn.txt
.\.venv\Scripts\python.exe -X utf8 -m scripts.train_tcn cv --run v1
```

### Zgłoszenie: plik z predykcją

Zgłaszamy średnią 6 modeli z `weights/tcn_v1/` z dekoderem Viterbiego. Opis po angielsku dla organizatorów:
[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md). Polecenie uruchamiasz sam, w zwykłym terminalu, z plikiem wynikowym
poza projektem:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.predict_tcn --atd-xml <xml> --atd-raw <txt> --out <plik.csv> --threads 4
```

Trwa ok. 30 s i kończy się oficjalną walidacją. Ten sam komputer i ta sama liczba wątków dają identyczny plik.

Przerwany przebieg wznawia się tą samą komendą od pierwszego nieukończonego foldu.
`final --run v1` uruchamia jednorazowy test na J03WN1 z ustawieniami przebiegu. **To nie jest wynik konkursowy.**

## Instrukcja krok po kroku

**Na tym komputerze środowisko jest już przygotowane.** Projekt zawiera punkt
startowy, przykład do nauki i pierwszą eksperymentalną metodę z kroku 1.

1. Otwórz PowerShell i przejdź do folderu projektu:

   ```powershell
   Set-Location -LiteralPath 'C:\Users\Sebastian\Documents\More than Hack 2027'
   ```

2. Uruchom przykład:

   ```powershell
   .\run.ps1 demo
   ```

   Program sprawdzi oficjalny kod, przygotuje sztuczne pozycje zawodników i piłki,
   uruchomi cztery warianty oficjalnego baseline'u, sprawdzi ich CSV i obliczy oceny
   względem wymyślonych odpowiedzi referencyjnych. Baseline to prosta metoda startowa:
   najbliższy zawodnik plus wygładzanie odpowiedzi w czasie.
   Poczekaj, aż wróci znak zachęty terminala. Poprawny przebieg zawiera cztery
   komunikaty `OK`, tabelę wyników i końcowe `Done`.

3. Sprawdź projekt:

   ```powershell
   .\run.ps1 check
   ```

   To sprawdzenie wersji oficjalnego kodu, zależności i testów na danych syntetycznych.
   Oczekiwany koniec: `Ran 53 tests` i `OK`. Testy kontrolują m.in. numerację klatek,
   niedozwolone etykiety, obsługę `X`, przerwy `D` i zabezpieczenia syntetycznego wejścia.

4. Zajrzyj do wyników, jeśli chcesz zobaczyć pliki:

   - [submissions/](outputs/synthetic/submissions/) — cztery wygenerowane CSV;
   - [scoreboard.csv](outputs/synthetic/evaluation/scoreboard.csv) — tabela ocen;
   - [logs/](outputs/synthetic/logs/) — szczegóły działania oficjalnych skryptów.

5. Przy kolejnym podejściu powtórz `demo` i `check`. Nie trzeba ponownie instalować
   środowiska ani ręcznie przygotowywać danych. CSV powstają z kodu; nie poprawiamy
   ręcznie odpowiedzi w plikach wynikowych.

Na nowym komputerze lub gdy brakuje `.venv` najpierw uruchom `./setup.ps1`.
Jeżeli terminal nie znajduje `run.ps1`, wróć do kroku 1. Gdy widzisz `ERROR` lub
`FAILED`, wykonanie wymaga sprawdzenia; nie oznacza to zaliczonego testu.
Komunikat o zmienionym wejściu syntetycznym jest powodem do sprawdzenia plików,
a nie do podmieniania ich prywatnymi danymi.

## Jak rozumieć wyniki

**`OK` przy CSV oznacza poprawny format pliku**, a nie poprawne rozpoznanie całego meczu.
Każdy wynik ma 199 319 wierszy — po jednej odpowiedzi na klatkę sztucznego nagrania.

- `macroF1` ocenia etykiety poszczególnych klatek, z równą wagą klas `H/A/D`.
- `spellF1` ocenia odtworzenie całych odcinków posiadania. To główna metryka konkursu.
- Obie wartości mają zakres 0–1; większa oznacza lepszy wynik według danej metryki.
  Nie są prostym procentem poprawnych odpowiedzi.

Sprawdzenie wykonane **27 października 2026** na naszym syntetycznym przykładzie:

| Wariant | macroF1 | spellF1 |
|---|---:|---:|
| raw — bez wygładzania | 0,8812 | 0,0427 |
| smooth25 | 0,9679 | 0,9994 |
| smooth50 | 0,9314 | 0,9985 |
| smooth125 | 0,8000 | 0,9985 |

W tym przykładzie krótkie, celowo zasymulowane błędy obserwacji rozbijają wynik `raw`
na wiele małych posiadań. Wygładzanie ogranicza te przeskoki, dlatego `spellF1` rośnie.
Ten prosty przykład pomaga zrozumieć metrykę. **Nie dowodzi skuteczności na prawdziwym
meczu i nie służy do wyboru najlepszego wariantu konkursowego.**

Oba polecenia zakończyły się poprawnie: cztery pełne CSV przeszły walidację,
a wszystkie testy przeszły. Prywatnych danych konkursowych nie otwierano.
Żadnego pliku z demo nie wysyłamy jako zgłoszenia.

## Sprawdzenie czystości danych

[scripts/data_quality.py](scripts/data_quality.py) opisuje stan pliku trackingu samymi liczbami
zbiorczymi: luki w piłce, skoki pozycji, liczbę i ciągłość torów, duplikaty. Testy sprawdzają,
że znajduje dokładnie tyle błędów, ile wstawiono do sztucznych danych.

Na prawdziwych danych uruchamiasz go sam, bez asystenta: skopiuj ten jeden plik do folderu poza
projektem i użyj zwykłego terminala. Nie otwieraj danych ani raportu w edytorze z wtyczką AI.

```powershell
& 'C:\sciezka\do\projektu\.venv\Scripts\python.exe' -I -X utf8 .\data_quality.py analyze `
  --atd-xml 'C:\sciezka\do\danych\BAR-ATH_tactical_atd.xml' `
  --atd-raw 'C:\sciezka\do\danych\BAR-ATH_tactical_atd.txt' `
  --preset bar-ath --name konkurs --out .\raport
```

Powstają `quality_report.md` do czytania i `quality_report.json` do porównań. Oczekiwany komunikat
to `OK: all published counts match`, czyli zgodność z liczbami opublikowanymi przez organizatorów.
Polecenie `compare` zestawia kilka raportów JSON. Przed przekazaniem raportu komukolwiek, zwłaszcza
do chmurowego LLM, sprawdź, czy pozwala na to zaakceptowany regulamin.

## Gdzie co jest

```text
README.md              <- ta instrukcja
setup.ps1              <- przygotowanie środowiska, zwykle jeden raz
run.ps1                <- wygodne uruchamianie poleceń
phase1.py              <- łączy generator, baseline i walidację
official/              <- niezmieniony kod i dokumentacja organizatorów
scripts/               <- generator sztucznych danych i kontrola środowiska
solution/              <- nasze metody: krok 1 i dekoder odcinków
datasets/idsse/        <- publiczny zbiór IDSSE, poza Gitem
tests/                 <- testy formatu, bezpieczeństwa ścieżek i metryk
data/synthetic/        <- sztuczne dane, tworzone przez demo
outputs/synthetic/     <- lokalne CSV, wyniki i raporty
data/synthetic-step1/  <- trudniejsze scenariusze do kroku 1
outputs/synthetic-step1/ <- porównanie metod na tych samych scenariuszach
outputs/idsse/         <- raport jakości, definicji etykiet i metod na IDSSE
requirements.txt       <- dokładne wersje zależności
THIRD_PARTY_NOTICES.md <- źródła i licencje
AGENTS.md              <- zasady pracy asystentów nad projektem
```

Kod zespołu rozwijamy poza `official/`. Organizacyjnie najważniejsze są trzy miejsca:
**`official/` = punkt odniesienia, `data/` = wejście, `outputs/` = wynik.**

Wiedzę o grze, hipotezy i publiczne źródła zbieramy w [analizie piłkarskiej](docs/badania/analiza-pilkarska.md).

## Opcjonalnie: sprawdzenie jednego CSV

```powershell
# Sprawdzenie jednego pliku, bez odczytywania jakiegokolwiek GT.
.\run.ps1 validate .\outputs\synthetic\submissions\baseline_smooth50.csv
```

Walidacja wymaga dokładnie dwóch kolumn `frame,label`, 199 319 wierszy,
numerów klatek 0–199318 po kolei i etykiet `H`, `A`, `D`, `X`.
Sprawdza poprawność pliku, a nie trafność odpowiedzi.
Korzysta z oficjalnej funkcji walidacji i dodatkowo pilnuje dwóch kolumn wymaganych
przez dokumentację zgłoszenia.

W folderze wyników znajdziesz cztery baseline'y: `raw`, `smooth25`, `smooth50`,
`smooth125`. Ich nazwy i parametry pochodzą od organizatorów. Liczba przy `smooth`
oznacza połowę szerokości okna: `smooth50` analizuje ±50 klatek.
`run_manifest.json` zapisuje wersję źródeł, środowisko i hashe plików wynikowych.

Demo można uruchamiać ponownie: sprawdza pochodzenie wejścia i odtwarza wyniki.
Jeżeli wejście syntetyczne zostało zmienione, zatrzymuje się zamiast je nadpisywać.
Ścieżka demo jest ustalona w projekcie; zastępuje ewentualną zewnętrzną wartość
`MTAH_DATA_DIR`, bez odczytywania wskazanego przez nią katalogu.

## Zasady na obecny etap

- Nie otwieramy ani nie pobieramy prywatnego pakietu konkursowego.
- Nie pobraliśmy nawet publicznego 5-minutowego GT. Zgodnie z regulaminem nie wolno
  używać go do treningu, strojenia ani wyboru modelu.
- Nie tworzymy ręcznych etykiet ani wyjątków dla klatek konkursowego meczu.
- Prywatne dane i ich fragmenty nie mogą trafiać do chmurowych LLM bez wymaganej
  pisemnej zgody. Dotyczy to również zawartości logów i wyników narzędzi.
- `.gitignore` wyklucza dane, predykcje, modele i typowe materiały meczowe.
  Przed publikacją nadal trzeba sprawdzić listę i treść dodawanych plików.
- Warunki użycia i usunięcia prywatnych materiałów po udziale wynikają z zaakceptowanego
  regulaminu v2; licencja MIT publicznego kodu ich nie zastępuje.

Szczegóły: [reguły](official/RULES.md), [zgłoszenie](official/SUBMISSION.md),
[metryki](official/EVALUATION.md), [źródła i licencje](THIRD_PARTY_NOTICES.md).

## Środowisko i oficjalna wersja

Publiczne źródło: [repozytorium organizatorów](https://github.com/metrica-sports/more-than-a-hack-2027).

Przypięty commit: `f8f2629ff1e0985f81d7e59d8413678514ee1530`.
Pobrano tylko wybrane pliki kodu i dokumentacji; pełna lista z hashami jest w
`official/UPSTREAM.json`. Nie uruchamiaj teraz komend z oryginalnego README w
`official/`, które zakładają dostęp do prawdziwych danych lub publicznego GT.

Środowisko projektu wymaga Python 3.11+. Lokalnie sprawdzono Python 3.12.14.
`setup.ps1` tworzy `.venv`. Gdy dostępny jest dołączony runtime, korzysta z jego
już zainstalowanych pakietów i sprawdza dokładne wersje. Na innym komputerze może
pobrać zależności z `requirements.txt`. Nie pobiera danych konkursowych.

Możesz wskazać własny interpreter podczas pierwszej konfiguracji:

```powershell
.\setup.ps1 -Python 'C:\sciezka\do\python.exe'
```

Nie trzeba aktywować środowiska: `run.ps1` zawsze używa lokalnego `.venv`.
Na innych systemach odpowiednikiem jest `python phase1.py demo` w środowisku
z zainstalowanym `requirements.txt`.

**Następny krok:** wygenerować zniekształcone wersje pozostałych meczów IDSSE (`scripts/degrade_idsse.py`),
ustalić sposób zbliżenia etykiet do definicji konkursu i dopiero wtedy stroić dekoder oraz trenować model,
z podziałem według meczów.
