# C5-alpha-0.1 — alternatywna metoda Ctrl Alt Elite

TCN + pełny N14 + dekoder C5. To dodatkowa metoda zespołu;
zgłoszony formularzem CSV pochodzi z modelu kolegi. Ta gałąź go nie zastępuje.
Nie zawiera danych ani konkursowych CSV. Wagi pozostają lokalnie poza Git.

## Instalacja

Środowisko referencyjne: Windows, Python 3.12.14, PyTorch 2.8.0+cpu,
dwa wątki CPU. Z katalogu repo, ze wskazanym interpreterem Python 3.12:

```powershell
.\setup.ps1 -Python python
.\.venv\Scripts\python.exe -s -X utf8 -m scripts.install_assets `
    --from-package 'D:\sciezka\code-and-weights.zip'
```

Pakiet oznacza lokalne archiwum kodu i wag C5 albo jego rozpakowany katalog
`code/`. Instalator odczytuje tylko siedem modeli z `configs/assets.json`
i sprawdza rozmiary oraz SHA-256. Nie kopiuje danych, CSV, logów ani cache.
Nie używaj sześciu wag kolegi z `weights/tcn_v1/`: to inna architektura.
Klon samej gałęzi wymaga osobnego dostarczenia lokalnego pakietu wag.

`setup.ps1` tworzy osobny venv bez dziedziczenia pakietów, instaluje przypięte
zależności CPU i wykonuje `pip check`. Na Linux odpowiednikiem jest własny
venv Python 3.12 i instalacja `requirements-nn.txt`. Identyczności między
systemami nie zakładamy bez sprawdzenia.

## Kontrola syntetyczna

```powershell
.\run.ps1 check
.\run.ps1 demo
.\run.ps1 improve
```

`check` sprawdza `official/`, zależności, modele i testy rzeczywistej inferencji
na wymyślonych trajektoriach. Brak PyTorch lub wag oznacza błąd, nie pominięcie.
`demo` generuje 6000 klatek, `improve` — 199319 klatek i 46 slotów, maksymalnie
22 widocznych graczy. Oba uruchamiają dwa świeże procesy, oficjalną kontrolę
formatu i porównanie CSV bajt po bajcie. Wymuszają syntetyczne `MTAH_DATA_DIR`.
Nowe wejścia trafiają do `data/synthetic-c5/`, wyniki do `outputs/synthetic-c5/`.
To test oprogramowania, nie wynik konkursowy. Wcześniejszych wyników nie nadpisujemy.

## Eksport z jawnych plików ATD

Polecenie dla uprawnionego operatora; nie wykonano go na danych konkursowych
w ramach przygotowania tej gałęzi:

```powershell
$dataDir = 'D:\sciezka\pakiet-organizatora'
$resultDir = 'D:\sciezka\prywatny-wynik'
.\.venv\Scripts\python.exe -s -X utf8 -m scripts.predict_c5 `
    --atd-xml "$dataDir\tactical\BAR-ATH_tactical_atd.xml" `
    --atd-raw "$dataDir\tactical\BAR-ATH_tactical_atd.txt" `
    --out "$resultDir\C5.csv"
if ($LASTEXITCODE -ne 0) { throw 'Inferencja lub walidacja nie przeszla' }
```

`dataDir` to katalog całego pakietu zawierający `tactical/`. Eksporter czyta
tylko wskazane XML/TXT, nie szuka danych i nie używa `MTAH_DATA_DIR`.
CSV ma `frame,label`, 199319 wierszy, klatki 0–199318 i H/A/D/X.
ATD 1–199319 przechodzi na zerową numerację CSV. Okresy gry i reset między nimi
są zgodne ze źródłem i jawne w `configs/c5.json`; poza okresami pozostaje X.

Sprawdzany jest dokładny zestaw wag, ich hashe i format wyniku.
Istniejący CSV nie jest nadpisywany. Bez GT, filmu i ponownego treningu.
Tożsamość konkretnego pliku wymaga osobnego porównania bajtów i SHA-256 dwóch
świeżych eksportów u operatora. Hash CSV kolegi nie jest referencją C5.
Wejścia, wyniki i logi prywatnego wykonania pozostają poza Git i usługami AI.

## Metoda i pochodzenie

37 cech przy 25 fps zasila pięć zamrożonych TCN B+CE. Ich prawdopodobieństwa
H/A/D są uśredniane z zachowaną symetryzacją drużyn. Pełny N14 dostarcza stan
i dowody kontroli piłki, przerw oraz wznowień. C5 miesza TCN z meta-modelem
log-liniowo przy alpha=0,1; Viterbi i reguły wyrównują granice i chronią plateau D.
Parametry nie są dobierane podczas uruchomienia.

Źródło: commit `d99253c24e739735510f76be40f1adaaafe39344`.
`SOURCE_MANIFEST.json` zapisuje hashe źródeł i migrację. Inferencja jest
w `solution/`, zamrożony N14 w `solution/n14/`. Nie ma runtime AST/exec,
podmiany funkcji, zmiany `sys.path` ani zależności od historycznych `outputs/`.
TCN i meta-model pochodzą z badań na publicznym IDSSE; Linear N14 z syntetyki.
Gałąź zawiera inferencję, bez treningu i dużych kampanii badawczych.
Pozostają błędy D, niejednoznaczne kontakty i luki trackingu; wyniki rozwojowe
nie dowodzą skuteczności na nieznanym meczu.

[Źródła i licencje](THIRD_PARTY_NOTICES.md) ·
[Zasoby modelu](weights/c5-alpha-0.1/README.md).

## Sprawdzenie migracji — 17 listopada 2026

11 testów PASS, bez pominięć, w osobnym venv; integralność 17 plików upstreamu.
Demo i pełne 199319 klatek dały po dwa identyczne CSV z poprawnym formatem.
Pełny syntetyczny CSV jest identyczny z zamrożonym oryginałem C5.
Na dwóch dodatkowych wejściach identyczne są wszystkie 20 tablic diagnostycznych,
w tym cechy, prawdopodobieństwa obu modeli i ślad N14. Raporty pozostają lokalnie.
