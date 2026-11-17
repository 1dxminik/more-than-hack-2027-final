# Ctrl Alt Elite — alternatywa C5-alpha-0.1

Status: lokalny szkielet C5, bez implementacji, wag i działającego eksportera.
Układ dokumentów i katalogów dostosowano do repozytorium
[zgłoszenia zespołu](https://github.com/1dxminik/more-than-hack-2027-final),
wersja `994b124c8c78036307faa915f08d59943bba52c6`.

## Relacja do zgłoszenia

Użytkownik potwierdził, że formularzem przesłano CSV metody kolegi.
Tamten plik oraz jego reprodukcja pozostają głównym zgłoszeniem.
C5-alpha-0.1 to dodatkowa metoda badawcza, bez zastępowania zgłoszonego wyniku.
CSV C5 przechowujemy prywatnie; nie dodajemy go do repo ani jako drugiego pliku
do zgłoszenia. Ewentualna zmiana zgłoszenia wymaga osobnego zlecenia.

Docelowo C5 powinno być osobną gałęzią opartą na `main` repo zgłoszeniowego.
Obecna gałąź szkieletowa nadal ma niezależną historię. Nie została połączona
z repo kolegi ani przygotowana jako gotowy PR. Szczegóły:
[organizacja gałęzi](docs/GALAZ-I-ZGLOSZENIE.md).

## Instalacja i reprodukcja

Nie ma jeszcze polecenia generującego CSV z tej gałęzi. Nie używamy instrukcji
`scripts.predict_tcn` kolegi do C5: architektura, zasoby i konfiguracja są inne.
Środowisko historycznego pakietu C5 to Python 3.12.14 i PyTorch 2.8.0+cpu;
nie zastępujemy go środowiskiem referencyjnym kolegi z Pythonem 3.12.10.

[Plan reprodukcji](docs/REPRODUKCJA.md) uwzględnia świeże środowisko CPU,
dokładny zestaw wag i ich hashe, jawne ścieżki, dwa przebiegi, oficjalną
walidację i porównanie finalnego CSV. Przygotowanie planu nie jest odbiorem
nowego układu. Przeniesienie wag z historycznych `outputs/` do `weights/`
wymaga również adaptacji kodu, a potem ponownej kontroli.

## Układ

| Miejsce | Rola |
| --- | --- |
| `TECHNICAL_REPORT.md` | Opis C5, wyniki i jawne ograniczenia |
| `THIRD_PARTY_NOTICES.md` | Pochodzenie i status licencji |
| `solution/` | Inferencja C5 i pełnego N14 — do przeniesienia |
| `scripts/` | Eksporter oraz kontrole zasobów — do przeniesienia |
| `weights/c5-alpha-0.1/` | Opis wymaganych wag; samych wag brak |
| `configs/` | Dokładna konfiguracja i metadane zasobów — do przeniesienia |
| `tests/` | Testy syntetyczne i odbiór modelu — do przeniesienia |
| `.github/workflows/` | Plan CI obejmującego również PyTorch i eksporter |
| `data/`, `outputs/` | Opisy katalogów lokalnych; bez danych i wyników |
| `experiments/`, `artifacts/` | Wcześniejsze warianty oraz indeks dorobku |
| `docs/` | Integracja gałęzi, reprodukcja i pochodzenie |

Pełne mapowanie do układu kolegi, w tym nieprzeniesione pliki instalacji,
zależności i `official/`: [struktura](docs/STRUKTURA.md).
Źródło C5: [przypięty commit](docs/POCHODZENIE.md).

Do historii Git nie dodajemy danych, CSV, wygenerowanych raportów, wag,
cache ani sekretów. Publiczny dostęp do wymaganych wag trzeba ustalić osobno.
Publikacji i wysłania zgłoszenia nie wykonano.
