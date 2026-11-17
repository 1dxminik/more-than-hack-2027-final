# C5-alpha-0.1 — szkielet alternatywnej metody

Status: przygotowana struktura. Kod modelu, wagi, zależności i polecenie
odtworzenia nie zostały jeszcze przeniesione. Ta gałąź nie generuje CSV.

Docelowa metoda: TCN + pełny N14 + dekoder C5, alpha = 0,10.
Jest to alternatywa badawcza zespołu. Główne zgłoszenie ma korzystać
z rozwiązania kolegi; jego kod i plik zgłoszenia przygotowywane są osobno.

| Katalog | Przeznaczenie |
| --- | --- |
| `solution/` | Inferencja C5 i niezbędne moduły |
| `configs/` | Finalna konfiguracja oraz wersje zasobów |
| `scripts/` | Przygotowanie środowiska, eksport i kontrola techniczna |
| `docs/` | Instrukcja reprodukcji, opis metody i ograniczenia |
| `experiments/` | Kod wcześniejszych wariantów i porównań |
| `artifacts/` | Indeks wersji i odsyłacze do dopuszczonych zasobów |
| `tests/` | Testy syntetyczne finalnego kodu |
| `models/` | Lokalne wagi, poza historią Git |
| `data/` | Lokalne wejścia, poza historią Git |
| `outputs/` | Lokalne wyniki, poza historią Git |

Źródło przyszłego przeniesienia zapisano w [docs/POCHODZENIE.md](docs/POCHODZENIE.md).
Lista prac przed udostępnieniem jest w [docs/README.md](docs/README.md).
Nie ma jeszcze działającego polecenia instalacji ani inferencji.

Do historii Git nie dodajemy danych, predykcji, wygenerowanych raportów,
wag, cache ani sekretów. Przed przeniesieniem kodu trzeba zachować wymagane
licencje i sprawdzić pochodzenie wszystkich komponentów. Publikacja wymaga
osobnego zlecenia.
