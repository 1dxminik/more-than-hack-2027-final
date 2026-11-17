# Mapowanie na repo kolegi

| Miejsce w repo kolegi | Odpowiednik lub plan dla C5 |
| --- | --- |
| `README.md` | Wprowadzenie do alternatywy i status przeniesienia |
| `TECHNICAL_REPORT.md` | Osobny opis C5; nie zastępuje dokumentu głównego zgłoszenia |
| `THIRD_PARTY_NOTICES.md` | Źródła, status licencji i wymagane uzupełnienia |
| `.gitattributes`, `.gitignore` | Stałe zakończenia linii i wykluczenia danych oraz wyników |
| `solution/` | Minimalna inferencja C5 i pełnego N14 — jeszcze bez kodu |
| `scripts/` | Dedykowany eksporter C5, sprawdzenie zasobów i powtarzalności — bez kodu |
| `weights/tcn_v1/` | Osobne `weights/c5-alpha-0.1/`; nie używać wag kolegi |
| `requirements.txt`, `requirements-nn.txt` | Do przeniesienia i przypięcia według środowiska C5 |
| `setup.ps1`, `run.ps1`, `phase1.py` | Do przygotowania z właściwymi poleceniami C5 i syntetycznymi kontrolami |
| `official/` | Do zachowania bez zmian z licencją i manifestem przy faktycznym przeniesieniu |
| `.github/workflows/check.yml` | Do przygotowania z instalacją PyTorch i testem finalnego runtime |
| `tests/` | Syntetyka, zgodność runtime, format i powtarzalność — do przeniesienia |
| `data/`, `outputs/` | Wyłącznie README w historii; wejścia i wyniki lokalnie |

Plików wykonawczych i manifestów wag nie zastępujemy atrapami udającymi
działający pipeline. Brakująca implementacja pozostaje jawna.

Dodatkowe `configs/` przechowuje docelowo osobną konfigurację C5.
`experiments/` jest na wybrane wcześniejsze warianty, `artifacts/` na indeks
ich wersji, a `docs/` na ograniczenia, reprodukcję i plan integracji.
Nie dodajemy surowych wyników, danych, checkpointów ani konkursowych CSV.
