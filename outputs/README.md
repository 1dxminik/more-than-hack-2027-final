# Wyniki lokalne

`synthetic/` powstaje po `./run.ps1 demo`:

- `submissions/`: cztery CSV wygenerowane oficjalnym baseline'em;
- `evaluation/scoreboard.csv`: wyniki na danych syntetycznych;
- `evaluation/reports/`: raporty oficjalnego scorera;
- `logs/`: pełne komunikaty niezmienionego baseline'u i scorera;
- `run_manifest.json`: wersje środowiska, źródło kodu i hashe wyników.

To pliki demonstracyjne. Nie wysyłaj ich jako rozwiązania konkursowego.
Wynik syntetyczny nie określa jakości rozwiązania na prawdziwym meczu.
Uwagi o niepewności statystycznej w oryginalnym logu scorera odnoszą się do jego
kalibracji na prawdziwym meczu; nie należy ich przenosić na to sztuczne demo.

Cała zawartość tego folderu poza tym README jest wykluczona z Git.

`synthetic-step1/` powstaje po `./run.ps1 improve`. Ma analogiczne podfoldery oraz
piąty CSV `step1_memory_dead.csv`. `comparison.json` pokazuje błędy według
syntetycznych sytuacji, także celowo trudnych. `run_manifest.json` zapisuje parametry
metody i hashe jej źródeł. Wyniki porównujemy wyłącznie w obrębie tego samego zestawu.
