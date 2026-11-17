# Źródła i licencje

Folder `official/` zawiera wybrane, niezmienione publiczne pliki projektu
[More Than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027).

- Autor: Metrica Sports; copyright 2026–2027.
- Licencja tych plików: [MIT](official/LICENSE).
- Wersja: `f8f2629ff1e0985f81d7e59d8413678514ee1530`.
- Źródło i hash każdego pliku: [official/UPSTREAM.json](official/UPSTREAM.json).
- Weryfikacja zgodności kopii: `./run.ps1 check`.

To wybrany zestaw plików, a nie pełny klon repozytorium. Odnośniki w oryginalnej
dokumentacji do notebooków, grafik, GT i wyników mogą prowadzić do plików, których
celowo tu nie pobrano. Lokalne polecenia opisuje główny README.md.

Licencja repozytorium organizatorów nie obejmuje prywatnego pakietu meczowego.
Nie pobrano żadnego GT ani predykcji organizatorów. Całe wejście demonstracyjne
powstaje w naszym generatorze z wymyślonych współrzędnych.

Publiczny zbiór IDSSE: Bassek i in., *Scientific Data* 2025,
[DOI 10.1038/s41597-025-04505-y](https://doi.org/10.1038/s41597-025-04505-y), właściciel danych DFL,
licencja CC BY 4.0. Pobierany lokalnie przez `scripts/fetch_idsse.py` z lustra
[pysport/idsse-data](https://huggingface.co/datasets/pysport/idsse-data). Nie jest częścią repozytorium;
do raportów przeliczamy go na tablice, etykiety H/A/D i zniekształcone warianty.
Wagi sieci w `weights/tcn_v1/` wytrenowano na zniekształconych meczach IDSSE (CC BY 4.0, uznanie
autorstwa jak wyżej). Zawierają tylko parametry sieci i średnie/odchylenia cech, bez klatek danych.
PyTorch (BSD-3-Clause) jest zależnością modelu (`requirements-nn.txt`).

Nie skopiowano kodu BallRadar, PathCRF, ligafooty ani football-cv-analytics.
Licencje ewentualnych przyszłych komponentów należy sprawdzić osobno.

Zależności środowiska: NumPy (BSD-3-Clause), pandas (BSD-3-Clause),
python-dateutil (Apache-2.0/BSD-3-Clause), six (MIT), tzdata (Apache-2.0;
materiał bazy IANA ma własne informacje o prawach). Ich pakiety zachowują własne licencje.
Nie narzucamy tutaj licencji na przyszły autorski kod zespołu.
