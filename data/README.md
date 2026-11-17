# Dane wejściowe

`synthetic/` powstaje po `./run.ps1 demo`. Zawiera wyłącznie wymyślony tracking,
minimalne metadane potrzebne baseline'owi i syntetyczne etykiety referencyjne.
Nie jest to pełny eksport EPTS do testowania kloppy ani danych wideo.

Nie umieszczaj tu teraz prywatnych danych konkursowych. Demo nie potrzebuje
filmów, Smart Tagging, homografii, dostępu do Drive ani publicznego GT.

Cała zawartość tego folderu poza tym README jest wykluczona z Git.

`synthetic-step1/` powstaje po `./run.ps1 improve`. Zawiera nowy zestaw wymyślonych
scenariuszy: zasłonięcia piłki, przerwy, wznowienia, braki zawodników i przypadki
niejednoznaczne. `synthetic_cases.json` służy tylko do oceny po wygenerowaniu
predykcji; nie jest wejściem algorytmu. Starszego zestawu `synthetic/` nie zmieniamy.
