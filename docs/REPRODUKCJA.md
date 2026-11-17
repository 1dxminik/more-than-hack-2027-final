# Plan instalacji i reprodukcji C5

To plan przyszłego przeniesienia. W szkielecie nie ma jeszcze działającego
polecenia eksportu; nie uruchamiamy C5 ani danych konkursowych.

1. Przenieść minimalny kod z przypiętego commita C5 i przejrzeć wszystkie importy.
   Historyczne `outputs/` zawierały zasoby używane przez runtime; nowy układ
   `weights/c5-alpha-0.1/` i `configs/` wymaga jawnej adaptacji tych ścieżek.
2. Przypiąć zależności C5 oraz wersję Pythona. Historyczny pakiet wskazuje
   Python 3.12.14 i `torch==2.8.0+cpu`; nie zastępować ich automatycznie
   wersjami modelu kolegi. Przewidzieć świeży venv bez `--system-site-packages`,
   właściwe źródło pakietu CPU i `pip check`. Nie trenować ponownie do samej
   reprodukcji gotowego wyniku.
3. Ustalić kompletną listę checkpointów, modelu Linear, meta-modelu, normalizacji
   i konfiguracji. Kontrola ma odrzucać brakujące, dodatkowe i zmienione pliki.
   SHA256SUMS musi dotyczyć C5, a nie sześciu checkpointów kolegi.
4. Uprawniony operator podaje jawnie katalog całego pakietu, zawierający
   `tactical/BAR-ATH_tactical_atd.xml` i `tactical/BAR-ATH_tactical_atd.txt`.
   Finalny katalog wyniku jest osobny i prywatny. Nie dodawać skanowania
   dysków ani fallbacku do prywatnej ścieżki. Syntetyczne demo i improve
   nadal muszą wymuszać własne wejścia syntetyczne.
5. Po przeniesieniu sprawdzić na syntetyce zgodność starego i nowego runtime,
   w tym pełny N14, reset połówek i wszystkie kanały C5. CI musi instalować
   również zależności sieci i faktycznie sprawdzać jej wczytanie oraz inferencję.
   Sam zielony check bez PyTorch nie jest odbiorem modelu.
6. Przed ewentualnym prywatnym wykonaniem osobno ustalić zakres i uprawnienia
   operatora. Następnie wykonać dwa świeże procesy, oficjalną walidację dokładnie
   wybranych plików i porównanie bajtów oraz lokalnego SHA-256 z referencją C5.
   Zapisać wersję kodu, środowisko i manifest. Składnia poleceń, testy na
   syntetyce i poprawny format CSV nie dowodzą zgodności finalnego pliku.

Nie używamy hasha CSV kolegi jako referencji C5. Prywatne pliki, hashe wykonania,
logi, rozkłady etykiet i dane nie są przenoszone do tej gałęzi ani do usług AI.
Nie stroimy na meczu, nie porównujemy predykcji z wideo i nie używamy GT.

Nie deklarujemy identyczności między systemami operacyjnymi lub kompilacjami
CPU bez sprawdzenia. Po przyjęciu nowego układu kodu wymagany jest ponowny
odbiór jego reprodukcji; historyczny P4 nie zastępuje tego kroku.
