# Gałąź C5 i główne zgłoszenie

Docelowy układ to jedno repo zgłoszeniowe zespołu:

- `main`: metoda kolegi, jej instrukcja i dokument techniczny odtwarzające
  CSV już wysłany przez formularz, zgodnie z informacją użytkownika;
- osobna gałąź `codex/c5-alpha-0.1`: C5 jako dodatkowa metoda badawcza,
  ze wskazaniem źródła, konfiguracji, zasobów i własnej reprodukcji;
- odnośnik z głównego README do alternatywy można dodać później osobno,
  bez zmiany domyślnego eksportera i znaczenia dokumentu zgłoszeniowego.

Nie ma potrzeby łączenia predykcji obu metod. Takie połączenie byłoby nową
metodą, wymagającą osobnego wykonania, reprodukcji i decyzji o zgłoszeniu.
Nie zmieniamy pliku w formularzu i nie składamy drugiego zgłoszenia.

CSV C5 oraz CSV kolegi pozostają poza publiczną historią Git.
Gałąź badawcza ani dodatkowy plik w repo nie są drugim ocenianym zgłoszeniem.

## Stan lokalny

Pracujemy w `codex/c5-alpha-0.1-structure`, z niezależnym commitem startowym
`d069cc6dd1e069f69b836c84fd1b107677f67448`. Próba pobrania publicznego `main`
i utworzenia worktree z jego historii została odrzucona przy autoryzacji
polecenia. Nie dodano remote i nie zmieniono `origin`.

Obecny szkielet wymaga późniejszego przeniesienia wybranych plików na gałąź
opartą na repo kolegi. Nie należy scalać całej niezależnej historii do `main`
ani używać `--allow-unrelated-histories` jako skrótu do tego zadania.
Publikacja, PR i kontakt z kolegą nie zostały wykonane.
