# Lokalne zasoby C5-alpha-0.1

Wymagane jest dokładnie siedem plików z `configs/assets.json`:
pięć `tcn_*.npz`, `linear.json` pełnego N14 i `meta.npz` dla alpha=0,1.
Manifest zawiera role, rozmiary, SHA-256 i nazwy członków pakietu źródłowego.
Runtime sprawdza komplet przed wczytaniem.

Instalacja: `python -m scripts.install_assets --from-package SCIEZKA_PAKIETU`.
Można wskazać lokalny `code-and-weights.zip` albo jego rozpakowany katalog kodu.
Kopiowane są tylko wymagane modele, bez danych, wyników i CSV.

Modele są dostępne lokalnie do testów tej gałęzi i ignorowane przez Git.
Klon gałęzi wymaga osobnego dostarczenia pakietu. Wag nie opublikowano.
