# CI do przygotowania po przeniesieniu kodu

Docelowy `check.yml` powinien instalować zależności bazowe i PyTorch CPU,
sprawdzać niezmienione `official/`, wczytać wymagane modele i wykonać inferencję
oraz eksport na niezależnej syntetyce. Brak sieci lub checkpointu nie może
prowadzić do cichego pominięcia testu głównej metody.

W szkielecie nie ma jeszcze workflow. CI nie pobiera danych konkursowych
i nie publikuje prywatnych wyników. Reprodukcja konkursowego CSV pozostaje
osobnym zadaniem uprawnionego operatora.
