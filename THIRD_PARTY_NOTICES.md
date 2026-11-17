# Pochodzenie i licencje

C5/N14 pochodzi z lokalnego projektu MoreThanHack, commit
`d99253c24e739735510f76be40f1adaaafe39344`.
Hashe źródeł i opis migracji są w `SOURCE_MANIFEST.json`.
Nie nadano tu nowej licencji na autorski kod lub wagi; jej ustalenie
i publiczne udostępnienie zasobów wymagają osobnej decyzji.

`official/` jest niezmienioną wybraną kopią publicznego kodu Metrica Sports
More Than a Hack 2027, wersja `f8f2629ff1e0985f81d7e59d8413678514ee1530`.
Zachowano [MIT](official/LICENSE) i [UPSTREAM.json](official/UPSTREAM.json).
Licencja publicznego kodu nie obejmuje materiałów konkursowych.

Źródło historycznego treningu TCN i meta-modelu: IDSSE, Bassek i in.,
Scientific Data 2025, DOI 10.1038/s41597-025-04505-y, właściciel DFL,
publiczne lustro `pysport/idsse-data`, CC BY 4.0. Zbioru nie rozprowadzamy.
Linear N14 jest zamrożonym modelem syntetycznym.

Zależności: PyTorch, NumPy i pandas (BSD-3-Clause), python-dateutil
(Apache-2.0/BSD-3-Clause), six (MIT), tzdata (Apache-2.0 oraz informacje
bazy IANA). Pozostałe pakiety zachowują własne licencje i są przypięte
w `requirements-nn.txt`.
