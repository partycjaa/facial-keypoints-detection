Celem projektu było stworzenie modelu lokalizującego kluczowe punkty na zdjęciach twarzy. 

W ramach pracy ze zbiorem obrazków twarzy udało się zaprojektować i przetestować kilka interesujących modeli:

# Lokalizowanie kluczowych punktów twarzy
Zaprojektowano i porównano wiele modeli lokalizowania punktów charakterystycznych twarzy. Okazało się, że najlepiej sprawdzają się płytsze architektury CNN, a dodanie dropoutu wydaje się poprawiać zdolność modelu do generalizacji. Pomimo tego, że część punktów w zbiorze treningowym była oznaczona niedokładnie, modele osiągały zadowalające rezultaty i były w stanie wiarygodnie lokalizować kluczowe elementy twarzy (oczy, nos, usta).

# Odszumianie obrazków twarzy
Zbudowany autoenkoder z powodzeniem usuwał szum z obrazków. Choć rezultaty bywały lekko rozmyte, należy zaznaczyć, że także oryginalne dane wejściowe były dość niewyraźne. Oczyszczone obrazy zachowywały ogólne cechy twarzy, co wskazuje na to, że model nauczył się skutecznie odfiltrowywać zakłócenia.

# Autoencoder
Zbudowany autoenkoder pozwolił na zakodowanie obrazu do 96-wymiarowej reprezentacji latentnej, a następnie na jego rekonstrukcję. W rekonstrukcjach widać było, że model pomijał mniej istotne detale (np. tło czy okulary), co pokazuje, jakie cechy obrazu uznaje za najważniejsze. Ciekawym efektem była czasem zmiana wyrazu twarzy (np. otwarte oczy zamiast przymkniętych). Mimo uproszczeń, twarze pozostawały rozpoznawalne, co sugeruje, że model nauczył się efektywnej reprezentacji twarzy.

# Wariacyjny autoencoder
VAE również mapował obrazy do 96-wymiarowego przestrzeni latentnej, ale dodatkowo narzucał strukturę probabilistyczną, co umożliwiło generowanie nowych, nieistniejących twarzy. Wygenerowane obrazy były często nieco zniekształcone lub niewyraźne (a czasem wręcz przerażające), ale w wielu przypadkach przypominały realistyczne ludzkie twarze. To potwierdza, że model nauczył się uogólnionej reprezentacji twarzy.

W przyszłości warto zbadać, które wymiary przestrzeni latentnej odpowiadają za konkretne cechy twarzy – jak np. uśmiech, okulary czy kształt twarzy – i spróbować je manipulować, by kontrolować cechy generowanych twarzy.
