import os
import shutil
import random


def copy_and_rename(src_path, dest_path):
    """
    Kopiert eine Datei von src_path nach dest_path und übernimmt dabei Metadaten
    (z.B. Änderungszeitstempel) via shutil.copy2.
    """
    shutil.copy2(src_path, dest_path)


def Chao_Non_Team():
    """
    Erzeugt für jede Datei im Ordner 'Chao\\Non-Team-Variante' 100 Kopien.
    In jeder Kopie wird die erste Zeile (Index 0) durch 'i\\t1\\n' ersetzt,
    wobei i von 0 bis 99 läuft.
    """
    dirPath = "Model_and_Data\\"
    directory = os.fsencode(dirPath + "Chao\\Non-Team-Variante")
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        for i in range(0, 100):
            # neue Zieldatei mit Suffix _i.txt im gleichen Ordner
            destPath = os.path.join(
                directory,
                os.fsencode(filename.split(".")[0] + '_' + str(i) + ".txt")
            )
            print(destPath)

            # Originaldatei nach destPath kopieren
            copy_and_rename(directory.decode('utf8') + '\\' + filename, destPath)

            # Inhalt der Originaldatei lesen
            zeilen = ""
            with open(directory.decode('utf8') + "\\" + filename, 'r', encoding='utf8') as f:
                zeilen = f.readlines()

            # erste Zeile mit neuem Wert für i überschreiben
            zeilen[0] = str(i) + '\t1\n'

            # modifizierten Inhalt in die neue Datei schreiben
            with open(
                directory.decode('utf8') + "\\" + filename.split(".")[0] + '_' + str(i) + ".txt",
                'w',
                encoding='utf8'
            ) as f:
                f.writelines(zeilen)


def Chao_Team():
    """
    Erzeugt für jede Datei im Ordner 'Chao\\Team-Variante' genau eine Kopie.
    In dieser Kopie werden:
      - Zeile 2 (Index 1) auf 'm 1\\n' gesetzt
      - Zeile 3 (Index 2) auf 'tmax <zufälliger Wert>\\n' gesetzt.
    """
    dirPath = ""Model_and_Data\\"
    directory = os.fsencode(dirPath + "Chao\\Team-Variante")
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        for i in range(0, 1):
            destPath = os.path.join(
                directory,
                os.fsencode(filename.split(".txt")[0] + '_' + str(i) + ".txt")
            )
            print(destPath)

            # Originaldatei kopieren
            copy_and_rename(directory.decode('utf8') + '\\' + filename, destPath)

            # Originalinhalt lesen
            zeilen = ""
            with open(directory.decode('utf8') + "\\" + filename, 'r', encoding='utf8') as f:
                zeilen = f.readlines()

            # Zeile 2: Mode-Parameter
            zeilen[1] = "m 1\n"
            # Zeile 3: tmax mit Zufallswert im Bereich [1.0, 110.0]
            zeilen[2] = "tmax " + str(round(random.uniform(1.0, 110.0), 1)) + '\n'

            # neue Datei mit angepassten Zeilen schreiben
            with open(
                directory.decode('utf8') + "\\" + filename.split(".txt")[0] + '_' + str(i) + ".txt",
                'w',
                encoding='utf8'
            ) as f:
                f.writelines(zeilen)


def Chao_Team_simple_change():
    """
    Variante ohne Kopien: geht alle .txt-Dateien im Ordner 'Chao\\Team-Variante' durch
    und setzt jeweils nur Zeile 2 (Index 1) auf 'm 1\\n'. tmax (Zeile 3) bleibt unverändert.
    Die Änderung wird atomar über eine temporäre Datei zurückgeschrieben.
    """
    dirPath = r"\Chao\Team-Variante"
    for filename in os.listdir(dirPath):
        if not filename.lower().endswith(".txt"):
            # nur .txt-Dateien bearbeiten
            continue

        path = os.path.join(dirPath, filename)

        # Dateiinhalt vollständig lesen
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if len(lines) < 2:
            # Dateien mit weniger als 2 Zeilen überspringen
            print(f"Übersprungen (zu wenig Zeilen): {path}")
            continue

        # Zeile 2 (Index 1) fest setzen
        lines[1] = "m 1\n"

        # atomisches Überschreiben via temporäre Datei
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            f.writelines(lines)
        os.replace(tmp, path)

        print(f"Updated: {path}")


def Chao_Team_Scores():
    """
    Erzeugt für jede Datei im Ordner 'Chao\\Team-Variante' eine Variante mit
    angehängtem Suffix '_H<i>'. Ab Zeile 4 (Index 3) wird für jede Zeile
    die dritte Spalte durch einen zufälligen Score (0–29) ersetzt.
    """
    dirPath = ""Model_and_Data\\"
    directory = os.fsencode(dirPath + "Chao\\Team-Variante")
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        for i in range(1, 2):
            destPath = os.path.join(
                directory,
                os.fsencode(filename.split(".txt")[0] + '_H' + str(i) + ".txt")
            )
            print(destPath)

            # Originaldatei kopieren
            copy_and_rename(directory.decode('utf8') + '\\' + filename, destPath)

            # Originalinhalt lesen
            zeilen = ""
            with open(directory.decode('utf8') + "\\" + filename, 'r', encoding='utf8') as f:
                zeilen = f.readlines()

            # Zeilen ab Index 3 (4. Zeile) modifizieren:
            # dritte Spalte wird durch Zufallswert ersetzt, Spalte 1 und 2 bleiben erhalten
            for i in range(3, len(zeilen)):
                zeile = zeilen[i].split("\t")
                zeilen[i] = zeile[0] + "\t" + zeile[1] + "\t" + str(random.randrange(start=30)) + "\n"

            # angepasste Zeilen in neue Datei schreiben
            with open(
                directory.decode('utf8') + "\\" + filename.split(".txt")[0] + '_' + str(i) + ".txt",
                'w',
                encoding='utf8'
            ) as f:
                f.writelines(zeilen)
