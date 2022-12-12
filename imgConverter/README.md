Konvertiert Bilder in ein einheitliches Format zum anlernen von Bilderkennungsalgorithmen


Benutzung (Alle Bilder aus dem Ordner input konvertieren und in output speichern):
Aufruf per python über die Kommandozeile: 

- pyhon imageconverter.py -d input -o output


Parameter:
'-f', '--file'            - Filename für einzelnes Bild
'-d', '--directory'       - Relativer Pfad zum Input Ordner
'-W', '--width'           - width, default 800 
'-H', '--height'          - height, default 800
'-fb', '--fillBlack'      - Bool ob fehlende Bildfläche schwarz ausgefüllt wird, default true
'-o', '--output'          - Relativer Pfad zum Ausgabeordner
'-op', '--outputPrefix'   - Prefix für die Benennung der Bilder (Name = Prefix + hochgezählte Nummer)
'-ft', '--fileType'       - Ausgabe Filetyp (bisher getestet: bmp, jpg, png), default bmp
