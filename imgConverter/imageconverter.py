import cv2
import os
import numpy as np
import argparse
from os.path import exists

# Parser für die CMD Parameter
parser = argparse.ArgumentParser( prog = 'PKIImageConverter', description = 'Konvertiert Bilder in ein einheitliches Format zum Anlernen von Bilderkennungsalgorithmen', epilog = '')
parser.add_argument('-f', '--file', default='' )
parser.add_argument('-d', '--directory', default='' )
parser.add_argument('-W', '--width', default='800', type=int )            # Height und Width mit Großem H und W, weil kleines h für help 
parser.add_argument('-H', '--height', default='800', type=int)       
parser.add_argument('-fb', '--fillBlack', default='True' , choices=['True', 'False'])
parser.add_argument('-o', '--output', default='' )
parser.add_argument('-op', '--outputPrefix', default='p' )
parser.add_argument('-ft', '--fileType', default='bmp' )


# Json zum verwalten der Settings - Defaultwerte können mit Aufrufparametern überschrieben werden
settings = {
    'width': 800,
    'height': 800,
    'fillWithBlack': True,
    'outputDir': '',    # Relativer Pfad zum Output Directory
    'outputFormat': 'bmp',
    'outputPrefix': 'p',
    'allowedInputFormats': ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif']
}

# Counter für die Benamung der Ausgabe Bilder
COUNTER = 1


# Funktion zum konvertieren von Bildern
def convertImage(inputImg, counter):
    if os.path.isfile(inputImg):
        # Name und Pfad der Ausgabe
        outputImg = os.path.join(settings['outputDir'], settings['outputPrefix'] + str(counter).zfill(3) + '.' + settings['outputFormat'])
        # Counter hochzählen, wenn Bild mit diesem Namen schon vorhanden ist
        while exists(outputImg):
            counter += 1
            outputImg = os.path.join(settings['outputDir'], settings['outputPrefix'] + str(counter).zfill(3) + '.' + settings['outputFormat'])
   
        # Try Except, weil teilweise Bilder von OpenCV nicht gelesen werden 
        try:
            # Bild einlesen
            img = cv2.imread(inputImg)
            
            # Bild skalieren
            scaleFactor = 1 # Faktor um den das Bild skaliert wird
            height = img.shape[0]
            width = img.shape[1]
            
            if height >= width:
                scaleFactor = settings['height'] / height
            else:
                scaleFactor = settings['width'] / width
                
            dim = (int(width*scaleFactor), int(height*scaleFactor)) # Neue Dimension des Bildes als Tuple
            resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
                        
            # schwarz auffüllen
            if settings['fillWithBlack']:
                offsetWidth = int( (settings['width'] - dim[0]) / 2 )
                offsetHeight = int( (settings['height'] - dim[1]) / 2 )
                            
                if offsetHeight > 0 or offsetWidth > 0:
                    # Leeres Schwarzes Bild erstellen
                    blackedImg = np.zeros((settings['height'], settings['width'],3), np.uint8) 
                    blackedImg[:,:] = (0,0,0)
                    
                    # In das Schwarze bild wird Mittig über die Offsets das eigentliche Bild reinkopiert
                    tmpImg = blackedImg.copy()
                    tmpImg[ offsetHeight:offsetHeight+dim[1], offsetWidth:offsetWidth+dim[0] ] = resized.copy()
                    resized = tmpImg
                         
            # Bild schreiben
            cv2.imwrite(outputImg, resized)
        except:
            print('FEHLER: ' + inputImg + ' konnte nicht konvertiert werden!')
    
'''
for filename in os.listdir('input'):
    f = os.path.join('input', filename)
    if os.path.isfile(f):
        convertImage(f, COUNTER)
        COUNTER += 1
'''        

if __name__ == '__main__':
    args = parser.parse_args()
    settings['height'] = args.height
    settings['width'] = args.width
    settings['fillWithBlack'] = eval(args.fillBlack)
    settings['outputDir'] = args.output
    settings['outputFormat'] = args.fileType
    settings['outputPrefix'] = args.outputPrefix
    
    if args.file != '':
        convertImage(inputImg, 1)
        
    if args.directory != '':
        for filename in os.listdir(args.directory):
            if os.path.splitext(filename)[1].lower() in settings['allowedInputFormats']:
                f = os.path.join(args.directory, filename)
                if os.path.isfile(f):
                    convertImage(f, COUNTER)
                    COUNTER += 1



