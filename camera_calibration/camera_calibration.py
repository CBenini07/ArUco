# ref.: https://learnopencv.com/camera-calibration-using-opencv/

import cv2
import numpy as np
import glob

# Defina o tamanho do tabuleiro de xadrez
chessboard_size = (8, 6)  # número de quadrados internos (cantos)
square_size = 0.023  # tamanho de cada quadrado do tabuleiro de xadrez em metros

# Prepare os pontos do objeto, como (0,0,0), (1,0,0), (2,0,0) ....,(6,4,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays para armazenar os pontos do objeto e os pontos da imagem de todas as imagens
objpoints = []  # pontos 3D no espaço do mundo real
imgpoints = []  # pontos 2D no plano da imagem

# Carregar todas as imagens do tabuleiro de xadrez
images = glob.glob('./calibration_images/*.png')

for fname in images:
    img = cv2.imread(fname)
    # Diminui resolução da imagem para facilitar processamento
    images = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Encontre os cantos do tabuleiro de xadrez
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # Se encontrado, adicione pontos do objeto e pontos da imagem
    if ret:
        # Refinar os cantos para precisão subpixel (aumenta o custo computacional)
        #   criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        #   corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners) 

        # Desenhe e visualize os cantos
        img = cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(25)

cv2.destroyAllWindows()

# Calibrar a câmera somente se pelo menos uma imagem foi encontrada
if objpoints and imgpoints:
    print('\niniciando calibragem...\n')
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Salvar os parâmetros da calibração
    np.savez('camera_calibration_params.npz', K=K, dist=dist)

    # Imprimir os parâmetros da câmera
    print("\n\nMatriz Intrínseca (K):")
    print(K)
    print("\nCoeficientes de Distorção (dist):")
    print(dist)
else:
    print("Nenhuma imagem de tabuleiro de xadrez foi encontrada.")
