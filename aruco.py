import cv2
import numpy as np

def detect_aruco_markers(cameraMatrix, distCoeffs):
    """
    Detecta marcadores ARUCO em um vídeo da webcam.

    Args:
        cameraMatrix: Matriz de câmera calibrada.
        distCoeffs: Coeficientes de distorção da câmera.
    """

    cap = cv2.VideoCapture(0)  # Abre a webcam padrão

    # Carrega o dicionário de marcadores ARUCO usando os métodos disponíveis
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    parameters = cv2.aruco.DetectorParameters()  # Criação dos parâmetros do detector
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)  # Cria o detector com o dicionário

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Detecta os marcadores ARUCO na imagem
        corners, ids, rejectedImgPoints = detector.detectMarkers(frame)

        # Desenha os marcadores detectados
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Exibe o resultado
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Exemplo de uso:
if __name__ == "__main__":
    # Carregar os parâmetros de calibração (já feito no seu código)
    with np.load('camera_calibration/webcam_calibration_params.npz') as X:
        cameraMatrix, distCoeffs = [X[i] for i in ('K', 'dist')]

    detect_aruco_markers(cameraMatrix, distCoeffs)
