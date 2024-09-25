import cv2
import numpy as np

def detect_aruco_markers(cameraMatrix, distCoeffs, video_path):
    """
    Detecta marcadores ARUCO em um vídeo fornecido.

    Args:
        cameraMatrix: Matriz de câmera calibrada.
        distCoeffs: Coeficientes de distorção da câmera.
        video_path: Caminho para o arquivo de vídeo .mp4
    """

    cap = cv2.VideoCapture(video_path)  # Abre o arquivo de vídeo .mp4

    if not cap.isOpened():
        print(f"Erro ao abrir o arquivo de vídeo: {video_path}")
        return

    # Carrega o dicionário de marcadores ARUCO usando os métodos disponíveis
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()  # Criação dos parâmetros do detector
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)  # Cria o detector com o dicionário

    while cap.isOpened():
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

    # Caminho do vídeo MP4
    video_path = "caminho_do_seu_video/video.mp4"

    detect_aruco_markers(cameraMatrix, distCoeffs, video_path)
