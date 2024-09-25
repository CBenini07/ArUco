#!/usr/bin/env python3
import math
import cv2
import numpy as np
import rospy
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from pyquaternion import Quaternion
import tf2_ros
from scipy.spatial.transform import Rotation as R

# Classe para publicar transformações no ROS
from tf import TransformBroadcaster


# Classe principal para detecção de marcadores ArUco
class ArucoCamera:
    def __init__(self):
        # Inicializa o nó ROS
        rospy.init_node('aruco_detector', anonymous=True)

        # Assina os tópicos de câmera para receber informações sobre a câmera e imagens
        self.camera_info_sub = rospy.Subscriber('/usb_cam/camera_info', CameraInfo, self.camera_info_callback)
        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback)

        # Inicializa as variáveis de calibração da câmera
        self.distCoeffs = None  # Coeficientes de distorção
        self.cameraMatrix = None  # Matriz de câmera

        # Inicializa a ponte para converter entre ROS Image e OpenCV
        self.bridge = CvBridge()

        # Carrega o dicionário de marcadores ArUco
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)

        # Define o tamanho do marcador ArUco em metros
        self.size_marker = 0.1

        # Parâmetros de detecção de marcadores
        self.parameters = cv2.aruco.DetectorParameters()

        # Configuração para o formato dos floats
        self.float_formatter = "{:.0f}".format

        # Inicializa o buffer para o TF (transformações entre frames)
        self.tfBuffer = tf2_ros.Buffer()

        # Inicializa o broadcaster para enviar transformações
        self.tfBroadcaster = tf2_ros.TransformBroadcaster()

    # Função chamada quando a informação da câmera é recebida
    def camera_info_callback(self, msg):
        # Extrai a matriz da câmera e os coeficientes de distorção da mensagem CameraInfo
        self.cameraMatrix = np.array(msg.K).reshape((3, 3))
        self.distCoeffs = np.array(msg.D)

    # Função para estimar a pose do marcador ArUco
    def estimate_pose(self, corners_, marker_size):
        # Define os pontos do marcador ArUco em 3D (no espaço real)
        marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                                  [marker_size / 2, marker_size / 2, 0],
                                  [marker_size / 2, -marker_size / 2, 0],
                                  [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)

        rvecs = []  # Vetores de rotação
        tvecs = []  # Vetores de translação
        for corner in corners_:
            # Estima a pose do marcador usando solvePnP (Resolve a transformação entre 2D e 3D)
            _, rotation, translation = cv2.solvePnP(marker_points, corner, self.cameraMatrix, self.distCoeffs, False,
                                                    cv2.SOLVEPNP_IPPE_SQUARE)
            rvecs.append(rotation)
            tvecs.append(translation)
        return rvecs, tvecs

    # Função chamada quando uma nova imagem é recebida
    def image_callback(self, msg):
        try:
            # Converte a imagem ROS para formato OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            print(e)
            return

        # Converte a imagem para escala de cinza (necessário para a detecção de marcadores ArUco)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detecta os marcadores ArUco na imagem
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)

        if len(corners) > 0:
            # Desenha os marcadores detectados na imagem
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Estima a pose dos marcadores (posição e orientação)
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.size_marker, self.cameraMatrix, self.distCoeffs)

            for i, marker_id in enumerate(ids):
                # Cria uma mensagem de transformação
                t = TransformStamped()
                t.header.frame_id = 'camera_depth_frame'
                t.child_frame_id = 'aruco marker'

                # Armazena a translação (posição) do marcador
                t.transform.translation.x = tvec[i][0][0]
                t.transform.translation.y = tvec[i][0][1]
                t.transform.translation.z = tvec[i][0][2]

                # Calcula a rotação em matriz e converte para quaternion
                rotation_matrix = np.eye(4)
                rotation_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvec[i][0]))[0]
                r = R.from_matrix(rotation_matrix[0:3, 0:3])
                quat = r.as_quat()

                # Armazena a rotação (orientação) como quaternion
                t.transform.rotation.x = quat[0]
                t.transform.rotation.y = quat[1]
                t.transform.rotation.z = quat[2]
                t.transform.rotation.w = quat[3]

                # Publica a transformação (posição e orientação) do marcador
                self.tfBroadcaster.sendTransform(t)

                # Desenha os eixos do marcador na imagem
                cv2.drawFrameAxes(frame, self.cameraMatrix, self.distCoeffs, rvec[i], tvec[i], self.size_marker / 2)

        # Mostra a imagem com os marcadores detectados
        cv2.imshow("ArUco Detection", frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    # Inicializa a classe ArucoCamera e roda o nó ROS
    aruco_detector = ArucoCamera()
    rospy.spin()  # Mantém o nó em execução
