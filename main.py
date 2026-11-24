import os
import time
import cv2
from deepface import DeepFace
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

load_dotenv()
RUTA_PERSONAS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "personas.autorizadas",
    "Personas1",
    "elegampi.jpg.jpeg"
)

print("Foto autorizada encontrada:", os.path.isfile(RUTA_PERSONAS))

API_KEY = os.getenv("ROBOFLOW_API_KEY")

if not API_KEY:
    raise ValueError("Falta ROBOFLOW_API_KEY en el archivo .env")

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
).configure(
    InferenceConfiguration(api_key_transport="header")
)


def main():
    camara = cv2.VideoCapture(0)

    if not camara.isOpened():
        print("No se pudo abrir la cámara.")
        return

    predicciones = []
    ultima_consulta = 0
    intervalo = 2.0
    persona_detectada = "Persona no reconocida"
    ultima_verificacion_rostro = 0

    print("FaceGuard iniciado. Presiona Q para salir.")

    try:
        while True:
            correcto, imagen = camara.read()

            if not correcto:
                print("No se pudo leer la cámara.")
                break

            ahora = time.time()
            
                        # Reconocer a la persona cada 3 segundos
            if ahora - ultima_verificacion_rostro >= 3:
                try:
                    resultado_rostro = DeepFace.find(
                        img_path=imagen,
                        db_path=os.path.dirname(
                            os.path.dirname(RUTA_PERSONAS)
                        ),
                        model_name="Facenet",
                        enforce_detection=False,
                        silent=True
                    )

                    if (
                        resultado_rostro
                        and not resultado_rostro[0].empty
                    ):
                        persona_detectada = "Persona autorizada"
                    else:
                        persona_detectada = "Persona no reconocida"

                except Exception as error:
                    print("Error de reconocimiento facial:", error)
                    persona_detectada = "No se pudo verificar"

                ultima_verificacion_rostro = time.time()

            # Consultar el modelo cada 2 segundos
            if ahora - ultima_consulta >= intervalo:
                if cv2.imwrite("captura_temporal.jpg", imagen):
                    try:
                        resultado = CLIENT.infer(
                            "captura_temporal.jpg",
                            model_id="hard-hat-workers/13"
                        )
                        predicciones = resultado.get("predictions", [])
                    except Exception as error:
                        print("Error de Roboflow:", error)

                    ultima_consulta = time.time()

            # Dibujar las detecciones sobre la imagen
            for objeto in predicciones:
                x = int(objeto["x"])
                y = int(objeto["y"])
                ancho = int(objeto["width"])
                alto = int(objeto["height"])

                x1 = int(x - ancho / 2)
                y1 = int(y - alto / 2)
                x2 = int(x + ancho / 2)
                y2 = int(y + alto / 2)

                clase = objeto["class"]
                confianza = objeto["confidence"] * 100
                etiqueta = f"{clase} {confianza:.0f}%"

                cv2.rectangle(
                    imagen, (x1, y1), (x2, y2), (0, 255, 0), 2
                )
                cv2.putText(
                    imagen, etiqueta, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
                clases = [
                objeto.get("class", "").lower()
                for objeto in predicciones
            ]

            if "helmet" in clases:
                estado = "CASCO DETECTADO - verificar posicion"
                color = (0, 255, 0)

            elif "person" in clases or "head" in clases:
                estado = "SIN CASCO DETECTADO"
                color = (0, 0, 255)

            else:
                estado = "SIN DETECCION"
                color = (0, 165, 255)

            cv2.putText(
                imagen,
                estado,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
            cv2.putText(
                imagen,
                persona_detectada,
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )
            cv2.imshow("FaceGuard - Deteccion de cascos", imagen)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camara.release()
        cv2.destroyAllWindows()

        if os.path.exists("captura_temporal.jpg"):
            os.remove("captura_temporal.jpg")


if __name__ == "__main__":
    main()