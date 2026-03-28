import cv2

from recognition_pipeline import UnifiedRecognitionService


def draw_results(frame, results):
    for face in results.get("faces", []):
        x1, y1, x2, y2 = face["bbox"]
        name = face["name"]
        confidence = face["confidence"]

        if name == "NAO ALUNO":
            color = (0, 0, 255)
            label = "NAO ALUNO"
        else:
            color = (0, 255, 0)
            label = f"{name} ({int(confidence * 100)}%)"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    for person in results.get("persons", []):
        x1, y1, x2, y2 = person["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

    for tracked_person in results.get("gestures", []):
        x1, y1, x2, y2 = tracked_person["bbox"]
        alerts = tracked_person["alerts"]
        track_id = tracked_person["track_id"]

        color = (0, 0, 255) if alerts else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"ID: {track_id}",
            (x1, max(0, y1 - 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

        for index, alert in enumerate(alerts):
            text_y = max(0, y1 - 10) + (index * 20)
            cv2.putText(
                frame,
                alert,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

    for detected_object in results.get("objects", []):
        x1, y1, x2, y2 = detected_object["bbox"]
        label = detected_object["label"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            frame,
            label,
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )


def main():
    service = UnifiedRecognitionService(run_in_parallel=True)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        raise RuntimeError("Nao foi possivel acessar a camera.")

    print("Sistema combinado iniciado. Pressione 'q' para sair.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = service.process_frame(frame)
            draw_results(frame, results)

            cv2.imshow("Reconhecimento Unificado", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
