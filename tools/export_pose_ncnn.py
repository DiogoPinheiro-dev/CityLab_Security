"""Exporta o peso de pose existente para comparacao opcional no Raspberry."""
import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    # O NCNN roda na forma fixa do export. No perfil rpi3, o frame de 640x480,
    # reduzido a 320x240, entra na pose em 416 como 320x416 (altura e largura):
    # exportado quadrado, em 416x416, a mao oculta caiu de 182 para 152 fases
    # nos videos de validacao, em 03/10/2026; em 320x416 ficou igual ao .pt.
    parser.add_argument("--imgsz", type=int, nargs="+", default=[640],
                        help="lado, ou altura e largura (rpi3: 320 416)")
    args = parser.parse_args(argv)
    if not args.model.is_file() or args.model.suffix != ".pt":
        parser.error("Informe um arquivo .pt de pose existente")
    if len(args.imgsz) > 2 or any(value <= 0 or value % 32 for value in args.imgsz):
        parser.error("imgsz: um ou dois valores positivos e multiplos de 32")
    destination = args.model.with_name(args.model.stem + "_ncnn_model")
    if destination.exists():
        parser.error("Diretorio NCNN ja existe; use uma copia do peso em outro diretorio")

    from ultralytics import YOLO

    model = YOLO(str(args.model.resolve()), task="pose")
    imgsz = args.imgsz[0] if len(args.imgsz) == 1 else args.imgsz
    result = model.export(format="ncnn", imgsz=imgsz, half=False, device="cpu")
    print(f"Exportado para: {result}")
    print("Valide caixas, keypoints, alertas e desempenho no Pi antes de ativar POSE_MODEL_PATH.")


if __name__ == "__main__":
    main()
