import fiftyone as fo
import fiftyone.zoo as foz
import os

# Classes do Open Images que vamos usar para popular nosso dataset inicial
# Mapeando para nossas classes do gestos.yaml
# "pessoa_com_arma_fogo" -> "Handgun", "Weapon"
# "pessoa_com_faca" -> "Knife"
# "pessoa_com_capacete_ou_capuz" -> "Helmet", "Cap", "Hat"
# As outras classes precisarão de coleta manual e anotação.
classes = ["Handgun", "Weapon", "Knife", "Helmet", "Cap", "Hat"]

# Criar os diretórios se não existirem
output_dir = "GestureRecon/Gestures"
split = "train" # Vamos baixar tudo como 'train' e depois separar manualmente

print(f"Baixando amostras para as classes: {classes}")

# Baixar subconjunto de imagens do Open Images
dataset = foz.load_zoo_dataset(
    "open-images-v6",
    split="train",
    label_types=["detections", "classifications"],
    classes=classes,
    max_samples=500 # Aumente este número para um dataset maior
)

# Exportar dataset no formato YOLOv5/YOLOv8
dataset.export(
    export_dir=output_dir,
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="detections",
    split=split,
    classes=classes
)

print(f"\nDataset exportado para a pasta: {output_dir}")
print("Próximos passos:")
print("1. Inspecione as imagens e labels gerados.")
print("2. Separe manualmente cerca de 20% dos arquivos de 'images/train' e 'labels/train' para as pastas 'images/val' e 'labels/val'.")
print("3. Anote manualmente as classes 'pessoa_ocultando_rosto' e 'pessoa_ocultando_objeto' usando uma ferramenta de anotação.")