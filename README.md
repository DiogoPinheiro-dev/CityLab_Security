## CityLab_Security
Repósitorio de um projeto feito na faculdade FATEC de são josé do rio preto, cuja funcionalidade é ajudar no controle da entrada de pessoas não alunas nos campus da faculdade, aumentando a segurança dos alunos e preservand a integridade da instituição. O projeto é, em sua maior parte, executado em python, tendo como os principais, para o funcionamento do projeto, os seguintes pacotes: ultralytics, insightface e opencvs.

## Autores
Mariana Lebrão Murauskas; Diogo de Lorenzi Pinheiro

## Orientador
Prof. Dr. Mário Henrique de Souza Pardo

## Execução do projeto
# Reconhecimento facial
Para gerar/atualizar o banco de dados, com as imagens de pessoas "valídadas" (serão reconhecidas como alunos), o arquivo cadastro.py, deve ser executado, logo após o diretório "alunos" ser atualizado.

```bash
$ py cadastro.py
```

A funcionalidade de reconhecimento facil, está inteiramente dentro do arquivo "reconhecimento.py". Esse arquivos, só deverá ser executado após a execução do arquivo "cadastro.py".

```bash
$ py reconhecimento.py
```

# Reconhecimento de gestos
O estado atual do projeto, só permite o treinamento do modelo YOLO, da ultralytics, por meio de imagens e arquivos .txt, cada arquivo .png tem seu correspondente, com o mesmo nome, em um arquivo .txt. Os arquivos .txt, tem a funcionalidade de marcar, para o modelo, que tipo de imagem é aquela, se é uma imagem com arma de fogo, uma pessoa em capuzada ou de capacete ou uma pessoa com arma branca. Já as imagens, são para o modelo gravar que tipo de imagem é aquela, seguindo o que ta escrito no arquivo .txt correspondente.s

Para executar, corretamente, esse treinamento, deve-se excutar, primeiramente, o arquivo "get_gestures.py". Esse arquivo, tem a funcionalidade de buscar na rede por imagens de pessoas com bonés, capacetes, armas (brancas ou de fogo), etc. Após a execução desse arquivo, uma limpeza de imagens deverá ser executa a mão, para a melhor eficiência do treinamento do modelo.

```bash
$ py get_gestures.py
```

Antes da limpeza de imagens a mão, pode ser executados dois arquivos, sem ordem específica, para uma filtragem inicial de imagens "ruins". O arquivo "analyze_quality.py", tem como sua funcionalidade marcar imagens que estejam borradas ou não e marcar imagens que estejam com baixa luminosidade, facilitando a limpeza das imagens para trainamento do modelo.

```bash
$ py analyze_quality.py
```

Outro arquivo útil para a limpeza de imagens, é o arquivo "check_math_files.py", ele busca, e gera uma lista, por imagens que possuem correspondencia como arquivos .txt e outra lista dos arquivos que não tem correspondência.

```bash
$ py check_match_files.py
```

Para finalizar, o treinamento do modelo, é executado um arquivo com o nome de "train.py", em que nele é chamado, apenas, o modelo e passado a ele os parâmetros para treinamento, que são as imagens.

```bash
py train.py
```

# Project Overview

This repository contains a project developed at **FATEC São José do Rio Preto** aiming to enhance campus security by controlling the entry of non-students. Its functionality helps increase the safety of students and preserve the integrity of the institution.

The project is predominantly executed in **Python**, utilizing the following key packages for its core functionality: **ultralytics**, **insightface**, and **opencv-python**.

# Authors

  * Mariana Lebrão Murauskas
  * Diogo de Lorenzi Pinheiro

# Advisor

  * Prof. Dr. Mário Henrique de Souza Pardo

# Project Execution

## Face Recognition

To **generate or update the database** with images of "validated" individuals (who will be recognized as students), the directory `"alunos"` (students) must first be updated with the new images. Afterward, the `cadastro.py` file should be executed.

> **Note:** This step registers the authorized faces.

```bash
$ py cadastro.py
```

The **facial recognition functionality** itself is entirely contained within the `reconhecimento.py` file. This file should only be executed *after* running `cadastro.py`.

```bash
$ py reconhecimento.py
```

## Gesture and Object Recognition

The current state of the project allows for the training of the **ultralytics YOLO model** using images and corresponding `.txt` annotation files. Each `.png` image has a counterpart `.txt` file with the same name.

The `.txt` files mark the object/gesture type for the model, such as:

  * A person with a **firearm**
  * A person wearing a **hood** or **helmet**
  * A person with a **knife/edged weapon**

The images are used for the model to learn the visual characteristics of these categories, guided by the corresponding `.txt` annotations.

## Image Preparation and Filtering

To correctly execute this training, the `get_gestures.py` file must be run first. This file is responsible for **fetching images from the web** of people with caps, helmets, weapons (edged or fire), etc.

```bash
$ py get_gestures.py
```

After executing this file, a **manual image cleanup** is required to ensure the best training efficiency for the model.

Before the manual cleanup, the following two files can be executed (in no specific order) for an initial **filtering of "bad" images**:

1.  **Image Quality Analysis:** The `analyze_quality.py` file flags images that are **blurred** or have **low luminosity**, making the cleanup process easier.

    ```bash
    $ py analyze_quality.py
    ```

2.  **Annotation File Check:** The `check_match_files.py` file is useful for cleanup as it finds and generates one list for images that **have a corresponding `.txt` annotation file** and another list for images that **do not**.

    ```bash
    $ py check_match_files.py
    ```

## Model Training

Finally, the model training is executed by running the `train.py` file. This script calls the model and passes the necessary parameters, which include the prepared image dataset, for training.

```bash
py train.py
```
