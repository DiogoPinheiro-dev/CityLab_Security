# CityLab\_Security Documentação

### Visão Geral do Projeto

Repósitorio de um projeto feito na faculdade **FATEC de São José do Rio Preto**, cuja funcionalidade é ajudar no controle da entrada de pessoas não alunas nos campus da faculdade, aumentando a segurança dos alunos e preservando a integridade da instituição.

O projeto é, em sua maior parte, executado em **Python**, tendo como os principais pacotes, para o funcionamento do projeto, os seguintes: **ultralytics**, **insightface** e **opencv-python**.

### Autores

  * Mariana Lebrão Murauskas
  * Diogo de Lorenzi Pinheiro

### Orientador

  * Prof. Dr. Mário Henrique de Souza Pardo

### Execução do Projeto

#### Reconhecimento Facial

Para **gerar/atualizar o banco de dados** com as imagens de pessoas "validadas" (serão reconhecidas como alunos), o diretório `"alunos"` deve ser atualizado. Logo após, o arquivo `cadastro.py` deve ser executado.

> **Nota:** Essa etapa registra as faces autorizadas.

```bash
$ py cadastro.py
```

A **funcionalidade de reconhecimento facial** está inteiramente dentro do arquivo `reconhecimento.py`. Esse arquivo só deverá ser executado *após* a execução do arquivo `cadastro.py`.

```bash
$ py reconhecimento.py
```

#### Reconhecimento de Gestos e Objetos

O estado atual do projeto só permite o treinamento do **modelo YOLO, da ultralytics**, por meio de imagens e arquivos de anotação `.txt` correspondentes. Cada arquivo `.png` tem seu correspondente, com o mesmo nome, em um arquivo `.txt`.

Os arquivos `.txt` têm a funcionalidade de marcar, para o modelo, que tipo de imagem é aquela, como:

  * Uma pessoa com **arma de fogo**
  * Uma pessoa **encapuzada** ou de **capacete**
  * Uma pessoa com **arma branca**

Já as imagens são para o modelo gravar que tipo de imagem é aquela, seguindo o que está escrito no arquivo `.txt` correspondente.

#### Preparação e Filtragem de Imagens

Para executar, corretamente, esse treinamento, deve-se executar, primeiramente, o arquivo `get_gestures.py`. Esse arquivo tem a funcionalidade de **buscar na rede por imagens** de pessoas com bonés, capacetes, armas (brancas ou de fogo), etc.

```bash
$ py get_gestures.py
```

Após a execução desse arquivo, uma **limpeza de imagens manual** deverá ser executa, para a melhor eficiência do treinamento do modelo.

Antes da limpeza de imagens manual, podem ser executados os seguintes dois arquivos (sem ordem específica) para uma **filtragem inicial de imagens "ruins"**:

1.  **Análise de Qualidade:** O arquivo `analyze_quality.py` tem como sua funcionalidade marcar imagens que estejam **borradas** ou que estejam com **baixa luminosidade**, facilitando a limpeza das imagens para treinamento do modelo.

    ```bash
    $ py analyze_quality.py
    ```

2.  **Verificação de Arquivos de Anotação:** Outro arquivo útil para a limpeza de imagens é o `check_match_files.py`. Ele busca, e gera uma lista, por imagens que **possuem correspondência** com arquivos `.txt` e outra lista dos arquivos que **não têm correspondência**.

    ```bash
    $ py check_match_files.py
    ```

#### Treinamento do Modelo

Para finalizar, o treinamento do modelo é executado por meio de um arquivo com o nome de `train.py`, em que nele é chamado, apenas, o modelo e são passados a ele os parâmetros para treinamento, que são as imagens.

```bash
py train.py
```

# English version 

### Project Overview

This repository contains a project developed at **FATEC São José do Rio Preto** aiming to enhance campus security by controlling the entry of non-students. Its functionality helps increase the safety of students and preserve the integrity of the institution.

The project is predominantly executed in **Python**, utilizing the following key packages for its core functionality: **ultralytics**, **insightface**, and **opencv-python**.

### Authors

  * Mariana Lebrão Murauskas
  * Diogo de Lorenzi Pinheiro

### Advisor

  * Prof. Dr. Mário Henrique de Souza Pardo

### Project Execution

#### Face Recognition

To **generate or update the database** with images of "validated" individuals (who will be recognized as students), the directory `"alunos"` (students) must first be updated with the new images. Afterward, the `cadastro.py` file should be executed.

> **Note:** This step registers the authorized faces.

```bash
$ py cadastro.py
```

The **facial recognition functionality** itself is entirely contained within the `reconhecimento.py` file. This file should only be executed *after* running `cadastro.py`.

```bash
$ py reconhecimento.py
```

#### Gesture and Object Recognition

The current state of the project allows for the training of the **ultralytics YOLO model** using images and corresponding `.txt` annotation files. Each `.png` image has a counterpart `.txt` file with the same name.

The `.txt` files mark the object/gesture type for the model, such as:

  * A person with a **firearm**
  * A person wearing a **hood** or **helmet**
  * A person with a **knife/edged weapon**

The images are used for the model to learn the visual characteristics of these categories, guided by the corresponding `.txt` annotations.

#### Image Preparation and Filtering

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

#### Model Training

Finally, the model training is executed by running the `train.py` file. This script calls the model and passes the necessary parameters, which include the prepared image dataset, for training.

```bash
py train.py
```
