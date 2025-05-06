# CPS841 Trabalho 1 
* Repetir (e possivelmente melhorar) os resultados obtidos com o Bloom WiSARD, utilizando os modelos BTHOWeN e ULEEN.
* Testar com os mesmos hiperparâmetros e com outros conjuntos de valores de parâmetros/entradas.
* Cooperação permitida entre grupos: compreensão dos modelos.
* Não cooperação: ajuste de hiperparâmetros e binarizações.

# Arquivos Importantes
* launch.json
    - Modifique isto para testar o BTHOWeN com parâmetros diferentes
    - Tem 3 configurações de inicialização
        - train_swept_models.py - Treina o modelo usando o BTHOWeN
        - evaluate.py - Avalia o modelo usando o BTHOWeN
        - full_run.py - Executa o modelo usando o ULEEN

* ULEEN/software_model/example.json
    - O modelo ULEEN usa isso para ler seus parâmetros

# Notes
* O ULEEN usa conjuntos de validação, mas não parece que o Bloom WiSARD ou o BTHOWeN usem.
    - Se o Bloom WiSARD não usa, então acho que não precisamos usar.
* Usei o mesmo código do BTHOWeN para carregar todos os conjuntos de dados UCI (basicamente tudo, exceto MNIST).
* Segui a mesma codificação de dados categóricos do Bloom WiSARD (ou seja, os rótulos a,b,c se tornam 0,1,2).
* Se você estiver usando Windows, poderá ter problemas para compilar devido às bibliotecas CUDA da Nvidia. Nesse caso, você pode excluí-las do arquivo requirements.txt principal e compilar sem elas. O código deve funcionar bem sem essas bibliotecas.


# Datasets from Bloom WiSARD paper (can use any of these names as input)
## Binary Classifications
* Adult
* Australian
* Banana
* Diabetes
* Liver
* Mushroom

## Multiclass classification
* Ecoli
* Glass
* Iris
* Letter
* MNIST
* Satimage
* Segment
* Shuttle
* Vehicle
* Vowel
* Wine

# BloomWiSARD Resultados

| Dataset    | Accuracy | Accuracy (Std) |
|------------|----------|----------------|
| Adult      | 0.718    | 0.0061495748   |
| Australian | 0.834    | 0.0223775813   |
| Banana     | 0.864    | 0.0057860498   |
| Diabetes   | 0.69    | 0.0262359291   |
| Liver      | 0.591    | 0.0483371899   |
| Mushroom   | 1.0    | 0.0000000000   |
| Ecoli      | 0.799    | 0.0202621531   |
| Glass      | 0.726    | 0.0367137219   |
| Iris       | 0.976    | 0.0215406592   |
| Letter     | 0.848    | 0.0045028728   |
| MNIST      | 0.915    | 0.0056781577   |
| Satimage   | 0.851    | 0.0057708318   |
| Segment    | 0.933    | 0.0080506388   |
| Shuttle    | 0.868    | 0.0122790440   |
| Vehicle    | 0.662    | 0.0238480121   |
| Vowel      | 0.876    | 0.0262235043   |
| Wine       | 0.926    | 0.0260741464   |

# References
* [BLoomwisard Paper](https://moodle.cos.ufrj.br/pluginfile.php/57035/mod_resource/content/1/1-s2.0-S0925231220305105-main-2.pdf)
* [bloomwisard github](https://github.com/leandro-santiago/bloomwisard)
    - Inclui todos os conjuntos de dados também. Usei-os para adicionar os conjuntos de dados ausentes

* [UCI Datasets](https://archive.ics.uci.edu/datasets)
    - Subconjunto destes usado pelo Bloom WiSARD

* [ULEEN](https://dl.acm.org/doi/epdf/10.1145/3629522)
* [ULEEN github](https://github.com/ZSusskind/ULEEN)

* [BTHOWeN](https://moodle.cos.ufrj.br/pluginfile.php/57024/mod_resource/content/1/3559009.3569680.pdf)
* [BTHOWen github](https://github.com/ZSusskind/BTHOWeN)
