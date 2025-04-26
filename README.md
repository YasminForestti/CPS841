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
* Acho que o ULEEN roda até você CTRL+c para pará-lo.
* O ULEEN usa conjuntos de validação, mas não parece que o Bloom WiSARD ou o BTHOWeN usem.
    - Se o Bloom WiSARD não usa, então acho que não precisamos usar.
* Usei o mesmo código do BTHOWeN para carregar todos os conjuntos de dados UCI (basicamente tudo, exceto MNIST).
* Segui a mesma codificação de dados categóricos do Bloom WiSARD (ou seja, os rótulos a,b,c se tornam 0,1,2).


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