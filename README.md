# Bootstrapping Multilingual Metadata Extraction<br>A Showcase in Cyrillic

Source code and evaluation details for the [SDP 2021](https://sdproc.org/2021/) paper “Bootstrapping Multilingual Metadata Extraction: A Showcase in Cyrillic.”

![](https://github.com/IllDepence/sdp2021/raw/master/schematic_overview.png)

### Contents

* directory `data_set`
    * code for data filtering
    * code for data set building
    * code for generating annotated plain text
    * code for generating TEIs
    * code for data set statistics
* directory `model_training_and_evaluation`
    * directory `BiLSTM`
        * code for model training
        * code for evaluation
    * directory `grobid`
        * code for Evaluation of vanilla Grobid
        * code for Correcting of the Grobid TEI training data


### Data Set

Our data set can be downloaded at [zenodo.org/record/4708696](https://zenodo.org/record/4708696).


### Cite As

```
@inproceedings{kssf-2021-cyrillic,
    title = {{Bootstrapping Multilingual Metadata Extraction: A Showcase in Cyrillic}},
    author = {Krause, Johan and
              Shapiro, Igor and
              Saier, Tarek and
              F{\"a}rber, Michael},
    booktitle = {Proceedings of the Second Workshop on Scholarly Document Processing},
    year = {2021}
}
```
