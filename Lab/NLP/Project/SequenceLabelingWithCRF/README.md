# SequenceLabelingWithCRF

Labelling dialogues with act tags using CRF suite in python

To run the project 

```
python baseline_crf.py train/ test/ model.crf.tagger
```
or

```
python advanced_crf.py train/ test/ model.crf.tagger
```

to evaluate the model
```
python evaluate_model.py dev/ model.crf.tagger
```
