Tutorial for creating MXNet .rec file with im2rec.py. Incluing Classification and ObjectDetection (both VOC and YOLO format annotations) tasks.

* Quick Start:

1. To generate .rec for classification task, run:
$ bash prepare_rec_clf.sh Dataset/Classification/PlantingSeeds/ seeds  

2. To generate .rec for object detection with YOLO-format annotations(txt files), run:
$ bash prepare_rec_yolo.sh ./Dataset/YOLO_format/images ./Dataset/YOLO_format/labels yolo.lst

3. To generate .rec for object detection with VOC-format annotations(xml files), run:
$ bash prepare_rec_voc.sh Dataset/VOC_format/multiClasses/ voc.lst

Note that for classification data, im2rec.py can be used to create .lst file, therefore we can set "train-ratio" and "test-ratio" for train-val-test split (if train-ratio+test-ratio<1 then val.rec will be created).
For object detection, .lst files should be customized by properly parsing your own label formats. Both VOC and Yolo formats are only examples to show this pipeline.

* Reference:
1. https://github.com/leocvml/mxnet-im2rec_tutorial
2. https://discuss.gluon.ai/t/topic/7585
