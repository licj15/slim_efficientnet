python -u train_image_classifier.py --train_dir=logs  --train_image_size=224  --model_name=mobilenet_v3_small  --dataset_name=imagenet --dataset_split_name=train --learning_rate=0.045*4 --preprocessing_name="inception_v2" --label_smoothing=0.1 --moving_average_decay=0.9999 --batch_size=96 --learning_rate_decay_factor=0.98 --num_epochs_per_decay=2.5/4 --dataset_dir=/userhome/ImageNet-Tensorflow/train_tfrecord --num_clones=4 > train_log.txt 2>&1


python -u eval_image_classifier.py --dataset_dir=/userhome/ImageNet-Tensorflow/validation_tfrecord --dataset_name=imagenet --dataset_split_name=validation --model_name=mobilenet_v3_small --checkpoint_path=logs/model.ckpt-XXXX >test_log.txt 2>&1
