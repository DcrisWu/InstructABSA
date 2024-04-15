python ../run_model.py -mode train -model_checkpoint ../model/tk-instruct-base-def-pos \
-experiment_name ate_check -task ate -output_dir ../Models \
-inst_type 2 \
-id_tr_data_path ../Dataset/SemEval14/Train/Laptops_Train.csv \
-id_te_data_path ../Dataset/SemEval14/Test/Laptops_Test.csv \
-ood_tr_data_path ../Dataset/SemEval14/Train/Restaurants_Train.csv \
-ood_te_data_path ../Dataset/SemEval14/Test/Restaurants_Test.csv \
-per_device_train_batch_size 16 -per_device_eval_batch_size 16 -num_train_epochs 4

# python ../run_model.py -mode train -model_checkpoint ../model/tk-instruct-base-def-pos -experiment_name ate_check -task ate -output_dir ../Models -inst_type 2 -id_tr_data_path ../Dataset/SemEval14/Train/Laptops_Train.csv -id_te_data_path ../Dataset/SemEval14/Test/Laptops_Test.csv -ood_tr_data_path ../Dataset/SemEval14/Train/Restaurants_Train.csv -ood_te_data_path ../Dataset/SemEval14/Test/Restaurants_Test.csv -per_device_train_batch_size 16 -per_device_eval_batch_size 16 -num_train_epochs 4