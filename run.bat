<<<<<<< Updated upstream
<<<<<<< Updated upstream
python run_image_classification.py --dataset_name imagefolder --output_dir ./dd/ --remove_unused_columns False --do_train --do_eval --learning_rate 2e-5 --num_train_epochs 2 --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --logging_strategy steps --logging_steps 10 --evaluation_strategy epoch --save_strategy epoch --load_best_model_at_end True --save_total_limit 3 --seed 1337 --ignore_mismatched_sizes --fp16
=======
python run_image_classification.py --dataset_name imagefolder --output_dir ./dd/ --remove_unused_columns False --do_train --do_eval --learning_rate 2e-5 --num_train_epochs 2 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --logging_strategy steps --logging_steps 10 --evaluation_strategy epoch --save_strategy epoch --load_best_model_at_end True --save_total_limit 3 --seed 1337 --ignore_mismatched_sizes --fp16
>>>>>>> Stashed changes
=======
python run_image_classification.py --dataset_name imagefolder --output_dir ./dd/ --remove_unused_columns False --do_train --do_eval --learning_rate 2e-5 --num_train_epochs 2 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --logging_strategy steps --logging_steps 10 --evaluation_strategy epoch --save_strategy epoch --load_best_model_at_end True --save_total_limit 3 --seed 1337 --ignore_mismatched_sizes --fp16
>>>>>>> Stashed changes
