#! /bin/bash
dataname=Adrenergic
model_name=dnn
task_type=regression
pca_components=100
lr=0.001
epochs=1000
batch_size=100

cd ..
mkdir "results/${dataname}/"
mkdir "results/${dataname}/${task_type}"
mkdir "results/${dataname}/${task_type}/${model_name}"
mkdir "results/${dataname}/${task_type}/${model_name}/models"
mkdir "results/${dataname}/${task_type}/${model_name}/predictions"

for run_number in 0; do
  python regression.py --dataname ${dataname} --run_number ${run_number} --pca_components ${pca_components} \
  --lr ${lr} --hidden_dims 100 100 --epochs ${epochs} --batch_size ${batch_size}
done