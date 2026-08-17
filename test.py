import random

# import wandb

# def test_wandb():
#         # Start a new wandb run to track this script.
#         run = wandb.init(
#         # Set the wandb entity where your project will be logged (generally your team name).
#         entity="scholarccc184-shenzhen-university",
#         # Set the wandb project where this run will be logged.
#         project="omnirad",
#         # Track hyperparameters and run metadata.
#         config={
#                 "learning_rate": 0.02,
#                 "architecture": "CNN",
#                 "dataset": "CIFAR-100",
#                 "epochs": 10,
#         },
#         )

#         # Simulate training.
#         epochs = 10
#         offset = random.random() / 5
#         for epoch in range(2, epochs):
#         acc = 1 - 2**-epoch - random.random() / epoch - offset
#         loss = 2**-epoch + random.random() / epoch + offset

#         # Log metrics to wandb.
#         run.log({"acc": acc, "loss": loss})

#         # Finish the run and upload any remaining data.
#         run.finish()

"""
 CUDA_VISIBLE_DEVICES=1   torchrun --master-port 8888 --nproc_per_node 1 \
        eval_scripts/model_evaluation.py \
        --cfg-path eval_configs/omnirad_evaluation.yaml \
        --dataset indiana_cxr,radvqa,slake_vqa,rsna,SLAKE,group_breast_us,kvasir

CUDA_VISIBLE_DEVICES=1   torchrun --master-port 8888 --nproc_per_node 1 \
        eval_scripts/model_evaluation.py \
        --cfg-path eval_configs/omnirad_evaluation.yaml \
        --dataset group_breast_us,kvasir

CUDA_VISIBLE_DEVICES=1   torchrun --master-port 8888 --nproc_per_node 1 \
        eval_scripts/model_evaluation.py \
        --cfg-path eval_configs/omnirad_evaluation.yaml \
        --dataset slake_vqa

CUDA_VISIBLE_DEVICES=1   torchrun --master-port 8888 --nproc_per_node 1 \
        eval_scripts/model_evaluation.py \
        --cfg-path eval_configs/minigptv2_evaluation.yaml \
        --dataset indiana_cxr,group_breast_us,kvasir
"""

if __name__ == "__main__":
    main()
