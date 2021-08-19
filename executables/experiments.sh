# semi_inductive/QBLP_semi_wd20_33_pairs_4.json
ilp run qblp --inductive-setting semi --dataset-name wd2033 --max-num-qualifier-pairs 4 --use-wandb True --num-epochs 352 --training-approach lcwa
# semi_inductive/QBLP_semi_wd20_25_pairs_6.json
ilp run qblp --inductive-setting semi --dataset-name wd2025 --max-num-qualifier-pairs 6 --use-wandb True --batch-size 256.0 --eval-batch-size 30.0 --label-smoothing 0.1 --learning-rate 0.0005353925050696774 --num-epochs 79 --embedding-dim 256.0 --transformer-hidden-dimension 768.0 --transformer-num-heads 4.0 --transformer-num-layers 4.0 --affine-transformation False --training-approach lcwa
# semi_inductive/QBLP_semi_wd20_33_pairs_2.json
ilp run qblp --inductive-setting semi --dataset-name wd2033 --max-num-qualifier-pairs 2 --use-wandb True --num-epochs 452 --training-approach lcwa
# semi_inductive/QBLP_semi_wd20_33_pairs_0.json
ilp run qblp --inductive-setting semi --dataset-name wd2033 --max-num-qualifier-pairs 0 --use-wandb True --batch-size 256.0 --eval-batch-size 30.0 --label-smoothing 0.15 --learning-rate 0.00035212939846635487 --num-epochs 222 --embedding-dim 128.0 --transformer-hidden-dimension 1024.0 --transformer-num-heads 2.0 --transformer-num-layers 2.0 --affine-transformation True --training-approach lcwa
# semi_inductive/QBLP_semi_wd20_25_pairs_2.json
ilp run qblp --inductive-setting semi --dataset-name wd2025 --max-num-qualifier-pairs 2 --use-wandb True --batch-size 192.0 --eval-batch-size 30.0 --label-smoothing 0.1 --learning-rate 0.0011047568126663207 --num-epochs 104 --embedding-dim 128.0 --transformer-hidden-dimension 512.0 --transformer-num-heads 2.0 --transformer-num-layers 2.0 --affine-transformation True --training-approach lcwa
# semi_inductive/QBLP_semi_wd20_25_pairs_4.json
ilp run qblp --inductive-setting semi --dataset-name wd2025 --max-num-qualifier-pairs 4 --use-wandb True --batch-size 128.0 --eval-batch-size 30.0 --label-smoothing 0.1 --learning-rate 0.00019679009980689841 --num-epochs 181 --embedding-dim 200.0 --transformer-hidden-dimension 768.0 --transformer-num-heads 4.0 --transformer-num-layers 4.0 --affine-transformation False --training-approach lcwa
# semi_inductive/QBLP_semi_wd20_25_pairs_0.json
ilp run qblp --inductive-setting semi --dataset-name wd2025 --max-num-qualifier-pairs 0 --use-wandb True --batch-size 512.0 --eval-batch-size 1000.0 --label-smoothing 0.15 --learning-rate 0.0001 --num-epochs 295 --embedding-dim 256.0 --transformer-hidden-dimension 1024.0 --transformer-num-heads 4.0 --transformer-num-layers 2.0 --affine-transformation True --training-approach lcwa
# semi_inductive/BLP_semi_wd20_25_pairs_0.json
ilp run blp --inductive-setting semi --dataset-name wd2025 --max-num-qualifier-pairs 0 --use-wandb True --batch-size 256.0 --eval-batch-size 30.0 --label-smoothing 0.1 --learning-rate 0.007666318259807473 --num-epochs 186 --embedding-dim 128.0 --training-approach lcwa
# semi_inductive/BLP_semi_wd20_33_pairs_0.json
ilp run blp --inductive-setting semi --dataset-name wd2033 --max-num-qualifier-pairs 0 --use-wandb True --batch-size 192.0 --eval-batch-size 30.0 --label-smoothing 0.15 --learning-rate 0.002084295761242309 --num-epochs 386 --embedding-dim 256.0 --training-approach lcwa
# semi_inductive/QBLP_semi_wd20_33_pairs_6.json
ilp run qblp --inductive-setting semi --dataset-name wd2033 --max-num-qualifier-pairs 6 --use-wandb True --batch-size 128.0 --eval-batch-size 30.0 --label-smoothing 0.1 --learning-rate 0.0006489254733778205 --num-epochs 80 --embedding-dim 128.0 --transformer-hidden-dimension 512.0 --transformer-num-heads 4.0 --transformer-num-layers 4.0 --affine-transformation True --training-approach lcwa
# fully_inductive/StarE_wd20_66_v2_pairs_2.json
ilp run stare --inductive-setting full --dataset-name wd2066 --max-num-qualifier-pairs 2 --use-wandb True --dataset-version v2 --batch-size 256 --eval-batch-size 10.0 --label-smoothing 0.1 --learning-rate 0.0001 --num-epochs 448 --embedding-dim 224.0 --transformer-hidden-dimension 960.0 --transformer-num-heads 2.0 --transformer-num-layers 2.0 --attention-dropout 0.4 --attention-slope 0.1 --gcn-dropout 0.4 --hidden-dropout 0.1 --attention-num-heads 2.0 --num-layers 2.0 --qualifier-aggregation attn --triple-qual-weight 0.8 --use-bias False
# fully_inductive/QBLP_wd20_100_v1_pairs_4.json
ilp run qblp --inductive-setting full --dataset-name wd20100 --max-num-qualifier-pairs 4 --use-wandb True --dataset-version v1 --batch-size 256 --eval-batch-size 30.0 --label-smoothing 0.1 --learning-rate 0.0005755058904477412 --num-epochs 280 --embedding-dim 128.0 --transformer-hidden-dimension 1024.0 --transformer-num-heads 4.0 --transformer-num-layers 4.0 --training-approach lcwa
# fully_inductive/QBLP_wd20_66_v2_pairs_6.json
ilp run qblp --inductive-setting full --dataset-name wd2066 --max-num-qualifier-pairs 6 --use-wandb True --dataset-version v2 --batch-size 192 --eval-batch-size 30.0 --label-smoothing 0.1 --learning-rate 0.002384251384729216 --num-epochs 39 --embedding-dim 256.0 --transformer-hidden-dimension 768.0 --transformer-num-heads 2.0 --transformer-num-layers 2.0 --training-approach lcwa
# fully_inductive/QBLP_wd20_66_v1_pairs_2.json
ilp run qblp --inductive-setting full --dataset-name wd2066 --max-num-qualifier-pairs 2 --use-wandb True --dataset-version v1 --batch-size 256 --eval-batch-size 30.0 --label-smoothing 0.15 --learning-rate 0.0009386787191242988 --num-epochs 574 --embedding-dim 128.0 --transformer-hidden-dimension 768.0 --transformer-num-heads 2.0 --transformer-num-layers 2.0 --training-approach lcwa
# fully_inductive/StarE_wd20_66_v1_pairs_4.json
# Skipping /home/wiss/berrendorf/inductive_pykeen/best_configs/fully_inductive/StarE_wd20_66_v1_pairs_4.json
# fully_inductive/QBLP_wd20_66_v1_pairs_4.json
ilp run qblp --inductive-setting full --dataset-name wd2066 --max-num-qualifier-pairs 4 --use-wandb True --dataset-version v1 --batch-size 960 --eval-batch-size 10.0 --label-smoothing 0.15 --learning-rate 0.0014451192932826325 --num-epochs 401 --embedding-dim 128.0 --transformer-hidden-dimension 768.0 --transformer-num-heads 4.0 --transformer-num-layers 3.0 --training-approach lcwa
# fully_inductive/QBLP_wd20_100_v2_pairs_6.json
ilp run qblp --inductive-setting full --dataset-name wd20100 --max-num-qualifier-pairs 6 --use-wandb True --dataset-version v2 --batch-size 512 --eval-batch-size 30.0 --label-smoothing 0.1 --learning-rate 0.0010499620497397613 --num-epochs 484 --embedding-dim 256.0 --transformer-hidden-dimension 768.0 --transformer-num-heads 4.0 --transformer-num-layers 2.0 --training-approach lcwa
# fully_inductive/StarE_wd20_100_v2_pairs_4.json
ilp run stare --inductive-setting full --dataset-name wd20100 --max-num-qualifier-pairs 4 --use-wandb True --dataset-version v2 --batch-size 512 --eval-batch-size 10.0 --label-smoothing 0.1 --learning-rate 0.000793745397531692 --num-epochs 827 --embedding-dim 160.0 --transformer-hidden-dimension 896.0 --transformer-num-heads 2.0 --transformer-num-layers 4.0 --attention-dropout 0.4 --attention-slope 0.4 --gcn-dropout 0.5 --hidden-dropout 0.4 --attention-num-heads 2.0 --num-layers 2.0 --qualifier-aggregation attn --triple-qual-weight 0.8 --use-bias False
# fully_inductive/QBLP_wd20_100_v1_pairs_0.json
ilp run qblp --inductive-setting full --dataset-name wd20100 --max-num-qualifier-pairs 0 --use-wandb True --dataset-version v1 --batch-size 896 --eval-batch-size 10.0 --label-smoothing 0.1 --learning-rate 0.00045484715659366737 --num-epochs 696 --embedding-dim 160.0 --transformer-hidden-dimension 640.0 --transformer-num-heads 4.0 --transformer-num-layers 4.0 --training-approach lcwa
# fully_inductive/StarE_wd20_100_v1_pairs_6.json
ilp run stare --inductive-setting full --dataset-name wd20100 --max-num-qualifier-pairs 6 --use-wandb True --dataset-version v1 --batch-size 704 --eval-batch-size 10.0 --label-smoothing 0.1 --learning-rate 0.00129082610513116 --num-epochs 855 --embedding-dim 160.0 --transformer-hidden-dimension 704.0 --transformer-num-heads 2.0 --transformer-num-layers 2.0 --attention-dropout 0.1 --attention-slope 0.2 --gcn-dropout 0.1 --hidden-dropout 0.4 --attention-num-heads 4.0 --num-layers 3.0 --qualifier-aggregation sum --triple-qual-weight 0.8 --use-bias False
# fully_inductive/BLP_wd20_100_v2_pairs_0.json
ilp run blp --inductive-setting full --dataset-name wd20100 --max-num-qualifier-pairs 0 --use-wandb True --dataset-version v2 --batch-size 320 --create-inverse-triples 0 --eval-batch-size 10.0 --learning-rate 0.0018551842330604567 --num-epochs 631 --embedding-dim 128.0 --training-approach lcwa
# fully_inductive/QBLP_wd20_66_v2_pairs_0.json
ilp run qblp --inductive-setting full --dataset-name wd2066 --max-num-qualifier-pairs 0 --use-wandb True --dataset-version v2 --batch-size 192 --eval-batch-size 30.0 --label-smoothing 0.1 --learning-rate 0.004013989204532026 --num-epochs 49 --embedding-dim 256.0 --transformer-hidden-dimension 512.0 --transformer-num-heads 2.0 --transformer-num-layers 2.0 --training-approach lcwa
# fully_inductive/StarE_wd20_66_v2_pairs_6.json
ilp run stare --inductive-setting full --dataset-name wd2066 --max-num-qualifier-pairs 6 --use-wandb True --dataset-version v2 --batch-size 1024 --eval-batch-size 10.0 --label-smoothing 0.1 --learning-rate 0.0010469104680997428 --num-epochs 997 --embedding-dim 256.0 --transformer-hidden-dimension 704.0 --transformer-num-heads 4.0 --transformer-num-layers 4.0 --attention-dropout 0.5 --attention-slope 0.1 --gcn-dropout 0.3 --hidden-dropout 0.4 --attention-num-heads 4.0 --num-layers 2.0 --qualifier-aggregation sum --triple-qual-weight 0.8 --use-bias False
# fully_inductive/StarE_wd20_66_v2_pairs_4.json
ilp run stare --inductive-setting full --dataset-name wd2066 --max-num-qualifier-pairs 4 --use-wandb True --dataset-version v2 --batch-size 896 --eval-batch-size 10.0 --label-smoothing 0.15 --learning-rate 0.0015193269883566618 --num-epochs 454 --embedding-dim 224.0 --transformer-hidden-dimension 1024.0 --transformer-num-heads 4.0 --transformer-num-layers 3.0 --attention-dropout 0.1 --attention-slope 0.3 --gcn-dropout 0.4 --hidden-dropout 0.1 --attention-num-heads 2.0 --num-layers 2.0 --qualifier-aggregation sum --triple-qual-weight 0.8 --use-bias False
# fully_inductive/StarE_wd20_100_v2_pairs_0.json
ilp run stare --inductive-setting full --dataset-name wd20100 --max-num-qualifier-pairs 0 --use-wandb True --dataset-version v2 --batch-size 960 --eval-batch-size 10.0 --label-smoothing 0.15 --learning-rate 0.0005455473041665988 --num-epochs 320 --embedding-dim 192.0 --transformer-hidden-dimension 512.0 --transformer-num-heads 4.0 --transformer-num-layers 3.0 --attention-dropout 0.3 --attention-slope 0.4 --gcn-dropout 0.4 --hidden-dropout 0.3 --attention-num-heads 4.0 --num-layers 2.0 --qualifier-aggregation sum --triple-qual-weight 0.8 --use-bias False
# fully_inductive/BLP_wd20_66_v2_pairs_0.json
ilp run blp --inductive-setting full --dataset-name wd2066 --max-num-qualifier-pairs 0 --use-wandb True --dataset-version v2 --batch-size 960 --eval-batch-size 10.0 --learning-rate 0.0093099562683248 --num-epochs 100 --embedding-dim 160.0 --training-approach lcwa
# fully_inductive/StarE_wd20_100_v1_pairs_2.json
ilp run stare --inductive-setting full --dataset-name wd20100 --max-num-qualifier-pairs 2 --use-wandb True --dataset-version v1 --batch-size 512 --eval-batch-size 10.0 --label-smoothing 0.15 --learning-rate 0.0018682677952032226 --num-epochs 643 --embedding-dim 160.0 --transformer-hidden-dimension 640.0 --transformer-num-heads 4.0 --transformer-num-layers 2.0 --attention-dropout 0.1 --attention-slope 0.4 --gcn-dropout 0.4 --hidden-dropout 0.3 --attention-num-heads 2.0 --num-layers 2.0 --qualifier-aggregation attn --triple-qual-weight 0.8 --use-bias False
# fully_inductive/StarE_wd20_100_v2_pairs_6.json
ilp run stare --inductive-setting full --dataset-name wd20100 --max-num-qualifier-pairs 6 --use-wandb True --dataset-version v2 --batch-size 960 --eval-batch-size 10.0 --label-smoothing 0.1 --learning-rate 0.004025459209365596 --num-epochs 580 --embedding-dim 128.0 --transformer-hidden-dimension 768.0 --transformer-num-heads 4.0 --transformer-num-layers 3.0 --attention-dropout 0.5 --attention-slope 0.1 --gcn-dropout 0.3 --hidden-dropout 0.1 --attention-num-heads 4.0 --num-layers 2.0 --qualifier-aggregation attn --triple-qual-weight 0.8 --use-bias False
# fully_inductive/StarE_wd20_66_v1_pairs_2.json
ilp run stare --inductive-setting full --dataset-name wd2066 --max-num-qualifier-pairs 2 --use-wandb True --dataset-version v1 --batch-size 960 --eval-batch-size 10.0 --label-smoothing 0.1 --learning-rate 0.0016242529646428693 --num-epochs 635 --embedding-dim 224.0 --transformer-hidden-dimension 704.0 --transformer-num-heads 4.0 --transformer-num-layers 3.0 --attention-dropout 0.3 --attention-slope 0.4 --gcn-dropout 0.3 --hidden-dropout 0.2 --attention-num-heads 2.0 --num-layers 3.0 --qualifier-aggregation attn --triple-qual-weight 0.8 --use-bias False
# fully_inductive/QBLP_wd20_66_v1_pairs_0.json
ilp run qblp --inductive-setting full --dataset-name wd2066 --max-num-qualifier-pairs 0 --use-wandb True --dataset-version v1 --batch-size 192 --eval-batch-size 30.0 --label-smoothing 0.15 --learning-rate 0.0003463091463067287 --num-epochs 207 --embedding-dim 200.0 --transformer-hidden-dimension 1024.0 --transformer-num-heads 4.0 --transformer-num-layers 3.0 --training-approach lcwa
# fully_inductive/StarE_wd20_100_v1_pairs_0.json
ilp run stare --inductive-setting full --dataset-name wd20100 --max-num-qualifier-pairs 0 --use-wandb True --dataset-version v1 --batch-size 192 --eval-batch-size 10.0 --label-smoothing 0.1 --learning-rate 0.0004935719069674829 --num-epochs 664 --embedding-dim 224.0 --transformer-hidden-dimension 960.0 --transformer-num-heads 2.0 --transformer-num-layers 4.0 --attention-dropout 0.2 --attention-slope 0.2 --gcn-dropout 0.4 --hidden-dropout 0.2 --attention-num-heads 2.0 --num-layers 3.0 --qualifier-aggregation attn --triple-qual-weight 0.8 --use-bias False
# fully_inductive/QBLP_wd20_66_v2_pairs_4.json
ilp run qblp --inductive-setting full --dataset-name wd2066 --max-num-qualifier-pairs 4 --use-wandb True --dataset-version v2 --batch-size 128 --eval-batch-size 30.0 --label-smoothing 0.1 --learning-rate 0.0028205180727749178 --num-epochs 28 --embedding-dim 128.0 --transformer-hidden-dimension 1024.0 --transformer-num-heads 2.0 --transformer-num-layers 2.0 --training-approach lcwa
# fully_inductive/StarE_wd20_100_v2_pairs_2.json
ilp run stare --inductive-setting full --dataset-name wd20100 --max-num-qualifier-pairs 2 --use-wandb True --dataset-version v2 --batch-size 896 --eval-batch-size 10.0 --label-smoothing 0.1 --learning-rate 0.002784250457508696 --num-epochs 753 --embedding-dim 192.0 --transformer-hidden-dimension 640.0 --transformer-num-heads 2.0 --transformer-num-layers 2.0 --attention-dropout 0.4 --attention-slope 0.1 --gcn-dropout 0.4 --hidden-dropout 0.1 --attention-num-heads 2.0 --num-layers 2.0 --qualifier-aggregation attn --triple-qual-weight 0.8 --use-bias False
# fully_inductive/StarE_wd20_66_v1_pairs_0.json
ilp run stare --inductive-setting full --dataset-name wd2066 --max-num-qualifier-pairs 0 --use-wandb True --dataset-version v1 --batch-size 192 --eval-batch-size 10.0 --label-smoothing 0.1 --learning-rate 0.0006153579367004525 --num-epochs 431 --embedding-dim 192.0 --transformer-hidden-dimension 704.0 --transformer-num-heads 4.0 --transformer-num-layers 2.0 --attention-dropout 0.4 --attention-slope 0.4 --gcn-dropout 0.3 --hidden-dropout 0.3 --attention-num-heads 2.0 --num-layers 3.0 --qualifier-aggregation sum --triple-qual-weight 0.8 --use-bias False
# fully_inductive/StarE_wd20_66_v2_pairs_0.json
ilp run stare --inductive-setting full --dataset-name wd2066 --max-num-qualifier-pairs 0 --use-wandb True --dataset-version v2 --eval-batch-size 10.0 --label-smoothing 0.15 --learning-rate 0.0013248043920739575 --num-epochs 339 --embedding-dim 224.0 --transformer-hidden-dimension 576.0 --transformer-num-heads 4.0 --transformer-num-layers 4.0 --gcn-dropout 0.3 --hidden-dropout 0.3 --attention-num-heads 4.0 --num-layers 2.0 --qualifier-aggregation sum --triple-qual-weight 0.8 --use-bias False
# fully_inductive/BLP_wd20_100_v1_pairs_0.json
ilp run blp --inductive-setting full --dataset-name wd20100 --max-num-qualifier-pairs 0 --use-wandb True --dataset-version v1 --batch-size 384 --eval-batch-size 10.0 --learning-rate 0.005035765756205903 --num-epochs 238 --embedding-dim 192.0 --training-approach lcwa
# fully_inductive/StarE_wd20_100_v1_pairs_4.json
ilp run stare --inductive-setting full --dataset-name wd20100 --max-num-qualifier-pairs 4 --use-wandb True --dataset-version v1 --batch-size 960 --eval-batch-size 10.0 --label-smoothing 0.1 --learning-rate 0.0015906597625936054 --num-epochs 768 --embedding-dim 224.0 --transformer-hidden-dimension 704.0 --transformer-num-heads 2.0 --transformer-num-layers 3.0 --attention-dropout 0.1 --attention-slope 0.3 --gcn-dropout 0.5 --hidden-dropout 0.4 --attention-num-heads 4.0 --num-layers 2.0 --qualifier-aggregation attn --triple-qual-weight 0.8 --use-bias False
# fully_inductive/QBLP_wd20_100_v1_pairs_6.json
ilp run qblp --inductive-setting full --dataset-name wd20100 --max-num-qualifier-pairs 6 --use-wandb True --dataset-version v1 --batch-size 1024 --eval-batch-size 10.0 --label-smoothing 0.1 --learning-rate 0.00033807732007455654 --num-epochs 552 --embedding-dim 128.0 --transformer-hidden-dimension 640.0 --transformer-num-heads 2.0 --transformer-num-layers 2.0 --training-approach lcwa
# fully_inductive/QBLP_wd20_100_v2_pairs_0.json
ilp run qblp --inductive-setting full --dataset-name wd20100 --max-num-qualifier-pairs 0 --use-wandb True --dataset-version v2 --batch-size 192 --eval-batch-size 30.0 --label-smoothing 0.1 --learning-rate 0.002070085196579545 --num-epochs 61 --embedding-dim 200.0 --transformer-hidden-dimension 512.0 --transformer-num-heads 2.0 --transformer-num-layers 4.0 --training-approach lcwa
# fully_inductive/QBLP_wd20_100_v1_pairs_2.json
ilp run qblp --inductive-setting full --dataset-name wd20100 --max-num-qualifier-pairs 2 --use-wandb True --dataset-version v1 --batch-size 512 --eval-batch-size 30.0 --label-smoothing 0.1 --learning-rate 0.002391592818904277 --num-epochs 247 --embedding-dim 128.0 --transformer-hidden-dimension 1024.0 --transformer-num-heads 4.0 --transformer-num-layers 3.0 --training-approach lcwa
# fully_inductive/QBLP_wd20_100_v2_pairs_2.json
ilp run qblp --inductive-setting full --dataset-name wd20100 --max-num-qualifier-pairs 2 --use-wandb True --dataset-version v2 --batch-size 192 --eval-batch-size 10.0 --label-smoothing 0.1 --learning-rate 0.003488188580850907 --num-epochs 90 --embedding-dim 160.0 --transformer-hidden-dimension 512.0 --transformer-num-heads 4.0 --transformer-num-layers 2.0 --training-approach lcwa
# fully_inductive/BLP_wd20_66_v1_pairs_0.json
ilp run blp --inductive-setting full --dataset-name wd2066 --max-num-qualifier-pairs 0 --use-wandb True --dataset-version v1 --batch-size 128 --eval-batch-size 10.0 --learning-rate 0.0024966987918393995 --num-epochs 234 --embedding-dim 224.0 --training-approach lcwa
# fully_inductive/QBLP_wd20_66_v1_pairs_6.json
ilp run qblp --inductive-setting full --dataset-name wd2066 --max-num-qualifier-pairs 6 --use-wandb True --dataset-version v1 --batch-size 256 --eval-batch-size 30.0 --label-smoothing 0.15 --learning-rate 0.0017571110732997439 --num-epochs 70 --embedding-dim 200.0 --transformer-hidden-dimension 1024.0 --transformer-num-heads 2.0 --transformer-num-layers 2.0 --training-approach lcwa
# fully_inductive/StarE_wd20_66_v1_pairs_6.json
# Skipping /home/wiss/berrendorf/inductive_pykeen/best_configs/fully_inductive/StarE_wd20_66_v1_pairs_6.json
# fully_inductive/QBLP_wd20_100_v2_pairs_4.json
ilp run qblp --inductive-setting full --dataset-name wd20100 --max-num-qualifier-pairs 4 --use-wandb True --dataset-version v2 --batch-size 128 --eval-batch-size 10.0 --label-smoothing 0.1 --learning-rate 0.0020618679875488125 --num-epochs 58 --embedding-dim 160.0 --transformer-hidden-dimension 768.0 --transformer-num-heads 2.0 --transformer-num-layers 4.0 --training-approach lcwa
# fully_inductive/QBLP_wd20_66_v2_pairs_2.json
ilp run qblp --inductive-setting full --dataset-name wd2066 --max-num-qualifier-pairs 2 --use-wandb True --dataset-version v2 --batch-size 128 --eval-batch-size 30.0 --label-smoothing 0.1 --learning-rate 0.0011416835908356704 --num-epochs 101 --embedding-dim 200.0 --transformer-hidden-dimension 1024.0 --transformer-num-heads 2.0 --transformer-num-layers 3.0 --training-approach lcwa
