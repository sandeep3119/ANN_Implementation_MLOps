# ANN_Implementation_MLOps
ANN implementation with MLOps project structure

### Project structure

```buildoutcfg
 -Base project
        -artifacts
               -models
               -plots
               -checkpoints
        -logs
            -general_logs
            -tensboard_logs
        -src
            -utils
               -(all utility files, like data management,model creation, etc)
            - __init__.py ( as src will be a package)
            - training.py (main training file- entry point)
            
        -config.yaml(contains all project and training related configuartions)
        -setup.py(project specific details)
```            

### Use Tensorboard
    tensorboard --logdir=log_dir/tensorboard_logs