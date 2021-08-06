from ray import tune

rt_hp_config = {
        'learning_rate': tune.grid_search([0.005, 0.01, 0.015, 0.001, 0.0001, 0.00001]),
        'epsilon': tune.grid_search([0.005, 0.01, 0.015, 0.1, 0.2]),
        'epsilon_decay': tune.grid_search([0.85, 0.95, 0.99, 0.995]),
        'epsilon_min': tune.grid_search([0.1]),
        'gamma': tune.grid_search([0.95, 0.99, 0.9999]),
        'num_hidden_nodes': tune.grid_search([20, 128, 1024, 8192]),
        'num_hidden_layers': tune.grid_search([2, 3, 4, 5, 6, 7, 8]),
        'batch_size': tune.grid_search([32]),
        # MLFlow config has to be passed
        "mlflow": {
            "experiment_name": "presslight_3",
            "tracking_uri": "http://10.195.1.7:5000"
        }
    }

hp_config = {
        'learning_rate': 0.015,
        'epsilon': 0.1,
        'epsilon_decay': 0.995,
        'epsilon_min': 0.1,
        'gamma': 0.95,
        'num_hidden_nodes': 20,
        'num_hidden_layers': 2,
        'batch_size': 32,
        # MLFlow config has to be passed
        "mlflow": {
            "experiment_name": "presslight_3",
            "tracking_uri": "http://10.195.1.7:5000"
        }
    }