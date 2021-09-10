# Dictionary storing network parameters.
params = {
    'batch_size': 128,  # Batch size.
    'num_epochs': 2,  # Number of epochs to train for.
    'learning_rate': 2e-4,  # Learning rate.
    'beta1': 0.5,
    'beta2': 0.999,
    'save_epoch': 1,  # After how many epochs to save checkpoints and generate test output.
    'dataset': 'MNIST'}  # Dataset to use. Choose from {MNIST, SVHN, CelebA, FashionMNIST}. CASE MUST MATCH EXACTLY!!!!!
