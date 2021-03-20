import tensorflow as tf
import os

from utils import read_images_from_data_folder, create_model, prepare_data_to_model, fit_model
from sklearn.model_selection import train_test_split


def check_reproducibility(n_repeat, seed):

    # Prepare data to ingest model
    data, target = read_images_from_data_folder(data_path='./data/')
    data, target = prepare_data_to_model(data, target)
    train_data, test_data, train_target, test_target = train_test_split(data,
                                                                        target,
                                                                        test_size=0.15,
                                                                        random_state=seed)

    # Train model multiple times
    results = []
    for i in range(n_repeat):

        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        tf.random.set_seed(seed)
        model = create_model(input_shape=data.shape[1:])
        model, history = fit_model(train_data, train_target, model, n_epochs=20, model_checkpoint=False, verbose=1)
        metrics_result = model.evaluate(test_data, test_target)

        results.append(metrics_result)

    # Check if we get the same results every time
    all_equal = all(elem == results[0] for elem in results)

    if all_equal:
        print("PASSED TEST : reproducibility achieved")
        print(f"Loss and accuracy are : {results[0]}")
    else:
        print("FAILED TEST : experiments are not reproducible")


def main():
    check_reproducibility(5, 1)


if __name__ == "__main__":
    main()
