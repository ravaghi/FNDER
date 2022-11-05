import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def process_data():
    dataset_path = os.path.join(BASE_DIR, "data")

    processed_train_path = os.path.join(dataset_path, 'train.csv')
    processed_val_path = os.path.join(dataset_path, 'val.csv')
    processed_test_path = os.path.join(dataset_path, 'test.csv')

    if os.path.exists(processed_train_path) and \
            os.path.exists(processed_val_path) and \
            os.path.exists(processed_test_path):
        print('Data already exists')
    else:
        print('Processing data')
        unprocessed_fake_data_path = os.path.join(dataset_path, 'Fake.csv')
        unprocessed_real_data_path = os.path.join(dataset_path, 'True.csv')

        fake = pd.read_csv(unprocessed_fake_data_path)
        real = pd.read_csv(unprocessed_real_data_path)

        fake['label'] = 0
        real['label'] = 1

        # Combine and shuffle
        combined = pd.concat([fake, real], axis=0)
        combined = combined.sample(frac=1).reset_index(drop=True)

        # Split dataset into train, val and test
        train = combined[:int(0.8 * len(combined))]
        val = combined[int(0.8 * len(combined)):int(0.9 * len(combined))]
        test = combined[int(0.9 * len(combined)):]

        # Save datasets
        train.to_csv(processed_train_path, index=False)
        val.to_csv(processed_val_path, index=False)
        test.to_csv(processed_test_path, index=False)

        # Delete old files
        os.remove(os.path.join(dataset_path, 'Fake.csv'))
        os.remove(os.path.join(dataset_path, 'True.csv'))


if __name__ == '__main__':
    process_data()
