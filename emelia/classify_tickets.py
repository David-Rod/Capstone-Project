from __future__ import absolute_import, division, print_function, \
    unicode_literals

import csv
import pandas as pd
import tensorflow as tf

import time
#123


def main():
    start_time = time.time()
    df = []
    line_count = 0
    file = 'ticketdata.csv'

    with open(file, encoding='utf8') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            end = len(row)
            start = end - 6
            df.append(row[start:end])
            line_count += 1

    df = pd.DataFrame(df, columns=['EventCause', 'DetectionMethod',
                                   'RestoreMethod', 'FixClassification',
                                   'Subsystem', 'Relevance'])

    df['EventCause'] = pd.Categorical(df['EventCause'])
    df['DetectionMethod'] = pd.Categorical(df['DetectionMethod'])
    df['RestoreMethod'] = pd.Categorical(df['RestoreMethod'])
    df['FixClassification'] = pd.Categorical(df['FixClassification'])
    df['Subsystem'] = pd.Categorical(df['Subsystem'])
    df['Relevance'] = pd.Categorical(df['Relevance'])

    df['EventCause'] = df.EventCause.cat.codes
    df['DetectionMethod'] = df.DetectionMethod.cat.codes
    df['RestoreMethod'] = df.RestoreMethod.cat.codes
    df['FixClassification'] = df.FixClassification.cat.codes
    df['Subsystem'] = df.Subsystem.cat.codes
    df['Relevance'] = df.Relevance.cat.codes

    target = df.pop('Relevance')

    dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))

    train_dataset = dataset.shuffle(len(df)).batch(1)

    def get_compiled_model():
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    model = get_compiled_model()
    model.fit(train_dataset, epochs=100, steps_per_epoch=20)

    end_time = time.time()
    runtime = round(end_time - start_time, 2)
    avg_time_per_ticket = round(runtime / line_count, 5)

    print("\n" * 2)
    print("############################ METRICS ############################")
    print("File: " + file)
    print("Runtime: " + str(runtime) + "s")
    print("Tickets processed: " + str(line_count))
    print("AVG Time per Ticket: " + str(avg_time_per_ticket) + "s")


if __name__ == "__main__":
    main()
