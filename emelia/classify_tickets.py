from __future__ import absolute_import, division, print_function, \
    unicode_literals
import csv
import numpy as np

from sklearn.model_selection import train_test_split
from learning_model import get_compiled_model
from data_processing import convert_array_to_np_array, encode_hex_values


# training model
def classify_data(alarm_data, classification_label_data, filepath,
                  input_dimension, input_num, dropout, output):
    """
        x_data is what is fed into the NN. This will be the encoded alarm
        values for each ticket.

        y_data are the encoded label values that correspond to each ticket.
    """
    x_data = alarm_data[:2132]
    y_data = classification_label_data[:2132]

    x_train, x_test, y_train, y_test = train_test_split(x_data,
                                                        y_data,
                                                        test_size=0.20,
                                                        random_state=0)

    model = get_compiled_model(input_dimension, input_num, dropout, output)

    # Epochs pass the data n times through the system
    history = model.fit(x_train,
                        y_train,
                        epochs=40,
                        batch_size=20,
                        validation_data=(x_test, y_test) #,
                        #verbose=2
                       )
    
    # Epoch with lowest validation loss value is found from history of model

    ##print(history.history['val_loss'])
    ##print(len(history.history['val_loss']))

    # best epoch is stored variable to retrain model
    best_epoch = history.history['val_loss'].index(min(history.history['val_loss']))
    print('Optimized validation loss value: epoch -' , best_epoch)
    
    # recompile model to retrain optimized model
    optimized_model = get_compiled_model(input_dimension, input_num, dropout, output)
    
    # Model retrained with best epoch and entire training data
    history_optimized_alldata = optimized_model.fit(x_data,
                                                    y_data,
                                                    epochs= best_epoch,
                                                    batch_size=20,
                                                    verbose=2)

    # saves the model in the filepath listed
     optimized_model.save('./models/' + filepath)

    # Eval accuracy of model on test data using the test labels in the file
    test_results = optimized_model.evaluate(x_test, y_test)
    train_results = optimized_model.evaluate(x_train, y_train)

    print(history_optimized_alldata.model)
    print("Test Evaluation: " + str(test_results))
    print("Train Evaluation: " + str(train_results))


def convert_test_data(filname):
    test_alarm_values = []
    with open('TestAlarms10.csv', encoding='utf8') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        for row in reader:
            test_alarm_values.append(row[3])

    # print(test_alarm_values)
    result_list = []
    for item in test_alarm_values:
        encoded_alarm = encode_hex_values(item)
        result_list.append(encoded_alarm)

    # test_alarm_values = encode_hex_values(test_alarm_values)
    # print(test_alarm_values)
    return convert_array_to_np_array(result_list)


def report_prediction_results(predicted_arr, label_arr):
    '''
    This function will take in a data set and provide the labels for the
    maximum confidence label index value. This will report the results as an
    array of values similar to the content of the ticket data file
    '''
    result_arr = []

    length_of_array = len(predicted_arr)
    num_of_vals = len(predicted_arr[0])

    # loops through rows
    for rows in range(length_of_array):
        greatest_percent = max(predicted_arr[rows])
        position = 0

        # gets the index of largest confidence value
        for index in range(num_of_vals):
            if predicted_arr[rows][index] == greatest_percent:
                position = index

        result_arr.append(label_arr[position])

    return result_arr


def report_ticket_id(filename):
    result_arr = []

    with open(filename, encoding='utf8') as csv_file:
        csv_reader = csv.reader(csv_file)
        # Skip first row in the file
        next(csv_reader)
        for row in csv_reader:
            result_arr.append(row[0])

    return result_arr


def validation(prediction_list, result_array, input_hex, filename):
    '''
    TODO: Convert this to doc string
    result.csv contains corresp. hex code,
    the correct classification (Actual),
    predicition (result) and if the
    prediction was correct (Correctly Classified)
    '''

    with open(filename, mode='w') as csv_file:
        fieldnames = ['Hex Code', 'Actual',
                      'Result', 'Correctly Classified']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()

        # Converting np arrays to lists
        # predict_actual = np.array(predict_actual).tolist()
        predict_actual = np.array(prediction_list).tolist()
        result_array = np.array(result_array).tolist()

        length_of_array = len(input_hex)
        num_of_vals = len(predict_actual[0])
        true_counter = 0

        # loops through rows
        for rows in range(length_of_array):
            greatest_percent = max(result_array[rows])
            boolean = False
            position = 0

            # gets the index of largest confidence value
            for index in range(num_of_vals):
                if result_array[rows][index] == greatest_percent:
                    position = index

            # if the index of largest confidence value equals the index of
            # actual classification
            for index in range(num_of_vals):
                if predict_actual[rows][index] == 1 and index == position:
                    boolean = True
                    true_counter += 1

            writer.writerow({'Hex Code': input_hex[rows],
                             'Actual': predict_actual[rows],
                             'Result': result_array[rows],
                             'Correctly Classified': boolean})
        csv_file.close()

    return true_counter / length_of_array
