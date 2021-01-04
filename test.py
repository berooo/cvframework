
TOP_K = 3

def to_hex(image_id):
    return '{0:0{1}x}'.format(image_id, 16)

def get_prediction_map(test_ids, train_ids_labels_and_scores):
    """Makes dict from test ids and ranked training ids, labels, scores."""

    prediction_map = dict()

    for test_index, test_id in enumerate(test_ids):
        hex_test_id = to_hex(test_id)

        aggregate_scores = {}
        for _, label, score in train_ids_labels_and_scores[test_index][:TOP_K]:
            if label not in aggregate_scores:
                aggregate_scores[label] = 0
            aggregate_scores[label] += score

        label, score = max(aggregate_scores.items(), key=operator.itemgetter(1))
        prediction_map[hex_test_id] = {'score': score, 'class': label}

    return prediction_map