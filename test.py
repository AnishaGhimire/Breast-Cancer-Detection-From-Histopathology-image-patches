import seaborn
import matplotlib.pyplot as plt
import numpy as np


# Function to generate Confusion Matrix figure
def plot_confusion_matrix(data, normalize, outputfile, total=0):
    seaborn.set(color_codes=True)
    plt.figure(1, figsize=(9, 6))

    labels = ['Cancerous', 'Not Cancerous']

    if normalize:
        seaborn.set(font_scale=1.4)
        plt.title('Confusion Matrix, Normalized')
        conf_data = np.array(data)/total
        seaborn.set(font_scale=2)
        ax = seaborn.heatmap(conf_data, annot=True,
                             cmap='Blues', cbar_kws={'label': 'Scale'})

    else:
        seaborn.set(font_scale=1.4)
        plt.title('Confusion Matrix')
        conf_data = data
        seaborn.set(font_scale=2)
        ax = seaborn.heatmap(conf_data, annot=True, cmap='Blues', cbar_kws={
                             'label': 'Scale'}, fmt='g')

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set(ylabel='Predicted Label', xlabel='True Label')

    plt.savefig(outputfile, bbox_inches='tight', dpi=300)

    plt.show()
    plt.close()


# Test Method
def test(loaders, model, criterion, use_cuda):
    test_loss = 0.0
    correct = 0.0
    total = 0.0
    all_preds = []
    all_targets = []
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        output = model(data)
        loss = criterion(output, target)
        test_loss += ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        pred = output.data.max(1, keepdim=True)[1]
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)

        # for metrics calculations
        for prediction in pred.tolist():
            all_preds.append(prediction)

        for targ in target.tolist():
            all_targets.append(targ)

    # Confusion matrix calculation
    for ii in range(len(all_targets)):
        if all_targets[ii] == 1:
            if all_preds[ii] == [1]:
                true_positive += 1
            elif all_preds[ii] == [0]:
                false_negative += 1
        elif all_targets[ii] == 0:
            if all_preds[ii] == [1]:
                false_positive += 1
            elif all_preds[ii] == [0]:
                true_negative += 1

    print('Test Loss: {:.6f}'.format(test_loss))
    print('Test Accuracy: %2d%% (%2d/%2d)' %
          (100. * correct / total, correct, total))
    print('-----------------------------------\n')
    print('Evaluation Metrics:\n')
    print('Total: {}\n'.format(len(all_preds)))
    print('Confusion Matrix')
    print('TP: {}    FP: {}\nFN: {}    TN: {}'.format(
        true_positive, false_positive, false_negative, true_negative))

    accuracy = (true_positive + true_negative)/(true_positive +
                                                false_positive + false_negative + true_negative)
    precision = true_positive/(true_positive + false_positive)
    recall = true_positive/(true_positive + false_negative)
    f1_score = 2 * ((precision * recall) / (precision + recall))

    print('\nAccuracy: %.2f%%' % (accuracy * 100.))
    print('Precision: %.2f%%' % (precision * 100.))
    print('Recall: %.2f%%' % (recall * 100.))
    print('F1 Score: %.2f%%' % (f1_score * 100.))

    confusion_data = [[true_positive, false_positive],
                      [false_negative, true_negative]]

    plot_confusion_matrix(confusion_data, False, 'confusion_matrix.png')
    plot_confusion_matrix(confusion_data, True,
                          'confusion_matrix_normalize.png', len(all_targets))