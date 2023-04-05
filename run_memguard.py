import argparse
from data_utils import LocationDataModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from arch import LocationModel

'''
NB: Only for Location Dataset as of now!
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run an experiment.')
    parser.add_argument('--dataset', type=str, default="Location", help='dataset')
    parser.add_argument('--expt', type=str, default="", help='experiment name')
    parser.add_argument('--def_epoch', type=int, default=400, help="amount defense trained for")
    parser.add_argument('--fold', type=int, default=1, help="data fold")
    parser.add_argument('--loss_ratio', type=float, default=0.3, help="amount to compromise memguard")

    args = parser.parse_args()

    if args.dataset != 'Location':
        print("unable to handle yet")
        exit()
        
    queryset = LocationDataModule(mode='query', k=0, calset=args.dataset , use_own=True, fold=args.fold)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### 1. Parse the query data
    print("Parsing query data...")
    x_eval = queryset.x_eval
    l_eval = queryset.l_eval
    y_eval = queryset.y_eval
    input_shape = queryset.input_shape
    output_dims = queryset.output_dims

    ### 2. Load the target model
    model = LocationModel.LocationMLP(input_shape, output_dims, dropout_probability=0.0).to(device)
    model_dict = torch.load(f'saves_new/{args.expt}/{args.dataset}/base/training_epoch200.pkl')
    model.load_state_dict(model_dict)
    model.eval()

    ### 3. Create the evaluate logits by hooking the second-to-last-layer and getting the activations
    f_evaluate=model(x_eval.to(device)).detach().cpu() #confidence score result of target model on evaluation dataset

    y_pred = f_evaluate.max(1)[1]
    accuracy = y_pred.eq(y_eval.view(-1)).sum().item() / len(y_eval)
    print("target model accuracy (30 classes)", accuracy)

    second_to_last_layer = list(model.children())[-1]

    def get_intermediate_output(module, input, output):
        global f_logits
        f_logits = output.detach().cpu()
    hook = second_to_last_layer.register_forward_hook(get_intermediate_output)

    model(x_eval.to(device))

    # sanity check logits
    print("Logits:")
    print(f_logits)

    hook.remove()

    f_evaluate_origin=np.copy(f_evaluate)  #keep a copy of original one
    f_evaluate_logits_origin=np.copy(f_logits)

    ### 4. Sort the prediction scores and use back_index to get the originals
    sort_index=np.argsort(f_evaluate,axis=1)
    back_index=np.copy(sort_index)

    for i in np.arange(back_index.shape[0]):
        back_index[i,sort_index[i,:]]=np.arange(back_index.shape[1])

    f_evaluate=torch.Tensor(np.sort(f_evaluate,axis=1))
    f_evaluate_logits=torch.Tensor(np.sort(f_logits,axis=1))

    print("f evaluate shape: {}".format(f_evaluate.shape))
    print("f evaluate logits shape: {}".format(f_evaluate_logits.shape))

    ### 5. Load the defense model and evaluate on undefended
    def_input = f_evaluate.shape[1:]
    def_model = LocationModel.DefenseMLP(def_input, 1).to(device)
    def_model_dict = torch.load(f'saves_new/{args.expt}/{args.dataset}/defense/defense=MNISTtraining_epoch{args.def_epoch}.pkl')
    def_model.load_state_dict(def_model_dict)
    def_model.eval()

    output = None
    with torch.no_grad():
        logits = def_model(f_evaluate_logits.to(device)).detach().cpu()
        output = logits[..., 0] # is this correct??
        print("ground truth")
        print(l_eval)
        criterion = nn.BCELoss()
        loss = criterion(output, l_eval)
        pred = torch.Tensor(np.where(np.asarray(output) >= 0.5, 1, 0))
        print("predictions")
        print(pred)
        accuracy = torch.sum((pred == l_eval)) / len(pred)

    print('evaluate loss on model:', loss.item())
    print('evaluate accuracy on model:', accuracy.item())

    ### 6. Set up loss for adversarial examples
    c1, c2, c3 = 1.0, 10.0, 0.1 # initial values
    loss1 = torch.abs(output)

    def get_grads(model_input, origin_value_placeholder, label_mask, c1_placeholder, c2_placeholder, c3_placeholder):
        label_mask = torch.tensor(label_mask)
        origin_value_placeholder = torch.tensor(origin_value_placeholder)
        model_input = torch.tensor(model_input, requires_grad=True)
        correct_label = torch.sum(label_mask * model_input, dim=1)
        wrong_label = torch.max((1 - label_mask) * model_input - 1e8 * label_mask, dim=1).values

        loss2 = F.relu(wrong_label - correct_label)
        loss3 = torch.sum(torch.abs(torch.subtract(F.softmax(model_input, dim=1), origin_value_placeholder))) # L-1 norm

        loss = c1_placeholder * loss1 + c2_placeholder * loss2 + c3_placeholder * loss3

        gradient_targetlabel = torch.autograd.grad(loss.sum(), model_input)[0]
        return loss, gradient_targetlabel
    
    ### 7. Perform the iteration
    label_mask_array=np.zeros([1, output_dims],dtype=np.float)
    result_array=np.zeros(f_evaluate.shape,dtype=np.float)
    result_array_logits=np.zeros(f_evaluate.shape,dtype=np.float)
    success_fraction=0.0
    max_iteration=300   #max iteration if can't find adversarial example that satisfies requirements
    np.random.seed(1000)

    print(f_evaluate.shape)
    print(f_evaluate_logits.shape)
    for test_sample_id in np.arange(0, f_evaluate.shape[0]):
        if test_sample_id % 100 == 0:
            print("test sample id: {}".format(test_sample_id))
        max_label = np.argmax(f_evaluate[test_sample_id, :])
        origin_value = np.copy(f_evaluate[test_sample_id, :]).reshape(1, output_dims)
        origin_value_logits = np.copy(f_evaluate_logits[test_sample_id, :]).reshape(1, output_dims)
        label_mask_array[0, :] = 0.0
        label_mask_array[0, max_label] = 1.0
        sample_f = np.copy(origin_value_logits)
        result_predict_scores_initial = def_model(torch.from_numpy(sample_f).to(device)).detach().cpu()

        ########## if the output score is already very close to 0.5, we can just use it for numerical reason
        if np.abs(result_predict_scores_initial - 0.5) <= 1e-5:
            success_fraction += 1.0
            result_array[test_sample_id, :] = origin_value[0, back_index[test_sample_id, :]]
            result_array_logits[test_sample_id, :] = origin_value_logits[0, back_index[test_sample_id, :]]
            continue
        last_iteration_result = np.copy(origin_value)[0, back_index[test_sample_id, :]]
        last_iteration_result_logits = np.copy(origin_value_logits)[0, back_index[test_sample_id, :]]
        success = True
        c3 = 0.1
        iterate_time = 1

        while success == True:
            sample_f = np.copy(origin_value_logits)
            j = 1
            result_max_label = -1
            result_predict_scores = result_predict_scores_initial
            while j < max_iteration and (max_label != result_max_label or (result_predict_scores - 0.5) * (result_predict_scores_initial - 0.5) > 0):                
                # Call the function defined earlier to get the gradients!!
                loss, gradient_values = get_grads(sample_f, origin_value, label_mask_array, c1, c2, c3)

                gradient_values = gradient_values / np.linalg.norm(gradient_values)
                sample_f = torch.subtract(torch.tensor(sample_f), 0.1 * gradient_values)
                result_predict_scores = def_model(sample_f.to(device)).detach().cpu()
                sample_f = np.asarray(sample_f)
                result_max_label = np.argmax(sample_f)
                j += 1

            if max_label != result_max_label:
                if iterate_time == 1:
                    print("failed sample for label not same for id: {},c3:{} not add noise".format(test_sample_id, c3))
                    success_fraction -= 1.0
                break

            sample_f = torch.tensor(sample_f)
            if ((def_model(sample_f.to(device)).detach().cpu() - 0.5) * (result_predict_scores_initial - 0.5)) > 0:
                if iterate_time == 1:
                    print("max iteration reached with id: {}, max score: {}, prediction_score: {}, c3: {}, not add noise".format(test_sample_id, torch.max(F.softmax(sample_f)), result_predict_scores, c3))
                break

            last_iteration_result[:] = F.softmax(sample_f, dim=0)[0, back_index[test_sample_id, :]]
            last_iteration_result_logits[:] = sample_f[0, back_index[test_sample_id, :]]
            iterate_time += 1
            c3 = c3 * 10
            if c3 > 100000:
                break
        success_fraction += 1.0
        result_array[test_sample_id, :] = last_iteration_result[:]
        result_array_logits[test_sample_id, :] = last_iteration_result_logits[:]
        
    print("Success fraction: {}".format(success_fraction/float(f_evaluate.shape[0])))

    ### 8. Save the new defense model
    y_pred_noisy = torch.Tensor(result_array).max(1)[1]
    print(y_pred_noisy)
    print(y_eval.view(-1))
    accuracy = y_pred_noisy.eq(y_eval.view(-1)).sum().item() / len(y_eval)
    print("defended model accuracy (30-classes)", accuracy)

    feval_sorted = torch.tensor(np.sort(f_evaluate_origin, axis=1))
    result_sorted = torch.tensor(np.sort(result_array, axis=1))
    predict_origin = def_model(feval_sorted.float().to(device)).detach().cpu()
    predict_modified = def_model(result_sorted.float().to(device)).detach().cpu()

    predict_origin = predict_origin[..., 0]
    predict_modified = predict_modified[..., 0]

    pred = torch.Tensor(np.where(np.asarray(predict_modified) >= 0.5, 1, 0))
    print("modified predictions..", pred)

    accuracy = torch.sum(pred == l_eval) / len(pred)
    print('evaluate accuracy on defended model:', accuracy.item())

    # want to use result_sorted as input to the target model during evaluation
    print("Saving the data...")
    if not os.path.exists(f'saves_new/{args.expt}/{args.dataset}/attack'):
        os.makedirs(f'saves_new/{args.expt}/{args.dataset}/attack')

    print(f"saves_new/{args.expt}/{args.dataset}/attack/fold={args.fold}noise_data_evaluation.npz")
    np.savez(f"saves_new/{args.expt}/{args.dataset}/attack/fold={args.fold}noise_data_evaluation.npz", 
             defense_output=result_array, defense_output_logits=result_array_logits, tc_output=f_evaluate_origin, 
             tc_output_logits=f_evaluate_logits_origin, predict_origin=predict_origin, predict_modified=predict_modified)