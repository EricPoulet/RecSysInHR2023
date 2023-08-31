import pandas as pd
import numpy as np

from datetime import datetime
import ast

import torch
from torch import Tensor

from sentence_transformers import SentenceTransformer

from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.loader import LinkNeighborLoader
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.seed import seed_everything

import tqdm

from graph_builder import build_graph
from model import Model
from custom_sampler import candidate_relations

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import sys
from torch_geometric.nn.models.lightgcn import BPRLoss

def resample_usr(sampled_data):
    edge_label_index = sampled_data["candidate", "is_shortlist", "job"].edge_label_index
    tmp_user = edge_label_index[0].tolist() + edge_label_index[0].tolist()
    tmp_item = []
    for i in range(0,len(edge_label_index[0])):
        rand_item_list = edge_label_index[1].tolist()
        if len(rand_item_list) !=1:
            del rand_item_list[i]
        rng = np.random.default_rng()
        tmp_item.append(rng.choice(rand_item_list))

    tmp_item_ = tmp_item + edge_label_index[1].tolist()
    edge_index = [tmp_user,tmp_item_]
    sampled_data["candidate", "is_shortlist", "job"].edge_label_index = torch.tensor(edge_index)
    sampled_data["candidate", "is_shortlist", "job"].edge_label = torch.tensor([1] * len(edge_label_index[0]) + [0] * len(edge_label_index[0]))
    sampled_data["candidate", "is_shortlist", "job"].input_id = torch.tensor( list( range( 0,len(edge_label_index[0]*2) ) ) )
    return sampled_data

def split_data(data):
    # For this, we first split the set of edges into
    # training (80%), validation (10%), and testing edges (10%).
    # Across the training edges, we use 70% of edges for message passing,
    # and 30% of edges for supervision.
    # We further want to generate fixed negative edges for evaluation with a ratio of 2:1.
    # Negative edges during training will be generated on-the-fly.
    # We can leverage the `RandomLinkSplit()` transform for this from PyG:
    transform = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        #disjoint_train_ratio=0.3,
        neg_sampling_ratio=1.0,
        is_undirected=True,
        add_negative_train_samples=False,
        edge_types=("candidate", "is_shortlist", "job"),
        rev_edge_types=("job", "rev_is_shortlist", "candidate"),
        # job to candidate
        #edge_types=("job", "shortlist", "candidate"),
        #rev_edge_types=("candidate", "rev_shortlist", "job"),
    )
    train_data, val_data, test_data = transform(data)
    return train_data, val_data, test_data

def train_loader(train_data, num_neigh=[20,10]):
    # Define seed edges:
    edge_label_index = train_data["candidate", "is_shortlist", "job"].edge_label_index
    edge_label = train_data["candidate", "is_shortlist", "job"].edge_label

    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=num_neigh,
        #neg_sampling_ratio=1.0,
        edge_label_index=(("candidate", "is_shortlist", "job"), edge_label_index),
        # job to candidate
        #edge_label_index=(("job", "shortlist", "candidate"), edge_label_index),
        edge_label=edge_label,
        batch_size=128,
        shuffle=True,
    )
    return train_loader

def test_loader(test_data):
    # Define seed edges:
    edge_label_index = test_data["candidate", "is_shortlist", "job"].edge_label_index
    edge_label = test_data["candidate", "is_shortlist", "job"].edge_label

    test_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[20, 10],
        #neg_sampling_ratio=1.0,
        edge_label_index=(("candidate", "is_shortlist", "job"), edge_label_index),
        edge_label=edge_label,
        batch_size=128,
        shuffle=True,
    )
    return test_loader

def val_loader(val_data, num_neigh=[20,10]):
    # Define seed edges:
    edge_label_index = val_data["candidate", "is_shortlist", "job"].edge_label_index
    edge_label = val_data["candidate", "is_shortlist", "job"].edge_label

    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=num_neigh,
        #neg_sampling_ratio=1.0,
        edge_label_index=(("candidate", "is_shortlist", "job"), edge_label_index),
        # job to candidate
        #edge_label_index=(("job", "shortlist", "candidate"), edge_label_index),
        edge_label=edge_label,
        batch_size=128,
        shuffle=True,
    )
    return val_loader

def train_w_tensorboard(model, train_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: '{device}'")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(0, 3):
        total_loss = total_examples = 0
        for sampled_data in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            sampled_data.to(device)
            pred = model(sampled_data)
            ground_truth = sampled_data["candidate", "is_shortlist", "job"].edge_label
            # job to candidate
            #ground_truth = sampled_data["job", "shortlist", "candidate"].edge_label
            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
        print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")
    return model

def evaluate(model, val_loader, verbose=False):
    model.eval()
    preds = []
    ground_truths = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        iterator = tqdm.tqdm(val_loader)
    else:
        iterator = val_loader
    total_loss = total_examples = 0
    for sampled_data in iterator:
        with torch.no_grad():
            # if loss BPR
            #sampled_data = resample_usr(sampled_data)

            sampled_data.to(device)
            pred = model(sampled_data)
            preds.append(pred)
            gt = sampled_data["candidate", "is_shortlist", "job"].edge_label
            ground_truths.append(gt)
            loss = F.binary_cross_entropy_with_logits(pred, gt)

            # if crossentropy
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()

    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()

    try:
        auc = roc_auc_score(ground_truth, pred)
    except ValueError:
        auc = 0
    f1 = f1_score(ground_truth, [int(x > 0.5) for x in pred], average='micro')
    prec = precision_score(ground_truth, [int(x > 0.5) for x in pred], average='micro')
    recall = recall_score(ground_truth, [int(x > 0.5) for x in pred], average='micro')
    if verbose:
        print()
        print(f"Validation AUC: {auc:.4f}")
        print(f"Validation Precision: {prec:.4f}")
        print(f"Validation Recall: {recall:.4f}")
        print(f"Validation F1: {f1:.4f}")
        print(f"Validation total loss: {total_loss / total_examples:.4f}")
    return auc, prec, recall, f1, total_loss / total_examples

def evaluate_recommendation(model, val_loader, verbose=False):
    model.eval()
    preds = []
    ground_truths = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    if verbose:
        iterator = tqdm.tqdm(val_loader)
    else:
        iterator = val_loader
    total_loss = total_examples = 0
    for sampled_data in iterator:
        optimizer.zero_grad()
        sampled_data = resample_usr(sampled_data)

        sampled_data.to(device)

        pred = model(sampled_data)
        preds.append(pred)

        ground_truth = sampled_data["candidate", "is_shortlist", "job"].edge_label
        ground_truths.append(ground_truth)

        pos_rank, neg_rank = pred.chunk(2)
        loss_fn = BPRLoss()
        loss = loss_fn(pos_rank, neg_rank)

        total_loss += float(loss) * pos_rank.numel()
        total_examples += pos_rank.numel()

    pred = torch.cat(preds, dim=0).cpu().detach().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    auc = roc_auc_score(ground_truth, pred)
    f1 = f1_score(ground_truth, [int(x > 0.5) for x in pred])
    prec = precision_score(ground_truth, [int(x > 0.5) for x in pred])
    recall = recall_score(ground_truth, [int(x > 0.5) for x in pred])
    if verbose:
        print()
        print(f"Validation AUC: {auc:.4f}")
        print(f"Validation Precision: {prec:.4f}")
        print(f"Validation Recall: {recall:.4f}")
        print(f"Validation F1: {f1:.4f}")
        print(f"Validation total loss: {total_loss / total_examples:.4f}")
    return auc, prec, recall, f1, total_loss / total_examples

def train(model, train_loader, val_loader, name_experiment, loss_BPR=False, verbose=False):
    writer = SummaryWriter(name_experiment)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    max_epoch = 2000
    max_val_decrease=10

    model.train()

    best_f1 = 0
    best_loss = 100
    counter = 0

    for epoch in range(max_epoch):
        if counter >= max_val_decrease:
            break
        total_loss = total_examples = 0
        iterator = train_loader
        for sampled_data in tqdm.tqdm(train_loader):
            optimizer.zero_grad()

            if loss_BPR:
                sampled_data = resample_usr(sampled_data)

            sampled_data.to(device)
            pred = model(sampled_data)
            ground_truth = sampled_data["candidate", "is_shortlist", "job"].edge_label
            # job to candidate
            #ground_truth = sampled_data["job", "shortlist", "candidate"].edge_label

            if loss_BPR:
                pos_rank, neg_rank = pred.chunk(2)
                loss_fn = BPRLoss()
                loss = loss_fn(pos_rank, neg_rank)

            else:
                loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
            loss.backward()
            optimizer.step()

            if loss_BPR:
                total_loss += float(loss) * pos_rank.numel()
                total_examples += pos_rank.numel()

            else:
                total_loss += float(loss) * pred.numel()
                total_examples += pred.numel()

        print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")

        if loss_BPR:
            auc, prec, recall, f1, val_loss = evaluate_recommendation(model, val_loader)
        else:
            auc, prec, recall, f1, val_loss = evaluate(model, val_loader)

        if writer is not None:
            writer.add_scalar('Loss/train', total_loss / total_examples, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Metrics/auc', auc, epoch)
            writer.add_scalar('Metrics/precision', prec, epoch)
            writer.add_scalar('Metrics/recall', recall, epoch)
            writer.add_scalar('Metrics/F1', f1, epoch)

        if f1 > best_f1:
            best_f1 = f1
            counter = 0

        #if val_loss < best_loss:
        #    best_loss = val_loss
        #    counter = 0
        #save_model = model
        else:
            counter += 1
    #writer.add_hparams(setting, {"hparams/AUC": auc, "hparams/Precision": prec, "hparams/Recall": recall, "hparams/F1": f1})
    writer.close()
    return model

def evaluate_topK(model, val_data,name_experiment):

    all_prec = []
    all_recall = []
    all_average_precision_score = []
    all_ndcg = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for u in tqdm.tqdm(val_data['candidate']['node_id']):

        edges = val_data["candidate", "is_shortlist", "job"]
        true_egdes_user = edges['edge_label_index'][0][edges['edge_label'] == 1]
        true_egdes_job = edges['edge_label_index'][1][edges['edge_label'] == 1]

        index_truth = [i for i, j in enumerate(true_egdes_user) if j == u]

        index_true_item = true_egdes_job[index_truth]

        all_items = list(val_data["job"]["node_id"])

        ground_truth = [1 if i in index_true_item else 0 for i in all_items]
        if sum(ground_truth) == 0:
            continue

        edge_label_index = torch.tensor([np.repeat(u.item(),len(all_items)),np.array(all_items)])
        edge_label = torch.Tensor(ground_truth)
        val_loader = LinkNeighborLoader(
            data=val_data,
            num_neighbors=[20, 10], # maybe change -1
            edge_label_index=(("candidate", "is_shortlist", "job"), edge_label_index),
            edge_label=edge_label,
            batch_size=3 * 128,
            shuffle=False,
        )
        local_pred = []
        local_ground_truth = []

        for sampled_data in val_loader:
            pred = model(sampled_data.to(device))
            local_pred += pred.tolist()
            local_ground_truth += sampled_data["candidate", "is_shortlist", "job"]["edge_label"]
            # job to candidate
            #local_ground_truth += sampled_data["job", "shortlist", "candidate"]["edge_label"]
        topk = sorted(zip(local_pred, local_ground_truth), key=lambda x: -x[0])[:10]
        topk_gt = [x[1].item() for x in topk]
        prec = sum(topk_gt)/len(topk_gt)
        recall = sum(topk_gt)/sum(local_ground_truth)
        ndcg = 0
        for i, x in enumerate(topk_gt):
            if x == 1:
                ndcg = 1/(i+1)
                break

        average_precision = 0
        count_top = 1

        for i,x in enumerate(topk_gt):
            if x == 1:
                average_precision += count_top/(i+1)
                count_top += 1

        if sum(topk_gt) != 0:
            average_precision = average_precision/sum(topk_gt)
        else:
            average_precision = 0

        all_prec.append(prec)
        all_recall.append(recall)
        all_average_precision_score.append(average_precision)
        all_ndcg.append(ndcg)

        print("prec@10: ", sum(all_prec)/len(all_prec))
        print("recall@10 : ", sum(all_recall)/len(all_recall))
        print("average_precision_score@10 : ", sum(all_average_precision_score)/len(all_average_precision_score))
        print("ndcg_score@10 : ", sum(all_ndcg)/len(all_ndcg))
    print()
    print("Metrics")
    print("prec@10: ", sum(all_prec)/len(all_prec))
    print("recall@10 : ", sum(all_recall)/len(all_recall))
    print("Mean average_precision_score@10 : ", sum(all_average_precision_score)/len(all_average_precision_score))
    print("ndcg_score@10 : ", sum(all_ndcg)/len(all_ndcg))

    # save metrics to text file
    with open(f'metrics{name_experiment}.txt', 'w') as f:
        f.write("prec@10: " + str(sum(all_prec)/len(all_prec)) + "\n")
        f.write("recall@10 : " + str(sum(all_recall)/len(all_recall)) + "\n")
        f.write("Mean average_precision_score@10 : " + str(sum(all_average_precision_score)/len(all_average_precision_score)) + "\n")
        f.write("ndcg_score@10 : " + str(sum(all_ndcg)/len(all_ndcg)) + "\n")

if __name__ == '__main__':

    # seed to allow reproductibility
    seed_everything(42)

    ablation_keys = ["no_skill", "no_contract", "no_origin", "no_exp", "no_salary", "no_zip", "no_category", "no_recruiter", "no_company", "no_concept"]
    ablation_indice = int(sys.argv[2])

    list_abl = 10 * [1]

    if ablation_indice == 0:
        list_abl[0] = 0
        list_abl[9] = 0
        name_experiment = sys.argv[1] + '_' + ablation_keys[ablation_indice]

    elif ablation_indice > -1:
        list_abl[ablation_indice] = 0
        name_experiment = sys.argv[1] + '_' + ablation_keys[ablation_indice]

    elif ablation_indice == -10:
        list_abl = 10 * [0]
        name_experiment = sys.argv[1] + '_' + 'base'

    else:
        name_experiment = sys.argv[1] + '_' + 'all'

    # Load graph data
    # This part is not provided due to RGPD compliance. Dataset is not public.
    # Build Graph pass a set of ablation parameters to build the graph
    # it returns a HeteroData object
    data = build_graph(list_abl)
    print("Graph built")

    candidate_relations = candidate_relations(data, data['candidate']['node_id'].numpy()[0:1000])

    print("Candidate relations built")

    # save data
    torch.save(data, 'data.pt')

    # Load data
    #data = torch.load('data.pt')

    # Split data into train, val, test
    train_data, val_data, test_data = split_data(data)
    print("Data splitted")

    # Instantiate model
    model = Model(hidden_channels=64, data=data, list_abl=list_abl)
    print("Model instantiated")

    num_neigh = [30,20,10]
    loss_BPR = False

    # load train data into train_loader
    train_loader = train_loader(train_data, num_neigh)
    print("Train loader loaded")

    val_loader = val_loader(val_data, num_neigh)
    print("Val loader loaded")

    # Train model
    model = train(model, train_loader, val_loader, name_experiment, loss_BPR)
    print("Model trained")

    # Save model
    torch.save(model.state_dict(), 'model.pt')

    # evaluate model
    evaluate_topK(model, val_data, name_experiment)
    print("Model evaluated")
