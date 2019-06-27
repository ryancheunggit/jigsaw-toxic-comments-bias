# evaluation code from https://www.kaggle.com/dborkan/benchmark-kernel
# loss function from https://www.kaggle.com/thousandvoices/simple-lstm
# bert training code from https://www.kaggle.com/yuval6967/toxic-bert-plain-vanila
# layer wise learning rate decay base on https://arxiv.org/abs/1905.05583 / https://arxiv.org/abs/1801.06146
import argparse
import os
import torch
import pickle
import numpy as np
import pandas as pd
from apex import amp
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorch_pretrained_bert import BertConfig
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import BertAdam
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import WarmupCosineSchedule, WarmupConstantSchedule
from datetime import datetime
from sklearn import metrics


parser = argparse.ArgumentParser(description='train model')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--model', type=str, default='bert-large-uncased-whole-word-masking',
                    choices=['bert-base-uncased', 'bert-large-uncased', 'bert-large-uncased-whole-word-masking'])
parser.add_argument('--grad_accum', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--initial_lr', type=float, default=2e-5)
parser.add_argument('--layer_wise_decay', type=float, default=.95)
parser.add_argument('--warmup', type=float, default=.05)
parser.add_argument('--epochs', type=int, default=4)
parser.add_argument('--valid_per_steps', type=int, default=10150)
parser.add_argument('--lr_schedule', type=str, default='constent')
parser.add_argument('--redo_token', action='store_true', default=False)

args = parser.parse_args()
print(args)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if args.model == 'bert-large-uncased-whole-word-masking':
    tensorflow_poretrained_path = '../model/wwm_uncased_L-24_H-1024_A-16/'
elif args.model == 'bert-base-uncased':
    tensorflow_poretrained_path = '../model/uncased_L-12_H-768_A-12/'
else:
    tensorflow_poretrained_path = '../model/uncased_L-24_H-1024_A-16/'

bert_config = BertConfig.from_json_file(tensorflow_poretrained_path + 'bert_config.json')
model_optim_checkpoint = '../model/{}/pytorch-model_optim.bin'.format(args.model)
model_checkpoint = '../model/{}/pytorch-model.bin'.format(args.model)
tokenizer = BertTokenizer.from_pretrained(tensorflow_poretrained_path)
token_filepath = '../data/{}_token_cache.pkl'.format(args.model)
start_id, end_id = tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])

LOG_FILE = '../model/{}/log.txt'.format(args.model)
N_FOLD = 10
EPOCHS = args.epochs
BATCH_SIZE = 32
MAX_LENGTH = 224
VALID_PER_STEPS = args.valid_per_steps
INITIAL_LR = args.initial_lr
ACCUMULATE_GRADS = args.grad_accum
REDO_TOKENIZATION = args.redo_token
WARMUP = args.warmup

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


with open(LOG_FILE, 'a') as f:
    f.write('start script at {}\n'.format(datetime.now()))


TOXICITY_COLUMN = 'target'
SUBGROUP_AUC = 'subgroup_auc'
BPSN_AUC = 'bpsn_auc'
BNSP_AUC = 'bnsp_auc'


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AUCMeter(object):
    """Store a window of label and predictions, calculate AUC on the fly."""
    def __init__(self, window_size=300):
        self.window_size = window_size
        self.reset()

    def reset(self):
        self.labels = []
        self.predictions = []

    def forget(self):
        if len(self.labels) >= self.window_size:
            self.labels.pop(0)
            self.predictions.pop(0)

    def update(self, labels, predictions):
        self.forget()
        self.labels.append(labels)
        self.predictions.append(predictions)

    @property
    def auc_(self):
        if np.vstack(self.labels).sum() > 0:
            return metrics.roc_auc_score(np.vstack(self.labels).ravel(), np.vstack(self.predictions).ravel())
        else:
            return 0


def compute_auc(y_true, y_pred):
    try:
        return metrics.roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan


def compute_subgroup_auc(df, subgroup, label, model_name):
    subgroup_examples = df[df[subgroup]]
    return compute_auc(subgroup_examples[label], subgroup_examples[model_name])


def compute_bpsn_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
    subgroup_negative_examples = df[df[subgroup] & ~df[label]]
    non_subgroup_positive_examples = df[~df[subgroup] & df[label]]
    examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
    return compute_auc(examples[label], examples[model_name])


def compute_bnsp_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
    subgroup_positive_examples = df[df[subgroup] & df[label]]
    non_subgroup_negative_examples = df[~df[subgroup] & ~df[label]]
    examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
    return compute_auc(examples[label], examples[model_name])


def compute_bias_metrics_for_model(dataset, subgroups, model, label_col, include_asegs=False):
    """Computes per-subgroup metrics for all subgroups and one model."""
    records = []
    for subgroup in subgroups:
        record = {
            'subgroup': subgroup,
            'subgroup_size': len(dataset[dataset[subgroup]])
        }
        record[SUBGROUP_AUC] = compute_subgroup_auc(dataset, subgroup, label_col, model)
        record[BPSN_AUC] = compute_bpsn_auc(dataset, subgroup, label_col, model)
        record[BNSP_AUC] = compute_bnsp_auc(dataset, subgroup, label_col, model)
        records.append(record)
    return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)


def calculate_overall_auc(df, model_name):
    true_labels = df[TOXICITY_COLUMN]
    predicted_labels = df[model_name]
    return metrics.roc_auc_score(true_labels, predicted_labels)


def power_mean(series, p):
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)


def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):
    bias_score = np.average([
        power_mean(bias_df[SUBGROUP_AUC], POWER),
        power_mean(bias_df[BPSN_AUC], POWER),
        power_mean(bias_df[BNSP_AUC], POWER)
    ])
    return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)


class JigsawDataset(Dataset):
    def __init__(self, mode, seq_token_ids, labels, start_id=101, end_id=102, max_length=MAX_LENGTH, random_mask=0):
        self.mode = mode
        self.seq_token_ids = seq_token_ids
        self.labels = labels
        self.start_id = start_id
        self.end_id = end_id
        self.max_length = max_length
        self.random_mask = random_mask
        self._n = len(seq_token_ids)

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        seq = self.seq_token_ids[idx]
        if len(seq) > self.max_length - 2:
            left = (self.max_length - 2) // 2
            right = self.max_length - 2 - left
            seq = seq[:left] + seq[-right:]
        seq = [self.start_id] + seq + [self.end_id]
        if len(seq) < self.max_length:
            seq += [0] * (self.max_length - len(seq))
        label = self.labels[idx] if self.mode == 'train' else None
        return torch.from_numpy(np.array(seq)), torch.from_numpy(label)


def corpus_to_seq_token_ids(corpus, tokenizer):
    seq_token_ids = []
    for text in tqdm(corpus):
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        seq_token_ids.append(token_ids)
    return np.array(seq_token_ids)


class CustomLoss(nn.Module):
    def __init__(self, loss_weight, use_sample_weight=True):
        super(CustomLoss, self).__init__()
        self.loss_weight = loss_weight
        self.use_sample_weight = use_sample_weight

    def forward(self, logits, targets):
        sample_weights = targets[:, 1:2]
        main_target_logits = logits[:, :1]
        main_target_labels = targets[:, :1]
        aux_target_logits = logits[:, 1:]
        aux_target_labels = targets[:, 2:]
        if self.use_sample_weight:
            main_target_bce_loss = nn.BCEWithLogitsLoss(weight=sample_weights)(main_target_logits, main_target_labels)
        else:
            main_target_bce_loss = nn.BCEWithLogitsLoss()(main_target_logits, main_target_labels)
        aux_target_bce_loss = nn.BCEWithLogitsLoss()(aux_target_logits, aux_target_labels)
        return (main_target_bce_loss * self.loss_weight) + aux_target_bce_loss


main_target_col = 'target'
identity_cols = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish', 'muslim', 'black', 'white',
                 'psychiatric_or_mental_illness']
aux_target_cols = ['soft_target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']

print('read competition data')
metadata = pd.read_csv('../data/train.csv')
metadata['comment_text'] = metadata['comment_text'].astype('str')
metadata['comment_text'] = metadata['comment_text'].fillna("['UNK']")
metadata.fillna(0, inplace=True)
metadata['soft_target'] = metadata['target'].values

# from kernel
weights = np.ones((metadata.shape[0],)) / 4
weights += (metadata[identity_cols].fillna(0).values >= .5).sum(axis=1).astype(bool).astype(np.int) / 4
weights += (((metadata['target'].values >= .5).astype(bool).astype(np.int) +
             (metadata[identity_cols].fillna(0).values < .5).sum(axis=1).astype(bool).astype(np.int)) > 1
            ).astype(bool).astype(np.int) / 4
weights += (((metadata['target'].values < .5).astype(bool).astype(np.int) +
             (metadata[identity_cols].fillna(0).values >= .5).sum(axis=1).astype(bool).astype(np.int)) > 1
            ).astype(bool).astype(np.int) / 2
loss_weight = 1 / weights.mean()
metadata['sample_weight'] = weights

metadata[[main_target_col] + identity_cols] = (metadata[[main_target_col] + identity_cols].values >= .5).astype(int)
mskf = MultilabelStratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=42)
kfolds = list(mskf.split(X=np.zeros(metadata.shape[0]), y=metadata[[main_target_col] + identity_cols]))

if not os.path.exists(token_filepath) or REDO_TOKENIZATION:
    seq_token_ids = corpus_to_seq_token_ids(metadata['comment_text'], tokenizer)
    pickle.dump(seq_token_ids, open(token_filepath, 'wb'))
else:
    seq_token_ids = pickle.load(open(token_filepath, 'rb'))

train_idx, dev_idx = kfolds[0]
train_meta = metadata.iloc[train_idx].copy()
dev_meta = metadata.iloc[dev_idx].copy()

print('convert text to token ids')
train_seq_token_ids = seq_token_ids[train_idx]
train_labels = train_meta[[main_target_col, 'sample_weight'] + aux_target_cols].values
dev_seq_token_ids = seq_token_ids[dev_idx]
dev_labels = dev_meta[[main_target_col, 'sample_weight'] + aux_target_cols].values

print('put data into dataloader')
train_ds = JigsawDataset(mode='train', seq_token_ids=train_seq_token_ids, labels=train_labels,
                         start_id=start_id, end_id=end_id, max_length=MAX_LENGTH)
train_dl = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
dev_ds = JigsawDataset(mode='train', seq_token_ids=dev_seq_token_ids, labels=dev_labels,
                       start_id=start_id, end_id=end_id, max_length=MAX_LENGTH)
dev_dl = DataLoader(dataset=dev_ds, batch_size=BATCH_SIZE*4, shuffle=False, drop_last=False)

print('set up model, optimizer and metric meter')

model = BertForSequenceClassification.from_pretrained(tensorflow_poretrained_path, num_labels=train_labels.shape[1] - 1)
model = model.to('cuda')

base_lr = INITIAL_LR
total_layers = 24 if 'large' in args.model else 12
lr_layer_wise_decay_rate = args.layer_wise_decay
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
if lr_layer_wise_decay_rate != 1:
    param_optimizer = list(model.named_parameters())
    embedding_layer_params = [
        {
            'params': [p for n, p in param_optimizer if ('embeddings' in n) and (not any(nd in n for nd in no_decay))],
            'lr': base_lr * lr_layer_wise_decay_rate ** total_layers,
            'weight_decay': 0.01
        },
            {'params': [p for n, p in param_optimizer if ('embeddings' in n) and (any(nd in n for nd in no_decay))],
            'lr': base_lr * lr_layer_wise_decay_rate ** total_layers,
            'weight_decay': 0.0
        }
    ]
    encoder_layer_params = [
        {
            'params': [p for n, p in param_optimizer
                         if ('encoder.layer.{}.'.format(layer_number) in n) and (not any(nd in n for nd in no_decay))
                    ],
            'lr':base_lr * lr_layer_wise_decay_rate ** (total_layers - layer_number),
            'weight_decay': 0.01
        }
        for layer_number in range(total_layers)
    ] + [
        {
            'params': [p for n, p in param_optimizer
                         if ('encoder.layer.{}.'.format(layer_number) in n) and any(nd in n for nd in no_decay)
                    ],
            'lr':base_lr * lr_layer_wise_decay_rate **(total_layers - layer_number),
            'weight_decay': 0.0
        }
        for layer_number in range(total_layers)
    ]
    all_reset = [
        {'params': model.bert.pooler.parameters()},
        {'params': model.classifier.parameters()}
    ]
    bert_parameters = embedding_layer_params + encoder_layer_params + all_reset
else:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    bert_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]


criterion = CustomLoss(loss_weight=loss_weight)
t_total = int(EPOCHS * len(train_ds) / BATCH_SIZE / ACCUMULATE_GRADS + 1)
if args.lr_schedule == 'constent':
    lr_scheduler = WarmupConstantSchedule(warmup=WARMUP, t_total=t_total)
else:
    lr_scheduler = WarmupCosineSchedule(warmup=WARMUP, t_total=t_total)
optimizer = BertAdam(bert_parameters, lr=INITIAL_LR, schedule=lr_scheduler)

model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
for param in model.parameters():
    param.requires_grad = True


message_template = '\r-- epoch {:>3} - iteration {:>8} - secs per batch {:4.2f} - train loss {:5.4f} - auc {:5.4f}'
dev_meta[[main_target_col] + identity_cols] = (dev_meta[[main_target_col] + identity_cols] > 0).astype(bool)
dev_meta['bert'] = 0

best_score = 0
print('start training')
for epoch in range(EPOCHS):
    tt0 = datetime.now()
    loss_meter = AverageMeter()
    auc_meter = AUCMeter()
    optimizer.zero_grad()
    for it, (x_batch, y_batch) in enumerate(train_dl):
        t0 = datetime.now()
        model.train()
        x_batch = x_batch.to('cuda')
        y_batch = y_batch.to('cuda').float()
        logits = model(x_batch, attention_mask=x_batch > 0, labels=None)
        loss = criterion(logits, y_batch)
        loss_meter.update(loss.detach().cpu().numpy())
        auc_meter.update(
            y_batch[:, 0].detach().cpu().numpy().reshape(-1, 1),
            torch.sigmoid(logits[:, 0]).detach().cpu().numpy().reshape(-1, 1)
        )
        if ACCUMULATE_GRADS > 1:
            loss = loss / ACCUMULATE_GRADS
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        if (it + 1) % ACCUMULATE_GRADS == 0:
            optimizer.step()
            optimizer.zero_grad()
        t1 = datetime.now()
        message = message_template.format(epoch, it, (t1 - t0).total_seconds(), loss_meter.avg, auc_meter.auc_)
        print(message, end='', flush=True)
        if it % 100 == 0:
            with open(LOG_FILE, 'a') as f:
                f.write(message + '\n')

        if (it + 1) % VALID_PER_STEPS == 0:
            print('\r\n', end='')
            predictions = []
            model.eval()
            with torch.no_grad():
                for idx, (x_batch, y_batch) in tqdm(enumerate(dev_dl), desc='validating', leave=False):
                    x_batch, y_batch = x_batch.to('cuda'), y_batch.to('cuda').float()
                    logits = model(x_batch, attention_mask=x_batch > 0, labels=None)
                    prediction = torch.sigmoid(logits).detach().cpu().numpy()
                    predictions.append(prediction)
            predictions = np.vstack(predictions)[:, 0]
            dev_meta['bert'] = predictions
            bias_metrics_df = compute_bias_metrics_for_model(dev_meta, identity_cols, 'bert', 'target')
            final_metric = get_final_metric(bias_metrics_df, calculate_overall_auc(dev_meta, 'bert'))
            print(bias_metrics_df)
            print(final_metric)
            if best_score < final_metric:
                torch.save(model.state_dict(), model_checkpoint)
                best_score = final_metric

            tt1 = datetime.now()
            print('-- {} batches finshed in {} minutes'.format(it + 1, (tt1 - tt0).total_seconds() // 60))
            t0 = datetime.now()
            loss_meter = AverageMeter()
            auc_meter = AUCMeter()
print(best_score)
