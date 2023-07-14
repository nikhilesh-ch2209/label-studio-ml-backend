import numpy as np
import torch
import numpy as np
import re
import os
import io
import logging

from functools import partial
from itertools import groupby
from operator import itemgetter
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
from collections import deque
from tensorboardX import SummaryWriter
from transformers import (
    BertTokenizer, BertForTokenClassification, BertConfig,
    RobertaConfig, RobertaForTokenClassification, RobertaTokenizer,
    DistilBertConfig, DistilBertForTokenClassification, DistilBertTokenizer,
    CamembertConfig, CamembertForTokenClassification, CamembertTokenizer,
    AutoConfig, AutoModelForTokenClassification, AutoTokenizer,
    BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
)
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import calc_slope
logger = logging.getLogger(__name__)


class Fit_Predict:
    def __init__(self):
        pass
    def fit_general_classifier(self, model, input_texts, output_labels_idx):
        model.fit(input_texts, output_labels_idx)

    def predict_general_classifier(self, model, input_texts):
        probabilities = model.predict_proba(input_texts)
        print('=== probabilities >', probabilities)
        predicted_label_indices = np.argmax(probabilities, axis=1)
        predicted_scores = probabilities[np.arange(len(predicted_label_indices)), predicted_label_indices]
        return predicted_label_indices, predicted_scores

    def fit_ner(
        self, completions, workdir=None, model_type='bert', pretrained_model='bert-base-uncased',
        batch_size=32, learning_rate=5e-5, adam_epsilon=1e-8, num_train_epochs=100, weight_decay=0.0, logging_steps=1,
        warmup_steps=0, save_steps=50, dump_dataset=True, cache_dir='~/.heartex/cache', train_logs=None,
        **kwargs
    ):
        train_logs = train_logs or os.path.join(workdir, 'train_logs')
        os.makedirs(train_logs, exist_ok=True)
        logger.debug('Prepare models')
        cache_dir = os.path.expanduser(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)

        model_type = model_type.lower()
        # assert model_type in MODEL_CLASSES.keys(), f'Input model type {model_type} not in {MODEL_CLASSES.keys()}'
        # assert pretrained_model in ALL_MODELS, f'Pretrained model {pretrained_model} not in {ALL_MODELS}'

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model, cache_dir=cache_dir)

        logger.debug('Read data')
        # read input data stream
        texts, list_of_spans = [], []
        for item in completions:
            texts.append(item['data'][self.value])
            list_of_spans.append(self.get_spans(item['annotations'][0]))

        logger.debug('Prepare dataset')
        pad_token_label_id = CrossEntropyLoss().ignore_index
        train_set = SpanLabeledTextDataset(
            texts, list_of_spans, tokenizer,
            cls_token_at_end=model_type in ['xlnet'],
            cls_token_segment_id=2 if model_type in ['xlnet'] else 0,
            sep_token_extra=model_type in ['roberta'],
            pad_token_label_id=pad_token_label_id
        )

        if dump_dataset:
            dataset_file = os.path.join(workdir, 'train_set.txt')
            train_set.dump(dataset_file)

        config = AutoConfig.from_pretrained(pretrained_model, num_labels=train_set.num_labels, cache_dir=cache_dir)
        model = AutoModelForTokenClassification.from_pretrained(pretrained_model, config=config, cache_dir=cache_dir)

        batch_padding = SpanLabeledTextDataset.get_padding_function(model_type, tokenizer, pad_token_label_id)

        train_loader = DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=batch_padding
        )

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        num_training_steps = len(train_loader) * num_train_epochs
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

        tr_loss, logging_loss = 0, 0
        global_step = 0
        if train_logs:
            tb_writer = SummaryWriter(logdir=os.path.join(train_logs, os.path.basename(workdir)))
        epoch_iterator = trange(num_train_epochs, desc='Epoch')
        loss_queue = deque(maxlen=10)
        for _ in epoch_iterator:
            batch_iterator = tqdm(train_loader, desc='Batch')
            for step, batch in enumerate(batch_iterator):

                model.train()
                inputs = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['input_mask'],
                    'labels': batch['label_ids'],
                    'token_type_ids': batch['segment_ids']
                }
                if model_type == 'distilbert':
                    inputs.pop('token_type_ids')

                model_output = model(**inputs)
                loss = model_output[0]
                loss.backward()
                tr_loss += loss.item()
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
                if global_step % logging_steps == 0:
                    last_loss = (tr_loss - logging_loss) / logging_steps
                    loss_queue.append(last_loss)
                    if train_logs:
                        tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar('loss', last_loss, global_step)
                    logging_loss = tr_loss

            # slope-based early stopping
            if len(loss_queue) == loss_queue.maxlen:
                slope = calc_slope(loss_queue)
                if train_logs:
                    tb_writer.add_scalar('slope', slope, global_step)
                if abs(slope) < 1e-2:
                    break

        if train_logs:
            tb_writer.close()

        model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(workdir)
        tokenizer.save_pretrained(workdir)
        label_map = {i: t for t, i in train_set.tag_idx_map.items()}

        return {
            'model_path': workdir,
            'batch_size': batch_size,
            'pad_token_label_id': pad_token_label_id,
            'dataset_params_dict': train_set.get_params_dict(),
            'model_type': model_type,
            'pretrained_model': pretrained_model,
            'label_map': label_map
        }

    def predict_ner(self, predict_loader, _model_type, _model, _label_map, from_name, to_name, **kwargs):

        results = []
        for batch in tqdm(predict_loader, desc='Prediction'):
            inputs = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['input_mask'],
                'token_type_ids': batch['segment_ids']
            }
            if _model_type == 'distilbert':
                inputs.pop('token_type_ids')
            with torch.no_grad():
                model_output = _model(**inputs)
                logits = model_output[0]

            batch_preds = logits.detach().cpu().numpy()
            argmax_batch_preds = np.argmax(batch_preds, axis=-1)
            max_batch_preds = np.max(batch_preds, axis=-1)
            input_mask = batch['input_mask'].detach().cpu().numpy()
            batch_token_start_map = batch['token_start_map']
            batch_strings = batch['strings']

            for max_preds, argmax_preds, mask_tokens, token_start_map, string in zip(
                max_batch_preds, argmax_batch_preds, input_mask, batch_token_start_map, batch_strings
            ):
                preds, scores, starts = [], [], []
                for max_pred, argmax_pred, mask_token, token_start in zip(max_preds, argmax_preds, mask_tokens, token_start_map):
                    if token_start != -1:
                        preds.append(_label_map[str(argmax_pred)])
                        scores.append(max_pred)
                        starts.append(token_start)
                mean_score = np.mean(scores) if len(scores) > 0 else 0

                result = []

                for label, group in groupby(zip(preds, starts, scores), key=lambda i: re.sub('^(B-|I-)', '', i[0])):
                    _, group_start, _ = list(group)[0]
                    if len(result) > 0:
                        if group_start == 0:
                            result.pop(-1)
                        else:
                            result[-1]['value']['end'] = group_start - 1
                    if label != 'O':
                        result.append({
                            'from_name': from_name,
                            'to_name': to_name,
                            'type': 'labels',
                            'value': {
                                'labels': [label],
                                'start': group_start,
                                'end': None,
                                'text': '...'
                            }
                        })
                if result and result[-1]['value']['end'] is None:
                    result[-1]['value']['end'] = len(string)
                results.append({
                    'result': result,
                    'score': float(mean_score),
                    'cluster': None
                })
        return results

    def predict_bert(self, tasks, **kwargs):
        if self.not_trained:
            print('Can\'t get prediction because model is not trained yet.')
            return []

        texts = [task['data'][self.value] for task in tasks]
        predict_dataloader = prepare_texts(texts, self.tokenizer, self.maxlen, SequentialSampler, self.batch_size)

        pred_labels, pred_scores = [], []
        for batch in predict_dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1]
            }
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs[0]

            batch_preds = logits.detach().cpu().numpy()

            argmax_batch_preds = np.argmax(batch_preds, axis=-1)
            pred_labels.extend(str(self.labels[i]) for i in argmax_batch_preds)

            max_batch_preds = np.max(batch_preds, axis=-1)
            pred_scores.extend(float(s) for s in max_batch_preds)

        predictions = []
        for predicted_label, score in zip(pred_labels, pred_scores):
            result = [{
                'from_name': self.from_name,
                'to_name': self.to_name,
                'type': 'choices',
                'value': {'choices': [predicted_label]}
            }]

            predictions.append({'result': result, 'score': score})
        return predictions

    def fit_bert(self, completions, workdir=None, cache_dir=None, **kwargs):
        input_texts = []
        output_labels, output_labels_idx = [], []
        label2idx = {l: i for i, l in enumerate(self.labels)}
        for completion in completions:
            # get input text from task data

            if completion['annotations'][0].get('skipped'):
                continue

            input_text = completion['data'][self.value]
            input_texts.append(input_text)

            # get an annotation
            output_label = completion['annotations'][0]['result'][0]['value']['choices'][0]
            output_labels.append(output_label)
            output_label_idx = label2idx[output_label]
            output_labels_idx.append(output_label_idx)

        new_labels = set(output_labels)
        added_labels = new_labels - set(self.labels)
        if len(added_labels) > 0:
            print('Label set has been changed. Added ones: ' + str(list(added_labels)))
            self.labels = list(sorted(new_labels))
            label2idx = {l: i for i, l in enumerate(self.labels)}
            output_labels_idx = [label2idx[label] for label in output_labels]

        tokenizer = BertTokenizer.from_pretrained(self.pretrained_model, cache_dir=cache_dir)

        train_dataloader = prepare_texts(input_texts, tokenizer, self.maxlen, RandomSampler, self.batch_size, output_labels_idx)
        model = self.reset_model(self.pretrained_model, cache_dir, device)

        total_steps = len(train_dataloader) * self.num_epochs
        optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        global_step = 0
        total_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(self.num_epochs, desc='Epoch')
        if self.train_logs:
            tb_writer = SummaryWriter(logdir=os.path.join(self.train_logs, os.path.basename(self.output_dir)))
        else:
            tb_writer = None
        loss_queue = deque(maxlen=10)
        for epoch in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc='Iteration')
            for step, batch in enumerate(epoch_iterator):
                model.train()
                batch = tuple(t.to(device) for t in batch)
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[2]}
                outputs = model(**inputs)
                loss = outputs[0]
                loss.backward()
                total_loss += loss.item()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
                if global_step % self.logging_steps == 0:
                    last_loss = (total_loss - logging_loss) / self.logging_steps
                    loss_queue.append(last_loss)
                    if tb_writer:
                        tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar('loss', last_loss, global_step)
                    logging_loss = total_loss

            # slope-based early stopping
            if len(loss_queue) == loss_queue.maxlen:
                slope = calc_slope(loss_queue)
                if tb_writer:
                    tb_writer.add_scalar('slope', slope, global_step)
                if abs(slope) < 1e-2:
                    break

        if tb_writer:
            tb_writer.close()

        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training  # noqa
        model_to_save.save_pretrained(workdir)
        tokenizer.save_pretrained(workdir)

        return {
            'model_path': workdir,
            'batch_size': self.batch_size,
            'maxlen': self.maxlen,
            'pretrained_model': self.pretrained_model,
            'labels': self.labels
        }



